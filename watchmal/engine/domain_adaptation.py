import torch
import torch.nn as nn
from watchmal.engine.reconstruction import ReconstructionEngine
from watchmal.model.dann import GradientReversalLayer, GradientReversalLayerModule

# generic imports
import numpy as np
from datetime import timedelta
from datetime import datetime
from abc import ABC, abstractmethod
import logging

# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch.nn.parallel import DistributedDataParallel

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader
from watchmal.utils.logging_utils import CSVLog

log = logging.getLogger(__name__)

class DANNEngine(ReconstructionEngine):
    def __init__(self, truth_key, model, rank, gpu, dump_path, pretrained_model_path=None, domain_pre_train_epochs=2, domain_in_train_itrs=2):
        super().__init__(truth_key, model, rank, gpu, dump_path)
        self.grl = GradientReversalLayerModule()
        self.domain_criterion = None
        self.grl_scheduler = lambda step=0: 1.0
        self.pretrained_model_path = pretrained_model_path
        self.domain_pre_train_epochs_k = domain_pre_train_epochs
        self.domain_in_train_itrs_r = domain_in_train_itrs

        self.train_adv_log = CSVLog(self.dump_path + f"log_train_adv_{self.rank}.csv")
        self.val_adv_log = CSVLog(self.dump_path + "log_val_adv.csv")

    def configure_loss(self, loss_config):
        self.criterion = instantiate(loss_config['classification'])
        self.domain_criterion = instantiate(loss_config['domain'])

    def configure_grl_scheduler(self, grl_config):
        self.grl_scheduler = instantiate(grl_config)

    def configure_optimizers(self, optimizer_config):
        """Instantiate an optimizer from a hydra config."""
        # self.optimizer = instantiate(optimizer_config, params=self.module.parameters())
        self.optimizer = instantiate(optimizer_config, params=self.module.label_predictor.parameters())
        self.optimizer_adv = instantiate(optimizer_config, params=self.module.domain_classifier.parameters())

    # override from ReconstructionEngine to deal with two datasets
    def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed):
        """
        Set up data loaders from loaders hydra configs for the data config, and a list of data loader configs.

        Parameters
        ==========
        data_config
            Hydra config specifying dataset.
        loaders_config
            Hydra config specifying a list of dataloaders.
        is_distributed : bool
            Whether running in multiprocessing mode.
        seed : int
            Random seed to use to initialize dataloaders.
        """
        
        for data_name, data_config_item in data_config.items():
            for loader_name, loader_config in loaders_config.items():
                if data_name in loader_name:
                    print(f"Creating dataloader for {loader_name} with config {data_config_item}")
                    self.data_loaders[loader_name] = get_data_loader(**data_config_item, **loader_config, is_distributed=is_distributed, seed=seed)
                
    def get_synchronized_metrics(self, metric_dict):
        """
        Gathers metrics from multiple processes using pytorch distributed operations for DistributedDataParallel

        Parameters
        ==========
        metric_dict : dict of torch.Tensor
            Dictionary containing values that are tensor outputs of a single process.

        Returns
        =======
        global_metric_dict : dict
            Dictionary containing mean of tensor values gathered from all processes
        """
        global_metric_dict = {}
        for name, tensor in zip(metric_dict.keys(), metric_dict.values()):
            if self.is_distributed:
                torch.distributed.reduce(tensor, 0)
                if self.rank == 0:
                    global_metric_dict[name] = tensor.item()/self.n_gpus
            else:
                if isinstance(tensor, torch.Tensor):
                    global_metric_dict[name] = tensor.item()
                else:
                    global_metric_dict[name] = tensor
        return global_metric_dict
    
    def load_pretrained_model(self):
        if self.pretrained_model_path is not None:
            checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
            
            state_dict = checkpoint['state_dict']
            print("pretrained: ", state_dict.keys())
            print("new model:  ", self.model.label_predictor.state_dict().keys())
            state_dict = {k.replace('feature_extractor.', ''): v for k, v in state_dict.items()}
            state_dict.pop("fc.weight", None)
            state_dict.pop("fc.bias", None)

            self.model.label_predictor.feature_extractor.load_state_dict(state_dict, strict=False)
            self.model.label_predictor.feature_extractor.fc = nn.Identity()
            log.info(f"Loaded pretrained model from {self.pretrained_model_path}")

    # USUALLY, foward() in regression_dann.py or classification_dann.py is called.
    def forward(self, train=True):
        features = self.model.feature_extractor(self.data)
        features_source = self.model.feature_extractor(self.source_data)

        # TODO: forward pass different for training f or r.
        # for now this works only for training f.
        
        if train:
            with torch.set_grad_enabled(train):
                # class output (predicted label for main classification task (mu, e, pi etc)) from source data.
                # this is the output of the model, which is the label/regression prediction from the class predictor.
                class_output = self.model.class_classifier(features_source)
                lambda_param = 1 # self.grl_scheduler.get_lambda()

                reverse_features = self.grl(features)
                domain_output = self.model.domain_classifier(reverse_features)
                
                class_loss = self.criterion(class_output, self.target)
                domain_labels = torch.cat([torch.zeros(len(self.source_data)), torch.ones(len(self.target_data))]).to(self.device)

                domain_loss = self.domain_criterion(domain_output, domain_labels.long())
                
                self.loss = class_loss - lambda_param * domain_loss
                
                metrics = {
                    'loss': self.loss.item(),
                    'class_loss': class_loss.item(),
                    'domain_loss': domain_loss.item()
                }
        else:
            with torch.set_grad_enabled(train):
                class_output = self.model.class_classifier(features)
                self.loss = self.criterion(class_output, self.target)
                metrics = {'loss': self.loss.item()}

        return {'class_output': class_output}, metrics

    def train(self, epochs=0, val_interval=20, num_val_batches=4, checkpointing=False, save_interval=None):
        if self.rank == 0:
            log.info(f"Training DANN for {epochs} epochs with {num_val_batches}-batch validation each {val_interval} iterations")
        
        self.load_pretrained_model()

        self.model.train()
        self.epoch = 0
        self.iteration = 0
        self.step = 0
        self.best_validation_loss = np.inf

        self.cum_itr_adv = 0
        
        source_train_loader = self.data_loaders["source_train"]
        target_train_loader = self.data_loaders["target_train"]

        source_val_iter = iter(self.data_loaders["source_validation"])
        target_val_iter = iter(self.data_loaders["target_validation"])
        

        start_time = datetime.now()
        step_time = start_time
        epoch_start_time = start_time
        
        self.set_requires_grads_for_models(False, True)
        steps_per_epoch = min(len(source_train_loader), len(target_train_loader))
        
        for k in range(self.domain_pre_train_epochs_k):
            print(f"Training adversary ONLY: {k}-th epoch")
            source_iter = iter(source_train_loader)
            target_iter = iter(target_train_loader)
            steps_per_epoch = steps_per_epoch 
            self.train_adversary(source_train_loader, target_train_loader,
                                 iterations=steps_per_epoch,
                                 print_interval=5, print_flag=False)

        
        train_f = True

        for self.epoch in range(epochs):
            if self.rank == 0:
                if self.epoch > 0:
                    log.info(f"Epoch {self.epoch} completed in {datetime.now() - epoch_start_time}")
                    epoch_start_time = datetime.now()
                log.info(f"Epoch {self.epoch+1} starting at {datetime.now()}")

            self.step = 0
            if self.is_distributed:
                source_train_loader.sampler.set_epoch(self.epoch)
                target_train_loader.sampler.set_epoch(self.epoch)

            steps_per_epoch = min(len(source_train_loader), len(target_train_loader))
            source_iter = iter(source_train_loader)
            target_iter = iter(target_train_loader)

            for self.step in range(steps_per_epoch):
                self.set_requires_grads_for_models(False, True)
                
                r_steps = self.domain_in_train_itrs_r
                self.train_adversary(source_train_loader, target_train_loader,
                                     iterations=r_steps,
                                     print_interval=5, print_flag=False)

                self.set_requires_grads_for_models(True, False)

                if not train_f:
                    self.set_requires_grads_for_models(False, False)

                
                
                source_iter = iter(source_train_loader)
                target_iter = iter(target_train_loader)

                source_data = next(source_iter)
                target_data = next(target_iter)

                self.source_data = source_data['data'].to(self.device)
                self.target_data = target_data['data'].to(self.device)
                # TODO: pretty confusing this here.
                # data is from two datasets
                self.data = torch.cat([self.source_data, self.target_data])
                # target is from source dataset only.
                self.target = source_data[self.truth_key].to(self.device)

                outputs, metrics = self.forward(train_f)
                if train_f:
                    self.backward()
                
                self.step += 1
                self.iteration += 1

                log_entries = {"iteration": self.iteration, "epoch": self.epoch, **metrics}
                self.train_log.log(log_entries)
                
                # should be refactored.
                if self.iteration % val_interval == 0:
                    if self.rank == 0:
                        previous_step_time = step_time
                        step_time = datetime.now()
                        average_step_time = (step_time - previous_step_time)/val_interval
                        print(f"Iteration {self.iteration}, Epoch {self.epoch+1}/{epochs}, Step {self.step}/{steps_per_epoch}"
                              f" Step time {average_step_time},"
                              f" Epoch time {step_time-epoch_start_time}"
                              f" Total time {step_time-start_time}")
                        print(f"  Training   {', '.join(f'{k}: {v:.5g}' for k, v in metrics.items())}")
                    self.validate(source_val_iter, target_val_iter,  num_val_batches, checkpointing)

            if self.rank == 0 and (save_interval is not None) and ((self.epoch+1) % save_interval == 0):
                self.save_state(suffix=f'_epoch_{self.epoch+1}')

            if self.scheduler is not None:
                self.scheduler.step()
                print(f"SCHEDULER, LR: {self.scheduler.get_last_lr()}")

            # self.grl_scheduler.step()

        self.train_log.close()
        if self.rank == 0:
            log.info(f"Epoch {self.epoch} completed in {datetime.now() - epoch_start_time}")
            log.info(f"Training {epochs} epochs completed in {datetime.now()-start_time}")
            self.val_log.close()
    
    def set_requires_grads_for_models(self, is_train_f: bool, is_train_r: bool):
        for param in self.model.label_predictor.parameters():
            param.requires_grad = is_train_f
        for param in self.model.domain_classifier.parameters():
            param.requires_grad = is_train_r

    def train_adversary(self, source_train_loader, target_train_loader, iterations = 2, print_interval=5, print_flag=False):
        # print(f"Training domain classifier {k}-th epoch")
        source_iter = iter(source_train_loader)
        target_iter = iter(target_train_loader)
        
        for step in range(iterations):
            source_data = next(source_iter)
            target_data = next(target_iter)

            if False:
                # REQUIRES: dataloaders to have SequentialSampler
                print("Source data sample:", source_data['data'][1])
                print("Target data sample:", target_data['data'][1])
                difference = source_data['data'][1] - target_data['data'][1]
                print("Difference:", difference)
                print("difference magnitude:", difference.norm())

            self.source_data = source_data['data'].to(self.device)
            self.target_data = target_data['data'].to(self.device)
            self.data = torch.cat([self.source_data, self.target_data])
            self.target = source_data[self.truth_key].to(self.device)

            _, domain_output = self.model(self.data, apply_grl=False)
            domain_labels = torch.cat([torch.zeros(len(self.source_data)), torch.ones(len(self.target_data))]).to(self.device)
            
            domain_labels = domain_labels.view(-1, 1).float()
            
            domain_loss = self.domain_criterion(domain_output, domain_labels)

            self.optimizer_adv.zero_grad()
            domain_loss.backward()
            self.optimizer_adv.step()

            log_entries = {"iteration": self.cum_itr_adv, "epoch_main": self.epoch, "domain_loss": domain_loss.item()}
            self.train_adv_log.log(log_entries)

            self.cum_itr_adv += 1


            if step % print_interval == 0 and print_flag:
                print(f"Iteration {step}, Step {step}")
                print(f"  Training Loss (adversary ONLY) {domain_loss.item()}")
                # print(f"  Domain output {domain_output}")
                # print(f"  Params {list(self.model.domain_classifier.parameters())[0]}")
                # print_gradients(self.model.domain_classifier)

    def validate(self, source_val_iter, target_val_iter, num_val_batches, checkpointing):
        self.model.eval()
        val_metrics = None

        for val_batch in range(num_val_batches):
            try:
                source_val_data = next(source_val_iter)
            except StopIteration:
                del source_val_iter
                if self.is_distributed:
                    self.data_loaders["source_validation"].sampler.set_epoch(self.data_loaders["source_validation"].sampler.epoch+1)
                source_val_iter = iter(self.data_loaders["source_validation"])
                source_val_data = next(source_val_iter)
            
            try:
                target_val_data = next(target_val_iter)
            except StopIteration:
                if self.is_distributed:
                    self.data_loaders["target_validation"].sampler.set_epoch(
                        self.data_loaders["target_validation"].sampler.epoch + 1)
                target_val_iter = iter(self.data_loaders["target_validation"])
                target_val_data = next(target_val_iter)

            # confusing thing: target and target_data are different things.
            self.source_data = source_val_data['data'].to(self.device)
            self.target_data = target_val_data['data'].to(self.device)
            self.data = torch.cat([self.source_data, self.target_data])
            self.target = source_val_data[self.truth_key].to(self.device)

            # Forward pass
            outputs, metrics = self.forward(False)

            if val_metrics is None:
                val_metrics = metrics
            else:
                for k, v in metrics.items():
                    val_metrics[k] += v

        val_metrics = {k: v/num_val_batches for k, v in val_metrics.items()}
        val_metrics = self.get_synchronized_metrics(val_metrics)
        
        if self.rank == 0:
            log_entries = {"iteration": self.iteration, "epoch": self.epoch, **val_metrics, "saved_best": False}
            print(f"  Validation {', '.join(f'{k}: {v:.5g}' for k, v in val_metrics.items())}", end="")
            if val_metrics["loss"] < self.best_validation_loss:
                print(" ... Best validation loss so far!")
                self.best_validation_loss = val_metrics["loss"]
                self.save_state(suffix="_BEST")
                log_entries["saved_best"] = True
            else:
                print("")
            if checkpointing:
                self.save_state()
            self.val_log.log(log_entries)

        self.model.train()

def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.norm()}")
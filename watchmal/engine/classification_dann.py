import torch

# from watchmal.engine.reconstruction import ReconstructionEngine
from watchmal.engine.domain_adaptation import DANNEngine


class DANNClassifierEngine(DANNEngine):
    """Engine for performing training or evaluation for a classification network."""
    def __init__(self, truth_key, model, rank, gpu, dump_path, label_set=None, pretrained_model_path=None, domain_pre_train_epochs=2, domain_in_train_itrs=2):
        """
        Parameters
        ==========
        truth_key : string
            Name of the key for the target labels in the dictionary returned by the dataloader
        model
            `nn.module` object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        gpu : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        label_set : sequence
            The set of possible labels to classify (if None, which is the default, then class labels in the data must be
            0 to N).
        """
        # create the directory for saving the log and dump files
        super().__init__(truth_key, model, rank, gpu, dump_path, pretrained_model_path, domain_pre_train_epochs, domain_in_train_itrs)
        self.softmax = torch.nn.Softmax(dim=1)
        self.label_set = label_set

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
        super().configure_data_loaders(data_config, loaders_config, is_distributed, seed)
        if self.label_set is not None:
            for name in loaders_config.keys():
                self.data_loaders[name].dataset.map_labels(self.label_set)

    def forward(self, train=True):
        _, features = self.module.label_predictor(self.data)
        features.requires_grad_(True)


        with torch.set_grad_enabled(train):
            # class loss
            class_output, features_source = self.module.label_predictor(self.source_data)
            features_source.requires_grad_(True)
            class_loss = self.criterion(class_output, self.target)

            # class accuracy
            softmax = self.softmax(class_output)
            predicted_labels = torch.argmax(class_output, dim=-1)
            class_accuracy = (predicted_labels == self.target).sum() / float(predicted_labels.nelement())

            # domain loss
            reverse_features = self.grl(features)
            domain_output = self.module.domain_classifier(reverse_features)            
            domain_labels = torch.cat([torch.zeros(len(self.source_data)), torch.ones(len(self.target_data))]).to(self.device)
            domain_labels = domain_labels.view(-1, 1).float()
            domain_loss = self.domain_criterion(domain_output, domain_labels)

            # domain accuracy
            predicted_domains = (domain_output > 0).float() 
            # print("domain labels", domain_labels)
            # print("domain output", domain_output)
            # print("predicted domains", predicted_domains)
            domain_accuracy = (predicted_domains == domain_labels).sum() / float(predicted_domains.nelement())
            # print("domain accuracy", domain_accuracy)

            # print(domain_output.grad_fn)

            # total loss
            lambda_param = 1
            self.loss = class_loss - lambda_param * domain_loss

            outputs = {'softmax': softmax}
            
            metrics = {
                'loss': self.loss.item(),
                'class_loss': class_loss.item(),
                'class_accuracy': class_accuracy.item(),
                'domain_loss': domain_loss.item(),
                'domain_accuracy': domain_accuracy.item(),
            }
        return outputs, metrics
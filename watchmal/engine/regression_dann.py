import torch

# from watchmal.engine.reconstruction import ReconstructionEngine
from watchmal.engine.domain_adaptation import DANNEngine


class DANNClassifierEngine(DANNEngine):
    """Engine for performing training or evaluation for a classification network."""
    def __init__(self, truth_key, model, rank, gpu, dump_path, output_center=0, output_scale=1, pretrained_model_path=None):
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
        super().__init__(truth_key, model, rank, gpu, dump_path, pretrained_model_path)
        self.output_center = output_center
        self.output_scale = output_scale

    # def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed):
    #     """
    #     Set up data loaders from loaders hydra configs for the data config, and a list of data loader configs.

    #     Parameters
    #     ==========
    #     data_config
    #         Hydra config specifying dataset.
    #     loaders_config
    #         Hydra config specifying a list of dataloaders.
    #     is_distributed : bool
    #         Whether running in multiprocessing mode.
    #     seed : int
    #         Random seed to use to initialize dataloaders.
    #     """
    #     super().configure_data_loaders(data_config, loaders_config, is_distributed, seed)
    #     if self.label_set is not None:
    #         for name in loaders_config.keys():
    #             self.data_loaders[name].dataset.map_labels(self.label_set)
    #     # if self.label_set is not None:
    #     #     for name in loaders_config.keys():
    #     #         if 'target' not in name:
    #     #             self.data_loaders[name].dataset.map_labels(self.label_set)

    def forward(self, train=True):
        # features are G_f([x_source, x_target]) == representation of source data and target data (concat)
        # features = self.model.feature_extractor(self.data)
        # # features from soure data, features from target data
        # features_source = self.model.feature_extractor(self.source_data)
        
        # DDP version of above code
        features = self.module.feature_extractor(self.data)
        features_source = self.module.feature_extractor(self.source_data)

        
        with torch.set_grad_enabled(train):
            class_output = self.module.class_classifier(features_source)
            
            lambda_param = 0.3 # self.grl_scheduler.get_lambda()

            model_out = self.grl(features)
            domain_output = self.module.domain_classifier(model_out)

            scaled_model_out = self.scale_values(model_out).float()
            scaled_target = self.scale_values(self.target).float()
            class_loss = self.criterion(scaled_model_out, scaled_target)
            domain_labels = torch.cat([torch.zeros(len(self.source_data)), torch.ones(len(self.target_data))]).to(self.device)

            domain_loss = self.domain_criterion(domain_output, domain_labels.long())
            predicted_domains = torch.argmax(domain_output, dim=-1)
            domain_accuracy = (predicted_domains == domain_labels).sum() / float(predicted_domains.nelement())
            
            self.loss = class_loss - lambda_param * domain_loss

            outputs = {"predicted_"+self.truth_key: model_out}

            metrics = {
                'loss': self.loss.item(),
                'class_loss': class_loss.item(),
                'domain_loss': domain_loss.item(),
                'domain_accuracy': domain_accuracy.item(),
            }
        return outputs, metrics
    
    def scale_values(self, data):
        scaled = (data - self.output_center) / self.output_scale
        return scaled
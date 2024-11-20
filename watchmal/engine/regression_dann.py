import torch

# from watchmal.engine.reconstruction import ReconstructionEngine
from watchmal.engine.domain_adaptation import DANNEngine


class DANNRegressionEngine(DANNEngine):
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

    def forward(self, train=True):
        # training f with GRL
        label_output, _ = self.module.label_predictor(self.source_data)

        _, features_all = self.module.label_predictor(self.data)
        features_all = self.module.grl(features_all)
        
        with torch.set_grad_enabled(train):
     
            lambda_param = 0.3

            
            # compute f loss (predictor loss) using data from source dataset
            scaled_label_output = self.scale_values(label_output).float()
            scaled_target = self.scale_values(self.target).float()
            f_loss = self.criterion(scaled_label_output, scaled_target)
            
            # compute r loss (domain loss) using data from both source and target datasets
            domain_output = self.module.domain_classifier(features_all)
            # print('domain_output', domain_output)
            domain_labels = torch.cat([torch.zeros(len(self.source_data)), torch.ones(len(self.target_data))]).to(self.device)
            domain_labels = domain_labels.view(-1, 1).float()    
            domain_loss = self.domain_criterion(domain_output, domain_labels)
            
            # predicted_domains = torch.argmax(domain_output, dim=-1)
            predicted_domains = (domain_output > 0).float()
            domain_accuracy = (predicted_domains == domain_labels).sum() / float(predicted_domains.nelement())
            
            # print("predicted_domains", predicted_domains)
            # print("domain acc from forward", domain_accuracy)

            self.loss = f_loss - lambda_param * domain_loss
            outputs = {"predicted_"+self.truth_key: label_output}
            metrics = {
                'loss': self.loss.item(),
                'class_loss': f_loss.item(),
                'domain_loss': domain_loss.item(),
                'domain_accuracy': domain_accuracy.item(),
            }
        return outputs, metrics
    
    def scale_values(self, data):
        scaled = (data - self.output_center) / self.output_scale
        return scaled
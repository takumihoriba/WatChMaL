from watchmal.engine.reconstruction import ReconstructionEngine
import torch
import torch.nn as nn
from watchmal.utils.logging_utils import CSVLog
# hydra imports
from hydra.utils import instantiate

class DANNEngine(ReconstructionEngine):
    def __init__(self, model, rank, gpu, dump_path):
        super().__init__('label', model, rank, gpu, dump_path)
        self.domain_criterion = nn.BCEWithLogitsLoss()
        self.alpha = 0  # Initialize alpha for gradient reversal

    def configure_loss(self, loss_config):
        self.class_criterion = instantiate(loss_config)

    def forward(self, train=True):
        class_output, domain_output = self.model(self.data, self.alpha)
        class_loss = self.class_criterion(class_output, self.target)
        domain_loss = self.domain_criterion(domain_output, self.data['domain'])
        self.loss = class_loss + domain_loss

        metrics = {
            "loss": self.loss,
            "class_loss": class_loss,
            "domain_loss": domain_loss
        }

        outputs = {
            "class_output": class_output,
            "domain_output": domain_output
        }

        return outputs, metrics

    def train(self, epochs=0, val_interval=20, num_val_batches=4, checkpointing=False, save_interval=None):
        # Update alpha based on training progress
        p = float(self.iteration) / epochs
        self.alpha = 2. / (1. + np.exp(-10 * p)) - 1

        super().train(epochs, val_interval, num_val_batches, checkpointing, save_interval)

    def validate(self, val_iter, num_val_batches, checkpointing):
        # Set a fixed alpha for validation
        self.alpha = 1.0
        super().validate(val_iter, num_val_batches, checkpointing)

    def evaluate(self, report_interval=20):
        # Set a fixed alpha for evaluation
        self.alpha = 1.0
        super().evaluate(report_interval)
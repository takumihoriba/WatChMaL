import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayerModule(nn.Module):
    def __init__(self, lambda_grad=1.0):
        super(GradientReversalLayerModule, self).__init__()
        self.lambda_grad = lambda_grad

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_grad)

class DANNModel(nn.Module):
    def __init__(self, label_predictor, domain_classifier):
        super(DANNModel, self).__init__()
        self.label_predictor = label_predictor
        self.domain_classifier = nn.Sequential(
            domain_classifier
        )
        self.grl = GradientReversalLayerModule()

    def forward(self, x, alpha=0, apply_grl=True):
        label_output, features = self.label_predictor(x)
        if apply_grl:
            reverse_features = self.grl(features)
        else:
            reverse_features = features
        domain_output = self.domain_classifier(reverse_features)
        return (label_output, domain_output)

class LabelPredictor(nn.Module):
    def __init__(self, feature_extractor, label_pred):
        super(LabelPredictor, self).__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_pred

    def forward(self, x):
        features = self.feature_extractor(x)
        label_output = self.label_predictor(features)
        return label_output, features


class FlexibleNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, output_dim=2, dropout_p=0.2):
        """
        Initialize the FlexibleNNClassifier.
        
        Parameters:
        - input_dim: int, the dimension of the input features.
        - hidden_dims: list of int, the dimensions of the hidden layers.
        - output_dim: int, the dimension of the output layer.
        - dropout_p: float, the dropout probability.
        """
        super(FlexibleNNClassifier, self).__init__()
        
        layers = []
        
        # fisrt layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_p))
        
        # subsequent hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
        
        # output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)




def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.norm()}")
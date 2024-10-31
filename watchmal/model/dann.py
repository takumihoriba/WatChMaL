import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        # clone or view as
        # return x.clone()
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # removed * ctx.alpha
        return grad_output.neg(), None
    
class GradientReversalLayerModule(nn.Module):
    def __init__(self, lambda_grad=1.0):
        super(GradientReversalLayerModule, self).__init__()
        self.lambda_grad = lambda_grad

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_grad)

class DANNModel(nn.Module):
    def __init__(self, feature_extractor, class_classifier, domain_classifier):
        super(DANNModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.class_classifier = class_classifier
        self.domain_classifier = nn.Sequential(
            domain_classifier
            # ,nn.Flatten(0, -1)  # This will flatten the output to 1D
        )
        self.grl = GradientReversalLayerModule()

    def forward(self, x, alpha=0):
        features = self.feature_extractor(x)
        class_output = self.class_classifier(features)
        reverse_features = self.grl(features)
        domain_output = self.domain_classifier(reverse_features)
        return (class_output, domain_output)

class SimpleNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, dropout_p=0.5):
        super(SimpleNNClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    
class FlexibleNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=2, dropout_p=0.2):
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
    

class PreTrainedFeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super(PreTrainedFeatureExtractor, self).__init__()


        self.base_model = pretrained_model
        
        # Remove the last fully connected layer
        # For ResNet, this is typically `fc`
        self.base_model.fc = nn.Identity()  # Replace with an identity function

    def forward(self, x):
        # Forward pass through the feature extractor
        return self.base_model(x)

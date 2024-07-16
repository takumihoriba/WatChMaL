import torch.nn as nn
from torch.autograd import Function

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DANNModel(nn.Module):
    def __init__(self, feature_extractor, class_classifier, domain_classifier):
        super(DANNModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.class_classifier = class_classifier
        self.domain_classifier = domain_classifier
        self.grl = GradientReversal.apply

    def forward(self, x, alpha):
        features = self.feature_extractor(x)
        class_output = self.class_classifier(features)
        reverse_features = self.grl(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        return class_output, domain_output
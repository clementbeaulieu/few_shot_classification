from collections import OrderedDict

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.checkpoint as cp

from . import encoders
from . import classifiers
from .modules import get_child_dict, Module, BatchNorm2d

from .models import register

class PrototypicalNetwork(nn.Module):

    # default distance logits
    def euclidean_distance(x, y):
        n = x.shape[0]
        m = y.shape[0]
        x = x.unsqueeze(1).expand(n, m, -1)
        y = y.unsqueeze(0).expand(n, m, -1)
        return torch.pow(x - y, 2).sum(2)

    def __init__(self, encoder, classifier, distance=euclidean_distance):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.distance = distance

    def forward(self, x):
        

@register('prototypical_net')
def maml(enc, clf):
    return PrototypicalNetwork(enc, clf)
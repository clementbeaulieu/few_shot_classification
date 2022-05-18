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

    def prototype(self, x_shot, x_query, y_shot, episode):
        """Args:
          x_shot (float tensor, [n_way * n_shot, C, H, W]): per episode support sets.
          x_query (float tensor, [n_way * n_query, C, H, W]): per episode query sets.
            (T: transforms, C: channels, H: height, W: width)
          y_shot (int tensor, [n_way * n_shot]): per episode support set labels.

        Returns:
          prototype points for x_query
        """
        n_way = x_shot.size(0)
        n_shot = x_query.size(1)
        assert x_shot.size(0) == x_query.size(0)

        


    
    def forward(self, x_shot, x_query, y_shot, inner_args, meta_train):
        """
        Args:
          x_shot (float tensor, [n_episode, n_way * n_shot, C, H, W]): support sets.
          x_query (float tensor, [n_episode, n_way * n_query, C, H, W]): query sets.
            (T: transforms, C: channels, H: height, W: width)
          y_shot (int tensor, [n_episode, n_way * n_shot]): support set labels.
          inner_args (dict, optional): inner-loop hyperparameters.
          meta_train (bool): if True, the model is in meta-training.

        Returns:
          logits (float tensor, [n_episode, n_way * n_shot, n_way]): predicted logits.
        """
        assert self.encoder is not None
        assert self.classifier is not None
        assert x_shot.dim() == 5 and x_query.dim() == 5
        assert x_shot.size(0) == x_query.size(0)

        # a dictionary of parameters that will be updated in the inner loop
        params = OrderedDict(self.named_parameters())
        for name in list(params.keys()):
            if not params[name].requires_grad or \
                    any(s in name for s in inner_args['frozen'] + ['temp']):
                params.pop(name)

        logits = []
        for ep in range(x_shot.size(0)):
            # inner-loop training
            self.train()
            if not meta_train:
                for m in self.modules():
                    if isinstance(m, BatchNorm2d) and not m.is_episodic():
                        m.eval()
            updated_params = self._adapt(
                x_shot[ep], y_shot[ep], params, ep, inner_args, meta_train)
            # inner-loop validation
            with torch.set_grad_enabled(meta_train):
                self.eval()
                logits_ep = self._inner_forward(
                    x_query[ep], updated_params, ep)
            logits.append(logits_ep)

        self.train(meta_train)
        logits = torch.stack(logits)
        return logits


@register('prototypical_net')
def maml(enc, clf):
    return PrototypicalNetwork(enc, clf)
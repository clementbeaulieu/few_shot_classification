import torch
import torch.nn as nn
import torch.nn.functional as F

def get_criterion(args):
    return{
        'maml': F.cross_entropy,
        'prototypical_net': prototypical_loss
    }[args.arch]


class PrototypicalLoss(nn.Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)

def euclidean_distance(x, y):
    n = x.shape[0]
    m = y.shape[0]
    x = x.unsqueeze(1).expand(n, m, -1)
    y = y.unsqueeze(0).expand(n, m, -1)
    return torch.pow(x - y, 2).sum(2)

def prototypical_loss(input, target, n_support):
    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target)
    n_classes = len(classes)
    n_query = target.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])
    query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input[query_idxs]
    dists = euclidean_distance(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val,  acc_val
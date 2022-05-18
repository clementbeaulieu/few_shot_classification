from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.checkpoint as cp

from . import encoders
from . import classifiers

models = {}

def register(arch):
    def decorator(cls):
        models[arch] = cls
        return cls
    return decorator

def make(arch, enc_name, enc_args, clf_name, clf_args):
    """
    Initializes a random meta model.

    Args:
      enc_name (str): name of the encoder (e.g., 'resnet12').
      enc_args (dict): arguments for the encoder.
      clf_name (str): name of the classifier (e.g., 'meta-nn').
      clf_args (dict): arguments for the classifier.

    Returns:
      model (<name>): a meta classifier with encoder enc and classifier clf for model <name>.
    """
    if arch is None:
        return None

    enc = encoders.make(enc_name, **enc_args)
    clf_args['in_dim'] = enc.get_out_dim()
    clf = classifiers.make(clf_name, **clf_args)
    model = models[arch](enc, clf)

    if torch.cuda.is_available():
        model.cuda()

    return model

def load(ckpt, load_clf=False, clf_name=None, clf_args=None):
    """
    Initializes a meta model with a pre-trained encoder.

    Args:
      ckpt (dict): a checkpoint from which a pre-trained encoder is restored.
      load_clf (bool, optional): if True, loads a pre-trained classifier.
        Default: False (in which case the classifier is randomly initialized)
      clf_name (str, optional): name of the classifier (e.g., 'meta-nn')
      clf_args (dict, optional): arguments for the classifier.
      (The last two arguments are ignored if load_clf=True.)

    Returns:
      model (<name>): a meta model with a pre-trained encoder.
    """
    enc = encoders.load(ckpt)
    if load_clf:
        clf = classifiers.load(ckpt)
    else:
        if clf_name is None and clf_args is None:
            clf = classifiers.make(ckpt['classifier'], **ckpt['classifier_args'])
        else:
            clf_args['in_dim'] = enc.get_out_dim()
            clf = classifiers.make(clf_name, **clf_args)
    arch = ckpt['arch']
    model = models[arch](enc, clf)
    return model

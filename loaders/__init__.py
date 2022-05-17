import torch
from loaders.mini_imagenet import MiniImageNetLoader, MetaMiniImageNetLoader

def get_loader(args):
    """get_loader
    :param name:
    """
    return {
        'mini_imagenet' : MiniImageNetLoader,
        'meta_mini_imagenet' : MetaMiniImageNetLoader,
    }[args.dataset]
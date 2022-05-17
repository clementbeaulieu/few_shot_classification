import random
import sys
import os

import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import datasets
import models
import utils

from loaders import get_loader
from args import parse_args

def main():
    global args
    if len(sys.argv) > 1:
        args = parse_args()
        print('----- Experiments parameters -----')
        for k, v in args.__dict__.items():
            print(k, ':', v)
    else:
        print('Please provide some parameters for the current experiment. Check-out args.py for more info!')
        sys.exit()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)

    # init random seeds
    utils.setup_env(args)

    ##### DataLoaders #####

    # init data loaders
    if args.meta:
        args.dataset = 'meta_' + args.dataset #to get appropriate loader if meta-training requested

    loader = get_loader(args)

    ##### test loader
    if args.meta:
        if args.test:
            test_set = loader(data_dir=args.data_dir, split='test', image_size=args.train_image_size, normalization=args.train_normalization, transform=args.train_transform, val_transform=args.val_transform, n_batch=args.train_n_batch, n_episode=args.train_n_episode, n_way=args.train_n_way, n_shot=args.train_n_shot, n_query=args.train_n_query)
            meta_collate_fn = test_set.meta_collate_fn
            utils.log('meta-test set: {} (x{}), {}'.format(test_set[0][0].shape, len(test_set), test_set.n_classes))
            test_loader = DataLoader(test_set, args.train_n_episode, collate_fn=meta_collate_fn, num_workers=args.workers, pin_memory=True)

    ##### Model #####

    assert args.load in ['best', 'latest']
    ckpt = torch.load(os.path.join(args.log_dir, args.load + '.pth'))
    inner_args = utils.config_inner_args(config.get('inner_args'))
    model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))

    if args.efficient:
        model.go_efficient()

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    ##### Evaluation #####

    model.eval()
    aves_va = utils.AverageMeter()
    va_lst = []

    for epoch in range(1, args.epoch + 1):
        for data in tqdm(test_loader, leave=False):
            x_shot, x_query, y_shot, y_query = data
            x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
            x_query, y_query = x_query.cuda(), y_query.cuda()

            if inner_args['reset_classifier']:
                if config.get('_parallel'):
                    model.module.reset_classifier()
                else:
                    model.reset_classifier()

            logits = model(x_shot, x_query, y_shot,
                           inner_args, meta_train=False)
            logits = logits.view(-1, args.test_n_way)
            labels = y_query.view(-1)

            pred = torch.argmax(logits, dim=1)
            acc = utils.compute_acc(pred, labels)
            aves_va.update(acc, 1)
            va_lst.append(acc)

        print('test epoch {}: acc={:.2f} +- {:.2f} (%)'.format(
            epoch, aves_va.item() * 100,
            utils.mean_confidence_interval(va_lst) * 100))


if __name__ == '__main__':
    main()
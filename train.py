import argparse
import os
import sys
import random
from collections import OrderedDict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.optimizers as optimizers

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

    '''if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)

    # initialize seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)'''

    # checkpoints
    utils.ensure_path(args.log_dir)
    utils.set_log_path(args.log_dir)
    writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard'))
    yaml.dump(config, open(os.path.join(args.log_dir, 'config.yaml'), 'w'))

    ##### Dataset #####

    ##### meta-train
    ###TODO
    train_set = datasets.make('meta-'+ args.dataset, **config['train'],root_path=args.data_dir)
    utils.log('meta-train set: {} (x{}), {}'.format(train_set[0][0].shape, len(train_set), train_set.n_classes))
    train_loader = DataLoader(train_set, config['train']['n_episode'], collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)

    ##### meta-val
    eval_val = False
    if config.get('val'):
        eval_val = True
        val_set = datasets.make('meta-'+ args.dataset, **config['val'],root_path=args.data_dir)
        utils.log('meta-val set: {} (x{}), {}'.format(val_set[0][0].shape, len(val_set), val_set.n_classes))
        val_loader = DataLoader(val_set, config['val']['n_episode'], collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)

    ##### Model and Optimizer #####

    inner_args = utils.config_inner_args(config.get('inner_args'))
    if config.get('load'):
        ckpt = torch.load(config['load'])
        config['encoder'] = ckpt['encoder']
        config['encoder_args'] = ckpt['encoder_args']
        config['classifier'] = ckpt['classifier']
        config['classifier_args'] = ckpt['classifier_args']
        model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
        optimizer, lr_scheduler = optimizers.load(ckpt, model.parameters())
        start_epoch = ckpt['training']['epoch'] + 1
        max_va = ckpt['training']['max_va']
    else:
        config['encoder_args'] = config.get('encoder_args') or dict()
        config['classifier_args'] = config.get('classifier_args') or dict()
        config['encoder_args']['bn_args']['n_episode'] = config['train']['n_episode']
        config['classifier_args']['n_way'] = config['train']['n_way']
        model = models.make(config['encoder'], config['encoder_args'],
                            config['classifier'], config['classifier_args'])
        optimizer, lr_scheduler = optimizers.make(
            config['optimizer'], model.parameters(), **config['optimizer_args'])
        start_epoch = 1
        max_va = 0.

    if args.efficient:
        model.go_efficient()

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

    ##### Training and evaluation #####

    # 'tl': meta-train loss
    # 'ta': meta-train accuracy
    # 'vl': meta-val loss
    # 'va': meta-val accuracy
    aves_keys = ['tl', 'ta', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    for epoch in range(start_epoch, config['epoch'] + 1):
        timer_epoch.start()
        aves = {k: utils.AverageMeter() for k in aves_keys}

        # meta-train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        np.random.seed(epoch)

        for data in tqdm(train_loader, desc='meta-train', leave=False):
            x_shot, x_query, y_shot, y_query = data
            x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
            x_query, y_query = x_query.cuda(), y_query.cuda()

            if inner_args['reset_classifier']:
                if config.get('_parallel'):
                    model.module.reset_classifier()
                else:
                    model.reset_classifier()

            logits = model(x_shot, x_query, y_shot,
                           inner_args, meta_train=True)
            logits = logits.flatten(0, 1)
            labels = y_query.flatten()

            pred = torch.argmax(logits, dim=-1)
            acc = utils.compute_acc(pred, labels)
            loss = F.cross_entropy(logits, labels)
            aves['tl'].update(loss.item(), 1)
            aves['ta'].update(acc, 1)

            optimizer.zero_grad()
            loss.backward()
            for param in optimizer.param_groups[0]['params']:
                nn.utils.clip_grad_value_(param, 10)
            optimizer.step()

        # meta-val
        if eval_val:
            model.eval()
            np.random.seed(0)

            for data in tqdm(val_loader, desc='meta-val', leave=False):
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
                logits = logits.flatten(0, 1)
                labels = y_query.flatten()

                pred = torch.argmax(logits, dim=-1)
                acc = utils.compute_acc(pred, labels)
                loss = F.cross_entropy(logits, labels)
                aves['vl'].update(loss.item(), 1)
                aves['va'].update(acc, 1)

        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, avg in aves.items():
            aves[k] = avg.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.end())
        t_elapsed = utils.time_str(timer_elapsed.end())
        t_estimate = utils.time_str(timer_elapsed.end() /
                                    (epoch - start_epoch + 1) * (config['epoch'] - start_epoch + 1))

        # formats output
        log_str = 'epoch {}, meta-train {:.4f}|{:.4f}'.format(
            str(epoch), aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'meta-train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'meta-train': aves['ta']}, epoch)

        if eval_val:
            log_str += ', meta-val {:.4f}|{:.4f}'.format(
                aves['vl'], aves['va'])
            writer.add_scalars('loss', {'meta-val': aves['vl']}, epoch)
            writer.add_scalars('acc', {'meta-val': aves['va']}, epoch)

        log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
        utils.log(log_str)

        # saves model and meta-data
        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'max_va': max(max_va, aves['va']),

            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
            if lr_scheduler is not None else None,
        }
        ckpt = {
            'file': __file__,
            'config': config,

            'encoder': config['encoder'],
            'encoder_args': config['encoder_args'],
            'encoder_state_dict': model_.encoder.state_dict(),

            'classifier': config['classifier'],
            'classifier_args': config['classifier_args'],
            'classifier_state_dict': model_.classifier.state_dict(),

            'training': training,
        }

        # 'epoch-last.pth': saved at the latest epoch
        # 'max-va.pth': saved when validation accuracy is at its maximum
        torch.save(ckpt, os.path.join(args.log_dir, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(args.log_dir, 'trlog.pth'))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(ckpt, os.path.join(args.log_dir, 'max-va.pth'))

        writer.flush()


if __name__ == '__main__':
    main()

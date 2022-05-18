import argparse
import os
import sys
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

import models
import utils
import utils.optimizers as optimizers

from loaders import get_loader
from args import parse_args

import yaml

import trainer

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

    # parallel computing
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    # init random seeds
    utils.setup_env(args)

    # init tensorboard summary is asked and initialize checkpoints
    utils.ensure_path(args.log_dir)
    utils.set_log_path(args.log_dir)
    yaml.dump(config, open(os.path.join(args.log_dir, 'config.yaml'), 'w'))
    writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard')) if args.tensorboard else None

    ##### DataLoaders (meta-learning format only for the moment) #####
    if not args.meta:
        sys.exit("Not Meta-Learning format.")
    
    args.dataset = 'meta_' + args.dataset #to get appropriate loader if meta-training requested
    loader = get_loader(args)

    ### train_loader
    train_set = loader(data_dir=args.data_dir, split='train', image_size=args.train_image_size, normalization=args.train_normalization, transform=args.train_transform, val_transform=args.val_transform, n_batch=args.train_n_batch, n_episode=args.train_n_episode, n_way=args.train_n_way, n_shot=args.train_n_shot, n_query=args.train_n_query)
    utils.log('meta-train set: {} (x{}), {}'.format(train_set[0][0].shape, len(train_set), train_set.n_classes))
    train_loader = DataLoader(train_set, args.train_n_episode, collate_fn=train_set.meta_collate_fn, num_workers=args.workers, pin_memory=True)

    ### val loader
    eval_val = False
    if args.val:
        eval_val = True
        val_set = loader(data_dir=args.data_dir, split='val', image_size=args.val_image_size, normalization=args.val_normalization, transform=args.train_transform, val_transform=args.val_transform, n_batch=args.val_n_batch, n_episode=args.val_n_episode, n_way=args.val_n_way, n_shot=args.val_n_shot, n_query=args.val_n_query)
        utils.log('meta-val set: {} (x{}), {}'.format(val_set[0][0].shape, len(val_set), val_set.n_classes))
        val_loader = DataLoader(val_set, args.val_n_episode, collate_fn=val_set.meta_collate_fn, num_workers=args.workers, pin_memory=True)
    
    ##### Model and Optimizer #####

    inner_args = utils.config_inner_args(config.get('inner_args'))

    ### for resume from load point best or latest
    if args.load in ['best', 'latest']:
        ckpt = torch.load(os.path.join(args.log_dir, args.load + '.pth'))
        args.arch = ckpt['arch']
        args.encoder = ckpt['encoder']
        config['encoder_args'] = ckpt['encoder_args']
        args.classifier = ckpt['classifier']
        config['classifier_args'] = ckpt['classifier_args']
        model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
        optimizer, lr_scheduler = optimizers.load(ckpt, model.parameters())
        start_epoch = ckpt['training']['epoch'] + 1
        max_va = ckpt['training']['max_va']
    else:
        config['arch'] = args.arch
        config['encoder_args'] = config.get('encoder_args') or dict()
        config['classifier_args'] = config.get('classifier_args') or dict()
        config['encoder_args']['bn_args']['n_episode'] = args.train_n_episode
        config['classifier_args']['n_way'] = args.train_n_way
        model = models.make(args.arch, args.encoder, config['encoder_args'], args.classifier, config['classifier_args'])
        optimizer, lr_scheduler = optimizers.make(args.optimizer, model.parameters(), **config['optimizer_args'])
        start_epoch = args.start_epoch
        max_va = -1.0

    if args.efficient:
        model.go_efficient()

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

    model.to(args.device)

    ### testing
    if args.test:
        test_set = loader(data_dir=args.data_dir, split='test', image_size=args.test_image_size, normalization=args.test_normalization, transform=args.test_transform, val_transform=args.test_transform, n_batch=args.test_n_batch, n_episode=args.test_n_episode, n_way=args.test_n_way, n_shot=args.test_n_shot, n_query=args.test_n_query)
        utils.log('meta-test set: {} (x{}), {}'.format(test_set[0][0].shape, len(test_set), test_set.n_classes))
        test_loader = DataLoader(test_set, args.test_n_episode, collate_fn=test_set.meta_collate_fn, num_workers=args.workers, pin_memory=True)

        trainer.meta_test(args, test_loader, model, inner_args, config)

        sys.exit()
    
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

    for epoch in range(start_epoch, args.epoch + 1):
        timer_epoch.start()
        aves = {k: utils.AverageMeter() for k in aves_keys}

        # meta-train
        model.train()
        
        if args.tensorboard:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        np.random.seed(epoch)

        for data in tqdm(train_loader, desc='meta-train', leave=False):
            x_shot, x_query, y_shot, y_query = data
            '''x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
            x_query, y_query = x_query.cuda(), y_query.cuda()'''

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
                '''x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
                x_query, y_query = x_query.cuda(), y_query.cuda()'''

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
                                    (epoch - start_epoch + 1) * (args.epoch - start_epoch + 1))

        # formats output
        log_str = 'epoch {}, meta-train {:.4f}|{:.4f}'.format(
            str(epoch), aves['tl'], aves['ta'])

        if args.tensorboard:
            writer.add_scalars('loss', {'meta-train': aves['tl']}, epoch)
            writer.add_scalars('acc', {'meta-train': aves['ta']}, epoch)

        if eval_val:
            log_str += ', meta-val {:.4f}|{:.4f}'.format(
                aves['vl'], aves['va'])
            if args.tensorboard:
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

            'optimizer': args.optimizer,
            'optimizer_args': config['optimizer_args'],
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
            if lr_scheduler is not None else None,
        }
        ckpt = {
            'file': __file__,
            'config': config,

            'arch': args.arch,

            'encoder': args.encoder,
            'encoder_args': config['encoder_args'],
            'encoder_state_dict': model_.encoder.state_dict(),

            'classifier': args.classifier,
            'classifier_args': config['classifier_args'],
            'classifier_state_dict': model_.classifier.state_dict(),

            'training': training,
        }

        # 'latest.pth': saved at the latest epoch
        # 'best.pth': saved when validation accuracy is at its maximum
        torch.save(ckpt, os.path.join(args.log_dir, 'latest.pth'))
        torch.save(trlog, os.path.join(args.log_dir, 'trlog.pth'))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(ckpt, os.path.join(args.log_dir, 'best.pth'))

        if args.tensorboard:
            writer.flush()

if __name__ == '__main__':
    main()
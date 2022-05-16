import sys
import argparse

import torch

def parse_args():
    parser = argparse.ArgumentParser(description='')

    # expriment settings

    # name of the experiment
    #parser.add_argument('--exp-name', default='convnet4_5_way_1_shot', type=str, help='name of experiment')

    # main folder for data storage
    parser.add_argument('--root-dir', type=str, default='/Users/theophilebeaulieu/Desktop/Clement/master_thesis/project/data')

    parser.add_argument('--config', type = str, default = None, help='configuration file')
    parser.add_argument('--name', help='model name', type=str, default=None)
    parser.add_argument('--tag',help='auxiliary information',type=str, default=None)
    parser.add_argument('--gpu',help='gpu device number',type=str, default='0')
    parser.add_argument('--efficient',help='if True, enables gradient checkpointing',action='store_true')

    # name of the dataset used in the experiment
    parser.add_argument('--dataset', default='mini-imagenet', type=str, help='name of dataset to train upon')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='To run inference on test set.')

    # train parameters
    parser.add_argument('--train-split', default='meta-train', type=str, help='train split')
    parser.add_argument('--train-image-size', default=84, type=int, help='train image size')
    parser.add_argument('--train-normalization', action='store_true', default=False, help='train normalization True or False (default False)')
    parser.add_argument('--train-n_batch', default=1, type=int, help='train number batches')
    parser.add_argument('--train-n_episode', default=4, type=int, help='train number episodes')
    parser.add_argument('--train-n_way', default=5, type=int, help='train number classes')
    parser.add_argument('--train-n_shot', default=1, type=int, help='train number shots per class')
    parser.add_argument('--train-n_query', default=15, type=int, help='train number queries')

    # val parameters
    parser.add_argument('--val-split', default='meta-train', type=str, help='train split')
    parser.add_argument('--val-image-size', default=84, type=int, help='train image size')
    parser.add_argument('--val-normalization', action='store_true', default=False, help='train normalization True or False (default False)')
    parser.add_argument('--val-n_batch', default=1, type=int, help='train number batches')
    parser.add_argument('--val-n_episode', default=4, type=int, help='train number episodes')
    parser.add_argument('--val-n_way', default=5, type=int, help='train number classes')
    parser.add_argument('--val-n_shot', default=1, type=int, help='train number shots per class')
    parser.add_argument('--val-n_query', default=15, type=int, help='train number queries')

    # model settings
    parser.add_argument('--encoder', type=str, default='convnet4', help='encoder')
    parser.add_argument('--encoder-args-bn-args-track-running-stats', action='store_true', default=False, help='encoder track running stats')   
    parser.add_argument('--classifier', type=str, default='logistic', help='classifier')

    parser.add_argument('--load', default='', type=str, metavar='PATH', help='which checkpoint to resume from. possible values["latest", "best", epoch]')

    # inner args settings
    parser.add_argument('--inner-args-n-step', type=int, default=5, help='inner args n_step')
    parser.add_argument('--inner-args-encoder_lr', type=float, default=0.01, help='encoder lr')
    parser.add_argument('--inner-args-classifier_lr', type=float, default=0.01, help='classifier lr')
    parser.add_argument('--inner-args-first-order', action='store_true', default=False, help='first order')

    # number of workers for the dataloader
    parser.add_argument('-j', '--workers', type=int, default=4)

    # training settings
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--step', type=int, default=20, help='frequency of updating learning rate')
    parser.add_argument('--epoch', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', default='adam', type=str, help='name of the optimizer')
    parser.add_argument('--optimizer-args-lr', type=float, default=0.01, help='learning rate')

    # misc settings
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--disable-cuda', action='store_true', default=False, help='disables CUDA training / using only CPU')
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', default=False, help='Use tensorboard to track and plot')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.encoder + '_' + args.dataset.replace('meta-', '') + '_{}_way_{}_shot'.format(args.train_n_way, args.train_n_shot)
    if args.tag is not None:
        args.name += '_' + args.tag

    # update args
    args.data_dir = '{}/{}'.format(args.root_dir, args.dataset)
    args.log_dir = '{}/runs/{}/'.format(args.data_dir, args.name)
    #args.res_dir = '%s/runs/%s/res' % (args.data_dir, args.name)
    args.out_pred_dir = '%s/runs/%s/pred' % (args.data_dir, args.name)

    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'

    assert args.data_dir is not None

    print(' '.join(sys.argv))
    print(args)

    return args
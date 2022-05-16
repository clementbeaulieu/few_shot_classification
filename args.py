import sys
import argparse

import torch

def parse_args():
    parser = argparse.ArgumentParser(description='')

    # expriment settings

    # name of the experiment
    parser.add_argument('--name', default='digitsum', type=str, help='name of experiment')
    parser.add_argument('--train-type', default='regression', type=str, help='type of learning task (regression, classification or unsupervised)')
    parser.add_argument('--val-type', default='regression', type=str, help='type of validation task (regression, classification, unsupervised or customed)')
    parser.add_argument('--test-type', default='regression', type=str, help='type of test task (regression, classification, unsupervised or customed)')
    parser.add_argument('--print-freq-train', type=int, default=10, help='print freq of batch values on training')
    parser.add_argument('--print-freq-val', type=int, default=10, help='print freq of batch values on training')

    # name of the dataset used in the experiment
    parser.add_argument('--dataset', default='miniimagenet', type=str, help='name of dataset to train upon')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='To run inference on test set.')

    # main folder for data storage
    parser.add_argument('--root-dir', type=str, default=None)

    # model settings
    parser.add_argument('--arch', type=str, default='digitsum_image', help='name of the architecture to be used')
    parser.add_argument('--model-name', type=str, default='digitsum_image50', help='name of the model to be used')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='which checkpoint to resume from. possible values["latest", "best", epoch]')

    # params for digitsum image experiment
    parser.add_argument('--min-size-train', type=int, default = 2, help='min size for train set sizes')
    parser.add_argument('--max-size-train', type=int, default = 10, help='max size for train set sizes')
    parser.add_argument('--min-size-val', type=int, default = 5, help='min size validation/test for set sizes')
    parser.add_argument('--max-size-val', type=int, default = 50, help='max size validation/test for set sizes')
    parser.add_argument('--dataset-size-train', type=int, default = 100000, help='size of the train dataset of sets')
    parser.add_argument('--dataset-size-val', type=int, default = 10000, help='size of the validation/test dataset of sets')
    parser.add_argument('--set-weight', type=str, default='mean', help='default set_weight metrics for set_MAP score (mean, linear or exp)')

    # params for classification tasks
    parser.add_argument('--num-classes', default=0, type=int)
    # number of workers for the dataloader
    parser.add_argument('-j', '--workers', type=int, default=4)

    # training settings
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--step', type=int, default=20, help='frequency of updating learning rate')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 70)')
    parser.add_argument('--optimizer', default='adam', type=str, help='name of the optimizer')
    parser.add_argument('--scheduler', default='StepLR', type=str, help='name of the learning rate scheduler')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='sgd momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-decay', default=0.995, type=float, metavar='lrd', help='learning rate decay (default: 0.995)')
    parser.add_argument('--criterion', default='mse', type=str, help='criterion to optimize')

    # misc settings
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
    parser.add_argument('--disable-cuda', action='store_true', default=False, help='disables CUDA training / using only CPU')
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',default=False, help='Use tensorboard to track and plot')

    args = parser.parse_args()

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
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

#************************************************************#
#************************* TRAINING *************************#
#************************************************************#


#************************************************************#
#*************************** TEST ***************************#
#************************************************************#

def meta_test(args, test_loader, model, inner_args, config):
    model.eval()
    aves_va = utils.AverageMeter()
    va_lst = []

    for epoch in range(1, args.epoch + 1):
        for data in tqdm(test_loader, leave=False):
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
            logits = logits.view(-1, args.test_n_way)
            labels = y_query.view(-1)

            pred = torch.argmax(logits, dim=1)
            acc = utils.compute_acc(pred, labels)
            aves_va.update(acc, 1)
            va_lst.append(acc)

        print('test epoch {}: acc={:.2f} +- {:.2f} (%)'.format(
            epoch, aves_va.item() * 100,
            utils.mean_confidence_interval(va_lst) * 100))
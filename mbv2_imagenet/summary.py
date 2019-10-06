from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from math import cos, pi, ceil
import random
import argparse
import os
import time
import numpy as np

from dataloader import  get_data_loader
from config import *

from sp_mbnetv2 import sp_mbnetv2 as mbnet
from sp_mbnetv2 import ConvBlock, InvertedResidual

from compute_flops import print_model_param_flops

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--data', default="", 
                    help='path to dataset', required=False)
parser.add_argument('-j', '--workers', default=25, type=int, metavar='N',
                    help='number of data loading workers (default: 25)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--load', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)', required=True)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

# Use CUDA
use_cuda = torch.cuda.is_available()
args.use_cuda = use_cuda

# Random seed
random.seed(0)
torch.manual_seed(0)
if use_cuda:
    torch.cuda.manual_seed_all(0)
    device = 'cuda'
    cudnn.benchmark = True
else:
    device = 'cpu'


# define loss function (criterion) and optimizer
num_classes = 1000

# Data loading code
train_loader, val_loader = \
    get_data_loader(args.data, train_batch_size=args.batch_size, test_batch_size=args.test_batch_size, workers=args.workers)


## loading pretrained model ##
assert args.load
assert os.path.isfile(args.load)
print("=> loading checkpoint '{}'".format(args.load))
checkpoint = torch.load(args.load)

model = mbnet(cfg=checkpoint['cfg'])
total_flops = print_model_param_flops(model, 224, multiply_adds=False) 
print(total_flops)

if args.use_cuda: 
    model.cuda()

selected_model_keys = [k for k in model.state_dict().keys() if not (k.endswith('.y') or k.endswith('.v') or k.startswith('net_params') or k.startswith('y_params') or k.startswith('v_params'))]
saved_model_keys = checkpoint['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
if len(selected_model_keys) == len(saved_model_keys):

    for k0, k1 in zip(selected_model_keys, saved_model_keys):
        new_state_dict[k0] = checkpoint['state_dict'][k1]   
    
    model_dict = model.state_dict()
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print('load from order match')
else:
    ## load form sp_mbnet model ##
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('module'):
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print('load from key match')
print("=> loaded checkpoint '{}' " .format(args.load))
del checkpoint


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval== 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n\n'
          .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


criterion = nn.CrossEntropyLoss().cuda()

print('testing acc before splitting')
validate(val_loader, model, criterion)



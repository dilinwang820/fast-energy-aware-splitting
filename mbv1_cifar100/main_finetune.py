from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import sys
from dataloader import  get_data_loader
from compute_flops import print_model_param_nums, print_model_param_flops

import json
from math import cos, pi

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)', required=True)
parser.add_argument('--load', default='', type=str, metavar='PATH',
                    help='path to the pruned/split model to be fine tuned', required=True)
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)', required=False)
parser.add_argument('--lr-mode', type=str, default='step', choices=['step', 'constant'],
                    help='learning rate scheduler mode. options are step, poly and cosine.')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)', required=False)
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--sp', action='store_true', default=False,
                    help='splitting settings')
parser.add_argument('--retrain', action='store_true', default=False,
                    help='retrain, otherwise, finetune')
parser.add_argument('--resume', action='store_true', default=False,
                    help='check existing models')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--warmup-epochs', type=int, default=0,
                    help='number of warmup epochs.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.sp:
    from sp_mbnet import sp_mbnet as mbnet
else:
    from mobilenetv1 import MobileNetV1 as mbnet

assert args.load

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

args.save = os.path.dirname(args.load)

checkpoint = torch.load(args.load)
if args.resume:
    model_save_path = args.load
else:
    training_mode = 'retrain' if args.retrain else 'finetune'
    args.save = os.path.join(args.save, training_mode)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    model_save_path = os.path.join(args.save, os.path.basename(args.load))


logging_file_path = model_save_path.replace(".pth.tar", ".log")

#########################################################
# create file handler which logs even debug messages 
import logging
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
fh = logging.FileHandler(logging_file_path, mode='w')

formatter = logging.Formatter('%(asctime)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

log.addHandler(fh)
log.addHandler(ch)
#########################################################

log.info('model save path: {}'.format(model_save_path))
log.info('log save path: {}'.format(logging_file_path))

##### check exsiting models ##
#if os.path.isfile(model_save_path) and args.resume:
#    pre_check = torch.load(model_save_path)
#    if pre_check['epoch'] == args.epochs and pre_check['cfg'] == checkpoint['cfg']:
#        print('no need to run, load from {}'.format(model_save_path))
#        sys.exit(0)

from dataloader import  get_data_loader
train_loader, test_loader = \
        get_data_loader(dataset = args.dataset, train_batch_size = args.batch_size, test_batch_size = args.test_batch_size, use_cuda=args.cuda)

model = mbnet(cfg=checkpoint['cfg'], dataset=args.dataset)
# load weights, otherwise, only the arch is used
if not args.retrain:
    model.load_state_dict(checkpoint['state_dict'])

if args.cuda: model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        #loss = criterion(output, target)
        avg_loss += loss.data
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            log.info('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()

    test_loss /= len(test_loader.dataset)
    log.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

prec1 = test()
best_prec1 = 0.
for epoch in range(args.epochs):
    # finetune & retrain:
    if epoch in [int(args.epochs*0.5), int(args.epochs*0.75)] and args.lr_mode == 'step':
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    train(epoch)
    prec1 = test()
    best_prec1 = max(prec1, best_prec1)

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'cfg': model.cfg,
        'acc': prec1,
        #ek: ev,
        'optimizer': optimizer.state_dict(),
    }, model_save_path)

print("Best accuracy: "+str(best_prec1))



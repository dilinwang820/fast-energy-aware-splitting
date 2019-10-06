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

from config import *

import json

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)', required=True)
parser.add_argument('--sr', action='store_true', default=False,
                    help='training with sparsity regularization')
parser.add_argument('--morphnet', action='store_true', default=False,
                    help='training with morphnet')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./saved_models', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--cfg', default=None, type=str, 
                    help='default cfg for vgg')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda') if args.cuda else torch.device('cpu')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

from mobilenetv1 import MobileNetV1 as mbnet
from mobilenetv1 import MbBlock, ConvBlock

#assert not (args.sr and args.sp)

prefix = '{}_base'.format(args.dataset)
if args.sr:
    prefix = '{}_bn_prune_base_s_{}'.format(args.dataset, args.s)
if args.morphnet:
    prefix = '{}_morphnet_base_s_{}'.format(args.dataset, args.s)

if args.cfg is not None:
    args.save = os.path.join(args.save, args.cfg)

logging_file = '{}.log'.format(prefix)
model_save_path = '{}.pth.tar'.format(prefix)

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = mbnet(dataset = args.dataset, cfg=args.cfg)
model.to(device)

#########################################################
# create file handler which logs even debug messages 
import logging
log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(args.save, logging_file), mode='w')
formatter = logging.Formatter('%(asctime)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

log.addHandler(fh)
log.addHandler(ch)
#########################################################

### check exsiting models ##
if os.path.isfile(os.path.join(args.save, model_save_path)):
    pre_check = torch.load(os.path.join(args.save, model_save_path))
    if pre_check['epoch'] == args.epochs and pre_check['cfg'] == model.cfg:
        print('no need to run, load from {}'.format(model_save_path))
        sys.exit(0)

train_loader, test_loader = \
        get_data_loader(dataset = args.dataset, train_batch_size = args.batch_size, test_batch_size = args.test_batch_size, use_cuda=args.cuda)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

#history_score = np.zeros((args.epochs - args.start_epoch + 1, 3))

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    layer_idx = 0
    for m in model.modules():
        if isinstance(m, MbBlock) or isinstance(m, ConvBlock):
            flops_inc = get_flops_inc_per_layer(model, layer_idx)
            layer_idx += 1
            if args.sr:
                r_lambda = args.s
            if args.morphnet:
                r_lambda = args.s * flops_inc

            if isinstance(m, MbBlock):
                m.bn2.weight.grad.data.add_(r_lambda*torch.sign(m.bn2.weight.data))  # L1
            elif isinstance(m, ConvBlock):
                m.bn.weight.grad.data.add_(r_lambda*torch.sign(m.bn.weight.data))  # L1


def train(epoch):
    model.train()
    #global history_score
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.data
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        if args.sr or args.morphnet:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            log.info('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
    #history_score[epoch][0] = avg_loss / len(train_loader)
    #history_score[epoch][1] = train_acc / float(len(train_loader))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()

    test_loss /= len(test_loader.dataset)
    log.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / float(len(test_loader.dataset))))
    return correct / float(len(test_loader.dataset))


def save_checkpoint(state, model_path):
    torch.save(state, model_path)
    #if is_best:
    #    shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


best_prec1 = 0.
curr_lr = args.lr
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [int(args.epochs*0.5), int(args.epochs*0.75)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
            curr_lr *= 0.1
    log.info('{}, {}'.format(epoch, curr_lr))

    train(epoch)
    prec1 = test()

    #history_score[epoch][2] = prec1
    #np.savetxt(os.path.join(args.save, 'record_{}.txt'.format(args.dataset)), history_score, fmt = '%10.5f', delimiter=',')
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    torch.save({
        'epoch': epoch + 1,
        'cfg': model.cfg,
        #'sr': args.sr,
        's': args.s,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        'acc': prec1,
    }, os.path.join(args.save, model_save_path))
   

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

from dataloader import  get_data_loader
from compute_flops import print_model_param_nums, print_model_param_flops
import sys

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)', required=True)
parser.add_argument('--load', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned', required=True)
parser.add_argument('--sp', action='store_true', default=False,
                    help='splitting settings')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                    help='input batch size for testing (default: 50)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-name', type=str, choices=['baseline', 'bn_finetune', 'bn_retrain', 'split_finetune', 'split_retrain', 'l1_retrain', 'l1_finetune', 'morph_retrain', 'morph_finetune'], required=True)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda') if args.cuda else torch.device('cpu')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.sp:
    from sp_mbnet import sp_mbnet as mbnet
else:
    from mobilenetv1 import MobileNetV1 as mbnet

assert args.load
assert os.path.isfile(args.load)
#    sys.exit(0)

checkpoint = torch.load(args.load)

model = mbnet(dataset=args.dataset, cfg=checkpoint['cfg'])

# load weights, otherwise, only the arch is used
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

train_loader, test_loader = \
        get_data_loader(dataset = args.dataset, train_batch_size = args.batch_size, test_batch_size = args.test_batch_size, use_cuda=args.cuda)


def test(model):
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
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

acc = test(model)

total_params = print_model_param_nums(model.cpu())
total_flops = print_model_param_flops(model.cpu(), 32)


results = {
    'load': args.load,
    'dataset': args.dataset,
    'model_name': args.model_name,
    'arch': 'mobilenetv1',
    'acc': acc,
    'cfg': model.cfg,
    'total_params': total_params,
    'total_flops': total_flops,
}
print(results)

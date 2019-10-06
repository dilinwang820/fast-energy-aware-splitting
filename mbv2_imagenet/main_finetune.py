import argparse, logging, os, math, time
import numpy as np
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from math import cos, pi
import random

from dataloader import  get_data_loader
from compute_flops import print_model_param_nums, print_model_param_flops

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default="", 
                    help='path to dataset', required=False)
parser.add_argument('-j', '--workers', default=25, type=int, metavar='N',
                    help='number of data loading workers (default: 25)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-mode', type=str, default='step',
                    help='learning rate scheduler mode. options are step, poly and cosine.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epochs at which learning rate decays. default is 40,60.')
parser.add_argument('--warmup-epochs', type=int, default=0,
                    help='number of warmup epochs.')
parser.add_argument('--label-smoothing', action='store_true',
                    help='use label smoothing or not in training. default is false.')
parser.add_argument('--no-cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--no-wd', action='store_true',
                    help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', action='store_true',
                    help='always load the previous saved model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--num-gpus', type=int, default=1,
                    help='number of gpus')
parser.add_argument('--retrain', action='store_true',
                    help='retrain the model')
parser.add_argument('--load', type=str,
                    help='checkpoint')
best_prec1 = 0


from sp_mbnetv2 import sp_mbnetv2 as mobilenetv2

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon = 0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss



def main():
    global args, best_prec1, device
    args = parser.parse_args()

    batch_size = args.batch_size * max(1, args.num_gpus)
    args.lr = args.lr * (batch_size / 256.)
    print(batch_size, args.lr, args.num_gpus)

    num_classes = 1000
    num_training_samples = 1281167
    args.num_batches_per_epoch = num_training_samples // batch_size


    assert os.path.isfile(args.load) and args.load.endswith(".pth.tar")
    args.save = os.path.dirname(args.load)
    training_mode = 'retrain' if args.retrain else 'finetune'
    args.save = os.path.join(args.save, training_mode)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    args.model_save_path = os.path.join(args.save, "epochs_{}_{}".format(args.epochs, os.path.basename(args.load)))
    args.distributed = args.world_size > 1

    ##########################################################
    ## create file handler which logs even debug messages
    #import logging
    #log = logging.getLogger() 
    #log.setLevel(logging.INFO)

    #ch = logging.StreamHandler()
    #fh = logging.FileHandler(args.logging_file_path)

    #formatter = logging.Formatter('%(asctime)s - %(message)s')
    #ch.setFormatter(formatter)
    #fh.setFormatter(formatter)
    #log.addHandler(fh)
    #log.addHandler(ch) 
    ##########################################################                                                                                                                     

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # Use CUDA
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda

    # Random seed
    random.seed(0)
    torch.manual_seed(0)
    if args.use_cuda:
        torch.cuda.manual_seed_all(0)
        device = 'cuda'
        cudnn.benchmark = True
    else:
        device = 'cpu'

    if args.evaluate == 1:
        device = 'cuda:0'



    assert os.path.isfile(args.load)
    print("=> loading checkpoint '{}'".format(args.load))
    checkpoint = torch.load(args.load)
 
    model = mobilenetv2(cfg=checkpoint['cfg'])
    cfg = model.cfg

    total_params = print_model_param_nums(model.cpu())
    total_flops = print_model_param_flops(model.cpu(), 224, multiply_adds=False) 
    print(total_params, total_flops)

    if not args.distributed:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model)

    ##### finetune #####
    if not args.retrain:
        model.load_state_dict(checkpoint['state_dict'])

    # define loss function (criterion) and optimizer
    if args.label_smoothing:
        criterion = CrossEntropyLabelSmooth(num_classes).to(device) 
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    ### all parameter ####
    no_wd_params, wd_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if ".bn" in name or '.bias' in name:
                no_wd_params.append(param)
            else:
                wd_params.append(param)
    no_wd_params = nn.ParameterList(no_wd_params)
    wd_params = nn.ParameterList(wd_params)

    optimizer = torch.optim.SGD([
                                {'params': no_wd_params, 'weight_decay':0.},
                                {'params': wd_params, 'weight_decay': args.weight_decay},
                            ], args.lr, momentum=args.momentum)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.model_save_path):
            print("=> loading checkpoint '{}'".format(args.model_save_path))
            checkpoint = torch.load(args.model_save_path)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model_save_path, checkpoint['epoch']))
        else:
            pass

    # Data loading code
    train_loader, val_loader = \
        get_data_loader(args.data, train_batch_size=batch_size, test_batch_size=32, workers=args.workers)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        #adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'cfg': cfg, 
            #'m': args.m,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, args.model_save_path)

        print('  + Number of params: %.3fM' % (total_params / 1e6))
        print('  + Number of FLOPs: %.3fG' % (total_flops / 1e9))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        if epoch < args.warmup_epochs:
            curr_lr = args.lr * (args.num_batches_per_epoch*epoch + i) / (args.warmup_epochs * args.num_batches_per_epoch)
        else:
            if args.lr_mode == 'cosine':
                N = (args.epochs - args.warmup_epochs) * args.num_batches_per_epoch
                T = i + (epoch - args.warmup_epochs) * args.num_batches_per_epoch
                curr_lr = args.lr * (1 + cos(pi * T / (N-1))) / 2
            elif args.lr_mode == 'step':
                step_epochs = [int(ep) for ep in args.lr_decay_epoch.split(',')]
                count = sum([1 for s in step_epochs if s <= epoch])
                curr_lr = args.lr * pow(self.lr_decay, count)
            else:
                raise NotImplementedError

        adjust_learning_rate(optimizer, curr_lr)
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t lr {lr:.4f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), lr=curr_lr, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


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
        input_var = torch.autograd.Variable(input, volatile=True)
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

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, filepath):
    torch.save(state, filepath)

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

#def adjust_learning_rate(optimizer, epoch):
def adjust_learning_rate(optimizer, lr):
    for params_group in optimizer.param_groups:
        params_group['lr'] = lr


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

if __name__ == '__main__':
    main()

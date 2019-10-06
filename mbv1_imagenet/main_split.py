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
from compute_flops import print_model_param_nums, print_model_param_flops
from sp_mbnet import sp_mbnet as mbnet
from sp_conv import SpConvBlock
from sp_mbnet import SpMbBlock
from config import *


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--data', default="", 
                    help='path to dataset', required=False)
parser.add_argument('-j', '--workers', default=25, type=int, metavar='N',
                    help='number of data loading workers (default: 25)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=4, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--label-smoothing', action='store_true',
                    help='use label smoothing or not in training. default is false.')
parser.add_argument('--load', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)', required=True)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--split-index', default="1", type=str,
                    help='#number of split', required=True)
parser.add_argument('--energy', action='store_true', default=False,
                    help='energy aware splitting')
parser.add_argument('--params', action='store_true', default=False,
                    help='parameter aware splitting')
parser.add_argument('--grow', type=float, default=0.2, 
                    help='split grow rate (default: 0.2)', required=True)
parser.add_argument('--exp-name', type=str, default=None, 
                    help='exp name', required=True)
parser.add_argument('--save', default='split/saved_models', type=str,
                    help='energy aware splitting')
#parser.add_argument('--prefix', default=None, type=str, metavar='PATH',
#                    help='split prefix', required=False)

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


# define loss function (criterion) and optimizer
num_classes = 1000
if args.label_smoothing:
    criterion = CrossEntropyLabelSmooth(num_classes).cuda() 
else:
    criterion = nn.CrossEntropyLoss().cuda()

# Data loading code
train_loader, val_loader = \
    get_data_loader(args.data, train_batch_size=args.batch_size, test_batch_size=32, workers=args.workers)


## loading pretrained model ##
assert args.load
assert os.path.isfile(args.load)
print("=> loading checkpoint '{}'".format(args.load))
checkpoint = torch.load(args.load)

model = mbnet(cfg=checkpoint['cfg'])
total_params = print_model_param_nums(model)
total_flops = print_model_param_flops(model, 224, multiply_adds=False) 
print(total_params, total_flops)

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


#from collections import OrderedDict
#new_state_dict = OrderedDict()
#for k, v in checkpoint['state_dict'].items():
#    if k.startswith('module'):
#        name = k[7:] # remove 'module.' of dataparallel
#        new_state_dict[name] = v
#model.load_state_dict(new_state_dict)
#print("=> loaded checkpoint '{}'".format(args.load))


## saving the results ##
#prefix = 'split' if not args.energy else 'energy_aware_split'
#if args.prefix is not None:
#    prefix = "{}_{}".format(args.prefix, prefix)
#
#if args.split_index == "1":
#    args.save = os.path.join(os.path.dirname(args.load), os.path.basename(args.load).replace(".pth.tar", ""))
#else:
#    args.save = os.path.dirname(args.load)
args.save = os.path.join(args.save, args.exp_name)
if args.energy:
    args.save = os.path.join(args.save, 'energy_aware')
if args.params:
    args.save = os.path.join(args.save, 'params_aware')

if not os.path.exists(args.save):
    os.makedirs(args.save)

#hyper_str = "lr_{}_local_{}_global_{}".format(args.lr, args.local_grow, args.global_grow)
#if args.label_smoothing:
#    hyper_str = "{}_label_smoothing".format(hyper_str)

#model_save_path = os.path.join(args.save, '{}_{}_{}.pth.tar'.format(prefix, hyper_str, args.split_index))
model_save_path = os.path.join(args.save, 'fast_grow_{}_split_{}.pth.tar'.format(args.grow,  args.split_index))
print(model_save_path)

##########################################################
## create file handler which logs even debug messages 
#import logging
#log = logging.getLogger()
#log.setLevel(logging.DEBUG)
#
#ch = logging.StreamHandler()
#fh = logging.FileHandler(os.path.join(args.save, logging_file_path))
#
#formatter = logging.Formatter('%(asctime)s - %(message)s')
#ch.setFormatter(formatter)
#fh.setFormatter(formatter)
#
#log.addHandler(fh)
#log.addHandler(ch)
##########################################################
def train(epoch):
    model.train() 
    train_acc = 0.

    min_eig_vals, min_eig_vecs = None, None
    for batch_idx, (input, target) in enumerate(train_loader):

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)

        optimizer_v.zero_grad()
        
        ## splitting aware forward ##
        output = model.sp_forward(input_var)
        #ce = F.cross_entropy(output, target_var)
        ce = criterion(output, target_var)
        y_grads = torch.autograd.grad(ce, model.y_params)

        eig_loss = []
        for sv, v in zip(y_grads, model.v_params):
            eig_loss.append((sv * v).sum([1,2,3]) / (v.pow(2).sum([1,2,3])))
        
        manual_v_grads = [] 
        for sv, v in zip(y_grads, model.v_params):
            vv = v.pow(2).sum([1,2,3], keepdim=True)
            vsv = (sv * v).sum([1,2,3], keepdim=True)
            manual_v_grads.append( 2.*(sv * vv -  v*vsv) / vv.pow(2) )
 
        if min_eig_vals is None or min_eig_vecs is None:
            min_eig_vals, min_eig_vecs = [], []
            for m in eig_loss: min_eig_vals.append(m.data.clone())
            for m in model.v_params: min_eig_vecs.append(m.data.clone())
        else:
            for k in range(len(eig_loss)):
                min_eig_vals[k] += eig_loss[k].data.clone()
                min_eig_vecs[k] += model.v_params[k].data.clone()

        loss = sum([el.sum() for el in eig_loss])

        #loss.backward() 
        #loss.backward() 
        # set gradient manually
        for v, mv_grad in zip(model.v_params, manual_v_grads):
            v.grad = mv_grad

        optimizer_v.step()

        # normalize v, ||v|| = 1
        for v in model.v_params:
            v.data =  v.data / v.pow(2).sum([1,2,3], keepdim=True).sqrt().data

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)], loss {:.2f}'.format(
                epoch, batch_idx * len(input), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),  sum([v.sum()/(1.+batch_idx) for v in min_eig_vals]).data ))

            #if batch_idx > 100:
            #    break

    for k in range(len(min_eig_vals)):
        min_eig_vals[k] /= float(len(train_loader))
        min_eig_vecs[k] /= float(len(train_loader))

    print('Train Epoch: loss {:.2f}'.format(sum([v.sum() for v in min_eig_vals]).data))
    return min_eig_vals, min_eig_vecs


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

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
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
optimizer_v = torch.optim.RMSprop(model.v_params, lr=args.lr, momentum=0.9, alpha=0.9)
##### fast gradient approximation ######
for k, m in enumerate(model.modules()):
    if k == 2: assert isinstance(m, SpConvBlock)
    if isinstance(m, SpMbBlock) or (k == 2 and isinstance(m, SpConvBlock)):
        m.reset_yv_()

print('testing acc before splitting')
validate(val_loader, model, criterion)

for epoch in range(args.epochs):
    if epoch > 0:
        for param_group in optimizer_v.param_groups:
            param_group['lr'] *= 0.2
    min_eig_vals, min_eig_vecs = train(epoch)
    #break
#print('testing acc after splitting')
#validate(val_loader, model, criterion)

##### select neurons ######
total = 0
for m in min_eig_vals:
    total += len(m)


##weights = torch.zeros(total)
#weights = [] #torch.zeros(total)
#index = 0
#for m in min_eig_vals:
#    size = len(m)
#    #weights[index:(index+size)] = m.data.clone()
#    weights.append(m.data.clone())
#    index += size
#
### local thre ##
#local_thre = []
#for i in np.unique(split_groups):
#    gi = np.where(split_groups == i)[0]
#    wi = torch.cat([weights[k] for k in gi])
#    y, _ = torch.sort(wi)
#    lti = ceil(len(wi) * args.local_grow)
#    local_thre.append(y[lti])
#
### global thre ##
#weights = torch.cat(weights)
#y, i = torch.sort(weights)
#thre_index = int(total * args.global_grow)
#global_thre = y[thre_index]
#
##y, i = torch.sort(weights)
##thre_index = int(total * args.global_grow)
##global_thre = y[thre_index]
#
#cfg_grow = []
#cfg_mask = []
#layer_idx = 0
#
#for k, m in enumerate(model.modules()):
#    if k == 2: assert isinstance(m, SpConvBlock)
#    if isinstance(m, SpMbBlock) or (k == 2 and isinstance(m, SpConvBlock)):
#        weight_copy = min_eig_vals[layer_idx].data.clone()
#
#        #sorted_weights, sorted_idx = torch.sort(weight_copy)
#        #thre_index = int(np.ceil(len(sorted_idx) * args.local_grow))
#        #local_thre = sorted_weights[thre_index]
#        #thre = np.max([local_thre.cpu().numpy(), global_thre.cpu().numpy()])
#        thre = np.max([local_thre[split_groups[layer_idx]].cpu().numpy(), global_thre.cpu().numpy()])
#
#        #weight_copy = weight_copy.cpu().numpy()
#        #mask = torch.tensor(weight_copy).lt(torch.tensor(thre)).float().to(device)
#        mask = torch.tensor(weight_copy).lt(torch.tensor(thre).to(device)).float().to(device)
#        cfg_grow.append(int(torch.sum(mask)))
#        cfg_mask.append(mask.clone())
#        layer_idx += 1
#        print('layer index: {:d} \t total channel: {:d} \t splitting channel: {:d}'.
#            format(k, mask.shape[0], int(torch.sum(mask))))
#    elif isinstance(m, nn.MaxPool2d):
#        cfg_grow.append('M')
#grow_ratio = sum([m.sum() for m in cfg_mask])/total
#cfg = [ c+v if isinstance(v, int) else (c+v[0], v[1]) for c, v in zip(cfg_grow, model.cfg)]
#

cfg_grow = []
cfg_mask = []

if args.energy or args.params:
    ## flops ##
    cfg = model.cfg

    params_inc_per_neuron, flops_inc_per_neuron = [], []
    for i, (c, r) in enumerate(zip(cfg, resolutions)):
        # number of channles in the previous layer
        flops_inc = get_flops_inc_per_layer(model, i)
        params_inc = get_params_inc_per_layer(model, i)

        flops_inc_per_neuron.append( np.log(flops_inc) )
        params_inc_per_neuron.append( np.log(params_inc) )

    target_variable = flops_inc_per_neuron if args.energy else params_inc_per_neuron
    MAX_RESOUCE = int(sum([args.grow *get_number_of_channels(c)*f for c, f in zip(cfg, target_variable)]))

    max_resource_per_group = {}
    for c, f, g in zip(cfg, target_variable, split_groups):
        max_resource_per_group[g] = max_resource_per_group.get(g, 0) + args.grow * get_number_of_channels(c) * f

    print(max_resource_per_group)
    # select maximum  total*grow neurons
    r"""
        \min_\beta \sum_{i} \beta_i \lambda_i
            s.t.  \sum_i \beta_i = c (number of neurons)
                  \sum_i \beta_i f_i < \tau
                  0 <= beta_i <= 1

        Minimize:
            c @ x

        Subject to:
            A_ub @ x <= b_ub
            A_eq @ x == b_eq
            lb <= x <= ub
    """
    all_lambda = torch.cat(min_eig_vals).cpu().numpy()
    # resource: flops or params

    #get_f = lambda layer_idx, fi, c: 1e9 if get_number_of_channels(max_splitcfg[layer_idx]) <= get_number_of_channels(c) else fi 
    #all_f = np.concatenate([[get_f(layer_idx, fi, c)]*get_number_of_channels(c) for layer_idx, (fi, c) in enumerate(zip(target_variable, cfg))])  
    all_f = np.concatenate([[fi]*get_number_of_channels(c) for fi, c in zip(target_variable, cfg)])
    # group id for each neuron
    all_g = np.concatenate([[gi]*get_number_of_channels(c) for gi, c in zip(split_groups, cfg)])
    from scipy.optimize import linprog
    n_max = total * args.grow# need the constrain
    #res = linprog(all_lambda, A_ub=np.expand_dims(all_f, 0), b_ub=MAX_RESOUCE, A_eq=np.ones((1, total)), b_eq=n_max, bounds=(0., 1.),
    #                options={"disp": True})

    ## grop level constrains ##
    A_ub, b_ub = [], []
    for i in np.unique(split_groups):
        idx_i = np.where( all_g == i)[0]
        A_ub_i = np.zeros(len(all_g))
        A_ub_i[idx_i] = 1.
        A_ub.append(A_ub_i * all_f) # flops constrains
        b_ub.append(max_resource_per_group[i])

    # check maximum cfg
    index = 0
    for i in range(len(cfg)):
        c = get_number_of_channels(cfg[i])
        residual = max(0, get_number_of_channels(max_splitcfg[i]) - get_number_of_channels(cfg[i]))
        A_ub_i = np.zeros(len(all_g))
        A_ub_i[index:index+c] = 1.
        A_ub.append(A_ub_i)
        b_ub.append(residual)
        index += c
    A_ub, b_ub = np.stack(A_ub), np.stack(b_ub)

    #print(A_ub.shape, b_ub.shape)
    #print(b_ub)
    print(all_f)

    #res = linprog(all_lambda, A_ub=np.expand_dims(all_f, 0), b_ub=MAX_RESOUCE, bounds=(0., 1.))
    res = linprog(all_lambda, A_ub=A_ub, b_ub=b_ub, bounds=(0., 1.), method='interior-point', options={'tol':1e-10})
    assert res.status == 0
    ### select top n_max neurons
    #sorted_idx = np.argsort(res.x)[::-1]
    #thre = res.x[sorted_idx[int(n_max)-1]]
    thre = 0.9

    layer_idx, start_idx = 0, 0
    for k, m in enumerate(model.modules()):
        if k == 2: assert isinstance(m, SpConvBlock)
        if isinstance(m, SpMbBlock) or (k == 2 and isinstance(m, SpConvBlock)):
            betas = res.x[start_idx:start_idx+len(min_eig_vals[layer_idx])]
            start_idx += len(min_eig_vals[layer_idx])
            
            curr_c = get_number_of_channels(cfg[layer_idx])
            max_c = get_number_of_channels(max_splitcfg[layer_idx])

            max_grow_c = max(0, min(curr_c, max_c - curr_c))

            weights_copy = np.copy(betas)
            sorted_idx = np.argsort(weights_copy)[::-1]
            betas[sorted_idx[max_grow_c:]] = 0.
            mask = torch.tensor(betas).float().ge(torch.tensor(thre).float()).float().to(device)

            cfg_grow.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            layer_idx += 1
            print('layer index: {:d} \t total channel: {:d} \t splitting channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg_grow.append('M')
        
    #sys.exit(0)
else:
    weights = [] #torch.zeros(total)
    index = 0
    for m in min_eig_vals:
        size = len(m)
        #weights[index:(index+size)] = m.data.clone()
        weights.append(m.data.clone())
        index += size
    
    ## global thre ##
    weights = torch.cat(weights)
    y, i = torch.sort(weights)
    thre_index = int(total * args.grow)
    global_thre = y[thre_index]

    layer_idx = 0
    for k, m in enumerate(model.modules()):
        if k == 2: assert isinstance(m, SpConvBlock)
        if isinstance(m, SpMbBlock) or (k == 2 and isinstance(m, SpConvBlock)):
            weight_copy = min_eig_vals[layer_idx].data.clone()
            
            #sorted_weights, sorted_idx = torch.sort(weight_copy)
            #thre_index = int(np.ceil(len(sorted_idx) * args.local_grow))
            #local_thre = sorted_weights[thre_index]
    
            #thre = np.max([local_thre[split_groups[layer_idx]].cpu().numpy(), global_thre.cpu().numpy()])
            thre = global_thre
    
            mask = torch.tensor(weight_copy).lt(torch.tensor(thre).to(device)).float().to(device)
            cfg_grow.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            layer_idx += 1
            print('layer index: {:d} \t total channel: {:d} \t splitting channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg_grow.append('M')

grow_ratio = sum([m.sum() for m in cfg_mask])/total
cfg = [ c+v if isinstance(v, int) else (c+v[0], v[1]) for c, v in zip(cfg_grow, model.cfg)]



print('old cfg')
print(model.cfg)
print('new cfg')
print(cfg)

##### copy weights and split #####
newmodel = mbnet(cfg=cfg) 

new_total_params = print_model_param_nums(newmodel) 
new_total_flops = print_model_param_flops(newmodel.cpu(), 224, multiply_adds=False) 
print('new_total_params', new_total_params)
print('new_total_flops', new_total_flops)

if args.use_cuda:
    newmodel.cuda()

layer_id_in_cfg = 0
start_mask = torch.zeros(3)
end_mask = cfg_mask[layer_id_in_cfg]
for k, (m0, m1) in enumerate(zip(model.modules(), newmodel.modules())):
    if k == 2: assert isinstance(m0, SpConvBlock)
    if (k == 2 and isinstance(m0, SpConvBlock)) or isinstance(m0, SpMbBlock):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        if isinstance(m0, SpMbBlock):
            # conv1, depth wise conv
            conv_weight = m0.conv1.weight.data.clone()
            m1.conv1.weight.data = torch.cat((conv_weight, conv_weight[idx0.tolist(), :, :, :].clone()), 0)  

            # bn1
            m1.bn1.weight.data = torch.cat((m0.bn1.weight.data.clone(), m0.bn1.weight.data[idx0.tolist()].clone()))
            m1.bn1.bias.data = torch.cat((m0.bn1.bias.data.clone(), m0.bn1.bias.data[idx0.tolist()].clone()))
            m1.bn1.running_mean =torch.cat((m0.bn1.running_mean.clone(), m0.bn1.running_mean[idx0.tolist()].clone()))
            m1.bn1.running_var = torch.cat((m0.bn1.running_var.clone(), m0.bn1.running_var[idx0.tolist()].clone()))
        
        if isinstance(m0, SpMbBlock):
            m0_conv = m0.sp_conv.conv2d
            m1_conv = m1.sp_conv.conv2d
            m0_bn = m0.sp_conv.bn
            m1_bn = m1.sp_conv.bn
        else:
            m0_conv = m0.conv2d
            m1_conv = m1.conv2d
            m0_bn = m0.bn
            m1_bn = m1.bn
       
        # copy conv weights
        conv_weight = m0_conv.weight.data.clone()
        conv_weight[:, idx0.tolist(), :, :] /= 2.
        w1 = torch.cat((conv_weight, conv_weight[:, idx0.tolist(), :, :]), 1)
        w1 = torch.cat((w1.clone(), w1[idx1.tolist(), :, :, :].clone()), 0)  
        eig_v = min_eig_vecs[layer_id_in_cfg]
        eig_v = eig_v / eig_v.pow(2).sum([1,2,3]).sqrt().view(-1, 1, 1, 1)
        eig_v = torch.cat((eig_v, eig_v[:, idx0.tolist(), :, :]), 1)
        #print(eig_v.shape, m1_conv.weight.shape)
        w1[idx1.tolist(), :, :, :] += 1e-2 * eig_v[idx1.tolist(), :, :, :]
        w1[conv_weight.size(0):, :, :, :] -= 1e-2 * eig_v[idx1.tolist(), :, :, :]

        m1_conv.weight.data = w1.clone()

        # copy bn weights 
        bn_weight = m0_bn.weight.data.clone()
        m1_bn.weight.data = torch.cat((bn_weight.clone(), bn_weight[idx1.tolist()].clone()))

        bn_bias = m0_bn.bias.data.clone()
        m1_bn.bias.data = torch.cat((bn_bias.clone(), bn_bias[idx1.tolist()].clone()))

        bn_running_mean = m0_bn.running_mean.clone()
        m1_bn.running_mean = torch.cat((bn_running_mean.clone(), bn_running_mean[idx1.tolist()].clone()))

        bn_running_var = m0_bn.running_var.clone()
        m1_bn.running_var = torch.cat((bn_running_var.clone(), bn_running_var[idx1.tolist()].clone()))

        layer_id_in_cfg += 1
        start_mask  = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):
            end_mask = cfg_mask[layer_id_in_cfg]
 
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        #if idx0.size == 0: continue
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        fc_weight = m0.weight.data.clone()
        fc_weight[:, idx0.tolist()] /= 2.
        fc_bias = m0.bias.data.clone()
        m1.weight.data = torch.cat((fc_weight, fc_weight[:, idx0.tolist()]), 1)
        m1.bias.data = m0.bias.data.clone()

# save as a dataparalle object
if args.use_cuda:
    newmodel = torch.nn.DataParallel(newmodel).cuda()

#print('testing acc after splitting')
validate(val_loader, newmodel, criterion)

torch.save( {
        'cfg': cfg, 
        'split_index': args.split_index,
        'state_dict': newmodel.state_dict(),
        'load': args.load,
        'grow': args.grow,
        'split_groups': split_groups,
        'min_eig_vals': min_eig_vals,
        'args': args,
    }, model_save_path)
    
    

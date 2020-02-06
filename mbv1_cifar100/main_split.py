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
from sp_mbnet import sp_mbnet as mbnet
from sp_mbnet import SpMbBlock, SpConvBlock
from config import * 
from compute_flops import print_model_param_nums, print_model_param_flops

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)', required=True)
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--load', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)', required=True)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--split-index', default="1", type=str,
                    help='#number of split', required=True)
parser.add_argument('--energy', action='store_true', default=False,
                    help='energy aware splitting')
parser.add_argument('--params', action='store_true', default=False,
                    help='paramter aware splitting')
parser.add_argument('--save', default='split/saved_models', type=str,
                    help='energy aware splitting')
parser.add_argument('--grow', type=float, default=-1, 
                    help='globally split grow rate (default: 0.2)', required=True)
parser.add_argument('--exp-name', type=str, default=None, 
                    help='exp name', required=True)
#parser.add_argument('--debug', action='store_true', default=False,
#                    help='debug mode')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda') if args.cuda else torch.device('cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader, test_loader = \
        get_data_loader(dataset = args.dataset, train_batch_size = args.batch_size, test_batch_size = args.test_batch_size, use_cuda=args.cuda)

args.save = os.path.join(args.save, args.exp_name)
if args.energy:
    args.save = os.path.join(args.save, 'energy_aware')
if args.params:
    args.save = os.path.join(args.save, 'params_aware')

logging_file_path = '{}_split_{}.log'.format(args.dataset, args.split_index)
model_save_path = 'fast_{}_{}.pth.tar'.format(args.dataset,  args.split_index)

if not os.path.exists(args.save):
    os.makedirs(args.save)

#### check exsiting models ##
#if os.path.isfile(os.path.join(args.save, model_save_path)):
#    pre_check = torch.load(os.path.join(args.save, model_save_path))
#    print(args)
#    print(pre_check['args'])
#    if args == pre_check['args']:
#        print('no need to run, load from {}'.format(model_save_path))
#        sys.exit(0)

#########################################################
# create file handler which logs even debug messages 
import logging
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(args.save, logging_file_path), mode='w')

formatter = logging.Formatter('%(asctime)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

log.addHandler(fh)
log.addHandler(ch)
#########################################################

assert args.load
assert os.path.isfile(args.load)
log.info("=> loading checkpoint '{}'".format(args.load))
checkpoint = torch.load(args.load)
model = mbnet(dataset=args.dataset, cfg=checkpoint['cfg']).to(device)

from collections import OrderedDict
new_state_dict = OrderedDict()

selected_model_keys = [k for k in model.state_dict().keys() if not (k.endswith('.y') or k.endswith('.v') or k.startswith('net_params') or k.startswith('y_params') or k.startswith('v_params'))]
saved_model_keys = checkpoint['state_dict']
if len(selected_model_keys) == len(saved_model_keys):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k0, k1 in zip(selected_model_keys, saved_model_keys):
        new_state_dict[k0] = checkpoint['state_dict'][k1]   
    
    model_dict = model.state_dict()
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
else:
    ## load form sp_mbnet model ##
    model.load_state_dict(checkpoint['state_dict'])
log.info("=> loaded checkpoint '{}' " .format(args.load))
del checkpoint

optimizer_v = optim.RMSprop(model.v_params, lr=args.lr, momentum=0.9, alpha=0.9)


def adaptive_weights_perturbation(W, scale=0.1):
    min_v = torch.min(W)
    max_v = torch.max(W)
    return torch.zeros_like(W).uniform_(scale*min_v, scale*max_v)


def train(epoch):
    model.train() 
    train_acc = 0.

    min_eig_vals, min_eig_vecs = None, None
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer_v.zero_grad()
        
        ## splitting aware forward ##
        output = model.sp_forward(data)
        ce = F.cross_entropy(output, target)
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
            log.info('Train Epoch: {} [{}/{} ({:.1f}%)]'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader)))
            #if batch_idx > 20:
            #    break

    for k in range(len(min_eig_vals)):
        min_eig_vals[k] /= len(train_loader)
        min_eig_vecs[k] /= len(train_loader)

    log.info('Train Epoch: loss {:.2f}'.format(sum([v.sum() for v in min_eig_vals]).data))
    return min_eig_vals, min_eig_vecs


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        #output = model.sp_forward(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()

    test_loss /= len(test_loader.dataset)
    log.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / float(len(test_loader.dataset))))
    return correct / float(len(test_loader.dataset))

for k, m in enumerate(model.modules()):
    if k == 2: assert isinstance(m, SpConvBlock)
    if isinstance(m, SpMbBlock) or (k == 2 and isinstance(m, SpConvBlock)):
        m.reset_yv_()
log.info('acc before splitting')
test(model)

for epoch in range(1, 1+args.epochs):
    if epoch % 2 == 0:
        for param_group in optimizer_v.param_groups:
            param_group['lr'] *= 0.2
    min_eig_vals, min_eig_vecs = train(epoch)
    #break

########################################
##### select neurons ######
########################################
print_model_param_flops(model.cpu(), 32)
model.to(device)

total = 0
for m in min_eig_vals:
    total += len(m)

cfg_grow = []
cfg_mask = []

block_weigths_norm = []
if args.energy or args.params:
    ## flops ##
    cfg = model.cfg

    params_inc_per_neuron, flops_inc_per_neuron = [], []
    for i, (c, r) in enumerate(zip(cfg, resolutions)):
        # number of channles in the previous layer

        flops_inc = get_flops_inc_per_layer(model, i)
        params_inc = get_params_inc_per_layer(model, i)

        flops_inc_per_neuron.append( np.sqrt(flops_inc) )
        #params_inc_per_neuron.append( np.log(params_inc) )
        params_inc_per_neuron.append( 1.0 )

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
        A_ub.append(A_ub_i * all_f)
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

    #res = linprog(all_lambda, A_ub=np.expand_dims(all_f, 0), b_ub=MAX_RESOUCE, bounds=(0., 1.))
    res = linprog(all_lambda, A_ub=A_ub, b_ub=b_ub, bounds=(0., 1.), method='interior-point')
    assert res.status == 0
    ### select top n_max neurons
    #sorted_idx = np.argsort(res.x)[::-1]
    #thre = res.x[sorted_idx[int(n_max)-1]]
    thre = 0.99

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
            log.info('layer index: {:d} \t total channel: {:d} \t splitting channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))

            if isinstance(m, SpConvBlock):
                conv_w = m.conv2d.weight
            else:
                conv_w = m.sp_conv.conv2d.weight
            block_weigths_norm.append( torch.sum(conv_w**2, [1,2,3]).sqrt() )

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
            log.info('layer index: {:d} \t total channel: {:d} \t splitting channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))

            if isinstance(m, SpConvBlock):
                block_weigths_norm.append( torch.norm(m.conv2d.weight, dim=0).mean() )
            else:
                block_weigths_norm.append( torch.norm(m.sp_conv.conv2d.weight, dim=0).mean() )

        elif isinstance(m, nn.MaxPool2d):
            cfg_grow.append('M')

grow_ratio = sum([m.sum() for m in cfg_mask])/total
cfg = [ c+v if isinstance(v, int) else (c+v[0], v[1]) for c, v in zip(cfg_grow, model.cfg)]

print(model.cfg)
print(cfg)

#eigen_vals = np.concatenate([ v.cpu().numpy() for v in min_eig_vals], axis=0)
#for v in block_weigths_norm:
#    print(v.shape)
#weight_vals = np.concatenate([v.detach().cpu().numpy() for v in block_weigths_norm])
#print(eigen_vals.shape, weight_vals.shape)
#
#plt.plot(eigen_vals )
#plt.savefig('min_eig_vals.png')
#plt.close()
#
#plt.plot(weight_vals)
#plt.savefig('block_weigths_norm.png')
#plt.close()
#sys.exit(0)

##################################
##### copy weights and split #####
##################################
newmodel = mbnet(dataset=args.dataset, cfg=cfg) 
newmodel.to(device)

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

        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        if isinstance(m0, SpMbBlock):
            # conv1, depth wise conv
            conv_weight = m0.conv1.weight.data.clone()
            #m1.conv1.weight.data = torch.cat((conv_weight, conv_weight[idx0.tolist(), :, :, :].clone()), 0)  
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


log.info(model.cfg)
log.info(newmodel.cfg)

print('acc after splitting')
test(newmodel)

#if not args.debug:
torch.save({
    'cfg': newmodel.cfg, 
    'split_index': args.split_index,
    'grow': args.grow,
    'min_eig_vals': min_eig_vals,
    'state_dict': newmodel.state_dict(),
    'args': args,
}, os.path.join(args.save, model_save_path))

print(os.path.join(args.save, model_save_path))
new_num_parameters = print_model_param_nums(newmodel.cpu())
new_num_flops = print_model_param_flops(newmodel.cpu(), 32)



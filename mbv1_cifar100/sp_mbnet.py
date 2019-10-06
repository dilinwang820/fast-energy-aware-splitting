'''MobileNetV1 in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Swish
from sp_conv import SpConvBlock

__all__ = ['sp_mbnet']

from config import *

class SpMbBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, activation='swish'):
        super(SpMbBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        
        self.sp_conv = SpConvBlock(in_planes, out_planes, kernel_size=1, stride=1, padding=0, activation=activation)

        self.nn_act = Swish() if activation == 'swish' else nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.nn_act(self.bn1(self.conv1(x)))
        out = self.sp_conv(out)
        return out

    def sp_forward(self, x):
        out = self.nn_act(self.bn1(self.conv1(x)))
        out = self.sp_conv.sp_forward(out)
        return out

    def reset_yv_(self):
        self.sp_conv.reset_yv_()


class sp_mbnet(nn.Module):

    def __init__(self, dataset='cifar10', cfg=None, activation='swish'):
        super(sp_mbnet, self).__init__()

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise NotImplementedError
        self.num_classes = num_classes

        self.activation = activation
        if cfg is None:
            cfg = splitcfg

        if isinstance(cfg, str):
            cfg = [int(c) for c in cfg.split("_")]
            cfg = [c0 if isinstance(c1, int) else (c0, c1[1]) for c0, c1 in zip(cfg, splitcfg)]

        self.cfg = cfg
        self.layers = self._make_layers()
        self.linear = nn.Linear(cfg[-1], num_classes)

        net_params, y_params, v_params = [], [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.endswith(".y"):
                    y_params.append(param)
                elif name.endswith(".v"):
                    v_params.append(param)
                else:
                    net_params.append(param)

        self.net_params = nn.ParameterList(net_params)
        self.y_params = nn.ParameterList(y_params)
        self.v_params = nn.ParameterList(v_params)


    def _make_layers(self,):
        layers = []

        in_planes = self.cfg[0]
        layers += [SpConvBlock(3, in_planes, kernel_size=3, stride=1, padding=1, activation=self.activation)]

        for i, x in enumerate(self.cfg[1:]):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(SpMbBlock(in_planes, out_planes, stride, activation=self.activation))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def sp_forward(self, x):
        out = x
        for l in self.layers:
            out = l.sp_forward(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def test():
    net = sp_mbnet()
    x = torch.randn(1,3,32,32)
    y = net.sp_forward(x)
    print(y.size())
    y = net(x)
    print(y.size())

test()


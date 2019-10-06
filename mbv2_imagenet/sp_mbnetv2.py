"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

from sp_conv import SpConvBlock
from config import *

__all__ = ['sp_mbnetv2']

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

class ConvBlock(nn.Module):

    def __init__(self, inp, oup, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.sp_conv = SpConvBlock(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.sp_conv(x)
        return out

    def sp_forward(self, x):
        out = self.sp_conv.sp_forward(x)
        return out

    def reset_yv_(self):
        self.sp_conv.reset_yv_()


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, dw_channels, out_channels, stride, identity, no_expand=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.no_expand = no_expand
        self.identity = identity

        if self.no_expand:
            dw_channels = in_channels
            assert not self.identity, 'no residual for the first mb block'
        if self.identity:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        if not self.no_expand:
            # first pointwise layer
            self.t_pw_conv = SpConvBlock(in_channels, dw_channels, kernel_size=1, stride=1, padding=0)

        # depthwise layer with stride
        self.dw_conv = nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(dw_channels)

        #splitting aware conv block, no activation
        self.b_pw_conv = SpConvBlock(dw_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=None)

        self.act_func = Swish()


    def forward(self, x):
        if not self.no_expand:
            out = self.t_pw_conv(x)
        else:
            out = x

        out = self.act_func( self.dw_bn(self.dw_conv(out)) )
        out = self.b_pw_conv(out)
        if self.identity:
            return out + self.residual(x) 
        return out

    def sp_forward(self, x):
        if not self.no_expand:
            out = self.t_pw_conv.sp_forward(x)
        else:
            out = x
        out = self.act_func( self.dw_bn(self.dw_conv(out)) )
        out = self.b_pw_conv.sp_forward(out)
        if self.identity:
            return out + self.residual(x)
        return out
   
    def reset_yv_(self,):
        if not self.no_expand:
            self.t_pw_conv.reset_yv_()
        self.b_pw_conv.reset_yv_()


class SpMobileNetV2(nn.Module):
    def __init__(self, input_size=224, cfg=None):
        super(SpMobileNetV2, self).__init__()

        num_classes = 1000
        self.num_classes = num_classes

        if cfg is None:
            cfg = splitcfg
        self.cfg = cfg 

        # building first layer
        assert input_size % 32 == 0

        input_channel = self.cfg[0][0] # first conv
        last_channel = self.cfg[-1][0] # last conv

        layers = [ConvBlock(3, input_channel, kernel_size=3, stride=self.cfg[0][1], padding=1)]
        # building inverted residual blocks
        block = InvertedResidual

        # first mobilenet block, no expand
        output_channel, s, i = self.cfg[1]
        layers.append(block(input_channel, input_channel, output_channel, s, i, no_expand=True))
        input_channel = output_channel 

        # other blocks with expand
        mb_cfg = cfg[2:-1]
        #print(mb_cfg)
        #for n_dw, n_out, s, i in self.cfg[1:-1]:
        for layer_idx in range(len(mb_cfg)//2):
            n_dw = mb_cfg[2*layer_idx][0]
            output_channel, s, i = mb_cfg[2*layer_idx+1]
            #print(layer_idx, n_dw, output_channel, s, i)
            layers.append(block(input_channel, n_dw, output_channel, s, i, no_expand=False))
            input_channel = output_channel

        # building last several layers
        layers += [ConvBlock(input_channel, last_channel, kernel_size=1, stride=self.cfg[-1][1], padding=0)]
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AvgPool2d(7)

        self.classifier = nn.Linear(last_channel, num_classes)

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

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    def sp_forward(self, x):
        for l in self.features:
            x = l.sp_forward(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def sp_mbnetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return SpMobileNetV2(**kwargs)


if __name__ == '__main__':
    net = sp_mbnetv2()
    x = Variable(torch.FloatTensor(2, 3, 224, 224))
    y = net(x)
    print(y.data.shape)

    print_cfg(net.cfg)

    from compute_flops import print_model_param_nums, print_model_param_flops
    total_flops = print_model_param_flops(net.cpu(), 224, multiply_adds=False) 


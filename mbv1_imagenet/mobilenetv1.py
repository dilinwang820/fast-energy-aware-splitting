'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = ['mobilenetv1']

defaultcfg = {
                '1.0': [32, 64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024,2), 1024],
                '0.75': [24, 48, (96, 2), 96, (192, 2), 192, (384, 2), 384, 384, 384, 384, 384, (768, 2), 768],
                '0.5': [16, 32, (64, 2), 64, (128, 2), 128, (256, 2), 256, 256, 256, 256, 256, (512, 2), 512],
                '0.25': [8, 16, (32, 2), 32, (64, 2), 64, (128, 2), 128, 128, 128, 128, 128, (256, 2), 256],
                'flat_v1': [32, 64, (128, 2), 128, (256, 2), 256, (256, 2), 256, 256, 256, 256, 256, (1024,2), 1024],
                'flat_v2': [32, 64, (96, 2), 96, (192, 2), 192, (384, 2), 384, 384, 384, 384, 384, (1024,2), 1024],
                'flat_v3': [32, 64, (96, 2), 96, (192, 2), 192, (384, 2), 384, 384, 384, 384, 384, (1280,2), 1280],
                'flat_v4': [32, 64, (96, 2), 96, (144, 2), 144, (216, 2), 216, 216, 216, 216, 216, (1280,2), 1280],
                'flat_v5': [32, 32, (64, 2), 64, (128, 2), 128, (256, 2), 256, 256, 256, 256, 256, (512, 2), 512],
            }

## 50% flops ##
amccfg =     [24, 48, (96, 2),  80,  (192, 2), 200, (328, 2), 352, 368, 360, 328, 400, (736, 2), 752]

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        init = 1.
        self.weight = Parameter(torch.Tensor(1).fill_(init))

    def forward(self, x):
        x = x * F.sigmoid(self.weight * x)
        return x

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.swish = Swish()

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.bn2(self.conv2(out)))
        out = self.swish(self.bn1(self.conv1(x)))
        out = self.swish(self.bn2(self.conv2(out)))
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, activation='swish'):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

        self.nn_act = Swish() if activation == 'swish' else nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.nn_act(self.bn(self.conv(x)))
        return out


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, cfg=None, amc=False, m='1.0'):
        super(MobileNet, self).__init__()

        if amc: cfg = amccfg

        if cfg is None:
            if 'flat' in m:
                cfg = defaultcfg[m]
            else:
                m = float(m)
                cfg = [int(v*m) if isinstance(v, int) else (int(v[0]*m), v[1]) for v in defaultcfg['1.0']]
        assert len(cfg) == len(defaultcfg['1.0'])

        self.num_classes = num_classes

        self.cfg = cfg
        self.features = self._make_layers()
        self.linear = nn.Linear(cfg[-1], num_classes)

    def _make_layers(self,):
        layers = []

        in_planes = self.cfg[0]
        #conv2d = nn.Conv2d(3, in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        #layers += [conv2d, nn.BatchNorm2d(32), nn.ReLU(inplace=True)]
        #layers += [conv2d, nn.BatchNorm2d(in_planes), Swish()]
        conv_block = ConvBlock(3, self.cfg[0], stride=2)
        layers.append(conv_block)

        for i, x in enumerate(self.cfg[1:]):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]

            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.features(x)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def mobilenetv1(**kwargs):
    return MobileNet(**kwargs)


def test():
    net = MobileNet(amc=True)
    from compute_flops import print_model_param_nums, print_model_param_flops
    #x = torch.randn(1,3,224,224)
    #y = net(x)
    #print(y.size())
    print_model_param_nums(net)
    print_model_param_flops(net)
test()

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, activation='swish'):
        super(SpConvBlock, self).__init__()

        assert activation in ['relu', 'swish'] or activation is None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation


        if isinstance(kernel_size, int):
            self.kh = self.kw = kernel_size
        else:
            assert len(kernel_size) == 2
            self.kh, self.kw = kernel_size
        
        if isinstance(padding, int):
            self.ph = self.pw = padding
        else:
            assert len(padding) == 2
            self.ph, self.pw = padding
        #self.padding = padding
        
        if isinstance(stride, int):
            self.dh = self.dw = stride
        else:
            assert len(stride) == 2
            self.dh, self.dw = stride
        #assert stride == 1
       
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(self.kh, self.kw), \
                                    stride=(self.dh, self.dw), padding=(self.ph, self.pw), bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

        self.y = nn.Parameter(torch.zeros(self.conv2d.weight.size()), requires_grad=True) #n_out * n_in * k * k
        self.v = nn.Parameter(torch.zeros(self.conv2d.weight.size()), requires_grad=True)

        self._initialize_weights()
        self.reset_yv_()

    def _initialize_weights(self):
        n = self.conv2d.weight.data.shape[2] * self.conv2d.weight.data.shape[3] * self.conv2d.weight.data.shape[0]
        self.conv2d.weight.data.normal_(0, math.sqrt(2. / n))

    def reset_yv_(self):
        #self.y.data.uniform_(-1e-4, 1e-4)  # very small weights
        self.y.data.zero_()  # very small weights
        self.v.data.uniform_(-0.1, 0.1) 


    def get_conv_patches(self, input):
        # Pad tensor to get the same output 
        x = F.pad(input, (self.pw, self.pw, self.ph, self.ph))  #(padding_left, padding_right, padding_top, padding_bottom)
        # get all image windows of size (kh, kw) and stride (dh, dw)
        patches = x.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        # Permute so that channels are next to patch dimension
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [batch_size, h, w, n_in, kh, kw]
        return patches

    def _swish(self, x):
        return x * torch.sigmoid(x)
    
    def _d_swish(self, x):
        s = torch.sigmoid(x)
        return  s + x * s * (1. - s)
    
    def _dd_swish(self, x):
        s = torch.sigmoid(x)
        return s*(1.-s) + s + x*s*(1.-s) - (s**2 +2.*x*s**2*(1.-s)) 


    def _dd_softplus(self, x, beta=3.):
        z = x * beta
        o = F.sigmoid(z)
        return beta * o * (1. - o)


    def forward(self, input):
        r"""
            regular forward
        """
        bn_out = self.bn(self.conv2d(input))
        if self.activation is None:
            out = bn_out
        elif self.activation == 'swish':
            out = self._swish(bn_out)
        else:
            out = F.relu(bn_out)
        return out # batch_size * n_out * h * w


    def sp_forward(self, input):
        self.bn.eval() # fix all bn weights!!!

        conv_out = self.conv2d(input) # batch size * n_out * h * w
        bn_out = self.bn(conv_out)  # batch size * n_out * h * w

        # batch normalization 
        # y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
        bn_coff = 1. / torch.sqrt(self.bn.running_var + 1e-5) * self.bn.weight # n_out
        bn_coff = bn_coff.view(1, -1, 1, 1) # 1, n_out, 1, 1

        if self.activation is None:
            act_sec_ord_grad = torch.ones_like(bn_out)
        elif self.activation == 'swish':
            act_sec_ord_grad = self._dd_swish(bn_out) # batch_size * n_out * h * w
        else:
            act_sec_ord_grad = self._dd_softplus(bn_out) # batch_size * n_out * h * w

        # now consider the inner linear terms, that would be the inner product between  
        patches = self.get_conv_patches(input) #[batch_size, h, w, n_in, kh, kw]    
        # the second order gradient can be view as a running sum over [h, w] independently
        batch_size, h, w, n_in, kh, kw = patches.size()
        n_out = self.out_channels

        dim = n_in * kh * kw
        dw = patches.reshape(batch_size, h, w, dim).permute(3, 0, 1, 2).reshape(dim, batch_size * h * w) # dim * [batch_size * h * w]

        y = self.y.reshape(n_out, dim) # n_out * dim
        v = self.v.reshape(n_out, dim)
        left = torch.matmul(y, dw).reshape(n_out, batch_size, h, w).permute(1, 0, 2, 3) * bn_coff # batch_size * n_out * h * w 
        right = torch.matmul(v, dw).reshape(n_out, batch_size, h, w).permute(1, 0, 2, 3) * bn_coff  # batch_size * n_out * h * w 

        aux = act_sec_ord_grad * left * right
        
        if self.activation is None:
            out = bn_out + aux
        elif self.activation == 'swish':
            out = self._swish(bn_out) + aux
        else:
            out = F.relu(bn_out) + aux

        return out


    def test_conv_grad(self, input):
        input = input[0].view(1, *input.size()[1:])
        assert input.size(0) == 1


        ### case 1, without bn normalization
        #conv_out = self.conv2d(input) 
        #out = F.softplus(conv_out)
        #w_grad = torch.autograd.grad(out.sum(), self.conv2d.weight)[0]  # all neurons have the same gradient, only depends on input
        #patches = self.get_conv_patches(input) # batch_size, h, w, n_in, kh, kw
        #batch_size, h, w, n_in, kh, kw = patches.size()
        #n_out = self.out_channels
        #d_sigma = F.sigmoid(conv_out) # batch_size * n_out * h * w

        #dw = d_sigma.view(batch_size, n_out, h, w, 1, 1, 1) * patches.view(batch_size, 1, h, w, n_in, kh, kw)
        #dw = dw.sum([2, 3]) # batch_size * n_out * n_in * kh * kw
        #print(( (w_grad - dw)**2).sum() )

        ### case 2, with bn normalization
        self.bn.eval() # fix the running var
        conv_out = self.conv2d(input) 

        bn_coff = 1. / torch.sqrt(self.bn.running_var + 1e-5) * self.bn.weight # n_out
        bn_coff = bn_coff.view(1, -1, 1, 1, 1) # 1, n_out, 1, 1
        bn_out = F.softplus(self.bn(conv_out))

        bn_w_grad = torch.autograd.grad(bn_out.sum(), self.conv2d.weight)[0]  # all neurons have the same gradient, only depends on input

        patches = self.get_conv_patches(input) # batch_size, h, w, n_in, kh, kw
        batch_size, h, w, n_in, kh, kw = patches.size()
        n_out = self.out_channels
        bn_d_sigma = F.sigmoid(self.bn(conv_out)) # batch_size * n_out * h * w

        bn_dw = bn_d_sigma.view(batch_size, n_out, h, w, 1, 1, 1) * patches.view(batch_size, 1, h, w, n_in, kh, kw)
        bn_dw = bn_dw.sum([2, 3]) # batch_size * n_out * n_in * kh * kw

        bn_dw = bn_dw * bn_coff

        #out = self.bn(conv_out)
        #torch.autograd.grad(out, conv_out)
        print(( (bn_w_grad - bn_dw)**2).sum() )

#if __name__ == '__main__':
#    #net = vgg()
#    x = Variable(torch.randn(16, 32, 28, 28))
#    sp_conv = SpConvBlock(32, 32, kernel_size=(2, 3), stride=(3,1), padding=(9,0), groups=32)
#    sp_conv.test_conv_grad(x)
#    #y = net.sp_forward(x)
#    #print(y.data.shape)
#
#    #print(net)
#
#    #for name, param in net.named_parameters():
#    #    if param.requires_grad:
#    #        print( name, param.data.shape)
#
#    #conv2d = nn.Conv2d(3, 10, kernel_size=3, padding=1, bias=False)
#    #y = conv2d(x)
#    #print(conv2d.weight.shape)
#    #g_conv_w = torch.autograd.grad(y.sum(), conv2d.weight)
#    #print(g_conv_w[0].shape)
#
#    #for m in net.modules():
#    #    if isinstance(m, SpConvBlock):
#    #        print(m.add_dummy())

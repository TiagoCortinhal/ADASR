import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1,1,1), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False

def conv(ni, nf, kernel, bias=True, init=False,stride=1,group=1,dilatation=1):
    c = nn.Conv2d(ni, nf, kernel, padding=kernel // 2, bias=bias,stride=stride,groups=group,dilation=dilatation)
    if init:
        kernel = icnr(c.weight, scale=2)
        c.weight.data.copy_(kernel)
    return c


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = init(subkernel)
    subkernel = subkernel.transpose(0, 1)
    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)
    kernel = subkernel.repeat(1, 1, scale ** 2)
    transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)
    kernel = kernel.transpose(0, 1)
    return kernel


class ResBlock(nn.Module):
    def __init__(self, conv, nf, kernel, bias=True, bn=False, act=True, res_scale=1.,group=1,dilatation=1):
        super(ResBlock, self).__init__()
        layers = [conv(nf, nf, kernel, bias=bias,group=group,dilatation=dilatation)]
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        if act:
            layers.append(nn.ReLU())
        layers.append(conv(nf, nf, kernel+1, bias=bias,group=group,dilatation=dilatation-1))
        self.block = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.block(x) * self.res_scale + x
        return res

class Extractor(nn.Module):
    def __init__(self, res, conv=conv, nf=64, mean=(0.4488, 0.4371, 0.4040)):
        super(Extractor, self).__init__()
        self.res = res
        self.nf = nf
        self.kernel = 2
        self.conv = conv

        init = [nn.Conv2d(3, 240,1,groups=3)]

        blocks = []
        for _ in range(self.res):
            blocks.append(ResBlock(self.conv, 240, self.kernel, res_scale=0.1,group=3,dilatation=2))
        blocks.append(nn.Conv2d(240, 120, 1,groups=3))
        blocks.append(self.conv(120, self.nf, self.kernel,dilatation=2))

        self.init = nn.Sequential(*init)
        self.blocks = nn.Sequential(*blocks)
        self.sub_mean_source = MeanShift(1, rgb_mean=mean)
        self.sub_mean_target = MeanShift(1, rgb_mean=(0.5137, 0.4156, 0.3649))

    def forward(self, x,domain='source'):
        if domain == 'source':
            x = self.sub_mean_source(x)
        else:
            x = self.sub_mean_target(x)
        init_x = self.init(x)
        x = self.blocks(init_x)
        return x
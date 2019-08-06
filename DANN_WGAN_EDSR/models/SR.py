import math
import torch
from torch import nn


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1,1,1), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False

def conv(ni, nf, kernel, bias=True, init=False):
    c = nn.Conv2d(ni, nf, kernel, padding=kernel // 2, bias=bias)
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
    def __init__(self, conv, nf, kernel, bias=True, bn=False, act=True, res_scale=1.):
        super(ResBlock, self).__init__()
        layers = [conv(nf, nf, kernel, bias=bias)]
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        if act:
            layers.append(nn.ReLU())
        layers.append(conv(nf, nf, kernel, bias=bias))
        self.block = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.block(x) * self.res_scale + x
        return res


def Upsampling(conv, scale, nf, bn=False, act=False, bias=True, init=True):
    layers = []

    for _ in range(int(math.log(scale, 2))):
        layers.append(conv(nf, 4 * nf, 3, bias, init=init))
        layers.append(nn.PixelShuffle(2))
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    if act:
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)

class EDSR(nn.Module):
    def __init__(self, scale, res, conv=conv,mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR, self).__init__()


        self.scale = scale
        self.res = res
        self.nf = 64
        self.kernel = 3
        self.conv = conv

        init = [self.conv(64, self.nf, self.kernel)]

        blocks = []
        for _ in range(self.res):
            blocks.append(ResBlock(self.conv, self.nf, self.kernel, res_scale=0.1))

        blocks.append(conv(self.nf, self.nf, self.kernel))

        end = [Upsampling(conv, self.scale, self.nf), conv(self.nf, 3, self.kernel)]

        self.init = nn.Sequential(*init)
        self.blocks = nn.Sequential(*blocks)
        self.end = nn.Sequential(*end)

        self.add_mean_source = MeanShift(1, rgb_mean=mean, sign=1)
        self.add_mean_target = MeanShift(1, rgb_mean=(0.5137, 0.4156, 0.3649), sign=1)

    def forward(self, xx,domain='source'):
        init_x = self.init(xx)
        x = self.blocks(init_x)
        x = self.end(x)
        if domain == 'source':
            x = self.add_mean_source(x)
        else:
            x = self.add_mean_target(x)
        return x

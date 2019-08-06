import torch
from torch import nn


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
            layers.append(nn.ReLU(inplace=True))
        layers.append(conv(nf, nf, kernel, bias=bias))
        self.block = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.block(x) * self.res_scale + x
        return res


def Upsampling(conv, scale, nf, bn=False, act=False, bias=True, init=True):
    layers = [conv(nf, 4 * nf, 3, bias, init=init), nn.PixelShuffle(scale)]

    if bn:
        layers.append(nn.BatchNorm2d(nf))
    if act:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class EDSR(nn.Module):
    def __init__(self, scale, res, conv=conv):
        super(EDSR, self).__init__()

        self.scale = scale
        self.res = res
        self.nf = 64
        self.kernel = 3
        self.conv = conv

        init = [self.conv(3, self.nf, self.kernel)]

        blocks = []
        for _ in range(self.res):
            blocks.append(ResBlock(self.conv, self.nf, self.kernel, res_scale=0.1))

        blocks.append(conv(self.nf, self.nf, self.kernel))

        end = [Upsampling(conv, self.scale, self.nf), conv(self.nf, 3, self.kernel)]

        self.init = nn.Sequential(*init)
        self.blocks = nn.Sequential(*blocks)
        self.end = nn.Sequential(*end)

    def forward(self, x):
        x = self.init(x)
        res = self.blocks(x)
        res += x
        x = self.end(res)
        return x

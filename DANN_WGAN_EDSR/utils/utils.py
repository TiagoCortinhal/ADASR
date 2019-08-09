import torch
from torch import nn
from torch.autograd import Variable
from utils.options import args
import math
LAMBDA = 10


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


def calculate_gradient_penalty(D, real_images, fake_images):
    eta = torch.FloatTensor(args.batch_size, 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(args.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.cuda()

    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.cuda()

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(
                                        prob_interpolated.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()



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
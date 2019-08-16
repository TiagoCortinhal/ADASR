import torch.nn as nn
from utils.utils import MeanShift
from utils.utils import conv



class ResBlock(nn.Module):
    def __init__(self, conv, nf, kernel, bias=True, bn=False, act=True, res_scale=1.,group=1,dilatation=1):
        super(ResBlock, self).__init__()
        layers = [conv(nf, nf, kernel, bias=bias,group=group,dilatation=dilatation)]
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        if act:
            layers.append(nn.ReLU())
        layers.append(conv(nf, nf, kernel-1, bias=bias,group=group,dilatation=dilatation-1))
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

        #init = [nn.Conv2d(3, 240,1,groups=3)]
        init = [nn.Conv2d(3, 240,self.kernel,padding=1,dilation=2,groups=3)]

        blocks = []
        for _ in range(self.res):
            blocks.append(ResBlock(self.conv, 240, self.kernel, res_scale=0.1,group=3,dilatation=2))
        blocks.append(nn.Conv2d(240, 120, 1,groups=3))
        blocks.append(self.conv(120, self.nf, self.kernel,dilatation=2))

        self.init = nn.Sequential(*init)
        self.blocks = nn.Sequential(*blocks)
        self.sub_mean = MeanShift(1, rgb_mean=mean)

    def forward(self, x):
        x = self.sub_mean(x)
        init_x = self.init(x)
        x = self.blocks(init_x)
        return x
from torch import nn

from utils.utils import MeanShift
from utils.utils import ResBlock
from utils.utils import Upsampling
from utils.utils import conv


class EDSR(nn.Module):
    def __init__(self, scale, res, conv=conv, mean=(0.4488, 0.4371, 0.4040)):
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

        self.add_mean = MeanShift(1, rgb_mean=mean, sign=1)

    def forward(self, xx):
        init_x = self.init(xx)
        x = self.blocks(init_x)
        x = self.end(x)
        x = self.add_mean(x)
        return x

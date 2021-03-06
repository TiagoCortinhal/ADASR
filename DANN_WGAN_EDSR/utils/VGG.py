import torch
import torch.nn as nn
import torchvision.models as models


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        for p in self.parameters():
            p.requires_grad = False
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        self.vgg = nn.Sequential(*modules[:8])

        self.sub_mean = MeanShift(1, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def forward(self, x):
        x = self.sub_mean(x)
        return self.vgg(x)

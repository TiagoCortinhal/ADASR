import torch
import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        self.vgg = nn.Sequential(*modules[:8])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.vgg(x)

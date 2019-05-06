import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn

from utils.DatasetManager import DatasetManager
from utils.options import args





def eval():
    from EDSR import EDSR
    os.makedirs(os.path.dirname('./out/8_4/hr'),
                exist_ok=True)

    data = DatasetManager(training=False, factor=8)
    dataMan = DataLoader(data, batch_size=1)

    EDSR = EDSR(2,16)
    a = torch.load('./8_4/training_SR_300.pth',
                      map_location=lambda storage, loc: storage)
    EDSR.load_state_dict(a.module.state_dict())

    EDSR.eval()

    for i, (X, y) in enumerate(dataMan):

        outputs = EDSR(X)
        torchvision.utils.save_image(y,'./out/8_4/hr/hr_{}.png'.format(i + 1))
        torchvision.utils.save_image(outputs[-1], './out/8_4/eval/300/eval_{}.png'.format(i + 1))
        torchvision.utils.save_image(X, './out/8_4/lr/lr_{}.png'.format(i + 1))

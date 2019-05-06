import os
import torch
import torchvision
from torch.utils.data import DataLoader

from utils.DatasetManager import DatasetManager


##TODO add args support for epochs

def eval():
    from EDSR import EDSR
    os.makedirs(os.path.dirname('./out/twores8_4/hr'),
                exist_ok=True)

    data = DatasetManager(training=False, factor=4)
    dataMan = DataLoader(data, batch_size=1)

    EDSR = EDSR(2, 16)
    a = torch.load('./twores8_4/training_SR_24.pth',
                   map_location=lambda storage, loc: storage)
    EDSR.load_state_dict(a.module.state_dict())

    EDSR.eval()

    for i, (X, y) in enumerate(dataMan):
        outputs = EDSR(X)
        torchvision.utils.save_image(y, './out/twores8_4/hr/hr_{}.png'.format(i + 1))
        torchvision.utils.save_image(outputs[-1], './out/twores8_4/eval/24/eval_{}.png'.format(i + 1))
        torchvision.utils.save_image(X, './out/twores8_4/lr/lr_{}.png'.format(i + 1))

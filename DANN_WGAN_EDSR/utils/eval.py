import os

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.feature_extractor import Extractor
from utils.datasetmanager import DatasetManager
from utils.options import args


def eval():
    from models.SR import EDSR
    hr = args.output_dir + '/hr' + str(args.factor) + "/"
    lr = args.output_dir + '/lr' + str(args.factor) + "/"
    eval = args.output_dir + '/' + str(args.eval_epoch) + "_" + str(args.factor) + '/eval/'
    os.makedirs(os.path.dirname(hr),
                exist_ok=True)
    os.makedirs(os.path.dirname(lr),
                exist_ok=True)

    os.makedirs(os.path.dirname(eval),
                exist_ok=True)

    data = DatasetManager(training=False, factor=args.factor)
    dataMan = DataLoader(data, batch_size=1)

    EDSR = EDSR(2, 20)
    EDSR = torch.nn.DataParallel(EDSR)
    a = torch.load(args.output_dir + '/training_SR_' + str(args.eval_epoch) + '.pth',
                   map_location=lambda storage, loc: storage)
    EDSR.load_state_dict(a)
    EDSR.eval()
    enc = Extractor(1)
    enc = torch.nn.DataParallel(enc)
    EDSR.eval()

    a = torch.load(args.output_dir + '/training_feature_extractor_' + str(args.eval_epoch) + '.pth',
                   map_location=lambda storage, loc: storage)
    enc.load_state_dict(a)
    enc.eval()
    EDSR = EDSR.module
    enc = enc.module
    print(str(args.eval_epoch))
    for i, (X, y) in enumerate(tqdm(dataMan)):
        outputs = EDSR(enc(X))
        torchvision.utils.save_image(y, hr + 'hr_{}.png'.format(i + 1))
        torchvision.utils.save_image(outputs[-1], eval + 'eval_{}.png'.format(i + 1))
        torchvision.utils.save_image(X, lr + 'lr_{}.png'.format(i + 1))


# def eval_3_models():


if __name__ == '__main__':
    from utils.scoring import scoring

    eval()
    scoring()

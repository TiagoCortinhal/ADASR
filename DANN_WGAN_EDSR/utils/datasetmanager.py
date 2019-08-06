import h5py
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LAMBDA = .1
PRINT_FREQ = 20
LOGS_FNAME = 'logs.tsv'
LOGS_ACC = 'acc.tsv'
PLOT_FNAME = 'plot.svg'
CKPT_PREFIX = 'networks'
CLASSES = 3
RATIO = 0.5
CROP_SIZE = 48
FOLDER = './domain_{}_{}'.format(CLASSES, RATIO)
INTERPOLATION = Image.BICUBIC


class DatasetManager(Dataset):

    def __init__(self, training=True, factor=8):
        self.training = training
        self.factor = factor
        if training:
            dataset = h5py.File('../training_domain_1_48_8.h5', 'r')
            self.X = dataset['patchesX']
            self.y = dataset['patchesY']

            dataset = h5py.File('../training_domain_1_48_4.h5', 'r')
            self.X3 = dataset['patchesX']
            self.y3 = dataset['patchesY']

            dataset = h5py.File('../training_domain_2_48_8.h5', 'r')
            self.X2 = dataset['patchesX']
            self.y2 = dataset['patchesY']

            dataset = h5py.File('../training_domain_2_48_4.h5', 'r')
            self.X4 = dataset['patchesX']
            self.y4 = dataset['patchesY']

        else:
            import os
            self.y_dir = 'hr/'
            self.root = '../domain_2_test/'
            self.y = os.listdir(self.root + self.y_dir)
            self.y.sort()

    def __getitem__(self, index):
        if self.training:
            return self.X[index % len(self.X)], \
                   self.y[index % len(self.y)], \
                   self.X2[index % len(self.X2)], \
                   self.y2[index % len(self.y2)], \
                   self.X3[index % len(self.X3)], \
                   self.y3[index % len(self.y3)], \
                   self.X4[index % len(self.X4)], \
                   self.y4[index % len(self.y4)]
        else:
            return transforms.Lambda(
                lambda x: transforms.ToTensor()(transforms.Resize((int(x.size[1] * 1 / self.factor),
                                                                   int(x.size[0] * 1 / self.factor)),
                                                                  interpolation=Image.BICUBIC)(
                    x)))(
                Image.open(self.root + self.y_dir + self.y[index])), \
                   transforms.Lambda(
                       lambda x: transforms.ToTensor()(transforms.Resize((int(x.size[1] * 1 / (self.factor / 2)),
                                                                          int(x.size[0] * 1 / (self.factor / 2))),
                                                                         interpolation=Image.BICUBIC)(
                           x)))(
                       Image.open(self.root + self.y_dir + self.y[index]))

    def __len__(self):
        if self.training:
            return max(len(self.X), len(self.X2), len(self.X3), len(self.X4))
        else:
            return len(self.y)

import os
import h5py
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
from utils.options import args


class DatasetManager(Dataset):

    def __init__(self, training=True,factor=1):
        self.training = training
        self.factor = factor
        if training:
            dataset = h5py.File('training_48_{}.h5'.format(args.factor), 'r')
            self.X = dataset['patchesX']
            self.y = dataset['patchesY']
        else:
            self.y_dir = 'hr/'
            self.root = '../data/div2k_valid/'
            self.y = os.listdir(self.root + self.y_dir)
            self.y.sort()

    def __getitem__(self, index):
        if self.training:
            return self.X[index], \
                   self.y[index]
        else:
            return transforms.Lambda(
                lambda x: transforms.ToTensor()(transforms.Resize((int(x.size[1] * 1 / self.factor),
                                                                   int(x.size[0] * 1 / self.factor)),
                                                                  interpolation=Image.BICUBIC)(
                    x)))(
                Image.open(self.root + self.y_dir + self.y[index])), \
                   transforms.Lambda(
                       lambda x: transforms.ToTensor()(transforms.Resize((int(x.size[1] * 1 / (self.factor/2)),
                                                                          int(x.size[0] * 1 / (self.factor/2))),
                                                                         interpolation=Image.BICUBIC)(
                           x)))(
                       Image.open(self.root + self.y_dir + self.y[index]))

    def __len__(self):
        if self.training:
            return len(self.X)
        else:
            return len(self.y)


class DatasetManager2(Dataset):

    def __init__(self, training=True, factor=1):
        self.training = training
        self.factor = factor
        if training:
            dataset = h5py.File('training_48_{}.h5'.format(args.factor), 'r')
            dataset2 = h5py.File('training_48_{}.h5'.format(args.factor * 2), 'r')
            self.X1 = dataset['patchesX']
            self.X2 = dataset2['patchesX']
            self.y1 = dataset['patchesY']
            self.y2 = dataset2['patchesY']
        else:
            self.y_dir = 'hr/'
            self.root = '../data/div2k_valid/'
            self.y = os.listdir(self.root + self.y_dir)
            self.y.sort()

    def __getitem__(self, index):
        if self.training:
            if index < len(self.X1):
                return self.X1[index], \
                       self.y1[index]
            else:
                return self.X2[index - len(self.X1)], \
                       self.y2[index - len(self.X1)]
        else:
            return transforms.Lambda(
                lambda x: transforms.ToTensor()(transforms.Resize((int(x.size[1] * 1 / self.factor),
                                                                   int(x.size[0] * 1 / self.factor)),
                                                                  interpolation=Image.BICUBIC)(
                    x)))(
                self.loader(self.root + self.y_dir + self.y[index])), \
                   transforms.Lambda(
                       lambda x: transforms.ToTensor()(transforms.Resize((int(x.size[1] * 1 / (self.factor/2)),
                                                                          int(x.size[0] * 1 / (self.factor/2))),
                                                                         interpolation=Image.BICUBIC)(
                           x)))(
                       self.loader(self.root + self.y_dir + self.y[index]))

    def __len__(self):
        if self.training:
            return len(self.X1) + len(self.X2)
        else:
            return len(self.y)

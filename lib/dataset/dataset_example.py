import os
import glob
import numpy as np

from imageio import imread

import torch
import torch.utils.data as data


class ExampleDataset(data.Dataset):
    def __init__(self, hw=(512, 1024), length=128, n_classes=10, random_lr_flip=False):
        self.hw = hw
        self.length = length
        self.n_classes = n_classes
        self.random_lr_flip = random_lr_flip

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ''' Implement your input/target loading '''
        x = torch.randn(3, *hw)
        gt = torch.randint(0, self.n_classes, self.hw)

        ''' Augmentation if needed '''
        if self.random_lr_flip:
            x = torch.flip(x, (-1,))
            gt = torch.flip(gt, (-1,))

        ''' Return the batch for your network.compute_losses '''
        return {'x': x, 'gt': gt}


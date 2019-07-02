# -*- coding: utf-8 -*-
"""
Supplementary code for a paper under review.
Do not distribute.
"""
import torch
import torchvision.datasets as dsets
from torchvision import transforms


class Data_Loader():
    def __init__(self, train, dataset, data_dir, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = data_dir
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize, self.imsize)))  # new/old version: Resize/Scale
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_imagenet(self):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.ImageFolder(self.path, transform=transforms)
        return dataset

    def load_cifar(self):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.CIFAR10(self.path, transform=transforms, download=True)
        return dataset

    def load_stl10(self):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.STL10(self.path, transform=transforms, download=True)
        return dataset

    def load_svhn(self):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.SVHN(self.path, transform=transforms, download=True)
        return dataset

    def loader(self):
        if self.dataset == 'imagenet':
            dataset = self.load_imagenet()
        elif self.dataset == 'cifar10':
            dataset = self.load_cifar()
        elif self.dataset == 'stl10':
            dataset = self.load_stl10()
        elif self.dataset == 'svhn':
            dataset = self.load_svhn()
        else:
            raise NotImplementedError('Available datasets: imagenet, '
                                      'cifar10, stl10 and svhn.')

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=self.shuf,
                                             num_workers=2,
                                             drop_last=True)
        return loader


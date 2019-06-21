# -*- coding: utf-8 -*-
"""
Supplementary code for paper under review by the International Conference on Machine Learning (ICML).
Do not distribute.
"""
import os
import sys
import torch
import warnings
from torch.utils.data import DataLoader


def setup_logger(_out_dir, _file_name='log.log'):
    """
    Sets up info logger.

    Raises:
        NotADirectoryError: _data_root is not a directory
    :param _out_dir: [str] directory to store the log file
    :return: [module]
    """
    if not os.path.exists(_out_dir):
        raise NotADirectoryError("Could not open {}".format(_out_dir))
    import logging
    _logger = logging.getLogger(os.path.splitext(os.path.basename(_file_name))[0])
    _handler = logging.FileHandler(os.path.join(_out_dir, _file_name))
    _handler.setFormatter(logging.Formatter('%(message)s'))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    return _logger


def _make_dir(_dir_name):
    """Creates a directory if it does not exists.

    Raises:
        TypeError: _dir_name is not string

    :param _dir_name: [str]
    :return: [None]
    """
    if not isinstance(_dir_name, str):
        raise TypeError("Got {}. Expected str.".format(type(_dir_name)))
    if not os.path.exists(_dir_name):
        os.makedirs(_dir_name)


def create_dirs(args):
    """Ensures required directories exist, or creates these if needed.

    In accordance with the selected options:
    1. If continuing, args.outdir has to be an existing directory
    2. If not continuing, and args.outdir exists, asks for confirmation
       to proceed.
    3. Creates args.outdir if it does not exist.
    4. Creates args.dataroot if it does not exist, s.t. the dataset can
       be later downloaded in the specified directory.
    5. Creates sub-directories: 'fake_samples', 'generator' and 'backup'
       in args.outdir, if selected these to be stored (args.sample_freq,
       args.generator_freq and args.backup_freq are strictly positive,
       respectively).

    It warns the user if the option continue is not activated and
    there exist a directory args.outdir.

    Raises:
        NotADirectoryError: continuing & non existing input directory
        TypeError: see _make_dir


    :param args: [argparse.Namespace]
    :return: [None]
    """
    if args.cont and not os.path.exists(args.outdir):
        raise NotADirectoryError("Could not open {}".format(args.outdir))
    elif not args.cont and os.path.exists(args.outdir):
        warnings.warn("Directory {} exists. Content may be "
                      "overwritten.".format(args.outdir))
    _make_dir(args.outdir)
    _make_dir(args.dataroot)

    if args.sample_freq > 0:
        _make_dir(os.path.join(args.outdir, 'fake_samples'))
    if args.generator_freq > 0:
        _make_dir(os.path.join(args.outdir, 'generator'))
    if args.backup_freq > 0:
        _make_dir(os.path.join(args.outdir, 'backup'))


def load_dataset(_dataset, _data_root, _verbose=False):
    """Wrapper of torchvision.datasets.*. Loads the selected dataset.

    Raises:
        NotADirectoryError: _data_root is not a directory
        NotImplementedError: unsupported dataset

    Todo:
        Currently only MNIST is supported. To be updated.

    :param _dataset: [str] dataset name, mnist|cifar10|imagenet|celeba|...
    :param _data_root: [str] directory where the data is/will be stored
    :param _verbose: [bool] if True prints info
    :return: [torchvision.dataset.<selected_dataset_class>]
    """
    if not os.path.exists(_data_root):
        raise NotADirectoryError("Could not open {}".format(_data_root))

    import torchvision.datasets as _datasets
    import torchvision.transforms as _transforms

    if _dataset == 'mnist':
        _data = _datasets.MNIST(_data_root, train=True, download=True,
                                transform=_transforms.Compose(
                                    [_transforms.ToTensor()]))
    else:
        raise NotImplementedError("{} not implemented.".format(_dataset))
    if _verbose:
        print('%s dataset loaded from: %s. Found %d samples.' %
              (_dataset, _data_root, len(_data)))
    return _data


def dataset_generator(dataset, batch_size, shuffle=True,
                      num_workers=0, drop_last=False):
    """Infinite generator over given dataset.

    Yields:
        Sample of dataset of size batch_size

    Raises:
        TypeError: dataset is not torch.utils.data (derived) object
        ValueError: see torch.utils.data.DataLoader

    See also docs [1].
        [1] https://pytorch.org/docs/stable/data.html

    Usage:
        sampler = dataset_generator(_)
        next(sampler)

    :param dataset: [derived from torch.utils.data] pre-loaded dataset
    :param batch_size: [int] batch size
    :param shuffle: [bool, optional] shuffle (& re-shuffle after full tra-
                    verse). Default: True
    :param num_workers: [int, optional] #subprocesses to load data. See [1].
                        Default: 0 (load in main process)
    :param drop_last: [bool, optional]: Drop left over [1]. Default: False
    :return: [_type] where _type is the the type of elements of dataset
             (normally torch tensor)
    """
    import torch.utils.data as _data
    if not isinstance(dataset, _data.Dataset):
        raise TypeError("Expected dataset object derived from torch."
                        "utils.data. Got: {}".format(type(dataset)))

    _dataloader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers,
                             drop_last=drop_last)
    _data_iter = iter(_dataloader)
    while True:
        try:
            _sample = _data_iter.next()[0]
        except StopIteration:
            _data_iter = iter(_dataloader)
            _sample = _data_iter.next()[0]
        yield _sample


def noise_generator(dataset, batch_size, drop_last=False, resample=True):
    """Infinite generator over dataset, where dataset is re-sampled ~N(0,1)

    Yields:
        Sample of dataset of size batch_size

    Raises:
        TypeError: dataset is not torch.FloatTensor object
        ValueError: see torch.utils.data.DataLoader

    Usage:
        sampler = noise_generator(_)
        next(sampler)

    :param dataset: [torch.FloatTensor] allocated tensor
    :param batch_size: [int] batch size
    :param drop_last: [bool, optional]: Drop left over [1]. Default: False
    :param resample: [bool, optional]: Re-sample dataset with N(0, 1) after
                     it is fully traversed. Default: True
    :return: [torch.FloatTensor] of size batch_size x dataset.size(1)
    """
    if not isinstance(dataset, torch.FloatTensor):
        raise TypeError("Expected dataset object derived from torch."
                        "utils.data. Got: {}".format(type(dataset)))
    _dataloader = DataLoader(dataset, batch_size=batch_size,
                             drop_last=drop_last)
    _data_iter = iter(_dataloader)
    while True:
        try:
            _sample = _data_iter.next()
        except StopIteration:
            if resample:
                dataset.normal_(0, 1)
            _dataloader = DataLoader(dataset, batch_size=batch_size,
                                     drop_last=drop_last)
            _data_iter = iter(_dataloader)
            _sample = _data_iter.next()
        yield _sample


def subsample(dataset, sub_sample, batch_size=50):
    """

    :param dataset: [torch.utils.data.Dataset]
    :param sub_sample: [torch.Tensor]
    :param batch_size: [int]
    :return: [None]
    """

    import torch.utils.data as _data
    if not isinstance(dataset, _data.Dataset):
        raise TypeError("Expected dataset object derived from torch."
                        "utils.data. Got: {}".format(type(dataset)))
    if not isinstance(sub_sample, torch.Tensor):
        raise TypeError("Expected torch tensor."
                        "Got: {}".format(type(sub_sample)))
    _sample_size = sub_sample.size(0)
    if _sample_size > len(dataset):
        raise ValueError("Size of the dataset: {}.".format(len(dataset)) +
                         "Size of the subsample: {}".format(_sample_size))

    _data_iter = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))

    for i in range(0, _sample_size, batch_size):
        data = _data_iter.next()[0].view(batch_size, -1)
        sub_sample[i:min(i + batch_size, _sample_size)].copy_(
            data[0:min(batch_size, _sample_size - i)])

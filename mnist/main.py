# -*- coding: utf-8 -*-
"""
Supplementary code for paper under review by the International Conference on Machine Learning (ICML).
Do not distribute.
"""
import os
import sys
import argparse
import torch
import warnings
import copy
import math
import torchvision.utils as vision_utils
import pretrained_model
import json
import numpy as np
from torch.autograd import Variable
from torch.distributions import bernoulli
from utility import setup_logger, create_dirs, \
    load_dataset, dataset_generator, noise_generator, \
    subsample


_NOISE_DIM = 128
_IMG_CHANNELS = 1


def parse_args():
    """Parses the input command line arguments.
    It warns the user if a GPU is detected, yet the 'cuda' option is not activated.
    :return: [argparse.Namespace]
    """

    def _positive_integer_input(value):
        """Checks if input is strictly positive.

        :param value: [str] command line input
        :return: [int] int(value)
        """
        _value = int(value)
        if _value <= 0:
            raise argparse.ArgumentTypeError("Expected strictly positive integer value. "
                                             "Got: {}".format(value))
        return _value

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='mnist | fashionmnist, default: %(default)s', )
    parser.add_argument('--dataroot', type=str, default='datasets',
                        help='path to dataset, default: %(default)s')
    parser.add_argument('--batch_size', type=_positive_integer_input, default=50,
                        help='batch size, default: %(default)s')
    parser.add_argument('--large_batch_size', type=_positive_integer_input, default=500,
                        help='batch size for calculating SVRG statistics, default: %(default)s')
    parser.add_argument('--image_size', type=_positive_integer_input, default=28,
                        help='height & width of the images, default: %(default)s')
    parser.add_argument('--outdir', type=str, default='results/test/',
                        help='output directory, default: %(default)s')
    parser.add_argument('--generator_freq', type=int, default=-1,
                        help='store generator frequency, use strictly positive to activate, '
                             'default: %(default)s')
    parser.add_argument('--sample_freq', type=int, default=100,
                        help='store sample frequency. Use strictly positive value to activate. '
                             'Default: %(default)s')
    parser.add_argument('--svrg_freq', type=int, default=-1,
                        help='frequency to update the SVRG statistics. '
                             'Defaults to one traverse of the dataset. '
                             'Use strictly positive number to redefine it.')
    parser.add_argument('--backup_freq', type=int, default=1000,
                        help='frequency to backup models and optimizer state. '
                             'Default: %(default)s')
    parser.add_argument('--seed', type=int, default=1,
                        help='fixed random seed, default: %(default)s')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='number of workers for loading the dataset, default: %(default)s')
    parser.add_argument('--iterations', type=_positive_integer_input, default=100000,
                        help='stopping criteria, total number of updates of the generator. '
                             'Default: %(default)s')
    parser.add_argument('--gan_type', type=str, default='gan',
                        help='Supported: gan, sngan, dragan. Todo: wgan, wgan_gp. '
                             'Default: %(default)s')
    parser.add_argument('--svrg', action='store_true', help='activate SVRG')
    parser.add_argument('--lrD', type=float, help='Discriminator learning rate. '
                                                  'Default: 1e-2 if svrg, 1e-4 otherwise')
    parser.add_argument('--lrG', type=float, help='Generator learning rate. '
                                                  'Default: 1e-2 if svrg, 1e-4 otherwise')
    parser.add_argument('--weight_decay', type=float, default=.0,
                        help='Recommended: 1e-4 (fashion-)MNIST and 1e-3 otherwise. '
                             'Default: %(default)s')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Hyper-parameter beta1 for the Adam optimizer (if used). '
                             'Default: %(default)s')
    parser.add_argument('--cuda', action='store_true',
                        help='enable Cuda/GPU, default: %(default)s')
    parser.add_argument('--optim', type=str, default='svrgadam',
                        choices=['sgd', 'adam', 'adadelta', 'svrgadam'])
    parser.add_argument('--verbose', action='store_true',
                        help='enable printing through iterations, default: %(default)s')
    parser.add_argument('--cont', action='store_true',
                        help='Continue training, default: %(default)s')
    parser.add_argument('--metric_freq', type=int, default=10,
                        help='performance estimation frequency (in generator updates), '
                             'use 0 to cancel it, default: %(default)s')
    parser.add_argument('--metric_sample_size', type=int, default=5000,
                        help='sample size for the metrics, default: %(default)s')
    parser.add_argument('--rst_freq', type=float, default=.1,
                        help='avg reset frequency, default: %(default)s')
    parser.add_argument('--reset', action='store_true',
                        help='reset averaging, default: %(default)s')
    _opt = parser.parse_args()
    _opt.dataset = _opt.dataset.lower()
    _opt.gan_type = _opt.gan_type.lower()
    if torch.cuda.is_available() and not _opt.cuda:
        warnings.warn("Detected CUDA device, run with --cuda")
    if not _opt.dataset == 'mnist':  # todo: add pre-trained classifiers for other datasets
        _opt.metric_freq = 0
    return _opt


def select_models(_dataset, _image_size, _weight_decay=None):
    """Loads the model definition for the selected dataset.

    Warns the user if non typical options are selected.
    This includes:
        1. For (fashion-)MNIST:
            - Image space of 1x28x28,
              thus models from models/dcgan28
            - no weight decay or of weight 1e-4
        2. (Todo)

    Returns:
        A triple of:
            gen_model: the generator class
            disc_model: the discriminator class
            product_dim: product of dimensions of real data space
    Raises:
        NotImplementedError: unsupported dataset

    Todo:
        Currently only MNIST is supported.

    :param _dataset: [str] dataset
    :param _image_size: [int] assuming square
    :param _weight_decay: [float, optional] hyper-parameter, default None
    :return: [triple (3-tuple)] gen_model, disc_model, product_dim
    """
    if 'mnist' in _dataset:  # set-up for (fashion)MNIST
        _img_channels = 1
        from models import dcgan28 as models
        _GeneratorNet = models.GeneratorCNN28
        _DiscriminatorNet = models.DiscriminatorCNN28
        if _image_size != 28:
            warnings.warn("Selected image space {}. "
                          "Expected 28.".format(_image_size))
        if _weight_decay is not None and _weight_decay not in [0, 1e-4]:
            warnings.warn("Recommended weight decay: 1e-4. "
                          "Got: {}".format(_weight_decay))
    else:
        raise NotImplementedError("Dataset {} not implemented. ".format(_dataset))
    return _GeneratorNet, _DiscriminatorNet, _image_size**2 * _img_channels


def init_networks(_GenNet, _DiscNet, _opt):

    # Target parameters ###################
    _g_net = _GenNet(noise_dim=_NOISE_DIM)
    _d_net = _DiscNet(spectral_norm=_opt.gan_type == 'sngan', img_size=_opt.image_size)

    if _opt.optim == 'sgd':
        _g_optim = torch.optim.SGD(_g_net.parameters(), lr=_opt.lrG, weight_decay=_opt.weight_decay)
        _d_optim = torch.optim.SGD(_d_net.parameters(), lr=_opt.lrD, weight_decay=_opt.weight_decay)
    elif _opt.optim == 'adam':
        _g_optim = torch.optim.Adam(_g_net.parameters(), lr=_opt.lrG, betas=(_opt.beta1, 0.999))
        _d_optim = torch.optim.Adam(_d_net.parameters(), lr=_opt.lrD, betas=(_opt.beta1, 0.999))
    elif _opt.optim == 'svrgadam':
        _g_optim = torch.optim.SvrgAdam(_g_net.parameters(), lr=_opt.lrG, betas=(_opt.beta1, 0.999))
        _d_optim = torch.optim.SvrgAdam(_d_net.parameters(), lr=_opt.lrD, betas=(_opt.beta1, 0.999))
    elif _opt.optim == 'adadelta':
        _g_optim = torch.optim.Adadelta(_g_net.parameters(), lr=_opt.lrG)
        _d_optim = torch.optim.Adadelta(_d_net.parameters(), lr=_opt.lrD)

    else:
        raise NotImplementedError('Supported optimizers: SGD, Adam, SvrgAdam, Adadelta')

    if _opt.cuda:
        _g_net = _g_net.cuda()
        _d_net = _d_net.cuda()

    # Extra-gradient ######################
    _extra_g_net = copy.deepcopy(_g_net)
    _extra_d_net = copy.deepcopy(_d_net)

    if _opt.optim == 'sgd':
        _extra_g_optim = torch.optim.SGD(_extra_g_net.parameters(), lr=_opt.lrG, weight_decay=_opt.weight_decay)
        _extra_d_optim = torch.optim.SGD(_extra_d_net.parameters(), lr=_opt.lrD, weight_decay=_opt.weight_decay)
    elif _opt.optim == 'adam':
        _extra_g_optim = torch.optim.Adam(_extra_g_net.parameters(), lr=_opt.lrG, betas=(_opt.beta1, 0.999))
        _extra_d_optim = torch.optim.Adam(_extra_d_net.parameters(), lr=_opt.lrD, betas=(_opt.beta1, 0.999))
    elif _opt.optim == 'svrgadam':
        _extra_g_optim = torch.optim.SvrgAdam(_extra_g_net.parameters(), lr=_opt.lrG, betas=(_opt.beta1, 0.999))
        _extra_d_optim = torch.optim.SvrgAdam(_extra_d_net.parameters(), lr=_opt.lrD, betas=(_opt.beta1, 0.999))
    elif _opt.optim == 'adadelta':
        _extra_g_optim = torch.optim.Adadelta(_extra_g_net.parameters(), lr=_opt.lrG)
        _extra_d_optim = torch.optim.Adadelta(_extra_d_net.parameters(), lr=_opt.lrD)
    else:
        raise NotImplementedError('Supported optimizers: SGD, Adam, Adadelta')

    # Dict-Group ##########################
    _generator = {'net': _g_net,
                  'optim': _g_optim,
                  'extra_net': _extra_g_net,
                  'extra_optim': _extra_g_optim}
    _discriminator = {'net': _d_net,
                      'optim': _d_optim,
                      'extra_net': _extra_d_net,
                      'extra_optim': _extra_d_optim}

    # SVRG  ###############################
    _g_snapshot = copy.deepcopy(_g_net) if opt.svrg else None
    _d_snapshot = copy.deepcopy(_d_net) if opt.svrg else None

    return _generator, _discriminator, _g_snapshot, _d_snapshot


def _requires_grad(_net, _bool=True):
    """Helper function which sets the requires_grad of _net to _bool.

    Raises:
        TypeError: _net is given but is not derived from nn.Module, or
                   _bool is not boolean

    :param _net: [nn.Module]
    :param _bool: [bool, optional] Default: True
    :return: [None]
    """
    if _net and not isinstance(_net, torch.nn.Module):
        raise TypeError("Expected torch.nn.Module. Got: {}".format(type(_net)))
    if not isinstance(_bool, bool):
        raise TypeError("Expected bool. Got: {}".format(type(_bool)))

    if _net is not None:
        for _w in _net.parameters():
            _w.requires_grad = _bool


def _update_grads_svrg(params, snapshot_params, mu):
    """Helper function which updates the accumulated gradients of
    params by subtracting those of snapshot and adding mu.

    Operates in-place.

    Raises:
        ValueError if the inputs have different lengths (the length
        corresponds to the number of layers in the network)

    :param params: [list of torch.nn.parameter.Parameter]
    :param snapshot_params: [torch.nn.parameter.Parameter]
    :param mu: [list of torch(.cuda).FloatTensor]
    :return: [None]
    """
    if not len(params) == len(snapshot_params) == len(mu):
        raise ValueError("Expected input of identical length. "
                         "Got {}, {}, {}".format(len(params),
                                                 len(snapshot_params),
                                                 len(mu)))
    for i in range(len(mu)):
        params[i].grad.data.sub_(snapshot_params[i].grad.data)
        params[i].grad.data.add_(mu[i])


def _backprop_gradient_penalty(net, input_data, dtype=torch.FloatTensor):
    """Helper function which accumulates 2nd order derivative

    # todo: doc & input validation

    :param net:
    :param input_data:
    :param dtype:
    :return: [None]
    """
    output_data = net(input_data)
    gradients = torch.autograd.grad(outputs=output_data, inputs=input_data,
                                    grad_outputs=dtype(output_data.size()).fill_(1),
                                    create_graph=True, retain_graph=True,
                                    only_inputs=True)[0]
    # gradient penalty weight is 10 in the literature
    gradient_penalty = 10 * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    gradient_penalty.backward()


def _extra_sync_nets(gen, disc):
    """ Helper function. Copies the current parameters to the t+1/2 parameters,
     stored as 'net' and 'extra_net', respectively.

    :param gen: [dict] with keys: net & extra_net
    :param disc: [dict] with keys: net & extra_net
    :return: [None]
    """
    gen['extra_net'].load_state_dict(gen['net'].state_dict())
    disc['extra_net'].load_state_dict(disc['net'].state_dict())


def train(gen, disc, _sampler, _noise_sampler, _dragan=False,
          _g_snapshot=None, _d_snapshot=None, _mu_g=None, _mu_d=None):
    """ Trains (DRA)GAN pair <gen, disc> for one iteration

    # todo: doc & input validation

    :param gen: [dict] of keys 'net' and 'optim'
    :param disc: [dict] of keys 'net' and 'optim'
    :param _sampler:
    :param _noise_sampler:
    :param _dragan:
    :param _g_snapshot:
    :param _d_snapshot:
    :param _mu_g:
    :param _mu_d:
    :return: [None]
    """
    cuda = next(gen['net'].parameters()).is_cuda
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    criterion = torch.nn.BCELoss()  # Binary Cross Entropy
    if cuda:
        criterion.cuda()

    _samples = Variable(next(_sampler).cuda() if cuda else next(_sampler))
    _noise = Variable(next(_noise_sampler).cuda() if cuda else next(_noise_sampler))
    _real_labels = Variable(dtype(_samples.data.size(0), 1).fill_(1))
    _fake_labels = Variable(dtype(_samples.data.size(0), 1).fill_(0))
    _extra_sync_nets(gen, disc)

    # Discriminator t + 1/2 ############################
    _requires_grad(disc['extra_net'], True)
    _requires_grad(gen['net'], False)
    disc['extra_net'].zero_grad()
    criterion(disc['extra_net'](_samples), _real_labels).backward()
    criterion(disc['extra_net'](gen['net'](_noise)), _fake_labels).backward()

    if _d_snapshot:
        _requires_grad(_d_snapshot, True)
        _d_snapshot.zero_grad()
        criterion(_d_snapshot(_samples), _real_labels).backward()
        # see line 12, Alg.3 of the paper
        criterion(_d_snapshot(_g_snapshot(_noise)), _fake_labels).backward()

    if _dragan:
        alpha = dtype(_samples.data.size(0), 1).uniform_().expand(_samples.size())
        x_hat = Variable(alpha * _samples.data + (1 - alpha) *
                         (_samples.data + 0.5 * _samples.data.std() *
                          dtype(_samples.size()).uniform_()),
                         requires_grad=True)
        _backprop_gradient_penalty(disc['extra_net'], x_hat, dtype)
        if _d_snapshot:
            _backprop_gradient_penalty(_d_snapshot, x_hat, dtype)

    if _d_snapshot:  # Modify the accumulated gradients (in-place)
        _update_grads_svrg(list(disc['extra_net'].parameters()),
                           list(_d_snapshot.parameters()),
                           _mu_d)
    disc['extra_optim'].step()

    # Generator t + 1/2 ############################
    _requires_grad(disc['net'], False)
    _requires_grad(gen['extra_net'], True)
    gen['extra_net'].zero_grad()

    _noise.data.copy_(next(_noise_sampler))
    criterion(disc['net'](gen['extra_net'](_noise)), _real_labels).backward()

    if _g_snapshot:
        _requires_grad(_d_snapshot, False)
        _requires_grad(_g_snapshot, True)
        _g_snapshot.zero_grad()
        criterion(_d_snapshot(_g_snapshot(_noise)), _real_labels).backward()

        _update_grads_svrg(list(gen['extra_net'].parameters()),
                           list(_g_snapshot.parameters()),
                           _mu_g)
    gen['extra_optim'].step()

    # Discriminator t + 1 ############################
    _samples = Variable(next(_sampler).cuda() if cuda else next(_sampler))
    _noise = Variable(next(_noise_sampler).cuda() if cuda else next(_noise_sampler))
    _requires_grad(disc['net'], True)
    _requires_grad(gen['extra_net'], False)
    disc['net'].zero_grad()
    criterion(disc['net'](_samples), _real_labels).backward()
    criterion(disc['net'](gen['extra_net'](_noise)), _fake_labels).backward()

    if _d_snapshot:
        _requires_grad(_d_snapshot, True)
        _d_snapshot.zero_grad()
        criterion(_d_snapshot(_samples), _real_labels).backward()
        # see line 12, Alg.3 of the paper
        criterion(_d_snapshot(_g_snapshot(_noise)), _fake_labels).backward()

    if _dragan:
        alpha = dtype(_samples.data.size(0), 1).uniform_().expand(_samples.size())
        x_hat = Variable(alpha * _samples.data + (1 - alpha) *
                         (_samples.data + 0.5 * _samples.data.std() *
                          dtype(_samples.size()).uniform_()),
                         requires_grad=True)
        _backprop_gradient_penalty(disc['net'], x_hat, dtype)
        if _d_snapshot:
            _backprop_gradient_penalty(_d_snapshot, x_hat, dtype)

    if _d_snapshot:  # Modify the accumulated gradients (in-place)
        _update_grads_svrg(list(disc['net'].parameters()),
                           list(_d_snapshot.parameters()),
                           _mu_d)

    disc['optim'].step()

    # Generator t + 1 ############################
    _requires_grad(disc['extra_net'], False)
    _requires_grad(gen['net'], True)
    gen['net'].zero_grad()

    _noise.data.copy_(next(_noise_sampler))
    criterion(disc['extra_net'](gen['net'](_noise)), _real_labels).backward()

    if _g_snapshot:
        _requires_grad(_d_snapshot, False)
        _requires_grad(_g_snapshot, True)
        _g_snapshot.zero_grad()
        criterion(_d_snapshot(_g_snapshot(_noise)), _real_labels).backward()

        _update_grads_svrg(list(gen['net'].parameters()),
                           list(_g_snapshot.parameters()),
                           _mu_g)

    gen['optim'].step()


def calculate_svrg_stats(gen_s, disc_s, data_set, noise_set,
                         batch_size, _dragan=False):
    """Calculates mu_d and mu_g (see Alg. 3)

    # todo: doc & input val

    :param gen_s:
    :param disc_s:
    :param data_set:
    :param noise_set:
    :param batch_size:
    :param _dragan:
    :return:
    """
    cuda = next(gen_s.parameters()).is_cuda
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    criterion = torch.nn.BCELoss()  # Binary Cross Entropy
    if cuda:
        criterion.cuda()
    _samples = Variable(dtype(batch_size, *data_set[0][0].size()))
    _noise = Variable(dtype(batch_size, *noise_set[0].size()))
    _real_labels = Variable(dtype(batch_size, 1).fill_(1))
    _fake_labels = Variable(dtype(batch_size, 1).fill_(0))
    _dataloader = torch.utils.data.DataLoader(data_set,
                                              batch_size=batch_size,
                                              drop_last=True)
    _noise_loader = torch.utils.data.DataLoader(noise_set,
                                                batch_size=batch_size,
                                                drop_last=True)
    _noise_sampler = iter(_noise_loader)

    # Calculate _mu_d ############################
    _requires_grad(gen_s, False)
    _requires_grad(disc_s, True)
    disc_s.zero_grad()

    for b_i, b_data in enumerate(_dataloader):
        _samples.data.copy_(b_data[0])
        _noise.data.copy_(next(_noise_sampler))
        criterion(disc_s(_samples), _real_labels).backward()
        criterion(disc_s(gen_s(_noise)), _fake_labels).backward()

        if _dragan:
            alpha = dtype(_samples.data.size(0), 1).uniform_().expand(_samples.size())
            x_hat = Variable(alpha * _samples.data + (1 - alpha) *
                             (_samples.data + 0.5 * _samples.data.std() *
                              dtype(_samples.size()).uniform_()),
                             requires_grad=True)
            _backprop_gradient_penalty(disc_s, x_hat, dtype)

    _mu_d = []

    for _w in disc_s.parameters():  # iterates per layer
        _w.grad.data.div_(len(_dataloader))
        _mu_d.append(_w.grad.data.clone())

    # Calculate _mu_g ############################
    _requires_grad(gen_s, True)
    _requires_grad(disc_s, False)
    gen_s.zero_grad()

    for _ in range(len(_dataloader)):
        _noise.data.copy_(next(_noise_sampler))
        criterion(disc_s(gen_s(_noise)), _real_labels).backward()

    _mu_g = []

    for _w in gen_s.parameters():  # iterates per layer
        _w.grad.data.div_(len(_dataloader))
        _mu_g.append(_w.grad.data.clone())

    return _mu_g, _mu_d


def update_avg(avg_net, net, avg_step):
    """In-place

    Todo: input val

    :param avg_net:
    :param net:
    :param avg_step:
    :return:
    """
    net_param = list(net.parameters())
    for i, p in enumerate(avg_net.parameters()):
        p.mul_((avg_step - 1)/avg_step)
        p.add_(net_param[i].div(avg_step))


def _calculate_metrics(classifier, net, sample, noise, n_classes=10, batch_size=100):

    if not isinstance(classifier, torch.nn.Module):
        raise TypeError("Expected torch.nn.Module. Got: {}".format(type(classifier)))
    if not isinstance(net, torch.nn.Module):
        raise TypeError("Expected torch.nn.Module. Got: {}".format(type(net)))

    if not noise.size(0) % batch_size == 0:
        raise NotImplementedError("Expected metric sample size % batch size == 0. "
                                  "Todo otherwise")

    cuda = next(classifier.parameters()).is_cuda
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    fake = torch.FloatTensor(sample.size())
    class_probas = torch.FloatTensor(sample.size(0), n_classes)
    vnoise = dtype(batch_size, noise.size(1))
    inception_predictions = []
    for j in range(0, noise.size(0), batch_size):
        vnoise.data.copy_(noise[j:j + batch_size])
        out_fake = net(vnoise).view(batch_size, -1)
        fake[j:j + batch_size].copy_(out_fake.data)

        out_class_probas = classifier(out_fake)
        class_probas[j:j + batch_size].copy_(out_class_probas.data)

        # for inception score
        pred = torch.nn.functional.softmax(out_class_probas).data.cpu().numpy()
        inception_predictions.append(pred)
    class_probas = class_probas.numpy()
    pred_prob = np.maximum(class_probas, 1e-20 * np.ones_like(class_probas))

    y_vec = 1e-20 * np.ones((len(pred_prob), n_classes), dtype=np.float)  # pred label distr
    gnd_vec = 0.1 * np.ones((1, n_classes), dtype=np.float)  # gnd label distr, uniform over classes

    for i, label in enumerate(pred_prob):
        y_vec[i, np.argmax(pred_prob[i])] = 1.0
    y_vec = np.sum(y_vec, axis=0, keepdims=True)
    y_vec = y_vec / np.sum(y_vec)

    label_entropy = np.sum(-y_vec * np.log(y_vec)).tolist()
    label_tv = np.true_divide(np.sum(np.abs(y_vec - gnd_vec)), 2).tolist()
    label_l2 = np.sum((y_vec - gnd_vec) ** 2).tolist()

    inception_predictions = np.concatenate(inception_predictions, 0)
    inception_scores = []
    for i in range(n_classes):
        part = inception_predictions[(i * inception_predictions.shape[0]
                                      // n_classes):((i + 1) * inception_predictions.shape[0]
                                                     // n_classes), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        inception_scores.append(np.exp(kl))

    return (label_entropy, label_tv, label_l2,
            float(np.mean(inception_scores)),
            float(np.std(inception_scores)))


def metrics(metrics_data, iteration):
    """Calculates entropy, TV, L2, and inception scores.

    Raises:
        TypeError
        ValueError

    :param metrics_data: [dict] with keys:
                        - classifier (torch.nn.Module),
                        - net (torch.nn.Module),
                        - sample (torch.Tensor),
                        - noise (torch.Tensor),
                        - results (list)
    :param iteration: [int]
    :return: [None]
    """
    required_keys = ['classifier', 'net', 'sample', 'noise', 'results']
    if not set(required_keys).issubset(metrics_data):
        raise ValueError("Expected keys: classifier, net, sample, noise and results.")
    if not isinstance(metrics_data['results'], list):
        raise TypeError("Expected list. Got: {}".format(type(metrics_data['results'])))

    e, tv, l2, is_m, is_std = _calculate_metrics(metrics_data['classifier'],
                                                 metrics_data['net'],
                                                 metrics_data['sample'],
                                                 metrics_data['noise'])

    m_result = {
        'iter': iteration,
        'entropy': e,
        'TV': tv,
        'L2': l2,
        'inception_mean': is_m,
        'inception_std': is_std
    }
    metrics_data['results'].append(m_result)

    if 'logger' in metrics_data:
        _str = 'i:%d;\t\E: %f;\t\TV: %f;\t\L2: %f;\tIS-mean: %f;\tIS-std: %f;' % \
               (iteration, e, tv, l2, is_m, is_std)
        metrics_data['logger'].info(_str)


if __name__ == "__main__":
    opt = parse_args()
    create_dirs(opt)
    logger = setup_logger(opt.outdir)
    logger.info(opt)
    if opt.verbose:
        print(opt)

    # Fix seed, use also random.seed(opt.seed) if it is used
    torch.manual_seed(opt.seed)
    if opt.cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        torch.cuda.manual_seed(opt.seed)
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Required for storing X ~ G(Z)
    if opt.sample_freq > 0:
        fixed_noise = Variable(dtype(100, _NOISE_DIM).normal_(0, 1))

    # load model definitions, according to selected dataset & create instances
    GenNet, DiscNet, space = select_models(opt.dataset, opt.image_size, opt.weight_decay)
    generator, discriminator, g_snapshot, d_snapshot = init_networks(GenNet, DiscNet, opt)

    # Data loading. #######################
    # - As SVRG requires GD step, an additional data loader is instantiated which
    # uses larger batch size (opt.large_batch_size). Analogous hold for noise data.
    # - To ensure that in expectation the noise vanishes (what reduces SVRG to SGD
    # [*]), svrg_noise_sampler & noise_sampler use the same noise tensor. This
    # noise tensor is re-sampled from p_z by noise_sampler, after its full traverse.
    #
    # [*] Accelerating stochastic gradient descent using predictive variance reduction,
    # Johnson & Zhang, Advances in Neural Information Processing Systems, 2013.

    dataset = load_dataset(opt.dataset, opt.dataroot, opt.verbose)
    data_sampler = dataset_generator(dataset, opt.batch_size,
                                     num_workers=opt.n_workers, drop_last=True)
    _n_batches = len(dataset) // opt.batch_size
    svrg_freq_sampler = bernoulli.Bernoulli(torch.tensor([1 / _n_batches]))
    noise_dataset = torch.FloatTensor(2 * len(dataset), _NOISE_DIM).normal_(0, 1)
    noise_sampler = noise_generator(noise_dataset, opt.batch_size,
                                    drop_last=True, resample=True)
    logger.info("{} loaded. Found {} samples, resulting in {} mini-batches.".format(
        opt.dataset, len(dataset), _n_batches))
    avg_rst_freq_sampler = bernoulli.Bernoulli(opt.rst_freq)
    avg_step = 1
    avg_g = copy.deepcopy(generator['net'])
    avg_d = copy.deepcopy(discriminator['net'])
    _requires_grad(avg_g, False)
    _requires_grad(avg_d, False)

    # Metrics. ############################
    if opt.metric_freq > 0:
        metrics_data = {
            'logger': setup_logger(opt.outdir, 'metrics.log'),
            'classifier': pretrained_model.mnist(pretrained='./mnist_clf.pth'),
            'net': generator['net'] if opt.reset else avg_g,
            'noise': torch.FloatTensor(opt.metric_sample_size, _NOISE_DIM).normal_(0, 1),
            'sample': torch.FloatTensor(opt.metric_sample_size, _IMG_CHANNELS * opt.image_size ** 2),
            'results': []}
        subsample(dataset, metrics_data['sample'], batch_size=opt.batch_size)
        if opt.cont:
            with open(os.path.join(opt.outdir, 'metrics.json')) as data_file:
                metrics_data['results'] = json.load(data_file)

    start_iter = 0
    ################################################################################
    # train
    ################################################################################
    mu_g, mu_d = calculate_svrg_stats(g_snapshot, d_snapshot,
                                      dataset, noise_dataset,
                                      batch_size=opt.large_batch_size,
                                      _dragan='dra' in opt.gan_type) \
        if opt.svrg else (None, None)

    for iteration in range(start_iter, opt.iterations):
        if opt.verbose and iteration % 1000 == 0:
            print('Iter %d/%d' % (iteration + 1, opt.iterations))

        train(generator, discriminator, data_sampler, noise_sampler, 'dra' in opt.gan_type,
              g_snapshot, d_snapshot, mu_g, mu_d)

        # if opt.svrg and iteration > 0 and iteration % opt.svrg_freq == 0:
        if opt.svrg and iteration > 0 and svrg_freq_sampler.sample() == 1:

            if opt.reset and iteration > 0 and avg_rst_freq_sampler.sample() == 1:
                generator['net'].load_state_dict(avg_g.state_dict())
                discriminator['net'].load_state_dict(avg_d.state_dict())
                logger.info('Parameters copied from the avg at %d iteration.' % iteration)
                avg_step = 1

            g_snapshot.load_state_dict(generator['net'].state_dict())
            d_snapshot.load_state_dict(discriminator['net'].state_dict())
            mu_g, mu_d = calculate_svrg_stats(g_snapshot, d_snapshot,
                                              dataset, noise_dataset,
                                              batch_size=opt.large_batch_size,
                                              _dragan='dra' in opt.gan_type)
            logger.info("SVRG stats updated at %d-th iteration." % iteration)

        # avg nets
        if iteration > 0:
            update_avg(avg_g, generator['net'], avg_step)
            update_avg(avg_d, discriminator['net'], avg_step)
        avg_step += 1

        if opt.sample_freq > 0 and (iteration + 1) % opt.sample_freq == 0:
            samples = generator['net'](fixed_noise).data.cpu()
            samples = samples.view(100, *dataset[0][0].size())
            file_name = os.path.join(opt.outdir, 'fake_samples', '%08d.png' % iteration)
            vision_utils.save_image(samples, file_name, nrow=10)

        if opt.generator_freq > 0 and (iteration + 1) % opt.generator_freq == 0:
            file_name = os.path.join(opt.outdir, 'generator', '%08d.pth' % iteration)
            torch.save(generator['net'].state_dict(), file_name)

        if opt.metric_freq > 0 and (iteration + 1) % opt.metric_freq == 0:
            metrics(metrics_data, iteration)
            with open(os.path.join(opt.outdir, 'metrics.json'), 'w') as f:
                json.dump(metrics_data['results'], f, sort_keys=True, indent=4, separators=(',', ': '))

        if opt.backup_freq > 0 and (iteration + 1) % opt.backup_freq == 0:
            _dir = os.path.join(opt.outdir, 'backup')
            torch.save(generator['net'].state_dict(), _dir + '/gen.pth')
            torch.save(discriminator['net'].state_dict(), _dir + '/disc.pth')
            torch.save(generator['optim'].state_dict(), _dir + '/gen_optim.pth')
            torch.save(discriminator['optim'].state_dict(), _dir + '/disc_optim.pth')
            with open(os.path.join(_dir, "timestamp.txt"), "w") as fff:
                fff.write("%d" % iteration)

        # Todo: load if opt.cont is selected

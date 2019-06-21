# -*- coding: utf-8 -*-
import os
import time
import torch
import datetime
import copy

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.distributions import bernoulli

from sagan_models import Generator, Discriminator
# from resnet_models import Generator, Discriminator
from utils import *


class Trainer(object):
    def __init__(self, data_loader, config):
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

        self.data_loader = data_loader
        self.model = config.model
        self.adv_loss = config.adv_loss

        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel
        self.extra = config.extra

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_scheduler = config.lr_scheduler
        self.g_beta1 = config.g_beta1
        self.d_beta1 = config.d_beta1
        self.beta2 = config.beta2

        self.dataset = config.dataset
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        self.backup_freq = config.backup_freq
        self.bup_path = config.bup_path

        # Path
        self.optim = config.optim
        self.svrg = config.svrg
        self.avg_start = config.avg_start
        self.build_model()

        if self.svrg:
            self.mu_g = []
            self.mu_d = []
            self.g_snapshot = copy.deepcopy(self.G)
            self.d_snapshot = copy.deepcopy(self.D)
            self.svrg_freq_sampler = bernoulli.Bernoulli(torch.tensor([1 / len(self.data_loader)]))

        self.info_logger = setup_logger(self.log_path)
        self.info_logger.info(config)
        self.cont = config.cont

    def train(self):
        self.data_gen = self._data_gen()

        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

        if self.cont:
            start = self.load_backup()
        else:
            start = 0

        start_time = time.time()
        if self.svrg:
            self.update_svrg_stats()
        for step in range(start, self.total_step):

            # =================== SVRG =================== #
            if self.svrg and self.svrg_freq_sampler.sample() == 1:

                # ================= Update Avg ================= #
                if self.avg_start >= 0 and step > 0 and step >= self.avg_start:
                    self.update_avg_nets()
                    if self.avg_freq_restart_sampler.sample() == 1:
                        self.G.load_state_dict(self.avg_g.state_dict())
                        self.D.load_state_dict(self.avg_d.state_dict())
                        self.avg_step = 1
                        self.info_logger.info('Params updated with avg-nets at %d-th step.' % step)

                self.update_svrg_stats()
                self.info_logger.info("SVRG stats updated at %d-th step." % step)

            # ================= Train pair ================= #
            d_loss_real = self._update_pair(step)

            # --- storing stuff ---
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], Step [{}/{}]".format(elapsed, step + 1, self.total_step))

            if (step + 1) % self.sample_step == 0:
                save_image(denorm(self.G(fixed_z).data),
                           os.path.join(self.sample_path, 'gen', 'iter%08d.png' % step))
                save_image(denorm(self.G_avg(fixed_z).data),
                           os.path.join(self.sample_path, 'gen_avg', 'iter%08d.png' % step))
                save_image(denorm(self.G_ema(fixed_z).data),
                           os.path.join(self.sample_path, 'gen_ema', 'iter%08d.png' % step))

            if self.model_save_step > 0 and (step+1) % self.model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, 'gen', 'iter%08d.pth' % step))
                torch.save(self.G_avg.state_dict(),
                           os.path.join(self.model_save_path, 'gen_avg', 'iter%08d.pth' % step))
                torch.save(self.G_ema.state_dict(),
                           os.path.join(self.model_save_path, 'gen_ema', 'iter%08d.pth' % step))
            if self.backup_freq > 0 and (step+1) % self.backup_freq == 0:
                self.backup(step)

    def _data_gen(self):
        """ Data iterator

        :return: s
        """
        data_iter = iter(self.data_loader)
        while True:
            try:
                real_images, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)
            yield real_images

    def _update_pair(self, step):
        _lr_scheduler = self.lr_scheduler > 0 and step > 0 and step % len(self.data_loader) == 0
        self.D.train()
        self.G.train()
        real_images = tensor2var(next(self.data_gen))

        self._extra_sync_nets()
        if self.extra:
            # ================== Train D @ t + 1/2 ================== #
            self._backprop_disc(D=self.D_extra, G=self.G, real_images=real_images,
                                d_optim=self.d_optimizer_extra, svrg=self.svrg,
                                scheduler_d=self.scheduler_d_extra if _lr_scheduler else None)

            # ================== Train G @ t + 1/2 ================== #
            self._backprop_gen(G=self.G_extra, D=self.D, bsize=real_images.size(0),
                               g_optim=self.g_optimizer_extra, svrg=self.svrg,
                               scheduler_g=self.scheduler_g_extra if _lr_scheduler else None)

            real_images = tensor2var(next(self.data_gen))  # Re-sample

        # ================== Train D @ t + 1 ================== #
        d_loss_real = self._backprop_disc(G=self.G_extra, D=self.D, real_images=real_images,
                                          d_optim=self.d_optimizer, svrg=self.svrg,
                                          scheduler_d=self.scheduler_d if _lr_scheduler else None)

        # ================== Train G and gumbel @ t + 1 ================== #
        self._backprop_gen(G=self.G, D=self.D_extra, bsize=real_images.size(0),
                           g_optim=self.g_optimizer, svrg=self.svrg,
                           scheduler_g=self.scheduler_g if _lr_scheduler else None)

        # === Moving avg Generator-nets ===
        self._update_avg_gen(step)
        self._update_ema_gen()
        return d_loss_real

    def _normalize_acc_grads(self, net):
        """Divides accumulated gradients with len(self.data_loader)"""
        for _param in filter(lambda p: p.requires_grad, net.parameters()):
            _param.grad.data.div_(len(self.data_loader))

    def update_svrg_stats(self):
        self.mu_g, self.mu_d = [], []

        # Update mu_d  ####################
        self.d_optimizer.zero_grad()
        for _, _data in enumerate(self.data_loader):
            real_images = tensor2var(_data[0])
            self._backprop_disc(self.G, self.D, real_images, d_optim=None, svrg=False)
        self._normalize_acc_grads(self.D)
        for _param in filter(lambda p: p.requires_grad, self.D.parameters()):
            self.mu_d.append(_param.grad.data.clone())

        # Update mu_g  ####################
        self.g_optimizer.zero_grad()
        for _ in range(len(self.data_loader)):
            self._backprop_gen(self.G, self.D, self.batch_size, g_optim=None, svrg=False)
        self._normalize_acc_grads(self.G)
        for _param in filter(lambda p: p.requires_grad, self.G.parameters()):
            self.mu_g.append(_param.grad.data.clone())

        # Update snapshots  ###############
        self.g_snapshot.load_state_dict(self.G.state_dict())
        self.d_snapshot.load_state_dict(self.D.state_dict())

    @staticmethod
    def _update_grads_svrg(params, snapshot_params, mu):
        """Helper function which updates the accumulated gradients of
        params by subtracting those of snapshot and adding mu.

        Operates in-place.
        See line 12 & 14 of Algo. 3 in SVRG-GAN.

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

    def _backprop_disc(self, G, D, real_images, d_optim=None, svrg=False, scheduler_d=None):
        """Updates D (Vs. G).

        :param G:
        :param D:
        :param real_images:
        :param d_optim: if None, only backprop
        :param svrg:
        :return:
        """
        d_out_real = D(real_images)
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif self.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        else:
            raise NotImplementedError

        z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
        fake_images = G(z)
        d_out_fake = D(fake_images)

        if self.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        else:
            raise NotImplementedError

        # Backward + Optimize
        d_loss = d_loss_real + d_loss_fake
        if d_optim is not None:
            d_optim.zero_grad()
        d_loss.backward()
        if d_optim is not None:
            if svrg:  # d_snapshot Vs. g_snapshot
                d_out_real = self.d_snapshot(real_images)
                d_out_fake = self.d_snapshot(self.g_snapshot(z))
                if self.adv_loss == 'wgan-gp':
                    d_s_loss_real = - torch.mean(d_out_real)
                    d_loss_fake = d_out_fake.mean()
                elif self.adv_loss == 'hinge':
                    d_s_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                    d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
                else:
                    raise NotImplementedError

                d_loss = d_s_loss_real + d_loss_fake
                self.d_snapshot.zero_grad()
                d_loss.backward()

                self._update_grads_svrg(list(filter(lambda p: p.requires_grad, D.parameters())),
                                        list(filter(lambda p: p.requires_grad, self.d_snapshot.parameters())),
                                        self.mu_d)
            d_optim.step()
            if scheduler_d is not None:
                scheduler_d.step()

        if self.adv_loss == 'wgan-gp':  # Todo: add SVRG for wgan-gp
            raise NotImplementedError('SVRG-WGAN-gp is not implemented yet')
            # Compute gradient penalty
            alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
            interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
            out = D(interpolated)

            grad = torch.autograd.grad(outputs=out,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            # Backward + Optimize
            d_loss = self.lambda_gp * d_loss_gp

            if d_optim is not None:
                d_optim.reset_grad()
            d_loss.backward()
            if d_optim is not None:
                self.d_optimizer.step()
        return d_loss_real.data.item()

    def _backprop_gen(self, G, D, bsize, g_optim=True, svrg=False, scheduler_g=None):
        """Updates G (Vs. D).

        :param G:
        :param D:
        :param bsize:
        :param g_optim: if None only backprop
        :param svrg:
        :return:
        """
        z = tensor2var(torch.randn(bsize, self.z_dim))
        fake_images = G(z)

        g_out_fake = D(fake_images)  # batch x n
        if self.adv_loss == 'wgan-gp' or self.adv_loss == 'hinge':
            g_loss_fake = - g_out_fake.mean()

        if g_optim is not None:
            g_optim.zero_grad()
        g_loss_fake.backward()
        if g_optim is not None:
            if svrg:  # G_snapshot Vs. D_snapshot
                self.g_snapshot.zero_grad()
                if self.adv_loss == 'wgan-gp' or self.adv_loss == 'hinge':
                    (- self.d_snapshot(self.g_snapshot(z)).mean()).backward()
                else:
                    raise NotImplementedError
                self._update_grads_svrg(list(filter(lambda p: p.requires_grad, G.parameters())),
                                        list(filter(lambda p: p.requires_grad, self.g_snapshot.parameters())),
                                        self.mu_g)
            g_optim.step()
            if scheduler_g is not None:
                scheduler_g.step()
        return g_loss_fake.data.item()

    def build_model(self):
        # Models                    ###################################################################
        self.G = Generator(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.D = Discriminator(self.batch_size, self.imsize, self.d_conv_dim).cuda()
        # Todo: do not allocate unnecessary GPU mem for G_extra and D_extra if self.extra == False
        self.G_extra = Generator(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.D_extra = Discriminator(self.batch_size, self.imsize, self.d_conv_dim).cuda()
        if self.avg_start >= 0:
            self.avg_g = copy.deepcopy(self.G)
            self.avg_d = copy.deepcopy(self.D)
            self._requires_grad(self.avg_g, False)
            self._requires_grad(self.avg_d, False)
            self.avg_g.eval()
            self.avg_d.eval()
            self.avg_step = 1
            self.avg_freq_restart_sampler = bernoulli.Bernoulli(.1)

        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            self.G_extra = nn.DataParallel(self.G_extra)
            self.D_extra = nn.DataParallel(self.D_extra)
            if self.avg_start >= 0:
                self.avg_g = nn.DataParallel(self.avg_g)
                self.avg_d = nn.DataParallel(self.avg_d)
        self.G_extra.train()
        self.D_extra.train()

        self.G_avg = copy.deepcopy(self.G)
        self.G_ema = copy.deepcopy(self.G)
        self._requires_grad(self.G_avg, False)
        self._requires_grad(self.G_ema, False)

        # Logs, Loss & optimizers   ###################################################################
        grad_var_logger_g = setup_logger(self.log_path, 'log_grad_var_g.log')
        grad_var_logger_d = setup_logger(self.log_path, 'log_grad_var_d.log')
        grad_mean_logger_g = setup_logger(self.log_path, 'log_grad_mean_g.log')
        grad_mean_logger_d = setup_logger(self.log_path, 'log_grad_mean_d.log')

        if self.optim == 'sgd':
            self.g_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.G.parameters()),
                                               self.g_lr,
                                               logger_mean=grad_mean_logger_g,
                                               logger_var=grad_var_logger_g)
            self.d_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.D.parameters()),
                                               self.d_lr,
                                               logger_mean=grad_mean_logger_d,
                                               logger_var=grad_var_logger_d)
            self.g_optimizer_extra = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                                            self.G_extra.parameters()),
                                                     self.g_lr)
            self.d_optimizer_extra = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                                            self.D_extra.parameters()),
                                                     self.d_lr)
        elif self.optim == 'adam':
            self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()),
                                                self.g_lr, [self.g_beta1, self.beta2],
                                                logger_mean=grad_mean_logger_g,
                                                logger_var=grad_var_logger_g)
            self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()),
                                                self.d_lr, [self.d_beta1, self.beta2],
                                                logger_mean=grad_mean_logger_d,
                                                logger_var=grad_var_logger_d)
            self.g_optimizer_extra = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                             self.G_extra.parameters()),
                                                      self.g_lr, [self.g_beta1, self.beta2])
            self.d_optimizer_extra = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                             self.D_extra.parameters()),
                                                      self.d_lr, [self.d_beta1, self.beta2])
        elif self.optim == 'svrgadam':
            self.g_optimizer = torch.optim.SvrgAdam(filter(lambda p: p.requires_grad, self.G.parameters()),
                                                    self.g_lr, [self.g_beta1, self.beta2],
                                                    logger_mean=grad_mean_logger_g,
                                                    logger_var=grad_var_logger_g)
            self.d_optimizer = torch.optim.SvrgAdam(filter(lambda p: p.requires_grad, self.D.parameters()),
                                                    self.d_lr, [self.d_beta1, self.beta2],
                                                    logger_mean=grad_mean_logger_d,
                                                    logger_var=grad_var_logger_d)
            self.g_optimizer_extra = torch.optim.SvrgAdam(filter(lambda p: p.requires_grad,
                                                          self.G_extra.parameters()),
                                                          self.g_lr, [self.g_beta1, self.beta2])
            self.d_optimizer_extra = torch.optim.SvrgAdam(filter(lambda p: p.requires_grad,
                                                          self.D_extra.parameters()),
                                                          self.d_lr, [self.d_beta1, self.beta2])
        else:
            raise NotImplementedError('Supported optimizers: SGD, Adam, Adadelta')

        if self.lr_scheduler > 0:  # Exponentially decaying learning rate
            self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer,
                                                                      gamma=self.lr_scheduler)
            self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer,
                                                                      gamma=self.lr_scheduler)
            self.scheduler_g_extra = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer_extra,
                                                                            gamma=self.lr_scheduler)
            self.scheduler_d_extra = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer_extra,
                                                                            gamma=self.lr_scheduler)

        print(self.G)
        print(self.D)

    def _extra_sync_nets(self):
        """ Helper function. Copies the current parameters to the t+1/2 parameters,
         stored as 'net' and 'extra_net', respectively.

        :return: [None]
        """
        self.G_extra.load_state_dict(self.G.state_dict())
        self.D_extra.load_state_dict(self.D.state_dict())

    @staticmethod
    def _update_avg(avg_net, net, avg_step):
        """Updates average network."""
        # Todo: input val
        net_param = list(net.parameters())
        for i, p in enumerate(avg_net.parameters()):
            p.mul_((avg_step - 1) / avg_step)
            p.add_(net_param[i].div(avg_step))

    @staticmethod
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

    def update_avg_nets(self):
        self._update_avg(self.avg_g, self.G, self.avg_step)
        self._update_avg(self.avg_d, self.D, self.avg_step)
        self.avg_step += 1

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))

    def backup(self, iteration):
        """Back-ups the networks & optimizers' states.

        Note: self.g_extra & self.d_extra are not stored, as these are copied from
        self.G & self.D at the beginning of each iteration. However, the optimizers
        are backed up.

        :param iteration: [int]
        :return: [None]
        """
        torch.save(self.G.state_dict(), os.path.join(self.bup_path, 'gen.pth'))
        torch.save(self.D.state_dict(), os.path.join(self.bup_path, 'disc.pth'))
        torch.save(self.G_avg.state_dict(), os.path.join(self.bup_path, 'gen_avg.pth'))
        torch.save(self.G_ema.state_dict(), os.path.join(self.bup_path, 'gen_ema.pth'))

        torch.save(self.g_optimizer.state_dict(), os.path.join(self.bup_path, 'gen_optim.pth'))
        torch.save(self.d_optimizer.state_dict(), os.path.join(self.bup_path, 'disc_optim.pth'))
        torch.save(self.g_optimizer_extra.state_dict(), os.path.join(self.bup_path, 'gen_extra_optim.pth'))
        torch.save(self.d_optimizer_extra.state_dict(), os.path.join(self.bup_path, 'disc_extra_optim.pth'))

        with open(os.path.join(self.bup_path, "timestamp.txt"), "w") as fff:
            fff.write("%d" % iteration)

    def load_backup(self):
        """Loads the Backed-up networks & optimizers' states.

        Note: self.g_extra & self.d_extra are not stored, as these are copied from
        self.G & self.D at the beginning of each iteration. However, the optimizers
        are backed up.

        :return: [int] timestamp to continue from
        """
        if not os.path.exists(self.bup_path):
            raise ValueError('Cannot load back-up. Directory {} '
                             'does not exist.'.format(self.bup_path))

        self.G.load_state_dict(torch.load(os.path.join(self.bup_path, 'gen.pth')))
        self.D.load_state_dict(torch.load(os.path.join(self.bup_path, 'disc.pth')))
        self.G_avg.load_state_dict(torch.load(os.path.join(self.bup_path, 'gen_avg.pth')))
        self.G_ema.load_state_dict(torch.load(os.path.join(self.bup_path, 'gen_ema.pth')))

        self.g_optimizer.load_state_dict(torch.load(os.path.join(self.bup_path, 'gen_optim.pth')))
        self.d_optimizer.load_state_dict(torch.load(os.path.join(self.bup_path, 'disc_optim.pth')))
        self.g_optimizer_extra.load_state_dict(torch.load(os.path.join(self.bup_path, 'gen_extra_optim.pth')))
        self.d_optimizer_extra.load_state_dict(torch.load(os.path.join(self.bup_path, 'disc_extra_optim.pth')))

        with open(os.path.join(self.bup_path, "timestamp.txt"), "r") as fff:
            timestamp = [int(x) for x in next(fff).split()]  # read first line
            if not len(timestamp) == 1:
                raise ValueError('Could not determine timestamp of the backed-up models.')
            timestamp = int(timestamp[0]) + 1

        self.info_logger.info("Loaded models from %s, at timestamp %d." %
                              (self.bup_path, timestamp))
        return timestamp

    def _update_avg_gen(self, n_gen_update):
        """ Updates the uniform average generator. """
        l_param = list(self.G.parameters())
        l_avg_param = list(self.G_avg.parameters())
        if len(l_param) != len(l_avg_param):
            raise ValueError("Got different lengths: {}, {}".format(len(l_param), len(l_avg_param)))

        for i in range(len(l_param)):
            l_avg_param[i].data.copy_(l_avg_param[i].data.mul(n_gen_update).div(n_gen_update + 1.).add(
                                      l_param[i].data.div(n_gen_update + 1.)))

    def _update_ema_gen(self, beta_ema=0.9999):
        """ Updates the exponential moving average generator. """
        l_param = list(self.G.parameters())
        l_ema_param = list(self.G_ema.parameters())
        if len(l_param) != len(l_ema_param):
            raise ValueError("Got different lengths: {}, {}".format(len(l_param), len(l_ema_param)))

        for i in range(len(l_param)):
            l_ema_param[i].data.copy_(l_ema_param[i].data.mul(beta_ema).add(
                l_param[i].data.mul(1-beta_ema)))

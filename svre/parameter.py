# -*- coding: utf-8 -*-
import argparse


def str2bool(v):
    return v.lower() in ('true')


def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='sagan', choices=['sagan', 'qgan'])
    parser.add_argument('--adv_loss', type=str, default='hinge', choices=['wgan-gp', 'hinge'])
    parser.add_argument('--imsize', type=int, default=64)
    parser.add_argument('--g_num', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--version', type=str,
                        help='name of sub-directory: ./results/<dataset>/'
                             '<version>[_svrg]_optim[_avgFreq]/'
                             '{data, logs, models, samples, attn, backup}')

    # Training setting
    parser.add_argument('--total_step', type=int, default=500000,
                        help='how many times to update the generator, default: %(default)s')
    parser.add_argument('--d_iters', type=float, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--g_beta1', type=float, default=0.0)
    parser.add_argument('--d_beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)

    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=True)
    parser.add_argument('--dataset', type=str, default='celeb',
                        choices=['imagenet', 'cifar10', 'lsun', 'svhn'])
    parser.add_argument('--cont', action='store_true', help='continue training, default: %(default)s')

    # Path
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_path', type=str, default='./results/<dataset>/<version>/logs')
    parser.add_argument('--model_save_path', type=str, default='./results/<dataset>/<version>/models')
    parser.add_argument('--sample_path', type=str, default='./results/<dataset>/<version>/samples')
    parser.add_argument('--bup_path', type=str, default='./results/<dataset>/<version>/backup')

    # Step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--backup_freq', type=int, default=10000,
                        help='Frequency to backup models and optim states, default: %(default)s')

    # Random seed
    parser.add_argument('--seed', type=int, default=1, help='fixed random seed, default: %(default)s')

    # SVRG
    parser.add_argument('--svrg', action='store_true', help='activate SVRG, default: %(default)s')
    parser.add_argument('--optim', type=str, default='svrgadam', choices=['sgd', 'adam', 'svrgadam'])
    parser.add_argument('--extra', type=str2bool, default=False)
    parser.add_argument('--avg_start', type=int, default=-1,
                        help='Select warm start iteration. Use negative value to cancel averaging. '
                             'Default: %(default)s')
    parser.add_argument('--lr_scheduler', type=float, default=-1,
                        help='Gamma parameter for a learning rate scheduler. Use negative value to cancel it. '
                             'Default: %(default)s')
    _args = parser.parse_args()
    if _args.version is None:
        _args.version = 'extra' if _args.extra else 'gan'
        _args.version += '_svrg' if _args.svrg else ''
        _args.version += '_' + _args.optim
        if _args.avg_start >= 0:
            _args.version += '_avg%d' % _args.avg_start
        _args.version += '/G%f_D%f' % (_args.g_lr, _args.d_lr)
        if _args.optim != 'sgd':
            _args.version += '_beta_%.1f' % _args.g_beta1
        if _args.lr_scheduler > 0:
            _args.version += '_gamma_%.2f' % _args.lr_scheduler

    _args.log_path = _args.log_path.replace("<dataset>", _args.dataset)
    _args.model_save_path = _args.model_save_path.replace("<dataset>", _args.dataset)
    _args.sample_path = _args.sample_path.replace("<dataset>", _args.dataset)
    _args.bup_path = _args.bup_path.replace("<dataset>", _args.dataset)

    _args.log_path = _args.log_path.replace("<version>", _args.version)
    _args.model_save_path = _args.model_save_path.replace("<version>", _args.version)
    _args.sample_path = _args.sample_path.replace("<version>", _args.version)
    _args.bup_path = _args.bup_path.replace("<version>", _args.version)

    if 'cifar' in _args.dataset or 'svhn' in _args.dataset:
        _args.imsize = 32
    return _args

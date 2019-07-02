# -*- coding: utf-8 -*-
"""
Supplementary code for a paper under review.
Do not distribute.
"""
from parameter import *
from trainer import Trainer
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder


def main(config):
    cudnn.benchmark = True

    data_loader = Data_Loader(config.train, config.dataset, config.data_dir, config.imsize,
                             config.batch_size, shuf=config.train)

    for _subdir in ['gen', 'gen_avg', 'gen_ema']:
        make_folder(config.model_save_path, _subdir)
        make_folder(config.sample_path, _subdir)

    make_folder(config.log_path)
    if config.backup_freq > 0:
        make_folder(config.bup_path)

    if config.model == 'sagan':
        trainer = Trainer(data_loader.loader(), config)
    elif config.model == 'qgan':
        trainer = qgan_trainer(data_loader.loader(), config)
    trainer.train()


if __name__ == '__main__':
    args = get_parameters()
    print(args)
    main(args)

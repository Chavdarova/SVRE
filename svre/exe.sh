#!/bin/sh

# source activate pytorch1  # conda env 

data_dir='~/data/CIFAR10/'
dataset=cifar10	# imagenet | cifar10 | stl10 | svhn 
g_lr=0.01		# type=float, default=0.0001
d_lr=0.04		# type=float, default=0.0004
bsize=64
avg_start=-1	# def:-1. Select warm start iteration. Use negative value to cancel averaging.
gamma=-1		# def:-1 (float) Gamma param for a learning rate scheduler. Use negative value to cancel it. Typically used: 0.99.


# Other params:
# --img_size: Default img_size is 64, 32 if CIFAR10 or SVHN
# --optim: type=str, default='svrgadam', choices=['sgd', 'adam', 'svrgadam']
# --cont	
# --extra activates extragradient, otherwise, vanilla GAN is used



python main.py \
	--dataset ${dataset} --adv_loss hinge \
	--sample_step 1000 \
	--data_dir ${data_dir} \
	--g_lr ${g_lr} \
	--d_lr ${d_lr} \
	--avg_start ${avg_start} \
    --lr_scheduler ${gamma} \
	--batch_size ${bsize} \
	--extra True --optim sgd \
	--svrg

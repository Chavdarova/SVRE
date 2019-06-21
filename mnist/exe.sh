#!/bin/sh

# Supplementary code for paper under review by the International Conference on Machine Learning (ICML).
# Do not distribute.



dataset=mnist			# mnsit
gan_type="gan"  		# gan, sngan, dragan, wgan
dataroot=~/datasets/MNIST
sample_freq=100
backup_freq=1000
lrD=0.001
lrG=0.001
optim=svrgadam  		# adam, svrgadam, sgd, adadelta
beta1=0.5


outdir="results/${dataset}/${gan_type}_g${lrG}_d${lrD}_beta_${beta1}_2"



python main.py \
	   --lrD ${lrD} --lrG ${lrG} \
	   --outdir ${outdir} \
	   --dataroot ${dataroot} \
	   --dataset ${dataset} \
	   --backup_freq ${backup_freq} \
	   --sample_freq ${sample_freq} \
	   --optim ${optim} \
	   --beta1 ${beta1} \
	   --svrg --cuda
	   

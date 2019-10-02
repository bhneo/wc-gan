#!/usr/bin/env bash
name=$(basename $0)
base_name=${name%.*}

name="${base_name}_baseline"
python run.py --name $name --dataset cifar10 --generator_adversarial_objective wgan\
 --discriminator_adversarial_objective wgan  --generator_block_norm d --generator_block_after_norm uconv\
 --generator_last_norm d --decomposition cholesky --group 1 --generator_last_after_norm uconv --discriminator_filters 128 --generator_filters 128\
 --discriminator_spectral 0 --gradient_penalty_weight 10 --lr_decay_schedule linear --number_of_epochs 100 --arc res --training_ratio 5 --generator_batch_multiple 2

name="${base_name}_pca"
python run.py --name $name --dataset cifar10 --generator_adversarial_objective wgan\
 --discriminator_adversarial_objective wgan  --generator_block_norm d --generator_block_after_norm uconv\
 --generator_last_norm d --decomposition pca --group 1 --generator_last_after_norm uconv --discriminator_filters 128 --generator_filters 128\
 --discriminator_spectral 0 --gradient_penalty_weight 10 --lr_decay_schedule linear --number_of_epochs 100 --arc res --training_ratio 5 --generator_batch_multiple 2

name="${base_name}_zca"
python run.py --name $name --dataset cifar10 --generator_adversarial_objective wgan\
 --discriminator_adversarial_objective wgan  --generator_block_norm d --generator_block_after_norm uconv\
 --generator_last_norm d --decomposition zca --group 1 --generator_last_after_norm uconv --discriminator_filters 128 --generator_filters 128\
 --discriminator_spectral 0 --gradient_penalty_weight 10 --lr_decay_schedule linear --number_of_epochs 100 --arc res --training_ratio 5 --generator_batch_multiple 2


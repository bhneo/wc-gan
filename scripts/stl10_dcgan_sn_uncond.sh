#!/usr/bin/env bash
name=$(basename $0)
base_name=${name%.*}

name="${base_name}_baseline"
python run.py --name $name --dataset stl10 --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge --generator_block_norm d --generator_block_after_norm uconv\
 --generator_last_norm d --decomposition cholesky --group 1 --generator_last_after_norm uconv --discriminator_filters 512 --generator_filters 512\
 --discriminator_spectral 1 --gradient_penalty_weight 0 --lr_decay_schedule linear --number_of_epochs 200 --arc dcgan --training_ratio 1 --generator_batch_multiple 1

name="${base_name}_pca"
python run.py --name $name --dataset stl10 --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge --generator_block_norm d --generator_block_after_norm uconv\
 --generator_last_norm d --decomposition pca --group 1 --generator_last_after_norm uconv --discriminator_filters 512 --generator_filters 512\
 --discriminator_spectral 1 --gradient_penalty_weight 0 --lr_decay_schedule linear --number_of_epochs 200 --arc dcgan --training_ratio 1 --generator_batch_multiple 1

name="${base_name}_zca"
python run.py --name $name --dataset stl10 --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge --generator_block_norm d --generator_block_after_norm uconv\
 --generator_last_norm d --decomposition zca --group 1 --generator_last_after_norm uconv --discriminator_filters 512 --generator_filters 512\
 --discriminator_spectral 1 --gradient_penalty_weight 0 --lr_decay_schedule linear --number_of_epochs 200 --arc dcgan --training_ratio 1 --generator_batch_multiple 1

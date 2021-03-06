#!/usr/bin/env bash
#Maximum achieved at 30k
name=$(basename $0)
base_name=${name%.*}

name="${base_name}_baseline"
python run.py --name $name --dataset cifar10 --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge  --generator_block_norm d --generator_block_after_norm ucconv\
 --generator_last_norm d --decomposition cholesky --group 1 --generator_last_after_norm uconv --discriminator_filters 256 --generator_filters 128\
 --discriminator_spectral 1 --gradient_penalty_weight 0 --lr_decay_schedule dropat30 --number_of_epochs 50 --gan_type PROJECTIVE --arc res --training_ratio 5 --generator_batch_multiple 2

name="${base_name}_pca"
python run.py --name $name --dataset cifar10 --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge  --generator_block_norm d --generator_block_after_norm ucconv\
 --generator_last_norm d --decomposition pca --group 1 --generator_last_after_norm uconv --discriminator_filters 256 --generator_filters 128\
 --discriminator_spectral 1 --gradient_penalty_weight 0 --lr_decay_schedule dropat30 --number_of_epochs 50 --gan_type PROJECTIVE --arc res --training_ratio 5 --generator_batch_multiple 2

name="${base_name}_zca"
python run.py --name $name --dataset cifar10 --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge  --generator_block_norm d --generator_block_after_norm ucconv\
 --generator_last_norm d --decomposition zca --group 1 --generator_last_after_norm uconv --discriminator_filters 256 --generator_filters 128\
 --discriminator_spectral 1 --gradient_penalty_weight 0 --lr_decay_schedule dropat30 --number_of_epochs 50 --gan_type PROJECTIVE --arc res --training_ratio 5 --generator_batch_multiple 2


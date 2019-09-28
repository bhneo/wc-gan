#!/usr/bin/env bash
name=$(basename $0)
name=${name%.*}
python run.py --name $name --dataset cifar10 --generator_adversarial_objective wgan\
 --discriminator_adversarial_objective wgan  --generator_block_norm d --generator_block_after_norm uconv\
 --generator_last_norm d --generator_last_after_norm uconv --discriminator_filters 128 --generator_filters 128\
 --gradient_penalty_weight 10 --lr_decay_schedule linear --number_of_epochs 100

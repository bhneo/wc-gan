#!/usr/bin/env bash
#Maximum achieved at 20k
python run.py --dataset cifar100 --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge --lr 2e-4 --generator_block_norm d --generator_block_after_norm ucconv\
 --generator_last_norm d --generator_last_after_norm uconv --discriminator_filters 256 --generator_filters 128\
 --spectral 1 --lr_decay_schedule linear --number_of_epochs 50 --gan_type PROJECTIVE

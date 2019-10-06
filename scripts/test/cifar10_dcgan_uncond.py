import os


base_name = os.path.basename(__file__).split('.')[0]

os.system('python run.py --name {} --dataset cifar10\
          --generator_adversarial_objective hinge --discriminator_adversarial_objective hinge\
          --generator_block_norm b --generator_last_norm b\
          --discriminator_norm d\
          --decomposition zca --iter_num 5 --group 32\
          --instance_norm 1\
          --generator_block_after_norm uconv --generator_last_after_norm uconv\
          --discriminator_filters 256 --generator_filters 256\
          --discriminator_spectral 1\
          --gradient_penalty_weight 0\
          --lr_decay_schedule linear --number_of_epochs 50  --arc dcgan --training_ratio 1 --generator_batch_multiple 1'.format(base_name))

os.system('python run.py --name {} --dataset cifar10\
          --generator_adversarial_objective hinge --discriminator_adversarial_objective hinge\
          --generator_block_norm b --generator_last_norm b\
          --discriminator_norm d\
          --decomposition zca --iter_num 5 --group 32\
          --instance_norm 0\
          --generator_block_after_norm uconv --generator_last_after_norm uconv\
          --discriminator_filters 256 --generator_filters 256\
          --discriminator_spectral 1\
          --gradient_penalty_weight 0\
          --lr_decay_schedule linear --number_of_epochs 50  --arc dcgan --training_ratio 1 --generator_batch_multiple 1'.format(base_name))


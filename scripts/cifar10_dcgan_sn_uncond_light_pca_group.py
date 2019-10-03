import os


base_name = os.path.basename(__file__).split('.')[0]

os.system('python run.py --name {} --dataset cifar10 --generator_adversarial_objective hinge\
          --discriminator_adversarial_objective hinge  --generator_block_norm d --generator_block_after_norm uconv\
          --generator_last_norm d --decomposition pca --group 32 --generator_last_after_norm uconv --discriminator_filters 256 --generator_filters 256\
          --discriminator_spectral 1 --gradient_penalty_weight 0 --lr_decay_schedule linear --number_of_epochs 50  --arc dcgan --training_ratio 1 --generator_batch_multiple 1'.format(base_name))

os.system('python run.py --name {} --dataset cifar10 --generator_adversarial_objective hinge\
          --discriminator_adversarial_objective hinge  --generator_block_norm d --generator_block_after_norm uconv\
          --generator_last_norm d --decomposition pca --group 1 --generator_last_after_norm uconv --discriminator_filters 256 --generator_filters 256\
          --discriminator_spectral 1 --gradient_penalty_weight 0 --lr_decay_schedule linear --number_of_epochs 50  --arc dcgan --training_ratio 1 --generator_batch_multiple 1'.format(base_name))

os.system('python run.py --name {} --dataset cifar10 --generator_adversarial_objective hinge\
          --discriminator_adversarial_objective hinge  --generator_block_norm d --generator_block_after_norm uconv\
          --generator_last_norm d --decomposition pca --group 16 --generator_last_after_norm uconv --discriminator_filters 256 --generator_filters 256\
          --discriminator_spectral 1 --gradient_penalty_weight 0 --lr_decay_schedule linear --number_of_epochs 50  --arc dcgan --training_ratio 1 --generator_batch_multiple 1'.format(base_name))

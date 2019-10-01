import os


base_name = os.path.basename(__file__).split('.')[0]

name = base_name + '_baseline'
print(name)
os.system('python run.py --name {} --dataset cifar10 --generator_adversarial_objective wgan\
          --discriminator_adversarial_objective wgan  --generator_block_norm d --generator_block_after_norm ucconv\
          --generator_last_norm d --decomposition cholesky --generator_last_after_norm uconv --discriminator_filters 128 --generator_filters 128\
          --discriminator_spectral 0 --gradient_penalty_weight 10 --lr_decay_schedule linear --number_of_epochs 100 --gan_type AC_GAN  --arc res --training_ratio 5 --generator_batch_multiple 2'.format(name))
name = base_name + '_pca'
print(name)
os.system('python run.py --name {} --dataset cifar10 --generator_adversarial_objective wgan\
          --discriminator_adversarial_objective wgan  --generator_block_norm d --generator_block_after_norm ucconv\
          --generator_last_norm d --decomposition pca --generator_last_after_norm uconv --discriminator_filters 128 --generator_filters 128\
          --discriminator_spectral 0 --gradient_penalty_weight 10 --lr_decay_schedule linear --number_of_epochs 100 --gan_type AC_GAN  --arc res --training_ratio 5 --generator_batch_multiple 2'.format(name))
name = base_name + '_zca'
print(name)
os.system('python run.py --name {} --dataset cifar10 --generator_adversarial_objective wgan\
          --discriminator_adversarial_objective wgan  --generator_block_norm d --generator_block_after_norm ucconv\
          --generator_last_norm d --decomposition zca --generator_last_after_norm uconv --discriminator_filters 128 --generator_filters 128\
          --discriminator_spectral 0 --gradient_penalty_weight 10 --lr_decay_schedule linear --number_of_epochs 100 --gan_type AC_GAN  --arc res --training_ratio 5 --generator_batch_multiple 2'.format(name))

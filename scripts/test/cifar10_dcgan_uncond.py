import os


base_name = os.path.basename(__file__).split('.')[0]

os.system('python run.py --name {} --dataset cifar10\
          --generator_adversarial_objective hinge --discriminator_adversarial_objective hinge\
          --generator_block_norm b --generator_last_norm b\
          --discriminator_norm d\
          --d_decomposition zca --d_iter_num 5 --d_whitten_m 16 --d_coloring_m 0 --d_instance_norm 1\
          --g_decomposition zca --g_iter_num 5 --g_whitten_m 16 --g_coloring_m 0 --g_instance_norm 1\
          --discriminator_coloring n\
          --generator_block_coloring uconv --generator_last_coloring uconv\
          --discriminator_filters 256 --generator_filters 256\
          --discriminator_spectral 1\
          --gradient_penalty_weight 0\
          --lr_decay_schedule linear --generator_lr 2e-4 --discriminator_lr 2e-4\
          --beta1 0 --beta2 0.9\
          --arc dcgan\
          --training_ratio 1 --generator_batch_multiple 1\
          --number_of_epochs 50 --batch_size 64'.format(base_name))



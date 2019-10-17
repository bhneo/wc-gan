import os


base_name = os.path.basename(__file__).split('.')[0]

os.system('python run.py --name {} --dataset cifar10\
          --generator_adversarial_objective hinge --discriminator_adversarial_objective hinge\
          --generator_block_norm d --generator_last_norm d\
          --generator_block_coloring uconv --generator_last_coloring uconv\
          --g_decomposition zca_wm --g_iter_num 5 --g_whitten_m 16 --g_coloring_m 0 --g_instance_norm 0\
          --discriminator_norm n\
          --discriminator_coloring n\
          --d_decomposition pca --d_iter_num 5 --d_whitten_m 4 --d_coloring_m 0 --d_instance_norm 0\
          --discriminator_filters 256 --generator_filters 256\
          --discriminator_spectral 1\
          --gradient_penalty_weight 0\
          --lr_decay_schedule linear --generator_lr 2e-4 --discriminator_lr 2e-4\
          --beta1 0 --beta2 0.9\
          --arc dcgan\
          --training_ratio 1 --generator_batch_multiple 1\
          --number_of_epochs 50 --batch_size 64'.format(base_name))

os.system('python run.py --name {} --dataset cifar10\
          --generator_adversarial_objective hinge --discriminator_adversarial_objective hinge\
          --generator_block_norm d --generator_last_norm d\
          --generator_block_coloring uconv --generator_last_coloring uconv\
          --g_decomposition zca --g_iter_num 5 --g_whitten_m 16 --g_coloring_m 0 --g_instance_norm 0\
          --discriminator_norm n\
          --discriminator_coloring n\
          --d_decomposition pca --d_iter_num 5 --d_whitten_m 4 --d_coloring_m 0 --d_instance_norm 0\
          --discriminator_filters 256 --generator_filters 256\
          --discriminator_spectral 1\
          --gradient_penalty_weight 0\
          --lr_decay_schedule linear --generator_lr 2e-4 --discriminator_lr 2e-4\
          --beta1 0 --beta2 0.9\
          --arc dcgan\
          --training_ratio 1 --generator_batch_multiple 1\
          --number_of_epochs 50 --batch_size 64'.format(base_name))

os.system('python run.py --name {} --dataset cifar10\
          --generator_adversarial_objective hinge --discriminator_adversarial_objective hinge\
          --generator_block_norm d --generator_last_norm d\
          --generator_block_coloring uconv --generator_last_coloring uconv\
          --g_decomposition cholesky_wm --g_iter_num 5 --g_whitten_m 16 --g_coloring_m 0 --g_instance_norm 0\
          --discriminator_norm n\
          --discriminator_coloring n\
          --d_decomposition pca --d_iter_num 5 --d_whitten_m 4 --d_coloring_m 0 --d_instance_norm 0\
          --discriminator_filters 256 --generator_filters 256\
          --discriminator_spectral 1\
          --gradient_penalty_weight 0\
          --lr_decay_schedule linear --generator_lr 2e-4 --discriminator_lr 2e-4\
          --beta1 0 --beta2 0.9\
          --arc dcgan\
          --training_ratio 1 --generator_batch_multiple 1\
          --number_of_epochs 50 --batch_size 64'.format(base_name))

os.system('python run.py --name {} --dataset cifar10\
          --generator_adversarial_objective hinge --discriminator_adversarial_objective hinge\
          --generator_block_norm d --generator_last_norm d\
          --generator_block_coloring uconv --generator_last_coloring uconv\
          --g_decomposition cholesky --g_iter_num 5 --g_whitten_m 16 --g_coloring_m 0 --g_instance_norm 0\
          --discriminator_norm n\
          --discriminator_coloring n\
          --d_decomposition pca --d_iter_num 5 --d_whitten_m 4 --d_coloring_m 0 --d_instance_norm 0\
          --discriminator_filters 256 --generator_filters 256\
          --discriminator_spectral 1\
          --gradient_penalty_weight 0\
          --lr_decay_schedule linear --generator_lr 2e-4 --discriminator_lr 2e-4\
          --beta1 0 --beta2 0.9\
          --arc dcgan\
          --training_ratio 1 --generator_batch_multiple 1\
          --number_of_epochs 50 --batch_size 64'.format(base_name))

os.system('python run.py --name {} --dataset cifar10\
          --generator_adversarial_objective hinge --discriminator_adversarial_objective hinge\
          --generator_block_norm d --generator_last_norm d\
          --generator_block_coloring uconv --generator_last_coloring uconv\
          --g_decomposition iter_norm_wm --g_iter_num 5 --g_whitten_m 16 --g_coloring_m 0 --g_instance_norm 0\
          --discriminator_norm n\
          --discriminator_coloring n\
          --d_decomposition pca --d_iter_num 5 --d_whitten_m 4 --d_coloring_m 0 --d_instance_norm 0\
          --discriminator_filters 256 --generator_filters 256\
          --discriminator_spectral 1\
          --gradient_penalty_weight 0\
          --lr_decay_schedule linear --generator_lr 2e-4 --discriminator_lr 2e-4\
          --beta1 0 --beta2 0.9\
          --arc dcgan\
          --training_ratio 1 --generator_batch_multiple 1\
          --number_of_epochs 50 --batch_size 64'.format(base_name))

os.system('python run.py --name {} --dataset cifar10\
          --generator_adversarial_objective hinge --discriminator_adversarial_objective hinge\
          --generator_block_norm d --generator_last_norm d\
          --generator_block_coloring uconv --generator_last_coloring uconv\
          --g_decomposition iter_norm --g_iter_num 5 --g_whitten_m 16 --g_coloring_m 0 --g_instance_norm 0\
          --discriminator_norm n\
          --discriminator_coloring n\
          --d_decomposition pca --d_iter_num 5 --d_whitten_m 4 --d_coloring_m 0 --d_instance_norm 0\
          --discriminator_filters 256 --generator_filters 256\
          --discriminator_spectral 1\
          --gradient_penalty_weight 0\
          --lr_decay_schedule linear --generator_lr 2e-4 --discriminator_lr 2e-4\
          --beta1 0 --beta2 0.9\
          --arc dcgan\
          --training_ratio 1 --generator_batch_multiple 1\
          --number_of_epochs 50 --batch_size 64'.format(base_name))

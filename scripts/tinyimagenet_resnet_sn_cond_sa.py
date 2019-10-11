import os


base_name = os.path.basename(__file__).split('.')[0]
name = base_name + '_baseline'
print(name)
os.system('python run.py --phase test --name {} --dataset tiny-imagenet\
          --generator_adversarial_objective hinge --discriminator_adversarial_objective hinge\
          --discriminator_norm n\
          --discriminator_coloring n\
          --d_decomposition zca --d_iter_num 5 --d_whitten_m 4 --d_coloring_m 0 --d_instance_norm 0\
          --generator_block_norm d --generator_block_coloring ufconv\
          --generator_last_norm d --generator_last_coloring uconv\
          --g_decomposition cholesky --g_iter_num 5 --g_whitten_m 16 --g_coloring_m 0 --g_instance_norm 0\
          --discriminator_filters 1024 --generator_filters 128\
          --discriminator_spectral 1\
          --gradient_penalty_weight 0\
          --lr_decay_schedule linear --generator_lr 2e-4 --discriminator_lr 2e-4\
          --beta1 0 --beta2 0.9\
          --arc res\
          --training_ratio 5 --generator_batch_multiple 2\
          --number_of_epochs 100 --batch_size 64\
          --gan_type PROJECTIVE --filters_emb 15'.format(name))

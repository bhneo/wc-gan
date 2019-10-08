#!/bin/bash
Gmethods=(zca)
Dmethods=(zca)
Gm=(16)
Dm=(16)
seeds=(1)
Count=0
dataset="cifar10"
arc="dcgan"
generator_adversarial_objective="hinge"
discriminator_adversarial_objective="hinge"
discriminator_filters=512
generator_filters=512
discriminator_spectral=1
generator_block_norm="d"
generator_block_coloring="uconv"
generator_last_norm="d"
generator_last_coloring="uconv"
g_iter_num=5
g_instance_norm=1
discriminator_norm="d"
discriminator_coloring="uconv"
d_iter_num=5
d_instance_norm=1
gradient_penalty_weight=0
lr_decay_schedule="linear"
generator_lr=2e-4
discriminator_lr=2e-4
beta1=0
beta2=0.9
number_of_epochs=100
batch_size=64
training_ratio=1
generator_batch_multiple=1

l=${#Gmethods[@]}
n=${#Dmethods[@]}
m=${#Gm[@]}
t=${#Dm[@]}
f=${#seeds[@]}

for ((a=0;a<$l;++a))
do 
   for ((i=0;i<$n;++i))
   do 
      for ((j=0;j<$m;++j))
      do	
        for ((k=0;k<$t;++k))
        do
          for ((b=0;b<$f;++b))
          do
        	#echo "Gm=${Gm[$j]}"
        	#echo "Dm=${Dm[$k]}"
                baseString="execute_${dataset}_${arc}_G${Gmethods[$a]}_Gins${g_instance_norm}_GW${Gm[$j]}_${discriminator_norm}_${discriminator_coloring}_D${Dmethods[$i]}_Dins${d_instance_norm}_DW${Dm[$k]}_s${seeds[$b]}_C${Count}"
                fileName="${baseString}.sh"
   	            echo "${baseString}"
                touch "${fileName}"
                echo  "#!/usr/bin/env bash
cd \"\$(dirname \$0)/..\" 
name=\$(basename \$0)
base_name=\${name%.*}
name=\"\${base_name}\"
CUDA_VISIBLE_DEVICES=${Count} python run.py --name \$name --dataset ${dataset} --arc ${arc}\\
 --generator_adversarial_objective ${generator_adversarial_objective} --discriminator_adversarial_objective ${discriminator_adversarial_objective}\\
 --discriminator_filters ${discriminator_filters} --generator_filters ${generator_filters}\\
 --discriminator_spectral ${discriminator_spectral}\\
 --generator_block_norm ${generator_block_norm} --generator_block_coloring ${generator_block_coloring} --generator_last_norm ${generator_last_norm} --generator_last_coloring ${generator_last_coloring}\\
 --g_decomposition ${Gmethods[$a]} --g_iter_num ${g_iter_num} --g_whitten_m ${Gm[$j]} --g_coloring_m ${Gm[$j]} --g_instance_norm ${g_instance_norm}\\
 --discriminator_norm ${discriminator_norm} --discriminator_coloring ${discriminator_coloring}\\
 --d_decomposition ${Dmethods[$i]} --d_iter_num ${d_iter_num} --d_whitten_m ${Dm[$k]} --d_coloring_m ${Dm[$k]} --d_instance_norm ${d_instance_norm}\\
 --gradient_penalty_weight ${gradient_penalty_weight}\\
 --lr_decay_schedule ${lr_decay_schedule}\\
 --number_of_epochs ${number_of_epochs}\\
 --training_ratio ${training_ratio} --generator_batch_multiple ${generator_batch_multiple} \\" >> ${fileName}
             
                echo  "nohup bash ${fileName} >output_${baseString}.out 2>&1 &" >> z_bash_excute.sh
               let Count=Count+1
               echo $Count
                if [ $Count -gt 7 ]
                then 
                  Count=0
                fi
           done
         done
      done
   done
done

#!/bin/bash
#SBATCH -JSampleCM
#SBATCH -N1 -n1
#SBATCH --mem-per-gpu 20GB
#SBATCH -G A100:1
#SBATCH -t 1:00:00
#SBATCH -oReport-%j.out


module load python/3.9.12-h6yxcg
module load anaconda3/2022.05.0.1
module load gcc/10.3.0-o57x6h
module load mvapich2/2.3.6
module load cuda/11.7.0-7sdye3

echo "Launching Sample"

srun ~/.conda/envs/torch/bin/python image_sample.py --batch_size 256 --training_mode consistency_distillation --sampler multistep --ts 0,22,39 --steps 40 --model_path target_model005000.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform
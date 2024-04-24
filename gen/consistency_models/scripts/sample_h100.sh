#!/bin/bash
#SBATCH -JSampleCM
#SBATCH -N1 -n1
#SBATCH --mem-per-gpu 80GB
#SBATCH -G H100:1
#SBATCH -t 02:00:00
#SBATCH -oReport-%j.out

module load python/3.10.10
module load anaconda3
module load gcc/12.3.0
module load openmpi/4.1.5
module load cuda/12.1.1

echo "Launching Sample"
echo "170"

# 512x512 edm model
srun ~/.conda/envs/h100_torch/bin/python image_sample.py --training_mode edm --batch_size 4 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path tristan_ema_0.9999432189950708_140000.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 512 --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --num_samples 40 --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras



# 256x256 edm model
# srun ~/.conda/envs/h100_torch/bin/python image_sample.py --training_mode edm --batch_size 16 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path tristan_ema_0.9999432189950708_319000.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 1087 --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras

# 64x64 cm model
# srun ~/.conda/envs/h100_torch/bin/python image_sample.py --batch_size 256 --training_mode consistency_distillation --sampler multistep --ts 0,22,39 --steps 40 --model_path model014500.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform
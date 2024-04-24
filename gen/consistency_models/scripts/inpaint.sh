#!/bin/bash
#SBATCH -JSampleCM
#SBATCH -N1 -n1
#SBATCH --mem-per-gpu 8GB
#SBATCH -G H100:1
#SBATCH -t 00:15:00
#SBATCH -oReport-%j.out

module load python/3.10.10
module load anaconda3
module load gcc/12.3.0
module load openmpi/4.1.5
module load cuda/12.1.1

echo "Launching Inpaint"

# inpainter call for the 256x256 pixel images
# srun ~/.conda/envs/h100_torch/bin/python inpaint.py --training_mode edm --batch_size 16 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --model_path tristan_model319000.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 1 --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras --sampler multistep --ts 0,22,39

# now for the 512
# srun ~/.conda/envs/h100_torch/bin/python inpaint.py --training_mode edm --batch_size 4 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --model_path tristan_model010000.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 512 --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --num_samples 1 --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras --sampler multistep --ts 0,22,39
#!/bin/bash
#SBATCH -JTrainCM
#SBATCH -N1 -n1
#SBATCH --mem-per-gpu 80GB
#SBATCH -G H100:1
#SBATCH -t 16:00:00
#SBATCH -oReport-%j.out

module load python/3.10.10
module load anaconda3
module load gcc/12.3.0
module load openmpi/4.1.5
module load cuda/12.1.1

echo "Launching Train"

cd ~/scratch/seg-replace/gen/consistency_models/scripts

# 515x515 ema
# srun ~/.conda/envs/h100_torch/bin/python edm_train.py --attention_resolutions 32,16,8 --class_cond False --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 4 --image_size 512 --lr 0.0001 --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 True --use_scale_shift_norm False --weight_decay 0.0 --weight_schedule karras --data_dir ../../../data/Messi --save_interval 10000

# 256x256 ema run
srun ~/.conda/envs/h100_torch/bin/python edm_train.py --attention_resolutions 32,16,8 --class_cond False --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 16 --image_size 256 --lr 0.0001 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 True --use_scale_shift_norm False --weight_decay 0.0 --weight_schedule karras --data_dir ../../../data/Messi --resume_checkpoint tristan_model324000.pt

# srun ~/.conda/envs/h100_torch/bin/python cm_train.py --training_mode consistency_training --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 200 --total_training_steps 50000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path ../../ckpts/cd_imagenet64_l2.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 64 --lr 0.0001 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir ../../../data/Messi --resume_checkpoint model011000.pt

# 256 consitency distill
# srun ~/.conda/envs/h100_torch/bin/python cm_train.py --training_mode consistency_training --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 200 --total_training_steps 100000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path ../../ckpts/checkpoint_26.pth --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 256 --lr 0.0001 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir ../../../data/Messi
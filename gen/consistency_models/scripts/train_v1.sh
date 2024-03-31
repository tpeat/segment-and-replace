#!/bin/bash
#SBATCH -JTrainCM
#SBATCH -N2 -n4
#SBATCH --mem-per-gpu 20GB
#SBATCH -G A100:4
#SBATCH -t 2:00:00
#SBATCH -oReport-%j.out


module load python/3.9.12-h6yxcg
module load anaconda3/2022.05.0.1
module load gcc/10.3.0-o57x6h
module load mvapich2/2.3.6
module load cuda/11.7.0-7sdye3

echo "Launching Train"

cd ~/scratch/seg-replace/gen/consistency_models/scripts

srun ~/.conda/envs/torch/bin/python cm_train.py --training_mode consistency_training --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 200 --total_training_steps 100000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path ../../ckpts/cd_imagenet64_l2.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 64 --image_size 64 --lr 0.001 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir ../../../assets/Messi_Filtered
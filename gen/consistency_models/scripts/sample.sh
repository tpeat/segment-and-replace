srun ~/.conda/envs/torch/bin/python image_sample.py --batch_size 256 --training_mode consistency_distillation --sampler multistep --ts 0,22,39 --steps 40 --model_path state_archive/model000050.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform
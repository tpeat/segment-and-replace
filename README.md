# Segment and Replace

Objective: Design novel method of segmenting humans from images and replacing them with targets of choice

What's important?
- Hiera is a hiearchical transformer based encoder
- Encoder is loaded with pretrained weights
- Immediately attach FPN to reproduce (1,256,256) binary mask
- Model is large > 50M params so need decent amount of gpu RAM

Requirements: saved in yaml
* torch with gpu compat
* timm

Future Upgrades:
* V1: replace with static person
* V2: text to image
* V3: image to image

Inspiriation:
- Segmant Anything (SAM)
- Inpaint Anything

## Env Setup

1. On slurm cluster allocate A100:
```sh
salloc -N1 -n1 --mem-per-gpu 20GB -t 8:00:00 -G A100:1
```

2. Load modules:
```sh
module load python/3.9.12-h6yxcg
module load anaconda3/2022.05.0.1
module load gcc/10.3.0-o57x6h
module load mvapich2/2.3.6
module load cuda/11.7.0-7sdye3
```
NOTE: we require gcc10.3 and mvapich2 for MPI

3. Create conda env:
```sh
conda create -n torch python=3.8
```

4. Get compatable torch version (with cuda11.7):
```sh
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

5. Install consistency models package:
```sh
cd <path to consitency models>
pip install -e .
```

6. Ensure installation works:
```sh
python # or ~/.conda/envs/torch/bin/python
>>> import cm
>>> import torch
>>> torch.rand(1).cuda()
```

# Log

## 03/30/2024

<details><summary>Tristan - setup Consistency Model library</summary>


Using openais consitency model library

Why? Don't want to worry about removing latent text vector entanglement from stable diffusion models. 

Having text embedding is a crutial component of latent diffuion models, so even if I implemented the Unet structure and pretrained-weights, I'm not confident the model would generate what we want.

Need mpi4py arch to make library work because the training is designed to be distributed. mpi4py requires mpi compiler which isn't compat with gcc12, so I had to revert to 10.3 if I want it to work which renders all previously installed packages useless.

Then lots of pace-quota exceeded issues when holding two envs at once.

</details>

## 03/31/2024

<details><summary>Tristan - sampling and training</summary>


Training Details:
- Only 162 images of Messi so far
- 2 nodes, four threads, 4 A100's
- Documentation on using MPI jobs on pace-ice found [here](https://gatech.service-now.com/technology?id=kb_article_view&sysparm_article=KB0042096#mpi-jobs)
- 100k steps, saved every 5k
- bs = 64, lr = 0.001

</details>

## 04/01/2024

<details><summary>Tristan - more images training</summary>


Training Details:
- Adarsh and Bijan got an additional 1k images from internet
- data naming scheme is: 0_<data_creator><sample_number>.jpg
- 1 nodes, 1 threads, 1 A100's
- changed save step to 500 and resumed from checkpoint 5k
- decreased learning rate to 0.0001
- still only ran for 500 iteraitons before nan losses found

Many many out of memory issues:
- increased GPU mem allocation to 80gb
- torch.cuda.empty_cache() after every checkpoint loaded
- max_split_size = 256 (not sure what this means)
- created a eddiscussion post for some aid on how to manage large models
- model takes like 40G

TODO:
- Going to relaunch training from scratch with lower learning rate to start
- Investigate how to normalize these images
    - find mean and std dev of dataset (going to be different from imagenet)
- Fix multiGPU issue

</details>

## 04/02/2024

<details><summary>Tristan - restarting trianing</summary>


Training Details:



</details>

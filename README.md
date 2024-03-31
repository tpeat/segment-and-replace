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

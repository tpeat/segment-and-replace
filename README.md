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

# Directory

inpainting.ipynb 
 - A Test Jupyter Notebook that uses an out-of-the-box inpainting model to give us a reference for how our model should hopefully look

#### /assets/
 - A folder to contain all object files like images, environment ymls, etc. necessary to run the model

/assets/Messi_Filtered
 - A folder containing all filtered (pre-processed and edited) images of Lionel Messi

/assets/Messi_Unfiltered
 - A folder containing all un-filtered images of Lionel Messi that need to be filtered

/assets/env/tristan_env.yml 
 - An Anaconda environment yml that is used to ensure all packages are keep consistent

/assets/cap.png
 - A generic, copyright free png of a hat to help increase our dataset 

/assets/sunglasses.png
 - A generic, copyright free png of sunglasses to help increase our dataset

#### /data/
 - A folder containing all relevant Jupyter Notebook's for data pre-processing

/data/DataAugment.ipynb
 - A Jupyter notebook for adding hats / sunglasses to pictures to increase the size of the dataset

/data/ImagePreprocessing.ipynb
 - A Jupyter notebook that runs facial recognition on our unfiltered image dataset in order to remove images without Messi's actual face in them.

/data/train_analysis.ipynb
 - A Jupyter notebook that is helping pre-process our image dataset for how the model expects the image filesnames

#### /gen/
 - A folder containing all relevant code for generating and visualizing images for our unsupervised learning method

##### /gen/consistency_models
 - A base model taken from [here](https://arxiv.org/abs/2303.01469) that we are focusing on Messi

/gen/main.ipynb 
 - A Jupyter notebook that was for testing training the consistency_models on a pre-made dataset to learn more about consistency_models

/gen/viz_samples.ipynb
 - A Jupyter notebook to visualize samples generated as .npz

#### /seg/
 - A folder containing all relevant code for segmenting an image (our supervised learning method)

/seg/hiera.py, /seg/hiera_mae.py, /seg/hiera_utils.py
 - Python files taken from Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles by Meta (temporarily for testing)

/seg/decoder.py
 - An implementation of a Feature Pyramid Network decoder

/seg/playground.ipynb
 - A Jupyter notebook for testing how a full model (Hiera) works and provides a baseline for our future implementation


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

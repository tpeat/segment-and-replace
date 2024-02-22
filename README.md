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
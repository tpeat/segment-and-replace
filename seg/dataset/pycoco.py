from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import requests
from io import BytesIO

dataDir='.'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

max_people = 3  # Define "crowded" as having more than 'max_people' annotations

# Initialize COCO
coco = COCO(annFile)

# Get category IDs for 'person'
catIds = coco.getCatIds(catNms=['person'])

# Get all images containing persons
imgIds = coco.getImgIds(catIds=catIds)

# Load images and filter out crowded scenes
for imgId in imgIds:
    img = coco.loadImgs(imgId)[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    annotations = coco.loadAnns(annIds)
    
    # Skip crowded images
    if len(annotations) > max_people:
        continue

    # Download the image
    response = requests.get(img['coco_url'])
    image = Image.open(BytesIO(response.content))
    image.save(f'{dataDir}/{dataType}/{img["file_name"]}')  # Save the actual image

    # Optional: Create and save mask of the people in the image
    if annotations:
        mask = np.zeros((img['height'], img['width']))
        for ann in annotations:
            mask += coco.annToMask(ann)
        mask = (mask * 255).astype(np.uint8)
        im = Image.fromarray(mask)
        im.save(f'{dataDir}/{dataType}/mask_{img["file_name"]}') 
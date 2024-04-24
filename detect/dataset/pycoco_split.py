from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import requests
from io import BytesIO

dataDir='.'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# Initialize COCO
coco = COCO(annFile)
catIds = coco.getCatIds()

# Get image IDs with people
all_catIds = coco.getCatIds()
person_cat = coco.getCatIds(catNms=['person'])
imgIds_people = coco.getImgIds(catIds=person_cat)

#Get image IDs without people
catIds = coco.getCatIds(catNms=['person'])
person_cat_id = catIds[0] if catIds else None
imgIds_no_people = []
for img_id in coco.imgs.keys():
    annIds = coco.getAnnIds(imgIds=img_id, catIds=person_cat_id, iscrowd=None)
    if not annIds:
        imgIds_no_people.append(img_id)


# #Load images and filter out crowded scenes
for imgId in imgIds_people:
    img = coco.loadImgs(imgId)[0]

    # Download the image
    response = requests.get(img['coco_url'])
    image = Image.open(BytesIO(response.content))
    image.save(f'{dataDir}/{dataType}/humans/{img["file_name"]}')  # Save the actual image

for imgId in imgIds_no_people:
    img = coco.loadImgs(imgId)[0]

    # Download the image
    response = requests.get(img['coco_url'])
    image = Image.open(BytesIO(response.content))
    image.save(f'{dataDir}/{dataType}/nonhumans/{img["file_name"]}')  # Save the actual image
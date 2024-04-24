import os
from PIL import Image
from PIL import UnidentifiedImageError
import blobfile as bf
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import numpy as np
from tqdm import tqdm

# path to messi photos
directory = '/home/hice1/tpeat3/scratch/seg-replace/data/Messi'

print(os.path.exists(directory))

def rename(directory):
    # Loop through all files in the directory
    for i, filename in enumerate(os.listdir(directory)):
        # Check if the file is a JPG file
        if filename.endswith('.jpg'):
            # Construct the new filename by prepending '0_' to the original filename
            new_filename = '0_' + 'adarsh' + str(i) + '.jpg'
            # Join the directory path and the new filename
            new_filepath = os.path.join(directory, new_filename)
            # Join the directory path and the original filename
            old_filepath = os.path.join(directory, filename)
            # Rename the file
            os.rename(old_filepath, new_filepath)
            print(f'Renamed "{filename}" to "{new_filename}"')

    print('Done renaming files.')

# rename(directory)

def remove_corrupt(directory):

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a JPG image
        if filename.endswith(".jpg"):
            try:
                # Try to open the image using PIL
                filepath = os.path.join(directory, filename)
                with bf.BlobFile(filepath, "rb") as f:
                    pil_image = Image.open(f)
                    pil_image.load()
                    pass
            except OSError:
                # If the image is corrupted, remove it
                print(f"Removing corrupted image: {filename}")
                os.remove(os.path.join(directory, filename))

# remove_corrupt(directory)

def find_transparent_pic(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add other file formats if necessary
            file_path = os.path.join(directory, filename)
            try:
                with Image.open(file_path) as img:
                    # Check if image has transparency
                    if img.mode == 'RGBA' or 'transparency' in img.info:
                        print(f"Image with transparency found: {filename}")
                        os.remove(file_path)

            except Exception as e:
                print("Error")
                        
# find_transparent_pic(directory)


class MessiDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = os.listdir(root)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = os.path.join(self.root, self.data[idx])

        img = Image.open(img_path)
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img) / 255.

        return img

def create_loader():
    root = 'Messi'
    transform = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
    dataset = MessiDataset(root, transform)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader

def get_mean_std(loader):
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(loader):
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs**2).sum(axis=[0, 2, 3])

    count = len(loader) * 64 * 64

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)
    return total_mean, total_std

# print(get_mean_std(create_loader()))


def unzip_npz(npz_path, output_path):
    data = np.load(npz_path)
    data = data['arr_0']

    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate through each item in the .npz file
    for idx, image_array in enumerate(data):
        # Assuming image_array is a numpy array representing an image
        # Convert the numpy array to an image
        image = Image.fromarray(image_array)
        
        # Construct the output file path
        output_file_path = os.path.join(output_path, f'image_{idx}.png')
        
        # Save the image
        image.save(output_file_path)
        print(f'Saved image {idx} as {output_file_path}')

npz_path = '/home/hice1/tpeat3/scratch/seg-replace/gen/consistency_models/scripts/319_samples_1087x256x256x3.npz'
output_path = 'assets/319_Messi_eval_256'
unzip_npz(npz_path, output_path)

def resize_images(directory, output_directory, size=(512,512)):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(directory, filename)
            output_file_path = os.path.join(output_directory, filename)
            try:
                with Image.open(file_path) as img:
                    resized_img = img.resize(size)
                    resized_img.save(output_file_path)
                    print(f'Resized and saved {filename} to {output_file_path}')
            except OSError as e:
                print(e)

# resize_images('Messi', 'Messi_512x512')

def center_crop_images(directory, output_directory, size=(256, 256)):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(directory, filename)
            output_file_path = os.path.join(output_directory, filename)
            try:
                with Image.open(file_path) as img:
                    # Calculate the area to crop
                    width, height = img.size
                    left = (width - size[0]) / 2
                    top = (height - size[1]) / 2
                    right = (width + size[0]) / 2
                    bottom = (height + size[1]) / 2

                    # Crop the center of the image
                    cropped_img = img.crop((left, top, right, bottom))
                    cropped_img.save(output_file_path)
                    print(f'Cropped and saved {filename} to {output_file_path}')
            except OSError as e:
                print(e)

# center_crop_images('Messi', 'Messi_256x256')


def duplicate_images(directory, target_count=2048):
    # Get a list of image files in the directory
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(files)

    if current_count >= target_count:
        print("No need to duplicate, the directory already has enough images.")
        return

    idx = 0  # To ensure unique filename for duplicates
    while current_count < target_count:
        for file in files:
            if current_count >= target_count:
                break  # Stop if we have reached the target count

            source_path = os.path.join(directory, file)
            file_name, file_extension = os.path.splitext(file)
            target_path = os.path.join(directory, f"{file_name}_copy{idx}{file_extension}")

            shutil.copy(source_path, target_path)
            print(f"Copied {source_path} to {target_path}")

            current_count += 1
            idx += 1

duplicate_images('assets/319_Messi_eval_256')
print(len(os.listdir('assets/319_Messi_eval_256')))

# duplicate_images('Messi_512x512')
# print(len(os.listdir('Messi_512x512')))
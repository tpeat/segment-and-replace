import os
from PIL import Image
from PIL import UnidentifiedImageError
import blobfile as bf

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

remove_corrupt(directory)
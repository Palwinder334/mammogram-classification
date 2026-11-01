# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:51:32 2024

@author: palwi
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Path to the folder containing original images
original_images_folder = r"path to the folder containing original images"
# original_images_folder = r"D:\PhD\PYTHON programs\all AIIMS mammo 178\benign"

# Directory to save augmented images
save_directory = r"path to save the augumented images"
# save_directory = r"D:\PhD\PYTHON programs\all AIIMS mammo 178\benign_aug"

# Create save_directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Define the augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Iterate over the images in the original images folder
for filename in os.listdir(original_images_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the original image
        original_image_path = os.path.join(original_images_folder, filename)
        original_image = Image.open(original_image_path)

        # Convert the image to grayscale
        original_image = original_image.convert('L')

        # Resize the image to a common size (e.g., 256x256)
        original_image = original_image.resize((256, 256))

        # Convert the image to a numpy array
        image_array = np.array(original_image)

        # Reshape the image array to match Keras' input shape
        image_array = image_array.reshape((1,) + image_array.shape + (1,))

        # Generate augmented images
        augmented_images = datagen.flow(
            image_array,
            batch_size=1,
            save_to_dir=save_directory,
            save_prefix=filename[:-4],  # Use the original image name as the prefix
            save_format='png'
        )

        # Generate and save augmented images
        counter = 1
        for augmented_image in augmented_images:
            augmented_filename = '{}({:02d}).png'.format(filename[:-4], counter)  # Append a number with leading zeros to the original image name
            augmented_image = augmented_image.reshape(image_array.shape[1:-1])  # Reshape the augmented image
            augmented_image = augmented_image.astype(np.uint8)  # Convert to uint8 data type
            augmented_image = Image.fromarray(augmented_image, mode='L')  # Convert back to PIL Image
            augmented_image.save(os.path.join(save_directory, augmented_filename))
            counter += 1

            if counter > 2:
                break

print("Augmentation complete!")
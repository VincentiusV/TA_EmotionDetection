import os
import random
import shutil

def copy_random_images(source_folder, destination_folder, num_images_to_copy):
    # List all files in source folder
    files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Shuffle the list
    random.shuffle(files)
    
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Copy random images to the destination folder
    for i in range(min(num_images_to_copy, len(files))):
        shutil.copy(files[i], destination_folder)
        print(f"Copied {files[i]} to {destination_folder}")

# Paths to the folders
source_folder = '../dataset/augmented_masked/surprise'
destination_folder = '../dataset/augmented/surprise'
num_images_to_copy = 1700  # Number of images you want to copy

copy_random_images(source_folder, destination_folder, num_images_to_copy)

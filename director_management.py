# Creating a Validation directory with 20% of the Training Data

import os
import shutil
import random

data_dir = '/Users/apple/Desktop/Brain_Tumor/'

train_dir = data_dir + 'Training/'

test_dir = data_dir + 'Testing/'

src_dir = train_dir

dest_dir = data_dir + 'Validation/'

validation_percent = 20

# Create the validation directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Loop through each class directory
for class_name in os.listdir(src_dir):
    # Exclude the .DS_Store file from the list of directories
    if class_name == '.DS_Store':
        continue

    class_dir = os.path.join(src_dir, class_name)

    # Create the validation class directory if it doesn't exist
    validation_class_dir = os.path.join(dest_dir, class_name)
    if not os.path.exists(validation_class_dir):
        os.makedirs(validation_class_dir)

    # Get a list of all image files in the class directory
    image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

    # Calculate the number of images to move to validation set
    num_validation = int(len(image_files) * validation_percent / 100)

    # Select a random subset of images to move to validation set
    validation_images = random.sample(image_files, num_validation)

    # Move the validation images to the validation directory
    for image in validation_images:
        src_path = os.path.join(class_dir, image)
        dest_path = os.path.join(validation_class_dir, image)
        shutil.move(src_path, dest_path)
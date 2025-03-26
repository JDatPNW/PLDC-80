import os
import shutil
import random

# Set the source and destination directories
dirname = "data_merged"
source_dir = f'./{dirname}/'
train_dir = f'./{dirname}_split/train'
test_dir = f'./{dirname}_split/test'

# Create destination directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Minimum images required per set for each class (e.g., 1 image for both train and test sets)
min_images_per_set = 1

# Loop through each class folder in the source directory
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    
    if os.path.isdir(class_path):  # Check if it's a directory
        # Create class directories in train and test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get a list of all images in the class directory
        images = os.listdir(class_path)
        
        # Ensure there are enough images to split between train and test
        if len(images) < 2 * min_images_per_set:
            raise ValueError(f"Not enough images in class {class_name} to split between train and test.")
        
        random.shuffle(images)  # Shuffle the list of images
        
        # Set aside at least one image for the test set, and the rest can follow the 80/20 split
        split_index = max(int(len(images) * 0.8), min_images_per_set)  # Ensure at least one image for both sets
        
        # Split images into train and test sets
        train_images = images[:split_index]
        test_images = images[split_index:]
        
        # If there's not enough test images, make sure at least one goes into the test set
        if len(test_images) < min_images_per_set:
            # Move some images from train to test
            test_images.extend(train_images[-min_images_per_set:])
            train_images = train_images[:-min_images_per_set]
        
        # Move images to the respective directories
        for img in train_images:
            shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        
        for img in test_images:
            shutil.move(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

print("Dataset has been successfully split into training and testing sets.")

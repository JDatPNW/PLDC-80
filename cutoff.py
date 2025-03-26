import os
import shutil
import random

# Define paths
src_root = "./data_augmented/train"
dst_root = "./PLDC80/train"
num_images = 3500  # Number of images to copy per class

# Ensure the destination root exists
os.makedirs(dst_root, exist_ok=True)

# Iterate through each class
for class_name in os.listdir(src_root):
    src_class_path = os.path.join(src_root, class_name)
    dst_class_path = os.path.join(dst_root, class_name)

    # Check if it's a directory
    if not os.path.isdir(src_class_path):
        continue

    # List all image files
    images = [f for f in os.listdir(src_class_path) if os.path.isfile(os.path.join(src_class_path, f))]

    # Ensure there are enough images
    if len(images) < num_images:
        print(f"Skipping {class_name}, not enough images ({len(images)})")
        continue

    # Randomly select 3600 images
    selected_images = random.sample(images, num_images)

    # Create destination class directory
    os.makedirs(dst_class_path, exist_ok=True)

    # Copy selected images
    for image in selected_images:
        shutil.copy(os.path.join(src_class_path, image), os.path.join(dst_class_path, image))

    print(f"Copied {num_images} images for class {class_name}")

print("Done!")


# Define paths
source_folder = "./data_augmented/test"
destination_folder = "./PLDC80/test"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Loop through items in the source folder
for item in os.listdir(source_folder):
    source_path = os.path.join(source_folder, item)
    dest_path = os.path.join(destination_folder, item)
    # Move the item
    shutil.move(source_path, dest_path)

print("Moving test data complete")

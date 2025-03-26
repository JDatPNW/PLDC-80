import os
import shutil

# Define paths
source_folder = "./pre_clean_data/cds"  # Adjust to your actual path
destination_folder = "./data/cds"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Iterate through both train and test directories
for split in ["train", "test"]:
    split_path = os.path.join(source_folder, split)
    
    if not os.path.exists(split_path):
        continue

    for class_folder in os.listdir(split_path):
        class_path = os.path.join(split_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        # Ensure class folder exists in the destination
        dest_class_path = os.path.join(destination_folder, class_folder)
        os.makedirs(dest_class_path, exist_ok=True)

        # Move all images
        for file_name in os.listdir(class_path):
            src_file = os.path.join(class_path, file_name)
            dest_file = os.path.join(dest_class_path, file_name)
            if os.path.isfile(src_file):  # Ensure it's a file before moving
                shutil.move(src_file, dest_file)

print("All images moved successfully!")

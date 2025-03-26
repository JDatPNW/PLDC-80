import os
import shutil

# Define paths
source_folder = "./pre_clean_data/diaMOS"
destination_folder = "./data/diaMOS"
excluded_folder = "healthy"
excluded_folder_2 = "curl"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Loop through items in the source folder
for item in os.listdir(source_folder):
    source_path = os.path.join(source_folder, item)
    dest_path = os.path.join(destination_folder, item)

    # Skip the excluded folder
    if item == excluded_folder:
        continue
    elif item == excluded_folder_2:
        continue

    # Move the item
    shutil.move(source_path, dest_path)

print("Move complete, excluding", excluded_folder)

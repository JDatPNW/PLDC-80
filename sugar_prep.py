import os
import shutil

# Define paths
source_folder_diseases = "./pre_clean_data/sugar/Diseases/Diseases"
source_folder_healthy = "./pre_clean_data/sugar/Healthy Leaves"
destination_folder = "./data/sugar"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Loop through items in the source folder
for item in os.listdir(source_folder_diseases):
    source_path = os.path.join(source_folder_diseases, item)
    dest_path = os.path.join(destination_folder, item)

    # Move the item
    shutil.move(source_path, dest_path)

# Loop through items in the source folder
for item in os.listdir(source_folder_healthy):
    source_path = os.path.join(source_folder_healthy, item)
    dest_path = os.path.join(destination_folder, item)

    # Move the item
    shutil.move(source_path, dest_path)

print("Move complete")

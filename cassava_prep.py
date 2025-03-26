import os
import shutil
import pandas as pd

# Load the CSV file
csv_file_path = './pre_clean_data/cassava/train.csv'  # Replace with the path to your CSV

# Define the image directory and the new destination directories
image_directory = './pre_clean_data/cassava/train_images/'  # Replace with the path to your image folder
output_directory = './data/cassava/'  # Replace with where you want the folders

# Class label mapping
class_mapping = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}

# Load the CSV file
df = pd.read_csv(csv_file_path)

# Create directories for each class if they don't exist
for class_name in class_mapping.values():
    class_dir = os.path.join(output_directory, class_name)
    os.makedirs(class_dir, exist_ok=True)

# Move images to respective class directories
for _, row in df.iterrows():
    image_name = row['image_id']
    label = row['label']
    class_name = class_mapping[label]
    
    src_path = os.path.join(image_directory, image_name)
    dst_path = os.path.join(output_directory, class_name, image_name)
    
    # Move the image to the corresponding class folder
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
    else:
        print(f"Image {image_name} not found!")

print("Images have been organized into class folders.")

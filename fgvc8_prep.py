import os
import shutil
import pandas as pd

# Define paths for the images and the CSV file
image_folder = './pre_clean_data/fgvc8/train_images'  # Replace with your images directory
csv_file_path = './pre_clean_data/fgvc8/train.csv'  # Replace with your CSV file path
destination_folder = './data/fgvc8/'  # Destination directory for organized dataset

# Create the destination directory if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Load the CSV file
df = pd.read_csv(csv_file_path)

to_delete = ['scab frog_eye_leaf_spot complex', 'complex', 'scab frog_eye_leaf_spot', 
             'frog_eye_leaf_spot complex', 'rust frog_eye_leaf_spot',  'powdery_mildew complex', 'rust complex']

# Loop through each unique label in the DataFrame
for label in df['labels'].unique():
    if any(disease in label for disease in to_delete):
        continue
    label_folder = os.path.join(destination_folder, label)
    os.makedirs(label_folder, exist_ok=True)  # Create a folder for each label
    
    # Get all images corresponding to the current label
    images_for_label = df[df['labels'] == label]['image']
    
    # Copy each image to the corresponding label folder
    for image_name in images_for_label:
        source_image_path = os.path.join(image_folder, image_name)
        if os.path.exists(source_image_path):  # Check if the image exists
            shutil.move(source_image_path, label_folder)  # Copy image to label folder

print("Images have been organized into folders based on labels.")

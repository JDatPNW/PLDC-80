import os
import shutil

# Define the paths to your train and test folders
train_dir = './pre_clean_data/sms/train'
test_dir = './pre_clean_data/sms/test'
val_dir = './pre_clean_data/sms/val'

output_dir = './data/sms/'

def organize_images_by_disease(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    for file in files:
        # Process only .jpg files
        if file.endswith(".jpg"):
            if any(disease in file for disease in ["anthracnose_fruit_rot", "blossom_blight", "gray_mold", "powdery_mildew_fruit"]):
                continue
            # Extract the disease name (part before the digit and file extension)
            disease_name = ''.join([i for i in file if not i.isdigit()]).replace('.jpg', '').rstrip('_')
            
            # Create a folder for the disease if it doesn't exist
            disease_folder = os.path.join(output_dir, disease_name)
            if not os.path.exists(disease_folder):
                os.makedirs(disease_folder)
            
            # Move the image to the appropriate folder
            src = os.path.join(folder_path, file)
            dst = os.path.join(disease_folder, file)
            shutil.move(src, dst)

# Organize both the train and test directories
organize_images_by_disease(train_dir)
organize_images_by_disease(test_dir)
organize_images_by_disease(val_dir)

print("Images have been organized into folders by disease.")

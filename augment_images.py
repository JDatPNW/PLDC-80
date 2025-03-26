import os
import cv2
import numpy as np
import shutil

# Define input and output directories
input_dir = f"./data_merged_split/"  # Main input directory containing all folders
output_dir = f"./data_augmented/"  # Directory to save augmented images
file_extension = ".JPG"

# Define augmentation parameters
resize_dim = (224, 224)
flip_horizontal = True
flip_vertical = True
rotation_angles = [-90, 90, 180]  # Rotation angles in degrees
zoom_factor = 0.25  # Zoom factor
brightness_factors = [-0.75, 0.75]  # Brightness adjustment factor
contrast_factors = [-0.25, 0.25]  # Contrast adjustment factor
channel_shift_intensity = 75  # Intensity of channel shift in color space transformation
noise_mean = 0  # Mean of the Gaussian noise
noise_stddev = 1  # Standard deviation of the Gaussian noise
hue_shift_factors = [-20, 20]  # Amount of adjustment in the hue channel of the HSV color space

# Function to perform data augmentation on images in a folder
def augment_images_in_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png')):  # Check for valid image files
            print(f"Augmenting image: {filename}")

            # Load image
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)
            img = cv2.resize(img, resize_dim)

            # Split images at the file extension to keep name without it
            filename_without_ext = os.path.splitext(filename)[0]

            # Apply horizontal and vertical flipping
            if flip_horizontal:
                flipped_img = np.fliplr(img)
                cv2.imwrite(os.path.join(output_folder, f"{filename_without_ext}_flip_horizontal{file_extension}"), flipped_img)
            if flip_vertical:
                flipped_img = np.flipud(img)
                cv2.imwrite(os.path.join(output_folder, f"{filename_without_ext}_flip_vertical{file_extension}"), flipped_img)

            # Apply rotation
            for angle in rotation_angles:
                rows, cols, _ = img.shape
                rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                rotated_img = cv2.warpAffine(img, rotation_matrix, (cols, rows))
                cv2.imwrite(os.path.join(output_folder, f"{filename_without_ext}_rotate_{angle}{file_extension}"), rotated_img)

            # Apply brightness adjustment
            for brightness in brightness_factors:
                brightness_img = cv2.convertScaleAbs(img, alpha=1 + brightness, beta=0)
                cv2.imwrite(os.path.join(output_folder, f"{filename_without_ext}_brightness_{brightness}{file_extension}"), brightness_img)

            # Apply contrast adjustment
            for contrast in contrast_factors:
                contrast_img = cv2.convertScaleAbs(img, alpha=1, beta=contrast * 255)
                cv2.imwrite(os.path.join(output_folder, f"{filename_without_ext}_contrast_{contrast}{file_extension}"), contrast_img)

            # Apply color space transformation
            for hue_shift in hue_shift_factors:
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
                shifted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
                cv2.imwrite(os.path.join(output_folder, f"{filename_without_ext}_hue_shift_{hue_shift}{file_extension}"), shifted_img)

            # Apply channel shift
            shifted_img = img.astype(np.int16) + np.random.randint(-channel_shift_intensity, channel_shift_intensity + 1, size=img.shape)
            shifted_img = np.clip(shifted_img, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, f"{filename_without_ext}_channel_shift{file_extension}"), shifted_img)

            # Apply histogram equalization
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            equalized_img = cv2.equalizeHist(gray_img)
            cv2.imwrite(os.path.join(output_folder, f"{filename_without_ext}_hist_eq{file_extension}"), equalized_img)

            # Apply Gaussian noise
            noise = np.random.normal(noise_mean, noise_stddev, img.shape).astype(np.uint8)
            noisy_img = cv2.add(img, noise)
            cv2.imwrite(os.path.join(output_folder, f"{filename_without_ext}_noise{file_extension}"), noisy_img)

            # Save untouched image
            cv2.imwrite(os.path.join(output_folder, f"{filename_without_ext}_untouched{file_extension}"), img)

for data_type in ['train']:
    data_folder = os.path.join(input_dir, data_type)
    if os.path.exists(data_folder):
        for class_folder in os.listdir(data_folder):
            input_class_folder = os.path.join(data_folder, class_folder)
            output_class_folder = os.path.join(output_dir, data_type, class_folder)
            print(f"\nAugmenting images in folder: {class_folder} ({data_type})")
            augment_images_in_folder(input_class_folder, output_class_folder)

print("\nData augmentation complete!")


# Define paths
source_folder = "./data_merged_split/test"
destination_folder = "./data_augmented/test"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Loop through items in the source folder
for item in os.listdir(source_folder):
    source_path = os.path.join(source_folder, item)
    dest_path = os.path.join(destination_folder, item)
    # Move the item
    shutil.move(source_path, dest_path)

print("Moving test data complete")

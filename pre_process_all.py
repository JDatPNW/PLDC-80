import os
import shutil
root_folder = "./data/"  # Change this to your root folder path
output_folder = "./data_merged/"

os.makedirs(output_folder, exist_ok=True)

for dataset_folder in os.listdir(root_folder):
    dataset_path = os.path.join(root_folder, dataset_folder)

    if os.path.isdir(dataset_path):  # Ensure it's a directory
        for class_folder in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_folder)

            if os.path.isdir(class_path):  # Ensure it's a directory
                if class_folder == "gls":
                    class_folder = "Gray Leaf Spot"
                elif class_folder == "nlb":
                    class_folder = "Northern Leaf Blight"
                elif class_folder == "nls":
                    class_folder = "Northern Leaf Spot"
                
                if dataset_folder == "fgvc8":
                    plant_type = "apple"
                elif dataset_folder == "cds":
                    plant_type = "corn"
                elif dataset_folder == "cassava":
                    plant_type = "cassava"
                elif dataset_folder == "diaMOS":
                    plant_type = "pear"
                elif dataset_folder == "paddy":
                    plant_type = "rice"
                elif dataset_folder == "pdd271":
                    plant_type = ""
                elif dataset_folder == "plantVillage":
                    plant_type = ""
                elif dataset_folder == "sms":
                    plant_type = "strawberry"
                elif dataset_folder == "sugar":
                    plant_type = "cane"

                new_class_name = f"{dataset_folder}_{plant_type}_{class_folder}"
                new_class_path = os.path.join(output_folder, new_class_name)

                # Rename the folder
                os.rename(class_path, new_class_path)
                print(f"Renamed: {class_folder} -> {new_class_name}")


# Define the folder pairs and their new combined names
folder_pairs = {
    # ("fgvc8_apple_healthy", "plantVillage__Apple___healthy"): "fgvc8_apple_healthy_AND_plantvillage_Apple___healthy",
    ("fgvc8_apple_rust", "plantVillage__Apple___Cedar_apple_rust"): "fgvc8_apple_rust_AND_plantvillage_Apple___Cedar_apple_rust",
    ("fgvc8_apple_scab", "plantVillage__Apple___Apple_scab"): "fgvc8_apple_scab_AND_plantvillage_Apple___Apple_scab",
    ("cds_corn_Northern Leaf Blight", "plantVillage__Corn_(maize)___Northern_Leaf_Blight"): "cds_corn_Northern Leaf Blight_AND_plantvillage_Corn_(maize)___Northern_Leaf_Blight",
    ("cds_corn_Gray Leaf Spot", "plantVillage__Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"): "cds_corn_Gray Leaf Spot_AND_plantvillage_Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
}

root_folder = output_folder

# Process each pair
for (folder_a, folder_b), new_folder in folder_pairs.items():
    path_a = os.path.join(root_folder, folder_a)
    path_b = os.path.join(root_folder, folder_b)
    new_path = os.path.join(root_folder, new_folder)

    # Create the new folder if it doesn't exist
    os.makedirs(new_path, exist_ok=True)

    # Move all images from both folders to the new folder
    for folder in [path_a, path_b]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    shutil.move(file_path, os.path.join(new_path, file))

            # Remove the empty folder after moving files
            os.rmdir(folder)
            print(f"Deleted: {folder}")

    print(f"Combined {folder_a} and {folder_b} into {new_folder}")

print("Merging completed!")
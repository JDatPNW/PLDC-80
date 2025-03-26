import os
import shutil

# Define source and destination paths
source_dir = "./pre_clean_data/pdd271"
destination_dir = "./data/pdd271"

# Mapping of folder IDs to new names
id_to_name = {
    "220/220": "Sweet_potato_healthy_leaf_220",
    "224/224": "Sweet_potato_sooty_mold_224",
    "227/227": "Sweet_potato_magnesium_deficiency_227",
    "135/135": "Soybean_downy_mildew_135",
    "246/246": "Mung_bean_brown_spot_246",
    "293/293": "radish_wrinkle_virus_disease_293",
    "295/295": "radish_mosaic_virus_disease_295",
    "297/297": "radish_black_spot_disease_297",
    "338/338": "leek_hail_damage_338",
    "339/339": "leek_gray_mold_disease_339"
}

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Move and rename the folders
for folder_id, new_name in id_to_name.items():
    src_path = os.path.join(source_dir, folder_id)
    dest_path = os.path.join(destination_dir, new_name)
    
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
        print(f"Moved '{src_path}' to '{dest_path}'")
    else:
        print(f"Folder '{src_path}' not found.")

print("Operation completed!")

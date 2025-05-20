# Remove corrupted images from dataset

import PIL
from PIL import Image
import os

folder_paths = [
    "Going_Modular/data/sushi_types/test/futomaki",
    "Going_Modular/data/sushi_types/test/uramaki",
    "Going_Modular/data/sushi_types/test/temaki",
    "Going_Modular/data/sushi_types/test/sashimi",
    "Going_Modular/data/sushi_types/test/nigiri",
    "Going_Modular/data/sushi_types/test/hosomaki",
    "Going_Modular/data/sushi_types/test/gunkan_maki",
    "Going_Modular/data/sushi_types/train/futomaki",
    "Going_Modular/data/sushi_types/train/uramaki",
    "Going_Modular/data/sushi_types/train/temaki",
    "Going_Modular/data/sushi_types/train/sashimi",
    "Going_Modular/data/sushi_types/train/nigiri",
    "Going_Modular/data/sushi_types/train/hosomaki",
    "Going_Modular/data/sushi_types/train/gunkan_maki",
]

# Identify and delete corrupted image in each of the folder file
for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        try:
            image = Image.open(os.path.join(folder_path, filename))
        except PIL.UnidentifiedImageError as e:
            print(f"Error in file {filename}: {e}")
            os.remove(os.path.join(folder_path, filename))
            print(f"Removed file {filename}")

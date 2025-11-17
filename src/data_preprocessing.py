import numpy as np
import pandas as pd
from PIL import Image
import os
from skimage.transform import resize
import csv
import zipfile
import shutil
from torchvision.transforms import Resize

def organize_files(src:str, dest:str) -> None:
    """
    Extracts files from a zipfile to a destination directory. 
    Some files may be redundant, so they will be removed.
    WARNING: The source directory will be deleted.

    Args:
        src (str): The path to the source directory.
        dest (str): The path to the destination directory.

    Returns:
        None
    """
    with zipfile.ZipFile(src, "r") as zip_file:
        zip_file.extractall(dest)

    for filename in os.listdir(dest):
        path = os.path.join(dest, filename)
        for filename in os.listdir(path):
            sub_path = os.path.join(path, filename)
            shutil.move(sub_path, dest)

        os.rmdir(path)

    os.remove(src)

def resize_images(images_path:str, save_path:str) -> None:

    """
    Resizes images to 224x224 pixels.

    Args:
        images_path (str): Path where the images are stored.
        save_path (str): Path where to save the resized images.

    Returns:
        None
    """

    img_dimensions = (224, 224)
    for filename in os.listdir(images_path):
        file_path = os.path.join(images_path, filename)
        img = Image.open(file_path, mode="r")
        resize = Resize(size=img_dimensions)
        resized_image = resize(img)
        resized_image.save(os.path.join(save_path, filename))

def main():
    path = "../data/raw/"
    for split in ["Train", "Test", "Val"]:
        src = os.path.join(path, f"{split}.zip")
        dest = os.path.join(path, split)
        if not os.path.exists(dest):
            organize_files(src, dest)

    processed_path = "../data/processed/"
    for split in ["Train", "Test", "Val"]:
        for category in ["Earthquake", "Fire", "Flood", "Normal"]:
            images_path = os.path.join(path, split)
            images_path = os.path.join(images_path, category)

            save_path = os.path.join(processed_path, split)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, category)
            os.makedirs(save_path, exist_ok=True)
            resize_images(images_path, save_path)

if __name__ == "__main__":
    main()
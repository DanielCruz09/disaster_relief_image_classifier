import numpy as np
import pandas as pd
from PIL import Image
import os
from skimage.transform import resize
import csv

def generate_labels(images_path:str, category:str) -> tuple:
    """
    Resizes images and generates labels for each image. Images will be resized to 224x224 pixels.

    Args:
        images_path (str): Path where the images are stored. 
        category (str): The type of natural disaster (Fire, Earthquake, Flood, or Normal)

    Returns:
        A list of numpy arrays containing the image data and labels.
    """
    resized_images = []
    img_dimensions = (224, 224)
    for filename in os.listdir(images_path):
        file_path = os.path.join(images_path, filename)
        if os.path.isfile(file_path):
            img = Image.open(file_path)
            img_data = np.asarray(file_path)
            resized = resize(img_data, img_dimensions)
            resized_images.append(resized)

    labels = [category for _ in range(len(resized_images))]

    return resized_images, labels


def write_to_csv(data: pd.DataFrame, split:str, category:str) -> None:
    """
    Writes the data to a CSV file.

    Args:
        data (pandas.DataFrame): The data stored as a Pandas DataFrame.
        split (str): The type of split (Train, Test, or Validation).
        category (str): The type of natural disaster (Fire, Earthquake, Flood, or Normal)

    Returns:
        None
    """

    path = f"./data/processed/{split}/{category}.csv"
    header = "Data,Label"
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

    data.to_csv(path)


def main():
    path = "./data/raw/"
    categories = ["Earthquake", "Fire", "Flood", "Normal"]
    for split in ["Train", "Test", "Valid"]:
        for cat in categories:
            img_path = os.path.join(path, split)
            resized_images, labels = generate_labels(img_path, cat)


if __name__ == "__main__":
    main()
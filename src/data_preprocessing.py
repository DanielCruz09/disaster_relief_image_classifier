import numpy as np
import pandas as pd
from PIL import Image
import os
from skimage.transform import resize
import csv

def generate_labels(images_path:str, category:str) -> pd.DataFrame:
    """
    Resizes images and generates labels for each image. Images will be resized to 224x224 pixels.

    Args:
        images_path (str): Path where the images are stored. 
        category (str): The type of natural disaster (Fire, Earthquake, Flood, or Normal)

    Returns:
        A dataframe containing the image data and labels.
    """
    resized_images = []
    img_dimensions = (224, 224)
    for filename in os.listdir(images_path):
        file_path = os.path.join(images_path, filename)
        for image_file in os.listdir(file_path):
            full_path = os.path.join(file_path, image_file)
            if os.path.isfile(full_path):
                img = Image.open(full_path, mode="r")
                img_data = np.asarray(img)
                resized = resize(img_data, img_dimensions)
                resized_images.append(resized)

    data = {
        "Data": resized_images,
        "Label": [category for _ in range(len(resized_images))]
    }
    df = pd.DataFrame(data=data)
    return df


def write_to_csv(data: pd.DataFrame, split:str, category:str) -> None:
    """
    Writes the data to a CSV file.

    Args:
        data (pandas.DataFrame): The data stored as a Pandas DataFrame.
        split (str): The type of split (Train, Test, or Val).
        category (str): The type of natural disaster (Fire, Earthquake, Flood, or Normal)

    Returns:
        None
    """

    path = f"../data/processed/{split}"
    os.makedirs(path, exist_ok=True)
    path = f"../data/processed/{split}/{category}.csv"
    header = "Data,Label"
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

    data.to_csv(path, index=False)


def main():
    path = "../data/raw/"
    categories = ["Earthquake", "Fire", "Flood", "Normal"]
    for split in ["Train", "Test", "Val"]:
        for cat in categories:
            img_path = os.path.join(path, split)
            df = generate_labels(img_path, cat)
            write_to_csv(df, split, cat)

if __name__ == "__main__":
    main()
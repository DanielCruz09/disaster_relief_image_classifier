from transformers import pipeline
from PIL import Image
import os
from natural_disaster_dataset import NaturalDisasterDataset
from pathlib import Path
from skimage.exposure import equalize_adapthist
from torch.utils.data import DataLoader
from resnet50 import ResNet50
import torch

def rename_directories(old_name, new_name):
    try:
        current_dir = Path(old_name)
        current_dir.rename(new_name)
    except FileNotFoundError:
        return   

def write_to_csv(line, write_path, header=None):
    mode = "a"
    if header:
        mode = "w"
        with open(write_path, mode=mode) as csvfile:
            csvfile.write(header)
            csvfile.write("\n")
        mode = "a"

    text = ""
    for item in line:
        text += str(item) + ","
    text = text[:-1]
    with open(write_path, mode=mode) as csvfile:
        csvfile.write(text)
        csvfile.write("\n")            

def main():
    image_path = "../data/processed/Train/"
    natural_disaster_dataset = NaturalDisasterDataset(root=image_path)
    loader = DataLoader(natural_disaster_dataset, batch_size=32, shuffle=True)
    # natural_disaster_dataset.load_sample()
    renamed = {
        "Normal": "Non_Damage",
        "Earthquake": "Land_Disaster",
        "Fire": "Fire_Disaster",
        "Flood": "Water_Disaster"
    }
    for name in renamed.keys():
        old_path = os.path.join(image_path, name)
        new_path = os.path.join(image_path, renamed[name])
        rename_directories(old_path, new_path)

    resnet50 = ResNet50(num_classes=len(renamed))
    # resnet50.train(epochs=5, train_loader=loader)
    weights = torch.load("model_weights.pth")
    resnet50.model.load_state_dict(weights["model_state_dict"])
    test_path = "../data/processed/Test/"
    test_dataset = NaturalDisasterDataset(root=test_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    for name in renamed.keys():
        old_path = os.path.join(test_path, name)
        new_path = os.path.join(test_path, renamed[name])
        rename_directories(old_path, new_path)
    resnet50.eval(test_loader=test_loader)

if __name__ == "__main__":
    main()

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

    resnet50 = ResNet50(num_classes=len(renamed), lr=0.001)
    resnet50.train(epochs=4, train_loader=loader)
    weights = torch.load("model_weights.pth")
    resnet50.model.load_state_dict(weights["model_state_dict"])
    test_path = "../data/processed/Test/"
    test_dataset = NaturalDisasterDataset(root=test_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    for name in renamed.keys():
        old_path = os.path.join(test_path, name)
        new_path = os.path.join(test_path, renamed[name])
        rename_directories(old_path, new_path)
    resnet50.eval(test_loader=test_loader, write_path="../results/resnet50_results.csv")

    val_path = "../data/processed/Val"
    for name in renamed.keys():
        old_path = os.path.join(val_path, name)
        new_path = os.path.join(val_path, renamed[name])
        rename_directories(old_path, new_path)
    val_dataset = NaturalDisasterDataset(root=val_path)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    resnet50.eval(test_loader=val_loader, write_path=None)

if __name__ == "__main__":
    main()

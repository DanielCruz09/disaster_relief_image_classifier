from transformers import pipeline
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset
import torch
from skimage import io, transform
import matplotlib.pyplot as plt

class NaturalDisasterDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for label in os.listdir(root):
            folder = os.path.join(root, label)
            for file in os.listdir(folder):
                self.image_paths.append(os.path.join(folder, file))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "category": label}


    
def load_sample(root:str):
    disaster_dataset = NaturalDisasterDataset(
        root=root,
        transform=None
    )

    categories_needed = {"Earthquake", "Flood", "Fire", "Normal"}
    shown = {}

    fig = plt.figure(figsize=(10, 3))

    for sample in disaster_dataset:
        category = sample["category"]

        # If we still need this category
        if category in categories_needed and category not in shown:
            shown[category] = sample["image"]

        # Stop if we have all 4 categories
        if len(shown) == len(categories_needed):
            break

    for i, (category, image) in enumerate(shown.items()):
        ax = plt.subplot(1, 4, i + 1)
        ax.imshow(image)
        ax.set_title(category)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def classify_images(model, image_path):

    batch_size = 32
    trainset = NaturalDisasterDataset(root=image_path)
    loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    

def main():
    classifier = pipeline("image-classification", model="Luwayy/disaster_images_model")
    image_path = "../data/processed/Train"
    load_sample(image_path)
    classify_images(classifier, image_path)


if __name__ == "__main__":
    main()
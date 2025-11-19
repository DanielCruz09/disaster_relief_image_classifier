from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset
import torch
from skimage import transform
import matplotlib.pyplot as plt

class NaturalDisasterDataset(Dataset):
    """
    A custom PyTorch Dataset that contains images of several types of natural disasters, 
    including earthquakes, fires, and floods.
    """
    def __init__(self, root:str, transform:any=None) -> None:
        """
        Creates a custom PyTorch dataset of natural disasters.

        Args:
            root (str): A path containing the images.
            transform (any): A type of transformation from the scikit-image library.

        Returns:
            None
        """
        self.root = root
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for label in os.listdir(root):
            folder = os.path.join(root, label)
            for file in os.listdir(folder):
                self.image_paths.append(os.path.join(folder, file))
                self.labels.append(label)

    def __len__(self) -> int:
        """
        Returns the length/size of the dataset.

        Args:
            None

        Returns:
            length (int): The length of the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx:int) -> dict:
        """
        Iterates through the dataset and returns a sample image.

        Args:
            idx (int): An index to the dataset.

        Returns:
            sample (dict): A dictionary containing the image and its label.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image) 

        sample = {"image": image, "category": label}
        return sample

    
    def load_sample(self) -> None:
        """
        Displays four sample images, one of each type of disaster.

        Args:
            None

        Returns:
            None
        """

        categories_needed = {"Normal", "Earthquake", "Fire", "Flood"}
        shown = {}

        fig = plt.figure(figsize=(10, 3))

        for sample in self:
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
from transformers import pipeline
from PIL import Image
import os
from natural_disaster_dataset import NaturalDisasterDataset
from pathlib import Path
from skimage.exposure import equalize_adapthist
from torch.utils.data import DataLoader
from resnet50 import ResNet50

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

def classify_images(model, image_path):

    batch_size = 32
    trainset = NaturalDisasterDataset(root=image_path, transform=equalize_adapthist)
    include_header = True
    index = 0
    write_path = "../results/results.csv"

    for true_label in os.listdir(image_path):
        class_dir = os.path.join(image_path, true_label)

        paths = []
        for file in os.listdir(class_dir):
            paths.append(os.path.join(class_dir, file))
            
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            images = [Image.open(path).convert("RGB") for path in batch]
            predictions = model(images)

            for path, result in zip(batch, predictions):
                predicted_label = result[0]["label"]
                if predicted_label == "Damaged_Infrastructure":
                    predicted_label = "Land_Disaster"
                if include_header:
                    write_to_csv(line=[index, true_label, predicted_label], write_path=write_path, header="Index,True,Predicted")
                else:
                    write_to_csv(line=[index, true_label, predicted_label], write_path=write_path)

                include_header = False

                index += 1
            

def main():
    classifier = pipeline("image-classification", model="Luwayy/disaster_images_model")
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
    resnet50.eval(loader)
    # classify_images(classifier, image_path)


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd

def create_indices(labels):
    mapping = {
        "Non_Damage": 0,
        "Land_Disaster": 1,
        "Fire_Disaster": 2,
        "Water_Disaster": 3
    }

    indices = list(mapping[category] for category in labels)
    return indices

def write_to_csv(predicted, actual, write_path, header):

    text = ""
    for x, y in zip(predicted, actual):
        if header:
            text += "Predicted,Actual\n"
        text += str(x.item()) + "," + str(y.item()) + "\n"
        header = False

    with open(write_path, "a") as file:
        file.write(text)

class ResNet50():

    def __init__(self, num_classes, lr=0.01, momentum=0.9):
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, self.num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

    def train(self, epochs, train_loader):
        last_loss = 0
        for epoch in range(epochs):
            self.model.train()
            current_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(data[inputs].float())
                indices = create_indices(data[labels])
                target = torch.tensor(indices)
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()
                current_loss += loss.item()
                last_loss = current_loss
            print(f"Epoch: {epoch + 1} \t Loss: {current_loss / len(train_loader)}")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epochs": epochs,
            "loss": last_loss
        }, "model_weights.pth")

    def eval(self, test_loader):
        self.model.eval()
        header = True
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                images, labels = data
                images = data[images].float()
                labels = data[labels]
                indices = create_indices(labels)
                labels = torch.tensor(indices)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += len(labels)
                correct += (predicted == labels).sum().item()
                write_to_csv(predicted, labels, write_path="../results/resnet50_results.csv", header=header)
                header = False
        
        print(f'Accuracy of the network on the test images: {round(100 * correct / total, 3)}%')

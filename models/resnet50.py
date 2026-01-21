import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class ResNet50():

    def __init__(self, num_classes, lr=0.01, momentum=0.9):
        self.model = models.resnet50(pretrained=True)
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, self.num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

    def train(self, epochs, train_loader):
        for epoch in range(epochs):
            current_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                # outputs = self.model(inputs)
                # loss = self.criterion(outputs, labels)
                outputs = self.model(data[inputs].float())
                # target = torch.tensor(data[labels])
                print(data[labels])
                with torch.tensor(data[labels]) as target:
                    loss = self.criterion(outputs, target)
                loss.backwards()
                self.optimizer.step()
                current_loss += loss.item()
            print(f"Epoch: {epoch + 1} \t Loss: {current_loss / len(train_loader)}")

    def eval(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = Image.open(images)
                images = np.asarray(images)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Accuracy of the network on the test images: {100 * correct / total}%')
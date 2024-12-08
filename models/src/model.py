import torch
from torch.jit import optimize_for_inference
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as fn

class VisionModel(nn.Module):
    def __init__(self, num_labels, resolution):
        self.num_conv_layers = 5
        self.linear_pooling = resolution // (2 ** self.num_conv_layers)

        super(VisionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.linear1 = nn.Linear(512 * self.linear_pooling * self.linear_pooling, resolution*2)
        self.linear2 = nn.Linear(resolution*2, num_labels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout_c = nn.Dropout2d(p=0.4)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(fn.relu(self.conv1(x)))
        x = self.pool(fn.relu(self.conv2(x)))
        x = self.dropout_c(x)
        x = self.pool(fn.relu(self.conv3(x)))
        x = self.pool(fn.relu(self.conv4(x)))
        x = self.dropout_c(x)
        x = self.pool(fn.relu(self.conv5(x)))

        #x = x.view(-1, 512 * self.linear_pooling * self.linear_pooling)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
 
        x = fn.relu(self.linear1(x))
        x =self.linear2(x)
        return x

def train_model(model, epochs, train, val):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0.0
        for image, label in train:
            image, label = image.to("cuda"), label.to("cuda")
            model.to("cuda")

            optimizer.zero_grad()

            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss  = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for image, label in val:
                image, label = image.to("cuda"), label.to("cuda")
                outputs = model(image)
                loss = criterion(outputs, label)
                val_loss += loss.item()

                _, pred = torch.max(outputs, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
        val_accuracy = 100 * correct / total


        print(f"Train loss: {train_loss / len(train):.4f}")
        print(f"Validation loss: {val_loss / len(val):.4f}, Validation Accuracy: {val_accuracy:.2f}")


def evaluate(model, data):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in data:
            image, label = image.to("cuda"), label.to("cuda")
            outputs = model(image)
            _, pred = torch.max(outputs, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()

    print(f"Accuracy: {100 * correct / total :.2f}%")


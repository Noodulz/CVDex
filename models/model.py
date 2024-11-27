import torch
import torch.nn as nn
import torch.nn.functional as fn

class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.linear1 = nn.Linear(64 * 7 * 7, 128)
        self.linear2 = nn.Linear(128, 10)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(fn.relu(self.conv1(x)))
        x = self.pool(fn.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7 * 7)

        x = fn.relu(self.linear1(x))
        x =self.linear2(x)
        return x

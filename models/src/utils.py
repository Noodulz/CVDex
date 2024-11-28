import os
from torch import mean
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def data_loader(path, width, height):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((width, height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])

    dataset = datasets.ImageFolder(root=path, transform=transform)
    return dataset

def get_labels(dataset):
    return dataset.classes

def dataset_distribution(dataset):
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return (train_loader, val_loader, test_loader)



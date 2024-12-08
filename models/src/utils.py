import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split

def transform_composition(width, height):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5)
        ]), p=0.3),
        transforms.Resize((width, height)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform

def transform_autocomp(width, height):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((width, height)),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform

def data_loader(path, transform):
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


def stratified_distribution(dataset):
    labels = [label for _, label in dataset]

    train_val_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.1,
        stratify=labels,
    )

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.1,
        stratify=[labels[i] for i in train_val_indices],
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"Stratified loading sizes\nTraining Size: {len(train_dataset)}, Validation Size: {len(val_dataset)}, Test Size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return (train_loader, val_loader, test_loader)


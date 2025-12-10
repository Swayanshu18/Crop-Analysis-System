"""
Train EfficientNet-B0 for Plant Disease Classification
Dataset format:
data/plant/
    train/class_name/*.jpg
    val/class_name/*.jpg
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
import json


def get_data_transforms(img_size=224):
    """Augmentation suitable for PlantVillage"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct/total)*100:.2f}%"
        })

    return running_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct/total)*100:.2f}%"
            })

    return running_loss / len(loader), correct / total


def main():
    CONFIG = {
        "data_dir": "data/plant",
        "img_size": 224,
        "epochs": 20,
        "batch_size": 16,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "save_path": "efficientnet_plant_best.pt",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    # transforms
    train_tf, val_tf = get_data_transforms(CONFIG["img_size"])

    # dataset
    train_dataset = ImageFolder(os.path.join(CONFIG["data_dir"], "train"), transform=train_tf)
    val_dataset = ImageFolder(os.path.join(CONFIG["data_dir"], "val"), transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    num_classes = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)

    # Load pretrained EfficientNet-B0
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    best_val_acc = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["save_path"])
            print("✓ Best model saved!")

    # Save history json
    with open("efficientnet_training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining Complete!")


if __name__ == "__main__":
    main()

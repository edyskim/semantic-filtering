#!/usr/bin/env python3
"""Train a downstream CNN classifier from CSV manifests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics import AUROC, Accuracy, F1Score
from torcheval.metrics import MulticlassAUPRC


# The five-seed protocol used in the paper is produced by running this script
# once per seed (0, 42, 84, 126, 168), each passed via --seed.


class ImageCsvDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["img_path"]).convert("RGB")
        return self.transform(image), int(row["label"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch: str, num_classes: int) -> torch.nn.Module:
    if arch == "resnet50":
        model = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "mobilenetv2":
        model = torchvision.models.mobilenet_v2(weights="MobileNet_V2_Weights.IMAGENET1K_V1")
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError("arch must be resnet50 or mobilenetv2")


def evaluate(model, loader, num_classes: int, device: torch.device) -> dict[str, float]:
    model.eval()
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    auroc = AUROC(task="multiclass", num_classes=num_classes).to(device)
    auprc = MulticlassAUPRC(num_classes=num_classes).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            accuracy.update(preds, labels)
            auroc.update(probs, labels)
            auprc.update(probs, labels)
            f1.update(preds, labels)

    return {
        "accuracy": float(accuracy.compute().cpu()),
        "auroc": float(auroc.compute().cpu()),
        "auprc": float(auprc.compute().cpu()),
        "f1": float(f1.compute().cpu()),
    }


def train_one_seed(train_csv: str, test_csv: str, arch: str, seed: int, epochs: int, output_json: str) -> None:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    num_classes = int(max(train_df["label"].max(), test_df["label"].max()) + 1)

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["label"],
        random_state=42,
    )

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
        ]
    )
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )

    labels = train_df["label"].to_numpy()
    classes, counts = np.unique(labels, return_counts=True)
    weights = {cls: len(labels) / count for cls, count in zip(classes, counts)}
    sample_weights = torch.DoubleTensor([weights[label] for label in labels])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(ImageCsvDataset(train_df, train_transform), batch_size=64, sampler=sampler, num_workers=4)
    val_loader = DataLoader(ImageCsvDataset(val_df, test_transform), batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(ImageCsvDataset(test_df, test_transform), batch_size=128, shuffle=False, num_workers=4)

    model = build_model(arch, num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                _ = model(images.to(device))

    metrics = evaluate(model, test_loader, num_classes, device)
    metrics.update({"seed": seed, "arch": arch, "epochs": epochs})
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--arch", choices=["resnet50", "mobilenetv2"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    train_one_seed(args.train_csv, args.test_csv, args.arch, args.seed, args.epochs, args.output_json)


if __name__ == "__main__":
    main()

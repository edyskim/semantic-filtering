#!/usr/bin/env python3
"""Train and evaluate one downstream classifier with the protocol of the article (Section 2.6).

The settings follow the training code that produced the reported results:
- a stratified 90/10 train-validation split within each class with a fixed random_state of 42,
- training augmentation: resize to 324, random resized crop to 224 (scale 0.5 to 1.0), horizontal and vertical
  flips, rotation, color jitter, perspective and affine transforms, Gaussian blur, ImageNet normalization, and
  random erasing, test images resized to 224 and normalized,
- a class-balanced sampler, batch size 64, SGD (momentum 0.9, weight decay 1e-4), a one-cycle learning rate
  peaking at 0.01, 30 epochs with mixed precision,
- evaluation of the checkpoint with the lowest validation loss on the test set.

The five-seed protocol of the article is obtained by running the script once per seed (0, 42, 84, 126, 168).
Input CSVs need an img_path column (path to the image file) and a class_name column. Class indices follow the
sorted class names of the training CSV.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics import AUROC, Accuracy, F1Score
from torcheval.metrics import MulticlassAUPRC

IMG_SIZE = 224
SPLIT_RANDOM_STATE = 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class ImageCsvDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), list(labels), transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with open(self.paths[index], "rb") as f:
            image = Image.open(f).convert("RGB")
        return self.transform(image), int(self.labels[index])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_transform() -> T.Compose:
    return T.Compose([
        T.Resize((IMG_SIZE + 100, IMG_SIZE + 100)),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value="random"),
    ])


def test_transform() -> T.Compose:
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """90/10 split within each class, fixed random_state for all runs (classes with one image stay in training)."""
    train, val = [], []
    for _, group in df.groupby("class_name"):
        if len(group) < 2:
            train.append(group)
            continue
        tr, va = train_test_split(group, test_size=0.1, random_state=SPLIT_RANDOM_STATE, shuffle=True)
        train.append(tr)
        val.append(va)
    train_df = pd.concat(train).reset_index(drop=True)
    val_df = pd.concat(val).reset_index(drop=True) if val else pd.DataFrame(columns=df.columns)
    return train_df, val_df


def build_model(arch: str, num_classes: int) -> torch.nn.Module:
    if arch == "resnet50":
        model = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif arch == "mobilenetv2":
        model = torchvision.models.mobilenet_v2(weights="MobileNet_V2_Weights.IMAGENET1K_V1")
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "vit_b_16":
        model = torchvision.models.vit_b_16(weights="ViT_B_16_Weights.IMAGENET1K_V1")
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError("arch must be resnet50, mobilenetv2, or vit_b_16")
    return model


def train(model, train_loader, val_loader, epochs: int, device: torch.device) -> torch.nn.Module:
    """Train for a fixed number of epochs and return the model with the lowest validation loss."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, total_steps=epochs * len(train_loader),
        pct_start=0.3, anneal_strategy="cos", final_div_factor=100)
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    best_model, best_val_loss = deepcopy(model), float("inf")

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        val_loss, steps = 0.0, 0
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            for images, labels in val_loader:
                val_loss += criterion(model(images.to(device)), labels.to(device)).item()
                steps += 1
        val_loss /= steps
        if val_loss < best_val_loss:
            best_val_loss, best_model = val_loss, deepcopy(model)
        print(f"epoch {epoch + 1}/{epochs}  validation loss {val_loss:.4f}")
    return best_model


def evaluate(model, loader, num_classes: int, device: torch.device):
    """Accuracy, AUROC, AUPRC, and macro-F1 on the test set, and the per-image class probabilities."""
    model.eval()
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    auroc = AUROC(task="multiclass", num_classes=num_classes).to(device)
    auprc = MulticlassAUPRC(num_classes=num_classes).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    probabilities = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            accuracy.update(preds, labels)
            auroc.update(probs, labels)
            auprc.update(probs, labels)
            f1.update(preds, labels)
            probabilities.append(probs.float().cpu().numpy())
    metrics = {"accuracy": accuracy.compute().item(), "auroc": auroc.compute().item(),
               "auprc": auprc.compute().item(), "f1": f1.compute().item()}
    return metrics, np.concatenate(probabilities, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, help="training pool after filtering (img_path, class_name)")
    parser.add_argument("--test_csv", required=True, help="test set (img_path, class_name)")
    parser.add_argument("--arch", choices=["resnet50", "mobilenetv2", "vit_b_16"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--predictions_csv", default=None, help="optional per-image test probabilities")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    classes = sorted(pool["class_name"].unique())
    label_map = {c: i for i, c in enumerate(classes)}
    train_df, val_df = stratified_split(pool)
    for frame in (train_df, val_df, test):
        frame["label"] = frame["class_name"].map(label_map)
    assert test["label"].notna().all(), "test classes must occur in the training CSV"

    seed_everything(args.seed)
    labels = train_df["label"].tolist()
    uniq, counts = np.unique(labels, return_counts=True)
    weight = {u: len(labels) / c for u, c in zip(uniq, counts)}
    sampler = WeightedRandomSampler(torch.tensor([weight[l] for l in labels], dtype=torch.double),
                                    num_samples=len(labels), replacement=True)
    # loader settings as in the reported runs (worker count and persistence change the augmentation random streams)
    train_loader = DataLoader(ImageCsvDataset(train_df["img_path"], labels, train_transform()),
                              batch_size=64, sampler=sampler, num_workers=5, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(ImageCsvDataset(val_df["img_path"], val_df["label"], test_transform()),
                            batch_size=64, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(ImageCsvDataset(test["img_path"], test["label"], test_transform()),
                             batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = train(build_model(args.arch, len(classes)).to(device), train_loader, val_loader, args.epochs, device)
    metrics, probabilities = evaluate(model, test_loader, len(classes), device)
    metrics.update({"seed": args.seed, "arch": args.arch, "epochs": args.epochs,
                    "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test)})
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.predictions_csv:
        out = test[["img_path", "class_name"]].copy()
        out["pred_label"] = [classes[i] for i in probabilities.argmax(axis=1)]
        for i, c in enumerate(classes):
            out[f"prob_{c}"] = probabilities[:, i]
        Path(args.predictions_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.predictions_csv, index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

from dino_finetune import DINOV2Classifier
from AAAI2026_aoi_dataset import AOIDataset  # 你之前写好的 dataset 类


def get_dataloader(dataset_root, excel_path, batch_size=16, split_ratio=0.85, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((490, 490)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset = AOIDataset(image_dir=dataset_root, excel_path=excel_path, transform=transform)

    total = len(dataset)
    train_len = int(split_ratio * total)
    val_len = total - train_len

    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def validate_epoch(model, val_loader, criterion, metrics):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            labels = labels.float().unsqueeze(1).cuda()

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    acc = total_correct / total_samples

    metrics.setdefault("val_loss", []).append(avg_loss)
    metrics.setdefault("val_acc", []).append(acc)

    print(f"Validation Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")


def finetune_dino(config, encoder):
    model = DINOV2Classifier(encoder=encoder, r=config.r, emb_dim=config.emb_dim, use_lora=config.use_lora).cuda()

    if config.lora_weights:
        model.load_parameters(config.lora_weights)

    dataset_root = "/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/标注2000/aoi_patches_train_2000"
    excel_path = "/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/标注2000/标注2000.xlsx"

    train_loader, val_loader = get_dataloader(dataset_root, excel_path, batch_size=config.batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for images, labels in loop:
            images = images.cuda()
            labels = labels.float().unsqueeze(1).cuda()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        metrics.setdefault("train_loss", []).append(epoch_loss / len(train_loader))

        if epoch % 2 == 0:
            validate_epoch(model, val_loader, criterion, metrics)
            os.makedirs(f"output/{config.exp_name}", exist_ok=True)
            model.save_parameters(f"output/{config.exp_name}/{epoch}.pt")

    model.save_parameters(f"output/{config.exp_name}/final.pt")
    with open(f"output/{config.exp_name}/metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="dino_aoi_cls")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--size", type=str, default="base")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--img_dim", type=int, nargs=2, default=(490, 490))
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    config = parser.parse_args()

    backbones = {
        "small": "vits14_reg",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    embedding_dims = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }
    config.emb_dim = embedding_dims[config.size]

    encoder = torch.hub.load("facebookresearch/dinov2", f"dinov2_{backbones[config.size]}").cuda()
    finetune_dino(config, encoder)
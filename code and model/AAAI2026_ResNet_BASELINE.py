import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

from AAAI2026_aoi_dataset import AOIDataset


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


def validate_epoch(model, val_loader, criterion):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            labels = labels.float().unsqueeze(1).cuda()
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(float)
            y_true.extend(labels.cpu().numpy().flatten())
            y_prob.extend(probs.flatten())
            y_pred.extend(preds.flatten())

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    return {
        "loss": total_loss / len(val_loader),
        "acc": acc,
        "auc": auc,
        "f1": f1,
        "precision": prec,
        "recall": rec
    }


def summarize_topk(metrics_list, topk=5):
    scores = [m["auc"] for m in metrics_list]
    topk_indices = np.argsort(scores)[-topk:]
    topk_metrics = [metrics_list[i] for i in topk_indices]
    summary = {}
    for key in topk_metrics[0].keys():
        values = np.array([m[key] for m in topk_metrics])
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }
    return summary


def train_resnet18(exp_name, dataset_root, excel_path, epochs=10, lr=1e-4, batch_size=8):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.cuda()

    os.makedirs(f"output/{exp_name}", exist_ok=True)

    train_loader, val_loader = get_dataloader(dataset_root, excel_path, batch_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    all_val_metrics = []

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in loop:
            images = images.cuda()
            labels = labels.float().unsqueeze(1).cuda()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        val_metrics = validate_epoch(model, val_loader, criterion)
        all_val_metrics.append(val_metrics)
        print(f"Val Epoch {epoch+1}: "
              f"Acc={val_metrics['acc']:.4f}, AUC={val_metrics['auc']:.4f}, "
              f"F1={val_metrics['f1']:.4f}, Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}")

        torch.save(model.state_dict(), f"output/{exp_name}/{epoch}.pt")

    torch.save(model.state_dict(), f"output/{exp_name}/final.pt")

    with open(f"output/{exp_name}/metrics_all.json", "w") as f:
        json.dump(all_val_metrics, f, indent=2)

    top5_summary = summarize_topk(all_val_metrics, topk=5)
    with open(f"output/{exp_name}/top5_summary.json", "w") as f:
        json.dump(top5_summary, f, indent=2)

    print("\n🏆 Top-5 Epochs Mean ± Std Summary:")
    for k, v in top5_summary.items():
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")


# Setup paths (adjust as needed)
dataset_root = "/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/标注2000/aoi_patches_train_2000"
excel_path = "/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/标注2000/标注2000.xlsx"

train_resnet18(
    exp_name="resnet18_full_eval",
    dataset_root=dataset_root,
    excel_path=excel_path,
    epochs=10,
    lr=1e-4,
    batch_size=8
)


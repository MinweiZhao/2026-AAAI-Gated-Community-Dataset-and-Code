import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, random_split
from PIL import Image, ImageFile
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

# === 简单图像数据集 ===
class ImageOnlyAOIDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, excel_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_excel(excel_path, header=1)
        df['label'] = df['label'].replace(-1, 0)
        df = df[['filename', 'label']].dropna()
        df['filename'] = df['filename'].astype(str)

        self.data = df[df['filename'].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, 'filename']
        label = self.data.loc[idx, 'label']
        img_path = os.path.join(self.image_dir, img_name)
        with open(img_path, 'rb') as f:
            image = Image.open(f).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# === DINOv2 + 分类器 ===
class DINOv2ImageClassifier(nn.Module):
    def __init__(self, encoder, emb_dim):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(self, x):
        feat = self.encoder(x)  # [B, emb_dim]
        return self.classifier(feat)  # [B, 1]

# === dataloader ===
def get_dataloader(dataset_root, excel_path, batch_size=8, split_ratio=0.85, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = ImageOnlyAOIDataset(dataset_root, excel_path, transform=transform)
    total = len(dataset)
    train_len = int(total * split_ratio)
    val_len = total - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

# === 验证 ===
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
            y_pred.extend(preds.flatten())
            y_prob.extend(probs.flatten())

    return {
        "loss": total_loss / len(val_loader),
        "acc": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }

# === 取 Top-K 结果并输出均值 ± std ===
def summarize_topk(metrics_list, topk=5):
    scores = [m["auc"] for m in metrics_list]
    topk_indices = np.argsort(scores)[-topk:]
    topk_metrics = [metrics_list[i] for i in topk_indices]
    summary = {}
    for key in topk_metrics[0].keys():
        vals = np.array([m[key] for m in topk_metrics])
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals))
        }
    return summary

# === 主训练 ===
def train(config):
    dataset_root = "/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/标注2000/aoi_patches_train_2000"
    excel_path = "/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/标注2000/标注2000.xlsx"

    train_loader, val_loader = get_dataloader(dataset_root, excel_path, batch_size=config.batch_size)

    base_encoder = torch.hub.load("facebookresearch/dinov2", f"dinov2_{config.backbone}").cuda()
    model = DINOv2ImageClassifier(base_encoder, emb_dim=config.emb_dim).cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    os.makedirs(f"output/{config.exp_name}", exist_ok=True)
    best_auc = 0
    all_metrics = []

    for epoch in range(config.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
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
        all_metrics.append(val_metrics)
        print(f"[VAL] Epoch {epoch+1}: AUC={val_metrics['auc']:.4f}, Acc={val_metrics['acc']:.4f}, "
              f"F1={val_metrics['f1']:.4f}, Prec={val_metrics['precision']:.4f}, Rec={val_metrics['recall']:.4f}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            # torch.save(model.state_dict(), f"output/{config.exp_name}/best.pt")
            # print("✅ Best model saved.")

    # 保存 final 模型和指标
    with open(f"output/{config.exp_name}/metrics_all.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # 统计 top-5 均值 ± 标准差
    top5_summary = summarize_topk(all_metrics, topk=5)
    with open(f"output/{config.exp_name}/top5_summary.json", "w") as f:
        json.dump(top5_summary, f, indent=2)

    print("\n🏆 Top-5 Epochs Summary:")
    for k, v in top5_summary.items():
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")

# === 参数入口 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="dinov2_image_baseline")
    parser.add_argument("--backbone", type=str, default="vitb14_reg", choices=["vits14_reg", "vitb14_reg", "vitl14_reg", "vitg14_reg"])
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    config = parser.parse_args()

    train(config)

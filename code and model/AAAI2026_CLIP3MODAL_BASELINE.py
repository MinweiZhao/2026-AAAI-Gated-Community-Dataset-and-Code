import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image, ImageFile
import pandas as pd
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# === 三模态数据集 ===
class MultiModalAOIDataset3Branch(Dataset):
    def __init__(self, image_dir, excel_path, transform, tokenizer):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer

        df = pd.read_excel(excel_path, header=1)
        df['label'] = df['label'].replace(-1, 0)
        df = df[['filename', 'label']].dropna()
        df['filename'] = df['filename'].astype(str)

        self.data = df[df['filename'].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        filename = row['filename']
        label = row['label']
        img_path = os.path.join(self.image_dir, filename)
        json_path = img_path.replace(".png", ".json")

        # 图像
        with open(img_path, 'rb') as f:
            image = Image.open(f).convert("RGB")
        image = self.transform(image)

        # 文本
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            text = meta.get("name", "")
        else:
            text = "Unknown"
            meta = {}

        encoded = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # 数值特征
        try:
            far = float(meta.get("FAR", 0.0))
            poi = float(meta.get("POI_Densit", 0.0))
        except:
            far, poi = 0.0, 0.0
        numeric_tensor = torch.tensor([far, poi], dtype=torch.float32)

        return image, input_ids, attention_mask, numeric_tensor, torch.tensor(label, dtype=torch.float32)

# === 三模态分类器 ===
class CLIP3BClassifier(nn.Module):
    def __init__(self, vision_encoder, text_encoder, numeric_dim=2, emb_img=768, emb_txt=512):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.numeric_proj = nn.Linear(numeric_dim, 128)

        self.classifier = nn.Sequential(
            nn.Linear(emb_img + emb_txt + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, image, input_ids, attention_mask, numeric):
        img_feat = self.vision_encoder(image).pooler_output  # [B, emb_img]
        txt_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # [B, emb_txt]
        num_feat = self.numeric_proj(numeric)  # [B, 128]
        fused = torch.cat([img_feat, txt_feat, num_feat], dim=1)
        return self.classifier(fused)

# === DataLoader ===
def get_dataloader(image_dir, excel_path, tokenizer, batch_size=8, split_ratio=0.85, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = MultiModalAOIDataset3Branch(image_dir, excel_path, transform, tokenizer)
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
        for image, input_ids, attention_mask, numeric, labels in val_loader:
            image = image.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            numeric = numeric.cuda()
            labels = labels.float().unsqueeze(1).cuda()

            logits = model(image, input_ids, attention_mask, numeric)
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

# === Top-K 结果汇总 ===
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

    print("🔍 加载 CLIP 模型...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    model = CLIP3BClassifier(
        vision_encoder=clip_model.vision_model,
        text_encoder=clip_model.text_model,
        emb_img=config.emb_dim_img,
        emb_txt=config.emb_dim_txt
    ).cuda()

    train_loader, val_loader = get_dataloader(dataset_root, excel_path, tokenizer, batch_size=config.batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    os.makedirs(f"output/{config.exp_name}", exist_ok=True)
    all_metrics = []

    for epoch in range(config.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for image, input_ids, attention_mask, numeric, labels in loop:
            image = image.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            numeric = numeric.cuda()
            labels = labels.float().unsqueeze(1).cuda()

            optimizer.zero_grad()
            logits = model(image, input_ids, attention_mask, numeric)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        val_metrics = validate_epoch(model, val_loader, criterion)
        all_metrics.append(val_metrics)
        print(f"[VAL] Epoch {epoch+1}: AUC={val_metrics['auc']:.4f}, Acc={val_metrics['acc']:.4f}, "
              f"F1={val_metrics['f1']:.4f}, Prec={val_metrics['precision']:.4f}, Rec={val_metrics['recall']:.4f}")

    # 保存指标
    with open(f"output/{config.exp_name}/metrics_all.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    top5_summary = summarize_topk(all_metrics, topk=5)
    with open(f"output/{config.exp_name}/top5_summary.json", "w") as f:
        json.dump(top5_summary, f, indent=2)

    print("\n🏆 Top-5 Epochs Summary:")
    for k, v in top5_summary.items():
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")

# === 参数入口 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="clip_3modality_fusion")
    parser.add_argument("--emb_dim_img", type=int, default=768)
    parser.add_argument("--emb_dim_txt", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    config = parser.parse_args()

    train(config)

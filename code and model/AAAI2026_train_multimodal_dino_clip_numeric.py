# train_multimodal3B_crossattention.py
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from AAAI2026_multimodal_dataset_3branch import MultiModalAOIDataset3Branch as MultiModalAOIDataset3B
from dino_finetune import DINOV2EncoderLoRA, DINOV2TextNumericCrossAttentionClassifier
# from model.dino_v2_multimodal3B_crossattention import DINOV2TextNumericCrossAttentionClassifier

def custom_collate_fn(batch):
    images, text_tokens_list, numeric_values, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    numeric_values = torch.stack(numeric_values, dim=0)
    labels = torch.tensor(labels, dtype=torch.float)

    text_tokens = {}
    for key in text_tokens_list[0].keys():
        text_tokens[key] = torch.nn.utils.rnn.pad_sequence(
            [item[key] for item in text_tokens_list], batch_first=True
        )

    return images, text_tokens, numeric_values, labels

def get_dataloader(dataset_root, excel_path, tokenizer, batch_size=16, split_ratio=0.85, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((490, 490)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = MultiModalAOIDataset3B(dataset_root, excel_path, transform=transform, tokenizer=tokenizer)

    total = len(dataset)
    train_len = int(split_ratio * total)
    val_len = total - train_len

    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)
    return train_loader, val_loader

def validate_epoch(model, val_loader, criterion, metrics):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, text_tokens, numeric_values, labels in val_loader:
            images = images.cuda()
            labels = labels.float().unsqueeze(1).cuda()
            numeric_values = numeric_values.cuda()
            text_tokens = {k: v.cuda() for k, v in text_tokens.items()}

            logits = model(images, text_tokens, numeric_values)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    acc = total_correct / total_samples
    metrics.setdefault("val_loss", []).append(avg_loss)
    metrics.setdefault("val_acc", []).append(acc)
    print(f"[VAL] Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    return acc

def train(config, encoder):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        torch_dtype=torch.float32,
        device_map=None,
        use_safetensors=True
    ).cuda()

    dataset_root = "/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/标注2000/aoi_patches_train_2000"
    excel_path = "/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/标注2000/标注2000.xlsx"

    train_loader, val_loader = get_dataloader(dataset_root, excel_path, tokenizer, batch_size=config.batch_size)

    model = DINOV2TextNumericCrossAttentionClassifier(
        encoder=encoder,
        text_encoder=text_encoder,
        r=config.r,
        emb_dim=config.emb_dim,
        use_lora=config.use_lora
    ).cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    metrics = {"train_loss": [], "val_loss": [], "val_acc": [], "train_acc": []}

    best_acc = 0.0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        correct, total = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for images, text_tokens, numeric_values, labels in loop:
            images = images.cuda()
            labels = labels.float().unsqueeze(1).cuda()
            numeric_values = numeric_values.cuda()
            text_tokens = {k: v.cuda() for k, v in text_tokens.items()}

            optimizer.zero_grad()
            logits = model(images, text_tokens, numeric_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        metrics["train_loss"].append(epoch_loss / len(train_loader))
        metrics["train_acc"].append(train_acc)
        print(f"[TRAIN] Epoch {epoch + 1}: Loss = {epoch_loss / len(train_loader):.4f}, Accuracy = {train_acc:.4f}")

        val_acc = validate_epoch(model, val_loader, criterion, metrics)

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(f"output/{config.exp_name}", exist_ok=True)
            model.save_parameters(f"output/{config.exp_name}/best.pt")
            print(f"✅ Saved Best Model (Acc={val_acc:.4f})")

    model.save_parameters(f"output/{config.exp_name}/final.pt")
    with open(f"output/{config.exp_name}/metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="multimodal3B_crossattention")
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--size", type=str, default="base")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_lora", action="store_true")
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

    base_encoder = torch.hub.load("facebookresearch/dinov2", f"dinov2_{backbones[config.size]}").cuda()
    encoder = DINOV2EncoderLoRA(base_encoder, r=config.r, emb_dim=config.emb_dim, use_lora=config.use_lora, use_fpn=False).cuda()

    train(config, encoder)

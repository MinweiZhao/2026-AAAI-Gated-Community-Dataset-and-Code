import os
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from AAAI2026_multimodal_dataset_3branch import MultiModalAOIDataset3Branch as MultiModalAOIDataset3B

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_numeric_data(dataset_root, excel_path):
    dataset = MultiModalAOIDataset3B(dataset_root, excel_path, transform=None, tokenizer=None)
    all_features = []
    all_labels = []
    for _, _, numeric, label in dataset:
        all_features.append(numeric.numpy())
        all_labels.append(label)
    return np.array(all_features), np.array(all_labels)

def evaluate(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    return {
        "loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "acc": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }

def summarize_metrics(metrics_list):
    summary = {}
    for key in metrics_list[0].keys():
        values = np.array([m[key] for m in metrics_list])
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }
    return summary

def train_mlp_kfold(config):
    dataset_root = "/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/标注2000/aoi_patches_train_2000"
    excel_path = "/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/标注2000/标注2000.xlsx"
    os.makedirs(f"output/{config.exp_name}", exist_ok=True)

    X, y = load_numeric_data(dataset_root, excel_path)
    input_dim = X.shape[1]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    all_fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(X[val_idx], dtype=torch.float32)
        y_train, y_val = torch.tensor(y[train_idx], dtype=torch.float32), torch.tensor(y[val_idx], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

        model = MLPClassifier(input_dim).cuda()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        # training
        model.train()
        for epoch in range(20):
            for xb, yb in train_loader:
                xb, yb = xb.cuda(), yb.unsqueeze(1).cuda()
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

        # validation
        model.eval()
        y_probs, y_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.cuda()
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                y_probs.extend(probs)
                y_trues.extend(yb.numpy())

        metrics = evaluate(np.array(y_trues), np.array(y_probs))
        all_fold_metrics.append(metrics)

        print(f"[Fold {fold+1}] "
              f"Acc={metrics['acc']:.4f}, AUC={metrics['auc']:.4f}, "
              f"F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

        torch.save(model.state_dict(), f"output/{config.exp_name}/fold{fold+1}.pt")

    summary = summarize_metrics(all_fold_metrics)
    with open(f"output/{config.exp_name}/top3cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n🏆 3-Fold Cross-Validation Summary:")
    for k, v in summary.items():
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="mlp_kfold")
    config = parser.parse_args()

    train_mlp_kfold(config)

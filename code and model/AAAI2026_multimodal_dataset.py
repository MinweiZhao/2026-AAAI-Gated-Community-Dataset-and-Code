# multimodal_dataset.py
import os
import pandas as pd
import json
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultiModalAOIDataset(Dataset):
    def __init__(self, image_dir, excel_path, transform=None, tokenizer=None):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer

        df = pd.read_excel(excel_path, header=1)
        df['label'] = df['label'].replace(-1, 0)
        df = df[['filename', 'label']]
        self.data = df.dropna()
        self.data['filename'] = self.data['filename'].astype(str)

        self.label_map = dict(zip(self.data['filename'], self.data['label']))
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.endswith('.png') and f in self.label_map
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        json_path = img_path.replace(".png", ".json")

        label = int(self.label_map[img_name])

        try:
            with open(img_path, 'rb') as f:
                image = Image.open(f).convert("RGB")
        except Exception as e:
            print(f"[警告] 图像加载失败: {img_path}, 错误: {e}")
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            name = meta.get("name", "")
            address = meta.get("address", "")
            text = f"{name}. Located at: {address}."
        else:
            text = "Unknown. Unknown address."

        if self.tokenizer:
            text_token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            text_token = {k: v.squeeze(0) for k, v in text_token.items()}
        else:
            text_token = None

        return image, text_token, label

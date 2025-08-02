import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class AOIDataset(Dataset):
    def __init__(self, image_dir, excel_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # 读取 Excel 文件
        df = pd.read_excel(excel_path, header=1)  # 从第二行作为表头
        print("列名如下：", df.columns.tolist())  # 👉 打印列名
        print(df['label'].value_counts(dropna=False))

        # 将 label 中为 -1 的值替换为 0
        df['label'] = df['label'].replace(-1, 0)

        # 显示修改后的统计
        print("修改后 label 值分布：")
        print(df['label'].value_counts(dropna=False))

        df = df[['filename', 'label']]  # 只提取必要列
        self.data = df.dropna()
        self.data['filename'] = self.data['filename'].astype(str)

        # 构造文件名 -> label 的映射
        self.label_map = dict(zip(self.data['filename'], self.data['label']))

        # 所有存在 label 的图像路径列表
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.endswith('.png') and f in self.label_map
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label = int(self.label_map[img_name])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# 用于测试和使用
if __name__ == "__main__":
    image_dir = "/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/标注2000/aoi_patches_train_2000"
    excel_path = "/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/标注2000/标注2000.xlsx"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = AOIDataset(image_dir=image_dir, excel_path=excel_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Example iteration
    for images, labels in dataloader:
        print(images.shape, labels)  # torch.Size([16, 3, 224, 224]) tensor([0, 1, ...])
        break

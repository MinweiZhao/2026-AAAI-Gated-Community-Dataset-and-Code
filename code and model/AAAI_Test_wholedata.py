import os
import json
import torch
import argparse
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision import transforms
from dino_finetune import DINOV2EncoderLoRA, DINOV2TextNumericCrossAttentionClassifier

@torch.no_grad()
def predict_folder(model, tokenizer, image_dir, output_file):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((490, 490)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    results = []

    image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    for img_name in tqdm(image_list, desc="Predicting"):
        img_path = os.path.join(image_dir, img_name)
        json_path = img_path.replace('.png', '.json')

        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).cuda()

            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                name = meta.get("name", "")
                text = f"{name}."
            else:
                meta = {}
                text = "Unknown."

            text_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            text_tokens = {k: v.cuda() for k, v in text_tokens.items()}

            numeric_tensor = torch.tensor([
                meta.get("FAR", 0.0),
                meta.get("POI_Densit", 0.0)
            ], dtype=torch.float32).unsqueeze(0).cuda()

            logits = model(image, text_tokens, numeric_tensor)
            prob = torch.sigmoid(logits).item()
            pred_label = 1 if prob > 0.5 else 0

        except (UnidentifiedImageError, Exception) as e:
            print(f"⚠️ 跳过损坏或异常图像: {img_name}，错误：{str(e)}")
            pred_label = -1

        results.append(f"{img_name},{pred_label}")

    with open(output_file, 'w') as f:
        for line in results:
            f.write(line + '\n')

    print(f"✅ 所有结果已保存到 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="multimodal_dino_clip_numeric_crossattention_v3")
    parser.add_argument("--image_dir", type=str, default="/home/ps/data/weimingzhang/zhongjian_Project/AAAI2026/whole_guangzhou/佛山数据集")
    parser.add_argument("--output_file", type=str, default="prediction_results_fs.txt")
    parser.add_argument("--size", type=str, default="base")
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--use_lora", action="store_true")
    args = parser.parse_args()

    # 加载模型结构和参数
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        torch_dtype=torch.float32,
        device_map=None,
        use_safetensors=True
    ).cuda()

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
    emb_dim = embedding_dims[args.size]

    base_encoder = torch.hub.load("facebookresearch/dinov2", f"dinov2_{backbones[args.size]}").cuda()
    encoder = DINOV2EncoderLoRA(base_encoder, r=args.r, emb_dim=emb_dim, use_lora=args.use_lora, use_fpn=False).cuda()

    model = DINOV2TextNumericCrossAttentionClassifier(
        encoder=encoder,
        text_encoder=text_encoder,
        r=args.r,
        emb_dim=emb_dim,
        use_lora=args.use_lora
    ).cuda()

    model_path = f"output/{args.exp_name}/best.pt"
    assert os.path.exists(model_path), f"❌ 模型文件不存在: {model_path}"
    model.load_parameters(model_path)

    # 预测并保存结果
    predict_folder(model, tokenizer, args.image_dir, args.output_file)

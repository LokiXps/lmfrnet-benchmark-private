import torch
import json
import os
from PIL import Image
from torchvision import transforms, datasets
from pathlib import Path
import sys

# モデル定義のパスを通す
sys.path.append('/home/loki/LMFRNet')

def main():
    # 設定
    model_path = "/home/loki/LMFRNet/outputs/experiment2/resnet18_s0/best_model.pt"
    samples_path = "/home/loki/public/samples.json"
    samples_dir = "/home/loki/public/samples"
    dataset_root = "/home/loki/LMFRNet/data/caltech101"

    print(f"Checking model: {model_path}")

    # 1. ラベルの整合性確認
    print("\n--- Checking Label Consistency ---")
    
    # torchvisionのCaltech101クラス定義を取得
    full_dataset = datasets.Caltech101(root=dataset_root, download=False)
    torch_classes = full_dataset.categories
    print(f"Torchvision classes: {len(torch_classes)}")
    print(f"First 5: {torch_classes[:5]}")
    
    # generate_samples.pyで生成したlabels.jsonを取得
    with open("/home/loki/public/labels.json", "r") as f:
        json_classes = json.load(f)
    print(f"JSON classes: {len(json_classes)}")
    print(f"First 5: {json_classes[:5]}")

    if torch_classes != json_classes:
        print("!!! WARNING: Class mismatch detected !!!")
        for i, (t, j) in enumerate(zip(torch_classes, json_classes)):
            if t != j:
                print(f"Mismatch at index {i}: Torch='{t}' vs JSON='{j}'")
                break
    else:
        print("Class definitions match.")

    # 2. モデル精度の検証
    print("\n--- Verifying Model Accuracy (Python) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ResNet18ロード
    from torchvision.models import resnet18
    model = resnet18(weights=None, num_classes=101)
    checkpoint = torch.load(model_path, map_location=device)
    
    # state_dictの修正
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 前処理（学習時と同じ）
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # サンプルデータの読み込み
    with open(samples_path, "r") as f:
        samples = json.load(f)

    correct = 0
    total = len(samples)

    for sample in samples:
        img_path = os.path.join(samples_dir, sample['filename'])
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            
        if predicted.item() == sample['label']:
            correct += 1
        
        # デバッグ: 最初の数枚だけ予測を表示
        if total > 0 and (correct + (total - len(samples))) <= 5: 
             pass # print(f"Img: {sample['filename']}, Label: {sample['label']}, Pred: {predicted.item()}")

    accuracy = 100 * correct / total
    print(f"\nResult: {correct}/{total} Correct")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

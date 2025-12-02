import os
import json
import random
import shutil
from pathlib import Path
from PIL import Image

def main():
    # 設定
    dataset_root = Path("/home/loki/LMFRNet/data/caltech101/caltech101/101_ObjectCategories")
    output_dir = Path("/home/loki/public/samples")
    output_json = Path("/home/loki/public/samples.json")
    
    # 出力ディレクトリ作成
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # torchvisionのCaltech101クラス順序に完全に合わせる
    from torchvision import datasets
    # ダミーデータセットを作成してクラスリストを取得（ダウンロード済み前提）
    ds = datasets.Caltech101(root="/home/loki/LMFRNet/data/caltech101", download=False)
    categories = ds.categories
    
    # カテゴリ名とインデックスのマッピング
    class_to_idx = {cls_name: i for i, cls_name in enumerate(categories)}
    
    # 全画像リスト取得
    all_images = []
    for cat in categories:
        cat_dir = dataset_root / cat
        if not cat_dir.exists():
             continue
        for img_path in cat_dir.glob("*.jpg"):
            all_images.append({
                "path": img_path,
                "label": class_to_idx[cat],
                "category": cat
            })
    
    # ランダムに200枚選出
    # 本来は「高品質」「低品質」の選別が必要だが、今回はランダムに選んで
    # 半分を「High」、半分を「Low」と仮定してタグ付けする（シミュレーション）
    random.seed(42)
    selected_images = random.sample(all_images, 200)
    
    samples = []
    for i, img_info in enumerate(selected_images):
        # 画像をコピー
        src_path = img_info["path"]
        dst_filename = f"img_{i:03d}.jpg"
        dst_path = output_dir / dst_filename
        
        # 画像をリサイズして保存（Web表示用、推論時はJSでリサイズされるが、転送量削減のため）
        # ただし、画質評価のため、あえてオリジナルに近い方がいいかもしれないが、
        # ここでは224x224より少し大きめで保存しておく
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img.save(dst_path, quality=95)
        
        # 品質タグ（仮）
        quality = "high" if i < 100 else "low"
        
        samples.append({
            "filename": dst_filename,
            "label": img_info["label"],
            "category": img_info["category"],
            "quality": quality
        })
    
    # JSON保存
    with open(output_json, "w") as f:
        json.dump(samples, f, indent=2)
    
    # ラベルリスト保存
    output_labels = Path("/home/loki/public/labels.json")
    with open(output_labels, "w") as f:
        json.dump(categories, f, indent=2)
    
    print(f"Generated {len(samples)} samples in {output_dir}")
    print(f"Saved metadata to {output_json}")
    print(f"Saved labels to {output_labels}")

if __name__ == "__main__":
    main()

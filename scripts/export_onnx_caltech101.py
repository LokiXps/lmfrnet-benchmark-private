import torch
import torchvision
import onnx
import sys
import os
import numpy as np

# Add LMFRNet to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../LMFRNet'))

def export_model(model, model_name, input_shape=(1, 3, 224, 224), output_dir="public/models_caltech101"):
    model.eval()
    dummy_input = torch.randn(*input_shape)
    # model = model.cuda() # Disable CUDA for export stability
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.onnx")
    
    print(f"Exporting {model_name} to {output_path}...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        training=torch.onnx.TrainingMode.EVAL,
    )
    
    # Force merge external data into the ONNX file
    print(f"Merging external data for {model_name}...")
    onnx_model = onnx.load(output_path)
    onnx.save(onnx_model, output_path)
    
    # Remove .data file if it exists (cleanup)
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        try:
            os.remove(data_file)
            print(f"Removed external data file: {data_file}")
        except OSError as e:
            print(f"Error removing {data_file}: {e}")
    
    print(f"Successfully exported {model_name} (single file).")
    return output_path

def main():
    # 実験2のログルート
    checkpoint_root = "/home/loki/LMFRNet/outputs/experiment2"
    output_dir = "/home/loki/public/models_caltech101"
    num_classes = 101
    
    # モデル定義
    models_config = [
        {
            "name": "lmfrnet",
            "path": os.path.join(checkpoint_root, "lmfrnet_s0/best_model.pt"),
            "type": "lmfrnet"
        },
        {
            "name": "lmfrnet_hires",
            "path": os.path.join(checkpoint_root, "lmfrnet_hires_s0/best_model.pt"),
            "type": "lmfrnet_hires"
        },
        {
            "name": "resnet18",
            "path": os.path.join(checkpoint_root, "resnet18_s0/best_model.pt"),
            "type": "resnet18"
        },
        {
            "name": "mobilenetv3_large",
            "path": os.path.join(checkpoint_root, "mobilenetv3_large_s0/best_model.pt"),
            "type": "mobilenetv3"
        }
    ]

    for config in models_config:
        print(f"\n--- Processing {config['name']} ---")
        if not os.path.exists(config['path']):
            print(f"Checkpoint not found: {config['path']}")
            continue

        try:
            # モデル作成
            if config['type'] == 'lmfrnet':
                from lmfrnet.OurLMFRNet import LMFRNet
                model = LMFRNet(num_classes=num_classes)
            elif config['type'] == 'lmfrnet_hires':
                from lmfrnet.LMFRNet_hires import LMFRNet as LMFRNetHires
                model = LMFRNetHires(num_classes=num_classes)
            elif config['type'] == 'resnet18':
                model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
            elif config['type'] == 'mobilenetv3':
                model = torchvision.models.mobilenet_v3_large(weights=None, num_classes=num_classes)
            
            # Load weights
            checkpoint = torch.load(config['path'], map_location='cpu')
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)
            model.eval()
            print(f"Loaded weights for {config['name']}")

            # --- Sanity Check ---
            print("Running sanity check on 5 samples...")
            from PIL import Image
            from torchvision import transforms
            import json
            
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            with open("/home/loki/public/samples.json", "r") as f:
                samples = json.load(f)[:5]
            
            correct = 0
            for s in samples:
                img = Image.open(os.path.join("/home/loki/public/samples", s['filename'])).convert("RGB")
                input_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    out = model(input_tensor)
                    pred = torch.argmax(out, 1).item()
                if pred == s['label']:
                    correct += 1
                # print(f"  {s['filename']}: Pred={pred}, Label={s['label']}")
            
            print(f"Sanity Check: {correct}/5 Correct")
            if correct < 3:
                print("!!! WARNING: Model accuracy seems low before export !!!")

            # Export
            export_model(model, config['name'], output_dir=output_dir)

        except Exception as e:
            print(f"Failed to export {config['name']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

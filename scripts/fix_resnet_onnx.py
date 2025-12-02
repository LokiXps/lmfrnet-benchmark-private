import torch
import torchvision
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import os
import onnx
from PIL import Image
from torchvision import transforms
import json

def main():
    print("--- ResNet18 Fix Attempt: Replace AdaptiveAvgPool ---")
    
    # 1. Model Setup
    num_classes = 101
    model = torchvision.models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # ★★★ FIX: Replace AdaptiveAvgPool2d with fixed AvgPool2d ★★★
    # ResNet18 outputs 7x7 feature maps for 224x224 input.
    # AdaptiveAvgPool2d((1,1)) is functionally equivalent to AvgPool2d(7) for this input size.
    # ONNX sometimes struggles with AdaptiveAvgPool.
    print("Replacing model.avgpool (AdaptiveAvgPool2d) with nn.AvgPool2d(7)...")
    model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
    
    # Load Weights
    weights_path = "/home/loki/LMFRNet/outputs/experiment2/resnet18_s0/best_model.pt"
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # State dict handling
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # 2. PyTorch Inference
    img_path = "/home/loki/public/samples/img_000.jpg"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        pt_out = model(input_tensor)
        pt_pred = torch.argmax(pt_out, 1).item()
        print(f"PyTorch Pred: {pt_pred}")

    # 3. ONNX Export
    onnx_path = "resnet18_fixed.onnx"
    dummy_input = torch.randn(1, 3, 224, 224)
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        training=torch.onnx.TrainingMode.EVAL
    )
    
    # Force merge external data (Single file)
    print(f"Merging external data for {onnx_path}...")
    onnx_model = onnx.load(onnx_path)
    onnx.save(onnx_model, onnx_path)
    
    # Cleanup .data file if exists
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
        print(f"Removed external data file: {data_file}")

    print(f"Exported to {onnx_path} (Single File)")

    # 4. ONNX Verification
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_pred = np.argmax(ort_outs[0])
    
    print(f"ONNX Pred: {onnx_pred}")
    
    if pt_pred == onnx_pred:
        print("\n✅ SUCCESS! The fix worked.")
        diff = np.abs(pt_out.numpy() - ort_outs[0]).max()
        print(f"Max Difference: {diff}")
    else:
        print("\n❌ FAILED. Still mismatch.")
        print(f"PyTorch: {pt_pred}, ONNX: {onnx_pred}")

if __name__ == "__main__":
    main()

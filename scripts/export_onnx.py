import torch
import torchvision
import onnx
import onnxruntime
import sys
import os
import numpy as np

# Add LMFRNet to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../LMFRNet'))

from lmfrnet.OurLMFRNet import LMFRNet

def export_model(model, model_name, input_shape=(1, 3, 32, 32), output_dir="models_onnx"):
    model.eval()
    dummy_input = torch.randn(*input_shape)
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
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
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

def verify_onnx(onnx_path, torch_model, input_shape=(1, 3, 32, 32)):
    print(f"Verifying {onnx_path}...")
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # PyTorch inference
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(dummy_input).numpy()
        
    # ONNX Runtime inference
    ort_session = onnxruntime.InferenceSession(onnx_path)
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Compare
    np.testing.assert_allclose(torch_out, ort_outs[0], rtol=1e-03, atol=1e-05)
    print(f"Verification passed for {onnx_path}!")

def main():
    output_dir = "models_onnx"
    os.makedirs(output_dir, exist_ok=True)
    
    # Base path for checkpoints
    checkpoint_root = "checkpoint"

    # 1. LMFRNet
    print("\n--- Processing LMFRNet ---")
    lmfrnet_path = os.path.join(checkpoint_root, "lmfrnet/ckpt.pth")
    if os.path.exists(lmfrnet_path):
        try:
            # The user's training script used default LMFRNet() which has num_classes=100
            model_lmfrnet = LMFRNet(num_classes=100)
            
            checkpoint = torch.load(lmfrnet_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                if 'net' in checkpoint:
                    state_dict = checkpoint['net']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
                
            model_lmfrnet.load_state_dict(new_state_dict)
            print("Loaded LMFRNet weights.")
            
            onnx_path = export_model(model_lmfrnet, "lmfrnet", output_dir=output_dir)
            verify_onnx(onnx_path, model_lmfrnet)
            
        except Exception as e:
            print(f"Failed to load/export LMFRNet: {e}")
    else:
        print(f"LMFRNet checkpoint not found at {lmfrnet_path}")

    # 2. MobileNetV3 Large
    print("\n--- Processing MobileNetV3 Large ---")
    mbv3_path = os.path.join(checkpoint_root, "mobilenetv3_large/ckpt.pth")
    if os.path.exists(mbv3_path):
        try:
            model_mbv3 = torchvision.models.mobilenet_v3_large(weights=None, num_classes=10)
            model_mbv3.features[0][0] = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            
            checkpoint = torch.load(mbv3_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                if 'net' in checkpoint:
                    state_dict = checkpoint['net']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
                
            model_mbv3.load_state_dict(new_state_dict)
            print("Loaded MobileNetV3 weights.")
            
            onnx_path = export_model(model_mbv3, "mobilenetv3_large", input_shape=(1, 3, 32, 32), output_dir=output_dir)
            verify_onnx(onnx_path, model_mbv3, input_shape=(1, 3, 32, 32))
        except Exception as e:
            print(f"Failed to export MobileNetV3: {e}")
    else:
        print(f"MobileNetV3 checkpoint not found at {mbv3_path}")

    # 3. ResNet18
    print("\n--- Processing ResNet18 ---")
    resnet_path = os.path.join(checkpoint_root, "resnet18/ckpt.pth")
    if os.path.exists(resnet_path):
        try:
            model_resnet = torchvision.models.resnet18(weights=None, num_classes=10)
            model_resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model_resnet.maxpool = torch.nn.Identity()
            
            checkpoint = torch.load(resnet_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                if 'net' in checkpoint:
                    state_dict = checkpoint['net']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
                
            model_resnet.load_state_dict(new_state_dict)
            print("Loaded ResNet18 weights.")
            
            onnx_path = export_model(model_resnet, "resnet18", input_shape=(1, 3, 32, 32), output_dir=output_dir)
            verify_onnx(onnx_path, model_resnet, input_shape=(1, 3, 32, 32))
        except Exception as e:
            print(f"Failed to export ResNet18: {e}")
    else:
        print(f"ResNet18 checkpoint not found at {resnet_path}")

if __name__ == "__main__":
    main()

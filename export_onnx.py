import os
import argparse
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from torchvision import transforms
from PIL import Image
import json
import warnings

from train import get_model

warnings.filterwarnings('ignore', category=UserWarning)


def load_pytorch_model(model_path, device):
    """Load trained PyTorch model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model_name = checkpoint['model_name']
    num_classes = checkpoint['num_classes']
    
    # Create model
    model = get_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, num_classes, model_name


def export_to_onnx(model, output_path, input_size=224, batch_size=1, 
                   opset_version=11, dynamic_batch=True):
    """Export PyTorch model to ONNX format"""
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)
    
    # Dynamic axes for variable batch size
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        dynamic_axes = None
    
    # Export model
    print(f"Exporting model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"Model exported successfully to {output_path}")
    
    return output_path


def verify_onnx_model(onnx_path):
    """Verify ONNX model"""
    print("\nVerifying ONNX model...")
    
    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Print model info
    print("\nModel Information:")
    print(f"- IR Version: {onnx_model.ir_version}")
    print(f"- Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
    print(f"- Opset Version: {onnx_model.opset_import[0].version}")
    
    # Get input/output info
    input_info = onnx_model.graph.input[0]
    output_info = onnx_model.graph.output[0]
    
    print(f"\nInput Information:")
    print(f"- Name: {input_info.name}")
    print(f"- Shape: {[dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                     for dim in input_info.type.tensor_type.shape.dim]}")
    
    print(f"\nOutput Information:")
    print(f"- Name: {output_info.name}")
    print(f"- Shape: {[dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                     for dim in output_info.type.tensor_type.shape.dim]}")
    
    return True


def compare_outputs(pytorch_model, onnx_path, input_size=224, num_tests=5):
    """Compare outputs between PyTorch and ONNX models"""
    print(f"\nComparing PyTorch and ONNX outputs with {num_tests} random inputs...")
    
    # Create ONNX runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    pytorch_model.eval()
    max_diff = 0
    avg_diff = 0
    
    for i in range(num_tests):
        # Generate random input
        dummy_input = torch.randn(1, 3, input_size, input_size)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input).numpy()
        
        # ONNX inference
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        diff = np.abs(pytorch_output - onnx_output)
        max_diff = max(max_diff, np.max(diff))
        avg_diff += np.mean(diff)
    
    avg_diff /= num_tests
    
    print(f"✓ Output comparison completed:")
    print(f"  - Maximum difference: {max_diff:.6f}")
    print(f"  - Average difference: {avg_diff:.6f}")
    
    if max_diff < 1e-3:
        print("✓ Models are numerically equivalent (difference < 1e-3)")
    elif max_diff < 1e-2:
        print("⚠ Small numerical differences detected (difference < 1e-2)")
    else:
        print("⚠ Significant numerical differences detected")
    
    return max_diff < 1e-2


def test_onnx_inference(onnx_path, test_image_path=None, input_size=224):
    """Test ONNX model inference with a real image"""
    print("\nTesting ONNX model inference...")
    
    # Create ONNX runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Prepare test image
    if test_image_path and os.path.exists(test_image_path):
        # Load and preprocess real image
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(test_image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).numpy()
        print(f"Using test image: {test_image_path}")
    else:
        # Use random input
        input_tensor = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
        print("Using random input tensor")
    
    # Run inference
    import time
    
    # Warmup
    for _ in range(5):
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        _ = ort_session.run(None, ort_inputs)
    
    # Measure inference time
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        outputs = ort_session.run(None, ort_inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
    
    # Get output
    output = outputs[0]
    probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)  # Softmax
    predicted_class = np.argmax(probabilities, axis=1)[0]
    confidence = probabilities[0, predicted_class]
    
    print(f"\n✓ Inference successful!")
    print(f"  - Average inference time: {avg_time:.2f} ms")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Predicted class: {predicted_class}")
    print(f"  - Confidence: {confidence:.2%}")
    
    return True


def optimize_onnx_model(onnx_path, optimized_path):
    """Optimize ONNX model for inference"""
    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.onnx_model import OnnxModel
        
        print("\nOptimizing ONNX model...")
        
        # Load model
        model = OnnxModel(onnx.load(onnx_path))
        
        # Optimize
        optimized_model = optimizer.optimize_model(
            onnx_path,
            model_type='bert',  # Generic optimization
            num_heads=0,
            hidden_size=0,
            opt_level=1,
            optimization_options=None,
            use_gpu=False
        )
        
        # Save optimized model
        optimized_model.save_model_to_file(optimized_path)
        
        # Check file sizes
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)  # MB
        
        print(f"✓ Model optimized successfully!")
        print(f"  - Original size: {original_size:.2f} MB")
        print(f"  - Optimized size: {optimized_size:.2f} MB")
        print(f"  - Size reduction: {(1 - optimized_size/original_size)*100:.1f}%")
        
        return optimized_path
        
    except ImportError:
        print("⚠ ONNX Runtime Transformers not installed. Skipping optimization.")
        print("  Install with: pip install onnxruntime-transformers")
        return onnx_path


def create_metadata_file(output_dir, model_info):
    """Create metadata JSON file with model information"""
    metadata = {
        "model_name": model_info["model_name"],
        "num_classes": model_info["num_classes"],
        "input_size": model_info["input_size"],
        "input_shape": [1, 3, model_info["input_size"], model_info["input_size"]],
        "output_shape": [1, model_info["num_classes"]],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "class_names": model_info.get("class_names", []),
        "export_info": {
            "framework": "PyTorch",
            "opset_version": model_info["opset_version"],
            "dynamic_batch": model_info["dynamic_batch"]
        }
    }
    
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved to {metadata_path}")
    return metadata_path


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'Using device: {device}')
    
    # Load PyTorch model
    print(f'\nLoading PyTorch model from {args.model_path}...')
    pytorch_model, num_classes, model_name = load_pytorch_model(args.model_path, device)
    print(f'✓ Model loaded: {model_name} with {num_classes} classes')
    
    # Move model to CPU for ONNX export
    pytorch_model = pytorch_model.cpu()
    
    # Export to ONNX
    onnx_path = export_to_onnx(
        pytorch_model,
        args.output_path,
        input_size=args.input_size,
        batch_size=args.batch_size,
        opset_version=args.opset_version,
        dynamic_batch=args.dynamic_batch
    )
    
    # Verify ONNX model
    if verify_onnx_model(onnx_path):
        print("✓ ONNX model verification passed")
    
    # Compare outputs
    if args.verify_outputs:
        if compare_outputs(pytorch_model, onnx_path, args.input_size):
            print("✓ Output comparison passed")
    
    # Test inference
    if args.test_image:
        test_onnx_inference(onnx_path, args.test_image, args.input_size)
    else:
        test_onnx_inference(onnx_path, None, args.input_size)
    
    # Optimize model
    if args.optimize:
        optimized_path = args.output_path.replace('.onnx', '_optimized.onnx')
        optimized_path = optimize_onnx_model(onnx_path, optimized_path)
    
    # Create metadata file
    if args.save_metadata:
        output_dir = os.path.dirname(args.output_path) or '.'
        model_info = {
            "model_name": model_name,
            "num_classes": num_classes,
            "input_size": args.input_size,
            "opset_version": args.opset_version,
            "dynamic_batch": args.dynamic_batch
        }
        
        # Load class names if available
        if args.class_names_file and os.path.exists(args.class_names_file):
            with open(args.class_names_file, 'r') as f:
                model_info["class_names"] = [line.strip() for line in f.readlines()]
        
        create_metadata_file(output_dir, model_info)
    
    print(f"\n✅ Export completed successfully!")
    print(f"ONNX model saved to: {onnx_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained PyTorch model (.pth)')
    parser.add_argument('--output_path', type=str, default='model.onnx',
                       help='Output path for ONNX model')
    parser.add_argument('--input_size', type=int, default=112,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for export')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset version')
    parser.add_argument('--dynamic_batch', action='store_true', default=True,
                       help='Enable dynamic batch size')
    parser.add_argument('--verify_outputs', action='store_true', default=True,
                       help='Verify outputs match between PyTorch and ONNX')
    parser.add_argument('--test_image', type=str, 
                       help='Test image path for inference testing')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize ONNX model for inference')
    parser.add_argument('--save_metadata', action='store_true', default=True,
                       help='Save model metadata to JSON file')
    parser.add_argument('--class_names_file', type=str,
                       help='Path to class names file')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU mode')
    
    args = parser.parse_args()
    main(args)

import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from train import get_model


def load_model(model_path, device):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model info from checkpoint
    model_name = checkpoint['model_name']
    num_classes = checkpoint['num_classes']
    
    # Create model
    model = get_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, num_classes


def preprocess_image(image_path, input_size=224):
    """Preprocess single image"""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image, image_tensor


def predict(model, image_tensor, device, top_k=5):
    """Make prediction on single image"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
    # Get top k predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    return top_probs, top_indices


def visualize_prediction(image, predictions, class_names=None, save_path=None):
    """Visualize prediction results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image')
    
    # Display predictions
    probs, indices = predictions
    y_pos = np.arange(len(probs))
    
    if class_names:
        labels = [class_names[idx] for idx in indices]
    else:
        labels = [f'Class {idx}' for idx in indices]
    
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.set_title('Top Predictions')
    
    # Add probability values on bars
    for i, (prob, label) in enumerate(zip(probs, labels)):
        ax2.text(prob, i, f'{prob:.2%}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f'Visualization saved to {save_path}')
    
    plt.show()


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.model_path}...')
    model, num_classes = load_model(args.model_path, device)
    print(f'Model loaded successfully! Number of classes: {num_classes}')
    
    # Load class names if provided
    class_names = None
    if args.class_names_file and os.path.exists(args.class_names_file):
        with open(args.class_names_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    
    # Process single image or directory
    if os.path.isfile(args.image_path):
        # Single image prediction
        print(f'\nProcessing image: {args.image_path}')
        image, image_tensor = preprocess_image(args.image_path, args.input_size)
        
        # Make prediction
        top_probs, top_indices = predict(model, image_tensor, device, args.top_k)
        
        # Print results
        print('\nTop predictions:')
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            if class_names:
                print(f'{i+1}. {class_names[idx]}: {prob:.2%}')
            else:
                print(f'{i+1}. Class {idx}: {prob:.2%}')
        
        # Visualize
        if args.visualize:
            visualize_prediction(image, (top_probs, top_indices), class_names, args.save_visualization)
    
    elif os.path.isdir(args.image_path):
        # Process directory
        image_files = [f for f in os.listdir(args.image_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        print(f'\nProcessing {len(image_files)} images from directory...')
        
        results = []
        for img_file in image_files:
            img_path = os.path.join(args.image_path, img_file)
            image, image_tensor = preprocess_image(img_path, args.input_size)
            top_probs, top_indices = predict(model, image_tensor, device, 1)
            
            pred_class = top_indices[0]
            pred_prob = top_probs[0]
            
            if class_names:
                pred_label = class_names[pred_class]
            else:
                pred_label = f'Class {pred_class}'
            
            results.append({
                'image': img_file,
                'prediction': pred_label,
                'confidence': pred_prob
            })
            
            print(f'{img_file}: {pred_label} ({pred_prob:.2%})')
        
        # Save results to file
        if args.save_results:
            import json
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f'\nResults saved to {args.save_results}')
    
    else:
        print(f'Error: {args.image_path} is neither a file nor a directory')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo for image classification')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--class_names_file', type=str, help='Path to class names file')
    parser.add_argument('--top_k', type=int, default=5, help='Show top k predictions')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--save_visualization', type=str, help='Path to save visualization')
    parser.add_argument('--save_results', type=str, help='Path to save results (for directory processing)')
    
    args = parser.parse_args()
    main(args)
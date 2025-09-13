import os
import argparse
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

from dataset import ImageClassificationDataset, get_data_transforms
from train import get_model
from torch.utils.data import DataLoader


def load_model(model_path, device):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model_name = checkpoint['model_name']
    num_classes = checkpoint['num_classes']
    
    model = get_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, num_classes


def evaluate_model(model, test_loader, device, num_classes):
    """Evaluate model on test set"""
    all_predictions = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # ROC AUC for multi-class
    if num_classes == 2:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc)
        },
        'per_class': {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist()
        },
        'confusion_matrix': cm.tolist()
    }
    
    return metrics, all_predictions, all_labels, cm


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f'Confusion matrix saved to {save_path}')
    
    plt.show()


def plot_per_class_metrics(metrics, class_names=None, save_path=None):
    """Plot per-class metrics"""
    precision = metrics['per_class']['precision']
    recall = metrics['per_class']['recall']
    f1 = metrics['per_class']['f1_score']
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(precision))]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision')
    bars2 = ax.bar(x, recall, width, label='Recall')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score')
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f'Per-class metrics plot saved to {save_path}')
    
    plt.show()


def generate_report(metrics, class_names=None, save_path=None):
    """Generate evaluation report"""
    report = []
    report.append("=" * 60)
    report.append("MODEL EVALUATION REPORT")
    report.append("=" * 60)
    
    # Overall metrics
    report.append("\nOVERALL METRICS:")
    report.append("-" * 30)
    for metric, value in metrics['overall'].items():
        report.append(f"{metric.upper():<15}: {value:.4f}")
    
    # Per-class metrics
    report.append("\n\nPER-CLASS METRICS:")
    report.append("-" * 30)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(metrics['per_class']['precision']))]
    
    report.append(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    report.append("-" * 56)
    
    for i, class_name in enumerate(class_names):
        precision = metrics['per_class']['precision'][i]
        recall = metrics['per_class']['recall'][i]
        f1 = metrics['per_class']['f1_score'][i]
        report.append(f"{class_name:<20} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    # Confusion matrix summary
    cm = np.array(metrics['confusion_matrix'])
    report.append("\n\nCONFUSION MATRIX SUMMARY:")
    report.append("-" * 30)
    report.append(f"Total Samples: {cm.sum()}")
    report.append(f"Correct Predictions: {cm.diagonal().sum()}")
    report.append(f"Incorrect Predictions: {cm.sum() - cm.diagonal().sum()}")
    
    # Most confused classes
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(cm_normalized, 0)
    
    report.append("\n\nMOST CONFUSED PAIRS:")
    report.append("-" * 30)
    
    confusion_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((cm[i, j], i, j))
    
    confusion_pairs.sort(reverse=True)
    for count, true_idx, pred_idx in confusion_pairs[:5]:
        true_class = class_names[true_idx]
        pred_class = class_names[pred_idx]
        report.append(f"{true_class} -> {pred_class}: {count} samples")
    
    report.append("\n" + "=" * 60)
    
    # Print report
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to {save_path}")
    
    return report_text


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.model_path}...')
    model, num_classes = load_model(args.model_path, device)
    print(f'Model loaded successfully! Number of classes: {num_classes}')
    
    # Create test dataset
    _, val_transform = get_data_transforms(args.input_size, data_augmentation=False)
    test_dataset = ImageClassificationDataset(
        args.data_dir,
        transform=val_transform,
        train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f'Test samples: {len(test_dataset)}')
    
    # Load class names
    class_names = test_dataset.classes
    
    # Evaluate model
    print('\nEvaluating model...')
    metrics, predictions, labels, cm = evaluate_model(model, test_loader, device, num_classes)
    
    # Generate and save classification report
    print('\n' + '=' * 60)
    print('CLASSIFICATION REPORT')
    print('=' * 60)
    print(classification_report(labels, predictions, target_names=class_names))
    
    # Generate custom report
    report = generate_report(metrics, class_names, 
                           os.path.join(args.output_dir, 'evaluation_report.txt'))
    
    # Save metrics to JSON
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {os.path.join(args.output_dir, 'metrics.json')}")
    
    # Plot confusion matrix
    if args.plot:
        plot_confusion_matrix(cm, class_names, 
                            os.path.join(args.output_dir, 'confusion_matrix.png'))
        
        # Plot per-class metrics
        plot_per_class_metrics(metrics, class_names,
                             os.path.join(args.output_dir, 'per_class_metrics.png'))
    
    # Save predictions
    if args.save_predictions:
        predictions_data = {
            'predictions': predictions.tolist(),
            'true_labels': labels.tolist(),
            'class_names': class_names
        }
        with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
            json.dump(predictions_data, f, indent=2)
        print(f"Predictions saved to {os.path.join(args.output_dir, 'predictions.json')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate image classification model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', 
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--input_size', type=int, default=112, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save_predictions', action='store_true', 
                       help='Save predictions to file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
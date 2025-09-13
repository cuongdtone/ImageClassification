import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dataset import create_data_loaders
from losses import get_loss_function


def get_model(model_name, num_classes, pretrained=True):
    """Get model from torchvision"""
    
    model_dict = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'mobilenet_v3_large': models.mobilenet_v3_large,
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not supported")
    
    # Load pretrained model
    model = model_dict[model_name](pretrained=pretrained)
    
    # Modify the final layer for our number of classes
    if 'resnet' in model_name:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif 'vgg' in model_name:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif 'densenet' in model_name:
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    elif 'efficientnet' in model_name:
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif 'mobilenet' in model_name:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    train_loader, val_loader, num_classes = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size
    )
    print(f'Number of classes: {num_classes}')
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Save class names to file
    class_names = train_loader.dataset.classes
    with open('class_names.txt', 'w', encoding='utf-8') as f:
        for class_id, class_name in enumerate(class_names):
            f.write(f'{class_id}: {class_name}\n')
    print(f'Saved class names to class_names.txt')
    
    # Analyze class data distribution
    print('\nClass distribution analysis:')
    class_counts = {}
    for label in train_loader.dataset.labels:
        class_name = class_names[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for class_id, class_name in enumerate(class_names):
        count = class_counts.get(class_name, 0)
        print(f'Class {class_id} ({class_name}): {count} images')
    
    # Create model
    model = get_model(args.model, num_classes, pretrained=args.pretrained)
    model = model.to(device)
    
    # Loss function
    criterion = get_loss_function(args.loss_type, num_classes=num_classes)
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    else:
        scheduler = None
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Tensorboard writer
    writer = SummaryWriter(f'runs/{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f'\nEpoch [{epoch+1}/{args.epochs}]')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print results
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': num_classes,
                'model_name': args.model
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f'Saved best model with accuracy: {best_acc:.2f}%')
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'num_classes': num_classes,
            'model_name': args.model
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest_model.pth'))
    
    writer.close()
    print(f'\nTraining completed! Best validation accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--model', type=str, default='mobilenet_v3_small', help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'none'])
    parser.add_argument('--loss_type', type=str, default='cross_entropy', 
                       choices=['cross_entropy', 'focal', 'label_smoothing'])
    parser.add_argument('--input_size', type=int, default=112, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3
"""
Universal train-test split script compatible with all OS
"""
import os
import shutil
from pathlib import Path
import random
import argparse

def train_test_split(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train/val/test sets"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Handle existing output directory
    if output_path.exists():
        response = input(f"'{output_dir}' exists. Delete (D) or Skip (S)? ").upper()
        if response == 'D':
            shutil.rmtree(output_path)
        else:
            print("Exiting...")
            return
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Create class subdirectories
        for split in ['train', 'val', 'test']:
            (output_path / split / class_name).mkdir(exist_ok=True)
        
        # Get all files in class directory
        files = list(class_dir.glob('*'))
        files = [f for f in files if f.is_file()]  # Only files
        random.shuffle(files)
        
        # Split files
        total = len(files)
        train_end = int(train_ratio * total)
        val_end = int((train_ratio + val_ratio) * total)
        
        # Copy files to respective directories
        for i, file in enumerate(files):
            if i < train_end:
                split_dir = 'train'
            elif i < val_end:
                split_dir = 'val'
            else:
                split_dir = 'test'
            
            shutil.copy2(file, output_path / split_dir / class_name / file.name)
        
        print(f"✓ {class_name}: {train_end} train, {val_end-train_end} val, {total-val_end} test")
    
    print(f"🎉 Dataset split completed! Output: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--input', required=True, help='Input dataset directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--train', type=float, default=0.8, help='Train ratio (default: 0.8)')
    parser.add_argument('--val', type=float, default=0.1, help='Validation ratio (default: 0.1)')
    parser.add_argument('--test', type=float, default=0.1, help='Test ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train + args.val + args.test - 1.0) > 0.001:
        print("❌ Error: Train + Val + Test ratios must sum to 1.0")
        return
    
    train_test_split(args.input, args.output, args.train, args.val, args.test)

if __name__ == '__main__':
    main()
    
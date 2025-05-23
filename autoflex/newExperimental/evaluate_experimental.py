#!/usr/bin/env python3
"""
Evaluation script for experimental vision transformers.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import os
import sys
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import experimental models
from experimental_vit import create_experimental_vit


def evaluate_model(
    architecture: str,
    checkpoint_path: str,
    data_root: str,
    batch_size: int = 128,
    img_size: int = 224,
    num_workers: int = 4
):
    """Evaluate a trained experimental model"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy from training: {checkpoint['val_acc']:.4f}")
    
    # Get number of classes and image size from checkpoint
    num_classes = checkpoint.get('num_classes', 1000)
    saved_img_size = checkpoint.get('img_size', img_size)
    
    # Create model
    model = create_experimental_vit(
        architecture_type=architecture,
        img_size=saved_img_size,
        num_classes=num_classes
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create validation transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(saved_img_size * 1.14)),
        transforms.CenterCrop(saved_img_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load validation dataset
    val_path = Path(data_root) / 'val'
    if val_path.exists():
        val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    else:
        # Try CIFAR-100 or CIFAR-10
        if 'cifar100' in str(data_root).lower():
            print("Using CIFAR-100 validation set")
            val_dataset = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
        else:
            print("Using CIFAR-10 validation set")
            val_dataset = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Evaluate
    print(f"\nEvaluating {architecture} model on validation set...")
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Top-5 accuracy
    top5_accuracy = top_k_accuracy_score(all_labels, all_probs, k=5)
    
    # Per-class accuracy
    per_class_correct = {}
    per_class_total = {}
    
    for label, pred in zip(all_labels, all_predictions):
        if label not in per_class_total:
            per_class_total[label] = 0
            per_class_correct[label] = 0
        
        per_class_total[label] += 1
        if label == pred:
            per_class_correct[label] += 1
    
    per_class_accuracy = {
        cls: per_class_correct[cls] / per_class_total[cls] 
        for cls in per_class_total
    }
    
    # Print results
    print("\n" + "="*60)
    print(f"Evaluation Results for {architecture}")
    print("="*60)
    print(f"Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
    print(f"Average per-class accuracy: {np.mean(list(per_class_accuracy.values())):.4f}")
    print(f"Min per-class accuracy: {min(per_class_accuracy.values()):.4f}")
    print(f"Max per-class accuracy: {max(per_class_accuracy.values()):.4f}")
    print("="*60)
    
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'per_class_accuracy': per_class_accuracy
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate experimental vision transformers')
    parser.add_argument('--architecture', type=str, required=True,
                       choices=['fourier', 'elfatt', 'mamba', 'kan', 'hybrid', 'mixed'],
                       help='Architecture type to evaluate')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory containing checkpoints')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--checkpoint-name', type=str, default='best_model_final.pt',
                       help='Name of checkpoint file to load')
    
    args = parser.parse_args()
    
    # Find checkpoint
    checkpoint_path = Path(args.checkpoint_dir) / args.architecture / args.checkpoint_name
    
    if not checkpoint_path.exists():
        # Try alternative names
        alt_path = Path(args.checkpoint_dir) / args.architecture / f"{args.architecture}_exp_best.pt"
        if alt_path.exists():
            checkpoint_path = alt_path
        else:
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            print(f"Available files in {Path(args.checkpoint_dir) / args.architecture}:")
            for f in (Path(args.checkpoint_dir) / args.architecture).glob("*.pt"):
                print(f"  - {f.name}")
            return
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Evaluate the model
    results = evaluate_model(
        architecture=args.architecture,
        checkpoint_path=str(checkpoint_path),
        data_root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Enhanced training script for experimental vision transformers with:
- Early stopping
- Model saving to ../../../experimentalmodels
- Proper checkpoint management
- Support for evaluation with perturbations
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from pathlib import Path
import os
import sys
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import experimental models
from experimental_vit import create_experimental_vit

# Import early stopping trainer
from early_stopping_trainer import EarlyStoppingTrainer


def get_data_loaders(data_root: str, batch_size: int, img_size: int = 224, num_workers: int = 4):
    """Create train and validation data loaders for Tiny-ImageNet"""
    
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load datasets
    train_path = Path(data_root) / 'train'
    val_path = Path(data_root) / 'val'
    
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset.classes)


def train_experimental_model(
    architecture: str,
    data_root: str,
    save_dir: str = "../../../experimentalmodels",
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.05,
    img_size: int = 224,
    warmup_epochs: int = 5,
    wandb_project: str = "experimental-vit",
    patience: int = 10,
    min_delta: float = 1e-4,
    cleanup_checkpoints: bool = True,
):
    """Train experimental model with early stopping and proper saving"""
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loaders
    train_loader, val_loader, num_classes = get_data_loaders(data_root, batch_size, img_size)
    print(f"Dataset loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Number of classes: {num_classes}")
    
    # Architecture-specific configurations
    arch_configs = {
        'fourier': {'batch_size': 128, 'lr': 1e-3},
        'elfatt': {'batch_size': 128, 'lr': 1e-3},
        'mamba': {'batch_size': 64, 'lr': 5e-4},
        'kan': {'batch_size': 64, 'lr': 5e-4},
        'hybrid': {'batch_size': 96, 'lr': 8e-4},
        'mixed': {'batch_size': 96, 'lr': 8e-4},
    }
    
    # Override with architecture-specific settings if available
    if architecture in arch_configs:
        arch_config = arch_configs[architecture]
        batch_size = arch_config.get('batch_size', batch_size)
        learning_rate = arch_config.get('lr', learning_rate)
        print(f"Using architecture-specific config: batch_size={batch_size}, lr={learning_rate}")
    
    # Create model
    model = create_experimental_vit(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        attention_type=architecture
    ).to(device)
    
    print(f"Model created: {architecture}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup
    def get_lr_scale(epoch, warmup_epochs, total_epochs):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: get_lr_scale(epoch, warmup_epochs, epochs)
    )
    
    # Create save directory structure
    save_path = Path(save_dir) / architecture
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project=wandb_project,
        name=f"{architecture}_tinyimagenet_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "architecture": architecture,
            "dataset": "tiny-imagenet-200",
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "img_size": img_size,
            "patience": patience,
            "min_delta": min_delta,
        }
    )
    
    # Initialize early stopping trainer
    trainer = EarlyStoppingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        save_dir=str(checkpoint_path),
        model_name=f"{architecture}_exp",
        cleanup_checkpoints=cleanup_checkpoints,
        verbose=True
    )
    
    # Train with early stopping
    print(f"\nStarting training for {architecture}...")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Minimum delta: {min_delta}")
    print(f"Save directory: {save_path}")
    
    results = trainer.train(
        epochs=epochs,
        criterion=criterion,
        log_wandb=True,
        log_prefix=f"{architecture}/"
    )
    
    # Save final best model in multiple formats
    if results['best_model_path']:
        # Load best checkpoint
        checkpoint = torch.load(results['best_model_path'], map_location='cpu')
        
        # Save main model file
        best_model_path = save_path / "best_model.pt"
        final_checkpoint = {
            'architecture': architecture,
            'epoch': checkpoint['epoch'],
            'model_state_dict': checkpoint['model_state_dict'],
            'val_acc': checkpoint['val_acc'],
            'val_loss': checkpoint['val_loss'],
            'num_classes': num_classes,
            'img_size': img_size,
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'warmup_epochs': warmup_epochs,
            }
        }
        torch.save(final_checkpoint, best_model_path)
        print(f"Best model saved to: {best_model_path}")
        
        # Save model for easy loading
        model_only_path = save_path / "model_state_dict.pt"
        torch.save(checkpoint['model_state_dict'], model_only_path)
        
        # Save training info
        info_path = save_path / "training_info.json"
        training_info = {
            'architecture': architecture,
            'best_epoch': checkpoint['epoch'],
            'best_val_acc': checkpoint['val_acc'],
            'best_val_loss': checkpoint['val_loss'],
            'total_epochs': results['epoch'],
            'early_stopped': results['early_stopped'],
            'train_time': results.get('total_time', 0),
            'dataset': 'tiny-imagenet-200',
            'num_classes': num_classes,
            'model_params': sum(p.numel() for p in model.parameters()),
            'save_date': datetime.now().isoformat(),
        }
        
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"Training info saved to: {info_path}")
    
    # Close wandb run
    wandb.finish()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train experimental vision transformers')
    parser.add_argument('--architecture', type=str, required=True,
                       choices=['fourier', 'elfatt', 'mamba', 'kan', 'hybrid', 'mixed'],
                       help='Architecture type to train')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to Tiny-ImageNet dataset root directory')
    parser.add_argument('--save-dir', type=str, default='../../../experimentalmodels',
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--wandb-project', type=str, default='experimental-vit',
                       help='Weights & Biases project name')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                       help='Minimum improvement delta for early stopping')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Keep all checkpoints (no cleanup)')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    results = train_experimental_model(
        architecture=args.architecture,
        data_root=args.data_root,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        warmup_epochs=args.warmup_epochs,
        wandb_project=args.wandb_project,
        patience=args.patience,
        min_delta=args.min_delta,
        cleanup_checkpoints=not args.no_cleanup,
    )
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Total epochs: {results['epoch']}")
    print(f"Early stopped: {results['early_stopped']}")


if __name__ == "__main__":
    main()
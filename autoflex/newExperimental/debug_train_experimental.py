#!/usr/bin/env python3
"""
Debug training script for experimental vision transformers.
Runs with batch size 1 and minimal samples for quick testing.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import debug configuration
from debug_config import setup_debug_environment, get_debug_config, DebugConfig

# Setup debug environment before importing other modules
setup_debug_environment()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Optional, Dict, Any
from tqdm import tqdm

# Import experimental models
from experimental_vit import create_experimental_vit
from early_stopping_trainer import EarlyStoppingTrainer


class DebugDataLoader:
    """Create debug data loaders with limited samples"""
    
    def __init__(self, data_root: str, debug_config: DebugConfig, img_size: int = 224):
        self.data_root = Path(data_root)
        self.debug_config = debug_config
        self.img_size = img_size
        
        # Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def get_transforms(self):
        """Get simplified transforms for debug mode"""
        # Simplified transforms for faster debugging
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        return train_transform, val_transform
    
    def load_dataset(self):
        """Load dataset and limit samples for debug mode"""
        train_transform, val_transform = self.get_transforms()
        
        # Try different dataset formats
        train_path = self.data_root / 'train'
        val_path = self.data_root / 'val'
        
        if train_path.exists() and val_path.exists():
            print(f"Loading ImageFolder dataset from {self.data_root}")
            train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
            val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
        else:
            # Try CIFAR-10 as fallback for testing
            print("Loading CIFAR-10 dataset for debug testing...")
            train_dataset = datasets.CIFAR10(
                str(self.data_root), 
                train=True, 
                download=True, 
                transform=train_transform
            )
            val_dataset = datasets.CIFAR10(
                str(self.data_root), 
                train=False, 
                download=True, 
                transform=val_transform
            )
        
        # Limit dataset size for debug mode
        train_indices = list(range(min(len(train_dataset), self.debug_config.num_samples_train)))
        val_indices = list(range(min(len(val_dataset), self.debug_config.num_samples_val)))
        
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        
        return train_subset, val_subset, len(train_dataset.classes)
    
    def get_loaders(self):
        """Get debug data loaders"""
        train_dataset, val_dataset, num_classes = self.load_dataset()
        
        loader_config = self.debug_config.get_dataloader_config()
        
        train_loader = DataLoader(train_dataset, **loader_config)
        val_loader = DataLoader(val_dataset, **{**loader_config, 'shuffle': False})
        
        return train_loader, val_loader, num_classes


def debug_train_experimental(
    architecture: str,
    data_root: str,
    checkpoint_dir: str = "debug_checkpoints"
):
    """Train experimental model in debug mode"""
    
    # Get debug configuration
    debug_cfg = get_debug_config()
    debug_cfg.print_debug_info()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Create debug data loaders
    debug_loader = DebugDataLoader(data_root, debug_cfg)
    train_loader, val_loader, num_classes = debug_loader.get_loaders()
    
    print(f"\n📊 Dataset info:")
    print(f"  - Training samples: {len(train_loader.dataset)}")
    print(f"  - Validation samples: {len(val_loader.dataset)}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Batch size: {debug_cfg.batch_size}")
    
    # Create model
    print(f"\n🏗️  Creating {architecture} model...")
    try:
        model = create_experimental_vit(
            architecture_type=architecture,
            img_size=debug_loader.img_size,
            num_classes=num_classes
        )
        model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Get model configuration
    model_config = debug_cfg.get_model_config('experimental')
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_config['learning_rate'],
        weight_decay=model_config['weight_decay']
    )
    
    # Simple scheduler for debug mode
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader) * debug_cfg.epochs
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir) / architecture
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    print(f"\n🚀 Starting debug training...")
    trainer = EarlyStoppingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=model_config['patience'],
        min_delta=1e-3,
        restore_best_weights=True,
        save_dir=str(checkpoint_path),
        model_name=f"{architecture}_debug",
        cleanup_checkpoints=False,  # Keep all checkpoints in debug mode
        verbose=True
    )
    
    # Train model
    try:
        results = trainer.train(
            epochs=debug_cfg.epochs,
            criterion=criterion,
            log_wandb=False,  # Disabled in debug mode
            log_prefix=f"{architecture}_debug/"
        )
        
        print(f"\n✅ Debug training completed successfully!")
        print(f"📊 Results:")
        print(f"  - Best validation accuracy: {results['best_val_acc']:.4f}")
        print(f"  - Best validation loss: {results['best_val_loss']:.4f}")
        print(f"  - Total epochs: {results['total_epochs']}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save debug checkpoint
    debug_checkpoint = {
        'architecture': architecture,
        'model_state_dict': model.state_dict(),
        'results': results,
        'debug_config': {
            'batch_size': debug_cfg.batch_size,
            'num_samples_train': debug_cfg.num_samples_train,
            'num_samples_val': debug_cfg.num_samples_val,
            'epochs': debug_cfg.epochs
        }
    }
    
    debug_path = checkpoint_path / 'debug_final.pt'
    torch.save(debug_checkpoint, debug_path)
    print(f"\n💾 Debug checkpoint saved to: {debug_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Debug training for experimental vision transformers')
    parser.add_argument('--architecture', type=str, default='fourier',
                       choices=['fourier', 'elfatt', 'mamba', 'kan', 'hybrid', 'mixed'],
                       help='Architecture type to train')
    parser.add_argument('--data-root', type=str, default='../cifar10',
                       help='Path to dataset root directory')
    parser.add_argument('--checkpoint-dir', type=str, default='debug_checkpoints',
                       help='Directory to save debug checkpoints')
    
    args = parser.parse_args()
    
    print(f"\n🔧 Debug Training: {args.architecture}")
    print("=" * 60)
    
    # Run debug training
    debug_train_experimental(
        architecture=args.architecture,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Debug training script for quick testing with minimal resources.
Runs with batch size 1 and limited dataset for fast iteration and testing.
Perfect for conda/pyenv environments and quick validation.
"""

import argparse
import torch
import os
import sys
from pathlib import Path
from debug_config import debug_config, is_debug_mode, setup_debug_environment
from flexible_models import create_model, BACKBONE_CONFIGS
from new_new import TinyImageNetDataset, set_seed
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

def create_debug_dataset(dataset_path: str, is_train: bool = True, limit: int = None, backbone_name: str = "vit_small"):
    """Create a limited dataset for debug mode"""
    if limit is None:
        limit = debug_config.num_samples_train if is_train else debug_config.num_samples_val
    
    # Determine image size based on backbone type
    # Custom ViT models (vit_small, vit_base) expect 64x64
    # All timm models are configured for 64x64 in this codebase
    # ResNet and VGG models expect 224x224
    if backbone_name in ['vit_small', 'vit_base'] or any(name in backbone_name for name in ['deit', 'swin']):
        img_size = 64
    else:
        img_size = 224
    
    # Standard transforms for debug mode
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create dataset
    try:
        dataset = TinyImageNetDataset(
            root_dir=dataset_path,
            split='train' if is_train else 'val',
            transform=transform
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Make sure the dataset path is correct and contains tiny-imagenet-200 data")
        sys.exit(1)
    
    # Limit dataset size for debug mode
    if len(dataset) > limit:
        indices = list(range(min(limit, len(dataset))))
        dataset = Subset(dataset, indices)
        print(f"🐛 Limited dataset to {len(dataset)} samples")
    
    return dataset

def debug_train_classification(
    dataset_path: str = "../tiny-imagenet-200",
    backbone_name: str = "vit_small",
    verbose: bool = True
):
    """
    Debug training for classification model with minimal resources.
    
    Args:
        dataset_path: Path to tiny-imagenet-200 dataset
        backbone_name: Backbone to use for training
        verbose: Whether to print detailed information
    """
    # Setup debug environment
    setup_debug_environment()
    set_seed(42)
    
    if verbose:
        debug_config.print_debug_info()
        print(f"\n🎯 Training {backbone_name} classification model in DEBUG MODE")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    
    # Create model
    try:
        model = create_model('classification', backbone_name, num_classes=200)
        model.to(device)
        print(f"✅ Model created: {backbone_name}")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False
    
    # Create debug datasets
    print(f"\n📂 Loading dataset from: {dataset_path}")
    train_dataset = create_debug_dataset(dataset_path, is_train=True, backbone_name=backbone_name)
    val_dataset = create_debug_dataset(dataset_path, is_train=False, backbone_name=backbone_name)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        **debug_config.get_dataloader_config()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=debug_config.batch_size,
        num_workers=debug_config.num_workers,
        pin_memory=debug_config.pin_memory,
        shuffle=False
    )
    
    print(f"📊 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Setup optimizer and loss
    config = debug_config.get_model_config('classification')
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\n🚀 Starting debug training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        print(f"\n📈 Epoch {epoch + 1}/{config['epochs']}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (data, targets) in enumerate(train_pbar):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            for data, targets in val_pbar:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"🎯 Epoch {epoch + 1} Results:")
        print(f"   Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"   Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
    
    print(f"\n✅ Debug training completed successfully!")
    print(f"🎉 Final validation accuracy: {val_acc:.2f}%")
    
    return True

def debug_test_all_backbones(dataset_path: str = "../tiny-imagenet-200"):
    """Test all available backbones in debug mode"""
    print("🔧 Testing all backbones in DEBUG MODE")
    print("=" * 60)
    
    # Get available backbones
    available_backbones = list(BACKBONE_CONFIGS.keys())
    print(f"📋 Found {len(available_backbones)} backbones to test")
    
    results = {}
    
    for backbone in available_backbones:
        print(f"\n🧪 Testing backbone: {backbone}")
        print("-" * 40)
        
        try:
            success = debug_train_classification(
                dataset_path=dataset_path,
                backbone_name=backbone,
                verbose=False
            )
            results[backbone] = "✅ PASS" if success else "❌ FAIL"
            print(f"Result: {results[backbone]}")
        except Exception as e:
            results[backbone] = f"❌ ERROR: {str(e)[:50]}..."
            print(f"Result: {results[backbone]}")
    
    # Print summary
    print(f"\n📊 Debug Test Summary")
    print("=" * 60)
    for backbone, result in results.items():
        print(f"{backbone:<25}: {result}")
    
    passed = sum(1 for r in results.values() if "✅" in r)
    total = len(results)
    print(f"\n🎯 Passed: {passed}/{total} ({100*passed/total:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Debug training with minimal resources")
    parser.add_argument("--dataset", type=str, default="../tiny-imagenet-200",
                        help="Path to tiny-imagenet-200 dataset")
    parser.add_argument("--backbone", type=str, default="vit_small",
                        help="Backbone to train (default: vit_small)")
    parser.add_argument("--test-all", action="store_true",
                        help="Test all available backbones")
    parser.add_argument("--list-backbones", action="store_true",
                        help="List all available backbones")
    
    args = parser.parse_args()
    
    if args.list_backbones:
        print("📋 Available backbones:")
        for backbone in sorted(BACKBONE_CONFIGS.keys()):
            print(f"  - {backbone}")
        return
    
    if args.test_all:
        debug_test_all_backbones(args.dataset)
    else:
        debug_train_classification(args.dataset, args.backbone)

if __name__ == "__main__":
    main()
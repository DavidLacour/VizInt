#!/usr/bin/env python3
"""
Main script for training and evaluating all models on CIFAR-10 dataset
Including: ViT, ResNet baselines, TTT, BlendedTTT, TTT3fc, and BlendedTTT3fc models
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from copy import deepcopy
import argparse
from datetime import datetime

# Import model architectures
from vit_implementation import create_vit_model
from baseline_models import SimpleResNet18
from ttt_model import TestTimeTrainer, train_ttt_model
from blended_ttt_model import BlendedTTT
from blended_ttt_training import train_blended_ttt_model
from ttt3fc_model import TestTimeTrainer3fc, train_ttt3fc_model
from blended_ttt3fc_model import BlendedTTT3fc
from blended_ttt3fc_training import train_blended_ttt3fc_model
from transformer_utils import set_seed

# Dataset configuration
DATASET_PATH = "../../cifar10"
CHECKPOINT_PATH = "../../cifar10checkpoints"
NUM_CLASSES = 10  # CIFAR-10 has 10 classes
IMG_SIZE = 32  # CIFAR-10 images are 32x32


def create_directories():
    """Create necessary directories for checkpoints"""
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(os.path.join(CHECKPOINT_PATH, "visualizations"), exist_ok=True)
    print(f"✅ Created checkpoint directory: {CHECKPOINT_PATH}")


def get_cifar10_transforms():
    """Get CIFAR-10 specific transforms"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010])
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010])
    ])
    
    return transform_train, transform_val


def load_cifar10_data():
    """Load CIFAR-10 dataset"""
    transform_train, transform_val = get_cifar10_transforms()
    
    # Download CIFAR-10 if not present
    train_dataset = datasets.CIFAR10(
        root=DATASET_PATH, 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    val_dataset = datasets.CIFAR10(
        root=DATASET_PATH, 
        train=False, 
        download=True, 
        transform=transform_val
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Loaded CIFAR-10 dataset")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def train_vit_model(train_loader, val_loader, model_name="vit", robust=False):
    """Train Vision Transformer model on CIFAR-10"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create ViT model adapted for CIFAR-10
    model = create_vit_model(
        img_size=IMG_SIZE,
        patch_size=4,  # Smaller patches for 32x32 images
        in_chans=3,
        num_classes=NUM_CLASSES,
        embed_dim=384,
        depth=8,
        head_dim=64,
        mlp_ratio=4.0,
        use_resnet_stem=True
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Model save path
    save_dir = os.path.join(CHECKPOINT_PATH, f"bestmodel_{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_acc = 0.0
    epochs = 100
    
    print(f"\n🚀 Training {model_name} ViT model on CIFAR-10...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Apply augmentation if robust training
            if robust and np.random.rand() > 0.5:
                # Simple augmentation for robustness
                noise = torch.randn_like(images) * 0.1
                images = images + noise
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 
                             'acc': f"{train_correct/train_total:.4f}"})
        
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_config': {
                    'img_size': IMG_SIZE,
                    'patch_size': 4,
                    'num_classes': NUM_CLASSES,
                    'embed_dim': 384,
                    'depth': 8
                }
            }, os.path.join(save_dir, "best_model.pt"))
            print(f"  ✅ New best model saved with val_acc: {val_acc:.4f}")
        
        scheduler.step()
    
    print(f"✅ Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return model, best_val_acc


def train_resnet_baseline(train_loader, val_loader, pretrained=False):
    """Train ResNet18 baseline on CIFAR-10"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if pretrained:
        # Use pretrained ResNet18 and adapt for CIFAR-10
        import torchvision.models as models
        model = models.resnet18(pretrained=True)
        # Modify first conv layer for 32x32 input
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # Remove maxpool for small images
        # Modify final layer for 10 classes
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model_name = "resnet18_pretrained"
    else:
        # Train from scratch
        model = SimpleResNet18(num_classes=NUM_CLASSES)
        model_name = "resnet18_baseline"
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Model save path
    save_dir = os.path.join(CHECKPOINT_PATH, f"bestmodel_{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_acc = 0.0
    epochs = 100
    
    print(f"\n🚀 Training {model_name} on CIFAR-10...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 
                             'acc': f"{train_correct/train_total:.4f}"})
        
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(save_dir, "best_model.pt"))
            print(f"  ✅ New best model saved with val_acc: {val_acc:.4f}")
        
        scheduler.step()
    
    print(f"✅ Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return model, best_val_acc


def train_ttt_models(train_loader, val_loader, base_model=None):
    """Train TTT and TTT3fc models on CIFAR-10"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load base model if not provided
    if base_model is None:
        base_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_main/best_model.pt")
        if os.path.exists(base_model_path):
            base_model = create_vit_model(
                img_size=IMG_SIZE, patch_size=4, in_chans=3, num_classes=NUM_CLASSES,
                embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
            )
            checkpoint = torch.load(base_model_path, map_location=device)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            base_model = base_model.to(device)
            print("✅ Loaded base model for TTT training")
        else:
            print("❌ Base model not found. Please train the main model first.")
            return None, None
    
    # Train original TTT model
    print("\n🚀 Training TTT model on CIFAR-10...")
    ttt_model = TestTimeTrainer(
        base_model=base_model,
        img_size=IMG_SIZE,
        patch_size=4,
        embed_dim=384
    ).to(device)
    
    # TTT training (simplified for CIFAR-10)
    save_dir_ttt = os.path.join(CHECKPOINT_PATH, "bestmodel_ttt")
    os.makedirs(save_dir_ttt, exist_ok=True)
    
    # Train TTT model
    optimizer = optim.AdamW(ttt_model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()  # For reconstruction
    
    best_val_loss = float('inf')
    epochs = 50
    
    for epoch in range(epochs):
        ttt_model.train()
        train_loss = 0.0
        
        for images, _ in tqdm(train_loader, desc=f"TTT Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            
            # Add noise for TTT training
            noisy_images = images + torch.randn_like(images) * 0.1
            
            optimizer.zero_grad()
            _, reconstructed = ttt_model(noisy_images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        ttt_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                noisy_images = images + torch.randn_like(images) * 0.1
                _, reconstructed = ttt_model(noisy_images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"TTT Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': ttt_model.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir_ttt, "best_model.pt"))
    
    # Train TTT3fc model
    print("\n🚀 Training TTT3fc model on CIFAR-10...")
    save_dir_ttt3fc = os.path.join(CHECKPOINT_PATH, "bestmodel_ttt3fc")
    os.makedirs(save_dir_ttt3fc, exist_ok=True)
    
    # Use the training function from ttt3fc_model
    train_ttt3fc_model(
        dataset_path=DATASET_PATH,
        base_model=base_model,
        model_save_dir=save_dir_ttt3fc,
        epochs=50,
        batch_size=128
    )
    
    return ttt_model, None  # Return trained models


def train_blended_models(train_loader, val_loader, base_model=None):
    """Train BlendedTTT and BlendedTTT3fc models on CIFAR-10"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load base model if not provided
    if base_model is None:
        base_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_main/best_model.pt")
        if os.path.exists(base_model_path):
            base_model = create_vit_model(
                img_size=IMG_SIZE, patch_size=4, in_chans=3, num_classes=NUM_CLASSES,
                embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
            )
            checkpoint = torch.load(base_model_path, map_location=device)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            base_model = base_model.to(device)
            print("✅ Loaded base model for Blended training")
        else:
            print("❌ Base model not found. Please train the main model first.")
            return None, None
    
    # Train BlendedTTT model
    print("\n🚀 Training BlendedTTT model on CIFAR-10...")
    blended_model = BlendedTTT(
        img_size=IMG_SIZE,
        patch_size=4,
        embed_dim=384,
        depth=8,
        num_classes=NUM_CLASSES
    ).to(device)
    
    save_dir_blended = os.path.join(CHECKPOINT_PATH, "bestmodel_blended")
    os.makedirs(save_dir_blended, exist_ok=True)
    
    # Train using the imported function (adapted for CIFAR-10)
    # Note: We'll need to create a simple training loop since the original expects TinyImageNet
    optimizer = optim.AdamW(blended_model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss()
    
    best_val_acc = 0.0
    epochs = 50
    
    for epoch in range(epochs):
        blended_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Blended Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Add noise for robustness
            noisy_images = images + torch.randn_like(images) * 0.1
            
            optimizer.zero_grad()
            outputs, reconstructed = blended_model(noisy_images)
            
            cls_loss = criterion(outputs, labels)
            rec_loss = reconstruction_criterion(reconstructed, images)
            loss = cls_loss + 0.1 * rec_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        blended_model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = blended_model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        train_acc = train_correct / train_total
        
        print(f"Blended Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': blended_model.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(save_dir_blended, "best_model.pt"))
    
    # Train BlendedTTT3fc model
    print("\n🚀 Training BlendedTTT3fc model on CIFAR-10...")
    save_dir_blended3fc = os.path.join(CHECKPOINT_PATH, "bestmodel_blended3fc")
    os.makedirs(save_dir_blended3fc, exist_ok=True)
    
    # Use the training function from blended_ttt3fc_training
    train_blended_ttt3fc_model(
        base_model=base_model,
        dataset_path=DATASET_PATH,
        model_save_dir=save_dir_blended3fc,
        epochs=50,
        batch_size=128
    )
    
    return blended_model, None


def evaluate_all_models(val_loader):
    """Evaluate all trained models on CIFAR-10"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    print("\n" + "="*80)
    print("📊 EVALUATING ALL MODELS ON CIFAR-10")
    print("="*80)
    
    # Model configurations
    model_configs = [
        ("Main ViT", "bestmodel_main", "vit"),
        ("Robust ViT", "bestmodel_robust", "vit"),
        ("ResNet18 Baseline", "bestmodel_resnet18_baseline", "resnet"),
        ("ResNet18 Pretrained", "bestmodel_resnet18_pretrained", "resnet"),
        ("TTT", "bestmodel_ttt", "ttt"),
        ("BlendedTTT", "bestmodel_blended", "blended"),
        ("TTT3fc", "bestmodel_ttt3fc", "ttt3fc"),
        ("BlendedTTT3fc", "bestmodel_blended3fc", "blended3fc"),
    ]
    
    for model_name, model_dir, model_type in model_configs:
        model_path = os.path.join(CHECKPOINT_PATH, model_dir, "best_model.pt")
        
        if not os.path.exists(model_path):
            print(f"⏭️  Skipping {model_name}: Model not found")
            continue
        
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Load model based on type
        if model_type == "vit":
            model = create_vit_model(
                img_size=IMG_SIZE, patch_size=4, in_chans=3, num_classes=NUM_CLASSES,
                embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
            )
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif model_type == "resnet":
            if "pretrained" in model_dir:
                import torchvision.models as models
                model = models.resnet18(pretrained=False)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
                model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            else:
                model = SimpleResNet18(num_classes=NUM_CLASSES)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif model_type == "ttt":
            base_model = create_vit_model(
                img_size=IMG_SIZE, patch_size=4, in_chans=3, num_classes=NUM_CLASSES,
                embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
            )
            model = TestTimeTrainer(base_model=base_model, img_size=IMG_SIZE, patch_size=4, embed_dim=384)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif model_type == "blended":
            model = BlendedTTT(img_size=IMG_SIZE, patch_size=4, embed_dim=384, depth=8, num_classes=NUM_CLASSES)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif model_type == "ttt3fc":
            base_model = create_vit_model(
                img_size=IMG_SIZE, patch_size=4, in_chans=3, num_classes=NUM_CLASSES,
                embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
            )
            model = TestTimeTrainer3fc(base_model=base_model, img_size=IMG_SIZE, patch_size=4, embed_dim=384)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif model_type == "blended3fc":
            model = BlendedTTT3fc(img_size=IMG_SIZE, patch_size=4, embed_dim=384, depth=8, num_classes=NUM_CLASSES)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(device)
        model.eval()
        
        # Evaluate
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Evaluating {model_name}"):
                images, labels = images.to(device), labels.to(device)
                
                # Handle different model outputs
                if model_type in ["ttt", "blended", "ttt3fc", "blended3fc"]:
                    outputs, _ = model(images)
                else:
                    outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        results[model_name] = accuracy
        print(f"✅ {model_name} Accuracy: {accuracy:.4f}")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Accuracy':>10}")
    print("-" * 40)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, acc) in enumerate(sorted_results):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        print(f"{medal} {name:<22} {acc:>10.4f}")
    
    return results


def create_performance_plots(results):
    """Create performance comparison plots"""
    save_dir = os.path.join(CHECKPOINT_PATH, "visualizations")
    
    # Bar plot of model performances
    plt.figure(figsize=(12, 8))
    models = list(results.keys())
    accuracies = list(results.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = plt.bar(models, accuracies, color=colors)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('CIFAR-10 Model Performance Comparison', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "cifar10_model_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Performance comparison plot saved to {save_path}")
    plt.close()
    
    # Group comparison plot
    plt.figure(figsize=(10, 8))
    
    # Group models by type
    baseline_models = {k: v for k, v in results.items() if 'ResNet' in k}
    vit_models = {k: v for k, v in results.items() if 'ViT' in k}
    ttt_models = {k: v for k, v in results.items() if 'TTT' in k and '3fc' not in k}
    ttt3fc_models = {k: v for k, v in results.items() if '3fc' in k}
    
    groups = {
        'Baseline (ResNet)': baseline_models,
        'ViT Models': vit_models,
        'TTT Models': ttt_models,
        'TTT-3FC Models': ttt3fc_models
    }
    
    x = 0
    for group_name, group_results in groups.items():
        if group_results:
            for i, (model, acc) in enumerate(group_results.items()):
                plt.bar(x, acc, label=model if i == 0 else "")
                plt.text(x, acc + 0.005, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
                x += 1
            x += 0.5  # Gap between groups
    
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('CIFAR-10 Performance by Model Group', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "cifar10_group_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Group comparison plot saved to {save_path}")
    plt.close()


def main():
    """Main training and evaluation pipeline for CIFAR-10"""
    parser = argparse.ArgumentParser(description="Train and evaluate all models on CIFAR-10")
    
    # Training options
    parser.add_argument("--train_main", action="store_true", help="Train main ViT model")
    parser.add_argument("--train_robust", action="store_true", help="Train robust ViT model")
    parser.add_argument("--train_baselines", action="store_true", help="Train ResNet baselines")
    parser.add_argument("--train_ttt", action="store_true", help="Train TTT models")
    parser.add_argument("--train_blended", action="store_true", help="Train Blended models")
    parser.add_argument("--train_all", action="store_true", help="Train all models")
    
    # Evaluation options
    parser.add_argument("--evaluate", action="store_true", help="Evaluate all trained models")
    parser.add_argument("--visualize", action="store_true", help="Create visualization plots")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    create_directories()
    
    # Load CIFAR-10 data
    train_loader, val_loader = load_cifar10_data()
    
    print("\n" + "="*80)
    print("🚀 CIFAR-10 MODEL TRAINING AND EVALUATION PIPELINE")
    print("="*80)
    print(f"📁 Dataset: {DATASET_PATH}")
    print(f"📁 Checkpoints: {CHECKPOINT_PATH}")
    print(f"🎯 Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Training phase
    if args.train_all or args.train_main:
        print("\n=== TRAINING MAIN VIT MODEL ===")
        train_vit_model(train_loader, val_loader, model_name="main", robust=False)
    
    if args.train_all or args.train_robust:
        print("\n=== TRAINING ROBUST VIT MODEL ===")
        train_vit_model(train_loader, val_loader, model_name="robust", robust=True)
    
    if args.train_all or args.train_baselines:
        print("\n=== TRAINING BASELINE MODELS ===")
        train_resnet_baseline(train_loader, val_loader, pretrained=False)
        train_resnet_baseline(train_loader, val_loader, pretrained=True)
    
    if args.train_all or args.train_ttt:
        print("\n=== TRAINING TTT MODELS ===")
        train_ttt_models(train_loader, val_loader)
    
    if args.train_all or args.train_blended:
        print("\n=== TRAINING BLENDED MODELS ===")
        train_blended_models(train_loader, val_loader)
    
    # Evaluation phase
    if args.evaluate or args.train_all:
        print("\n=== EVALUATION PHASE ===")
        results = evaluate_all_models(val_loader)
        
        if args.visualize or args.train_all:
            print("\n=== CREATING VISUALIZATIONS ===")
            create_performance_plots(results)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📁 All models and results saved in: {CHECKPOINT_PATH}")
    print(f"📊 Visualizations saved in: {os.path.join(CHECKPOINT_PATH, 'visualizations')}")


if __name__ == "__main__":
    main()
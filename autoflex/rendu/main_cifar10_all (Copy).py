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
from blended_ttt_cifar10 import BlendedTTTCIFAR10
from blended_ttt3fc_cifar10 import BlendedTTT3fcCIFAR10
from blended_ttt_training import train_blended_ttt_model
from ttt3fc_model import TestTimeTrainer3fc, train_ttt3fc_model
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
    patience = 5
    epochs_no_improve = 0
    
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
        
        # Save best model and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
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
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")
            
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"  🛑 Early stopping triggered after {epoch+1} epochs")
            print(f"  Best validation accuracy: {best_val_acc:.4f}")
            break
        
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
    patience = 5
    epochs_no_improve = 0
    
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
        
        # Save best model and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(save_dir, "best_model.pt"))
            print(f"  ✅ New best model saved with val_acc: {val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")
            
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"  🛑 Early stopping triggered after {epoch+1} epochs")
            print(f"  Best validation accuracy: {best_val_acc:.4f}")
            break
        
        scheduler.step()
    
    print(f"✅ Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return model, best_val_acc


def train_ttt_models(train_loader, val_loader, base_model=None, train_ttt=True, train_ttt3fc=True):
    """Train TTT and TTT3fc models on CIFAR-10"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check which models already exist
    ttt_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_ttt", "best_model.pt")
    ttt3fc_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_ttt3fc", "best_model.pt")
    
    ttt_exists = os.path.exists(ttt_model_path)
    ttt3fc_exists = os.path.exists(ttt3fc_model_path)
    
    # Override training flags based on existing models
    if ttt_exists and train_ttt:
        print(f"✓ TTT model already exists at {ttt_model_path}, skipping training")
        train_ttt = False
    if ttt3fc_exists and train_ttt3fc:
        print(f"✓ TTT3fc model already exists at {ttt3fc_model_path}, skipping training")
        train_ttt3fc = False
    
    # If both models exist, return early
    if not train_ttt and not train_ttt3fc:
        return None, None
    
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
    
    ttt_model = None
    ttt3fc_model = None
    
    # Train original TTT model if needed
    if train_ttt:
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
        criterion = nn.CrossEntropyLoss()  # For transformation prediction
        
        best_val_loss = float('inf')
        epochs = 50
        patience = 5
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            ttt_model.train()
            train_loss = 0.0
            
            for images, _ in tqdm(train_loader, desc=f"TTT Epoch {epoch+1}/{epochs}"):
                images = images.to(device)
                batch_size = images.size(0)
                
                # Create transformed images and labels
                transform_labels = torch.randint(0, 4, (batch_size,)).to(device)
                transformed_images = images.clone()
                
                for i in range(batch_size):
                    if transform_labels[i] == 1:  # Gaussian noise
                        transformed_images[i] = images[i] + torch.randn_like(images[i]) * 0.1
                    elif transform_labels[i] == 2:  # Rotation
                        transformed_images[i] = torch.rot90(images[i], 1, [1, 2])
                    elif transform_labels[i] == 3:  # Affine transformation
                        transformed_images[i] = torch.flip(images[i], [2])
                    # transform_labels[i] == 0 means no transformation
                
                optimizer.zero_grad()
                _, transform_logits = ttt_model(transformed_images)
                loss = criterion(transform_logits, transform_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            ttt_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)
                    batch_size = images.size(0)
                    
                    # Create transformed images and labels for validation
                    transform_labels = torch.randint(0, 4, (batch_size,)).to(device)
                    transformed_images = images.clone()
                    
                    for i in range(batch_size):
                        if transform_labels[i] == 1:  # Gaussian noise
                            transformed_images[i] = images[i] + torch.randn_like(images[i]) * 0.1
                        elif transform_labels[i] == 2:  # Rotation
                            transformed_images[i] = torch.rot90(images[i], 1, [1, 2])
                        elif transform_labels[i] == 3:  # Affine transformation
                            transformed_images[i] = torch.flip(images[i], [2])
                    
                    _, transform_logits = ttt_model(transformed_images)
                    loss = criterion(transform_logits, transform_labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f"TTT Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save({
                    'model_state_dict': ttt_model.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(save_dir_ttt, "best_model.pt"))
                print(f"  ✅ New best model saved with val_loss: {val_loss:.4f}")
            else:
                epochs_no_improve += 1
                print(f"  No improvement for {epochs_no_improve} epochs")
                
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"  🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"  Best validation loss: {best_val_loss:.4f}")
                break
    
    # Train TTT3fc model if needed
    if train_ttt3fc:
        print("\n🚀 Training TTT3fc model on CIFAR-10...")
        save_dir_ttt3fc = os.path.join(CHECKPOINT_PATH, "bestmodel_ttt3fc")
        os.makedirs(save_dir_ttt3fc, exist_ok=True)
        
        # Create TTT3fc model
        ttt3fc_model = TestTimeTrainer3fc(
            base_model=base_model,
            img_size=IMG_SIZE,
            patch_size=4,
            embed_dim=384
        ).to(device)
        
        # Training setup
        optimizer = optim.AdamW(ttt3fc_model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        epochs = 50
        patience = 5
        epochs_no_improve = 0
        
        # Training loop similar to TTT but with 3fc architecture
        for epoch in range(epochs):
            ttt3fc_model.train()
            train_loss = 0.0
            
            for images, _ in tqdm(train_loader, desc=f"TTT3fc Epoch {epoch+1}/{epochs}"):
                images = images.to(device)
                batch_size = images.size(0)
                
                # Create transformed images and labels
                transform_labels = torch.randint(0, 4, (batch_size,)).to(device)
                transformed_images = images.clone()
                
                for i in range(batch_size):
                    if transform_labels[i] == 1:  # Gaussian noise
                        transformed_images[i] = images[i] + torch.randn_like(images[i]) * 0.1
                    elif transform_labels[i] == 2:  # Rotation
                        transformed_images[i] = torch.rot90(images[i], 1, [1, 2])
                    elif transform_labels[i] == 3:  # Affine transformation
                        transformed_images[i] = torch.flip(images[i], [2])
                
                optimizer.zero_grad()
                _, transform_logits = ttt3fc_model(transformed_images)
                loss = criterion(transform_logits, transform_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            ttt3fc_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)
                    batch_size = images.size(0)
                    
                    # Create transformed images and labels for validation
                    transform_labels = torch.randint(0, 4, (batch_size,)).to(device)
                    transformed_images = images.clone()
                    
                    for i in range(batch_size):
                        if transform_labels[i] == 1:  # Gaussian noise
                            transformed_images[i] = images[i] + torch.randn_like(images[i]) * 0.1
                        elif transform_labels[i] == 2:  # Rotation
                            transformed_images[i] = torch.rot90(images[i], 1, [1, 2])
                        elif transform_labels[i] == 3:  # Affine transformation
                            transformed_images[i] = torch.flip(images[i], [2])
                    
                    _, transform_logits = ttt3fc_model(transformed_images)
                    loss = criterion(transform_logits, transform_labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f"TTT3fc Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save({
                    'model_state_dict': ttt3fc_model.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(save_dir_ttt3fc, "best_model.pt"))
                print(f"  ✅ New best model saved with val_loss: {val_loss:.4f}")
            else:
                epochs_no_improve += 1
                print(f"  No improvement for {epochs_no_improve} epochs")
                
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"  🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"  Best validation loss: {best_val_loss:.4f}")
                break
    
    return ttt_model, ttt3fc_model  # Return trained models


def train_blended_models(train_loader, val_loader, base_model=None, train_blended=True, train_blended3fc=True):
    """Train BlendedTTT and BlendedTTT3fc models on CIFAR-10"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check which models already exist
    blended_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_blended", "best_model.pt")
    blended3fc_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_blended3fc", "best_model.pt")
    
    blended_exists = os.path.exists(blended_model_path)
    blended3fc_exists = os.path.exists(blended3fc_model_path)
    
    # Override training flags based on existing models
    if blended_exists and train_blended:
        print(f"✓ Blended model already exists at {blended_model_path}, skipping training")
        train_blended = False
    if blended3fc_exists and train_blended3fc:
        print(f"✓ Blended3fc model already exists at {blended3fc_model_path}, skipping training")
        train_blended3fc = False
    
    # If both models exist, return early
    if not train_blended and not train_blended3fc:
        return None, None
    
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
    
    blended_model = None
    blended3fc_model = None
    
    # Train BlendedTTT model if needed
    if train_blended:
        print("\n🚀 Training BlendedTTT model on CIFAR-10...")
        blended_model = BlendedTTTCIFAR10(
            img_size=IMG_SIZE,
            patch_size=4,
            embed_dim=384,
            depth=8,
            num_classes=NUM_CLASSES
        ).to(device)
    
        save_dir_blended = os.path.join(CHECKPOINT_PATH, "bestmodel_blended")
        os.makedirs(save_dir_blended, exist_ok=True)
        
        # Training setup for BlendedTTT CIFAR-10
        optimizer = optim.AdamW(blended_model.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss()
        aux_criterion = nn.CrossEntropyLoss()  # For transform type prediction
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        best_val_acc = 0.0
        epochs = 50
        patience = 5
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            blended_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Blended Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                batch_size = images.size(0)
                
                # Create transformed images for auxiliary task
                transform_labels = torch.randint(0, 4, (batch_size,)).to(device)
                transformed_images = images.clone()
                
                for i in range(batch_size):
                    if transform_labels[i] == 1:  # Gaussian noise
                        transformed_images[i] = images[i] + torch.randn_like(images[i]) * 0.1
                    elif transform_labels[i] == 2:  # Rotation
                        transformed_images[i] = torch.rot90(images[i], 1, [1, 2])
                    elif transform_labels[i] == 3:  # Affine transformation
                        transformed_images[i] = torch.flip(images[i], [2])
                
                optimizer.zero_grad()
                
                # Forward pass
                class_logits, aux_outputs = blended_model(transformed_images)
                
                # Classification loss
                cls_loss = criterion(class_logits, labels)
                
                # Auxiliary loss (transform prediction)
                aux_loss = aux_criterion(aux_outputs['transform_type_logits'], transform_labels)
                
                # Combined loss
                loss = cls_loss + 0.5 * aux_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(class_logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation
            blended_model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    class_logits, _ = blended_model(images)
                    _, predicted = torch.max(class_logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = val_correct / val_total
            train_acc = train_correct / train_total
            
            print(f"Blended Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Update learning rate scheduler
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save({
                    'model_state_dict': blended_model.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(save_dir_blended, "best_model.pt"))
                print(f"  ✅ New best model saved with val_acc: {val_acc:.4f}")
            else:
                epochs_no_improve += 1
                print(f"  No improvement for {epochs_no_improve} epochs")
                
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"  🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"  Best validation accuracy: {best_val_acc:.4f}")
                break
    
    # Train BlendedTTT3fc model if needed
    if train_blended3fc:
        print("\n🚀 Training BlendedTTT3fc model on CIFAR-10...")
        save_dir_blended3fc = os.path.join(CHECKPOINT_PATH, "bestmodel_blended3fc")
        os.makedirs(save_dir_blended3fc, exist_ok=True)
        
        # Create BlendedTTT3fc model for CIFAR-10
        blended3fc_model = BlendedTTT3fcCIFAR10(
            img_size=IMG_SIZE,
            patch_size=4,
            embed_dim=384,
            depth=8,
            num_classes=NUM_CLASSES
        ).to(device)
        
        # Training setup
        optimizer = optim.AdamW(blended3fc_model.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss()
        aux_criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        best_val_acc = 0.0
        epochs = 50
        patience = 5
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            blended3fc_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Blended3fc Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                batch_size = images.size(0)
                
                # Create transformed images
                transform_labels = torch.randint(0, 4, (batch_size,)).to(device)
                transformed_images = images.clone()
                
                for i in range(batch_size):
                    if transform_labels[i] == 1:  # Gaussian noise
                        transformed_images[i] = images[i] + torch.randn_like(images[i]) * 0.1
                    elif transform_labels[i] == 2:  # Rotation
                        transformed_images[i] = torch.rot90(images[i], 1, [1, 2])
                    elif transform_labels[i] == 3:  # Affine transformation
                        transformed_images[i] = torch.flip(images[i], [2])
                
                optimizer.zero_grad()
                
                # Forward pass
                class_logits, aux_outputs = blended3fc_model(transformed_images)
                
                # Losses
                cls_loss = criterion(class_logits, labels)
                aux_loss = aux_criterion(aux_outputs['transform_type_logits'], transform_labels)
                loss = cls_loss + 0.5 * aux_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(class_logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation
            blended3fc_model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    class_logits, _ = blended3fc_model(images)
                    _, predicted = torch.max(class_logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = val_correct / val_total
            train_acc = train_correct / train_total
            
            print(f"Blended3fc Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Update learning rate scheduler
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save({
                    'model_state_dict': blended3fc_model.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(save_dir_blended3fc, "best_model.pt"))
                print(f"  ✅ New best model saved with val_acc: {val_acc:.4f}")
            else:
                epochs_no_improve += 1
                print(f"  No improvement for {epochs_no_improve} epochs")
                
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"  🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"  Best validation accuracy: {best_val_acc:.4f}")
                break
    
    return blended_model, blended3fc_model


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


def retrain_vit_model(train_loader, val_loader, model_path, model_name="vit", robust=False):
    """Retrain an existing ViT model with validation-based early stopping"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load existing model
    print(f"📂 Loading existing model from {model_path}")
    model = create_vit_model(
        img_size=IMG_SIZE,
        patch_size=4,
        in_chans=3,
        num_classes=NUM_CLASSES,
        embed_dim=384,
        depth=8,
        head_dim=64,
        mlp_ratio=4.0,
        use_resnet_stem=True
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate current performance
    print("📊 Evaluating current model performance...")
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Initial Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    initial_val_acc = 100 * val_correct / val_total
    print(f"📈 Initial validation accuracy: {initial_val_acc:.2f}%")
    
    # Retrain with early stopping
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)  # Lower LR for fine-tuning
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    best_val_acc = initial_val_acc
    patience = 10
    patience_counter = 0
    epochs = 50
    
    print(f"\n🔄 Retraining {model_name} model with early stopping...")
    
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
                noise = torch.randn_like(images) * 0.1
                images = images + noise
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss/len(train_loader), 'acc': 100*train_correct/train_total})
        
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
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc: {100*train_correct/train_total:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch
            }, model_path)
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
                break
        
        scheduler.step()
    
    print(f"🏁 Retraining completed. Best Val Acc: {best_val_acc:.2f}% (improvement: {best_val_acc - initial_val_acc:.2f}%)")
    return model


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
    
    # Mode options
    parser.add_argument("--skip_training", action="store_true", help="Skip training phase")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation phase")
    parser.add_argument("--retrain", action="store_true", help="Reload existing models and retrain with validation early stopping")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Default behavior: if no specific arguments are provided, train missing models and evaluate all
    if not any([args.train_main, args.train_robust, args.train_baselines, args.train_ttt, 
                args.train_blended, args.train_all, args.evaluate, args.visualize,
                args.skip_training, args.skip_evaluation]):
        print("No specific arguments provided. Using default behavior:")
        print("- Training missing models")
        print("- Evaluating all existing models")
        args.train_all = True  # This will check and train only missing models
        args.evaluate = True
        args.visualize = True
    
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
    
    # Show current mode
    if args.skip_training:
        print("\n⚠️  Training phase SKIPPED (--skip_training flag set)")
    if args.skip_evaluation:
        print("⚠️  Evaluation phase SKIPPED (--skip_evaluation flag set)")
    if args.retrain:
        print("\n🔄 RETRAIN MODE: Will reload and retrain existing models")
    
    # Retrain mode
    if args.retrain:
        print("\n=== RETRAIN MODE ===")
        
        # Check and retrain main model
        main_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_main", "best_model.pt")
        if os.path.exists(main_model_path):
            print(f"\n🔄 Retraining main ViT model...")
            retrain_vit_model(train_loader, val_loader, main_model_path, model_name="main", robust=False)
        else:
            print(f"❌ Main model not found at {main_model_path}")
        
        # Check and retrain robust model
        robust_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_robust", "best_model.pt")
        if os.path.exists(robust_model_path):
            print(f"\n🔄 Retraining robust ViT model...")
            retrain_vit_model(train_loader, val_loader, robust_model_path, model_name="robust", robust=True)
        else:
            print(f"❌ Robust model not found at {robust_model_path}")
        
        # Note: For ResNet, TTT, and Blended models, we would need separate retrain functions
        # which would be similar but adapted to their specific architectures
        print("\n⚠️  Note: Retraining for ResNet, TTT, and Blended models not yet implemented")
        
        # After retraining, proceed to evaluation if not skipped
        if not args.skip_evaluation:
            args.evaluate = True
    
    # Training phase (skip if --skip_training is set or if in retrain mode)
    elif not args.skip_training and (args.train_all or args.train_main):
        main_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_main", "best_model.pt")
        if os.path.exists(main_model_path):
            print(f"\n✓ Main ViT model already exists at {main_model_path}")
        else:
            print("\n=== TRAINING MAIN VIT MODEL ===")
            train_vit_model(train_loader, val_loader, model_name="main", robust=False)
    
    if not args.retrain and not args.skip_training and (args.train_all or args.train_robust):
        robust_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_robust", "best_model.pt")
        if os.path.exists(robust_model_path):
            print(f"\n✓ Robust ViT model already exists at {robust_model_path}")
        else:
            print("\n=== TRAINING ROBUST VIT MODEL ===")
            train_vit_model(train_loader, val_loader, model_name="robust", robust=True)
    
    if not args.retrain and not args.skip_training and (args.train_all or args.train_baselines):
        resnet_path = os.path.join(CHECKPOINT_PATH, "bestmodel_resnet18_baseline", "best_model.pt")
        resnet_pretrained_path = os.path.join(CHECKPOINT_PATH, "bestmodel_resnet18_pretrained", "best_model.pt")
        
        if os.path.exists(resnet_path):
            print(f"\n✓ ResNet baseline already exists at {resnet_path}")
        else:
            print("\n=== TRAINING RESNET BASELINE (from scratch) ===")
            train_resnet_baseline(train_loader, val_loader, pretrained=False)
            
        if os.path.exists(resnet_pretrained_path):
            print(f"\n✓ Pretrained ResNet baseline already exists at {resnet_pretrained_path}")
        else:
            print("\n=== TRAINING RESNET BASELINE (pretrained) ===")
            train_resnet_baseline(train_loader, val_loader, pretrained=True)
    
    if not args.retrain and not args.skip_training and (args.train_all or args.train_ttt):
        ttt_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_ttt", "best_model.pt")
        ttt3fc_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_ttt3fc", "best_model.pt")
        
        ttt_exists = os.path.exists(ttt_model_path)
        ttt3fc_exists = os.path.exists(ttt3fc_model_path)
        
        if ttt_exists:
            print(f"\n✓ TTT model already exists at {ttt_model_path}")
        if ttt3fc_exists:
            print(f"✓ TTT3fc model already exists at {ttt3fc_model_path}")
            
        if not ttt_exists or not ttt3fc_exists:
            print("\n=== TRAINING TTT MODELS ===")
            train_ttt_models(train_loader, val_loader)
    
    if not args.retrain and not args.skip_training and (args.train_all or args.train_blended):
        blended_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_blended", "best_model.pt")
        blended3fc_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_blended3fc", "best_model.pt")
        
        if os.path.exists(blended_model_path) and os.path.exists(blended3fc_model_path):
            print(f"\n✓ Blended models already exist:")
            print(f"  - Blended: {blended_model_path}")
            print(f"  - Blended3fc: {blended3fc_model_path}")
        else:
            print("\n=== TRAINING BLENDED MODELS ===")
            train_blended_models(train_loader, val_loader)
    
    # Evaluation phase (skip if --skip_evaluation is set)
    if not args.skip_evaluation and (args.evaluate or args.train_all):
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
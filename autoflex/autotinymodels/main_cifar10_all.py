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
from cifar10_healer_additions import TransformationHealerCIFAR10, HealerLossCIFAR10
from blended_ttt_training import train_blended_ttt_model
from ttt3fc_model import TestTimeTrainer3fc, train_ttt3fc_model
from blended_ttt3fc_training import train_blended_ttt3fc_model
from transformer_utils import set_seed

# Import ContinuousTransforms for OOD training
from new_new import ContinuousTransforms

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
        transforms.ToTensor(),

    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return transform_train, transform_val


def get_cifar10_transforms_no_norm():
    """Get CIFAR-10 transforms without normalization (for OOD training)"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return transform_train, transform_val


def get_cifar10_normalize():
    """Get CIFAR-10 normalization transform"""
    return transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                              std=[0.2023, 0.1994, 0.2010])


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


def load_cifar10_data_no_norm():
    """Load CIFAR-10 dataset without normalization (for OOD training)"""
    transform_train, transform_val = get_cifar10_transforms_no_norm()
    
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
    
    print(f"✅ Loaded CIFAR-10 dataset (without normalization)")
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
    
    # Create ContinuousTransforms for robust training
    if robust:
        continuous_transform = ContinuousTransforms(severity=0.3)
    
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
            
            # Apply continuous transformations if robust training
            if robust:
                batch_size = images.size(0)
                transformed_images = []
                
                for i in range(batch_size):
                    if np.random.rand() > 0.5:  # Apply transformations 50% of the time
                        # Randomly choose transformation type
                        transform_type = np.random.choice(continuous_transform.transform_types[1:])  # Skip 'no_transform'
                        # Apply transformation with random severity
                        transformed_img, _ = continuous_transform.apply_transforms(
                            images[i], 
                            transform_type=transform_type,
                            severity=np.random.uniform(0.0, 0.5),  # Random severity up to 0.5
                            return_params=True
                        )
                        transformed_images.append(transformed_img)
                    else:
                        transformed_images.append(images[i])
                
                images = torch.stack(transformed_images)
            
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
        
        # Create ContinuousTransforms for TTT training
        continuous_transform = ContinuousTransforms(severity=0.5)
        normalize = get_cifar10_normalize()
        
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
                
                # Apply continuous transformations to unnormalized images
                transformed_images = []
                transform_labels = []
                
                for i in range(batch_size):
                    # Randomly choose transformation type
                    transform_type = np.random.choice(continuous_transform.transform_types)
                    transform_type_idx = continuous_transform.transform_types.index(transform_type)
                    
                    # Apply transformation to unnormalized image
                    transformed_img, _ = continuous_transform.apply_transforms(
                        images[i], 
                        transform_type=transform_type,
                        severity=np.random.uniform(0.0, 1.0),  # Random severity
                        return_params=True
                    )
                    
                    # Normalize after transformation
                    transformed_img = normalize(transformed_img)
                    
                    transformed_images.append(transformed_img)
                    transform_labels.append(transform_type_idx)
                
                transformed_images = torch.stack(transformed_images)
                transform_labels = torch.tensor(transform_labels, device=device)
                
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
                    
                    # Apply continuous transformations for validation
                    transformed_images = []
                    transform_labels = []
                    
                    for i in range(batch_size):
                        # Randomly choose transformation type
                        transform_type = np.random.choice(continuous_transform.transform_types)
                        transform_type_idx = continuous_transform.transform_types.index(transform_type)
                        
                        # Apply transformation with fixed severity for validation
                        transformed_img, _ = continuous_transform.apply_transforms(
                            images[i], 
                            transform_type=transform_type,
                            severity=0.5,  # Fixed severity for validation
                            return_params=True
                        )
                        
                        # Normalize after transformation
                        transformed_img = normalize(transformed_img)
                        
                        transformed_images.append(transformed_img)
                        transform_labels.append(transform_type_idx)
                    
                    transformed_images = torch.stack(transformed_images)
                    transform_labels = torch.tensor(transform_labels, device=device)
                    
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
            embed_dim=384,
            num_classes=NUM_CLASSES
        ).to(device)
        
        # Training setup
        optimizer = optim.AdamW(ttt3fc_model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        
        # Create ContinuousTransforms for TTT3fc training
        continuous_transform = ContinuousTransforms(severity=0.5)
        normalize = get_cifar10_normalize()
        
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
                
                # Apply continuous transformations
                transformed_images = []
                transform_labels = []
                
                for i in range(batch_size):
                    # Randomly choose transformation type
                    transform_type = np.random.choice(continuous_transform.transform_types)
                    transform_type_idx = continuous_transform.transform_types.index(transform_type)
                    
                    # Apply transformation
                    transformed_img, _ = continuous_transform.apply_transforms(
                        images[i], 
                        transform_type=transform_type,
                        severity=np.random.uniform(0.0, 1.0),  # Random severity
                        return_params=True
                    )
                    
                    # Normalize after transformation
                    transformed_img = normalize(transformed_img)
                    
                    transformed_images.append(transformed_img)
                    transform_labels.append(transform_type_idx)
                
                transformed_images = torch.stack(transformed_images)
                transform_labels = torch.tensor(transform_labels, device=device)
                
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
                    
                    # Apply continuous transformations for validation
                    transformed_images = []
                    transform_labels = []
                    
                    for i in range(batch_size):
                        # Randomly choose transformation type
                        transform_type = np.random.choice(continuous_transform.transform_types)
                        transform_type_idx = continuous_transform.transform_types.index(transform_type)
                        
                        # Apply transformation with fixed severity for validation
                        transformed_img, _ = continuous_transform.apply_transforms(
                            images[i], 
                            transform_type=transform_type,
                            severity=0.5,  # Fixed severity for validation
                            return_params=True
                        )
                        
                        # Normalize after transformation
                        transformed_img = normalize(transformed_img)
                        
                        transformed_images.append(transformed_img)
                        transform_labels.append(transform_type_idx)
                    
                    transformed_images = torch.stack(transformed_images)
                    transform_labels = torch.tensor(transform_labels, device=device)
                    
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
        
        # Create ContinuousTransforms for BlendedTTT training
        continuous_transform = ContinuousTransforms(severity=0.5)
        
        # Get normalization transform
        normalize = get_cifar10_normalize()
        
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
                
                # Apply continuous transformations
                transformed_images = []
                transform_labels_list = []
                
                for i in range(batch_size):
                    # Randomly choose transformation type
                    transform_type = np.random.choice(continuous_transform.transform_types)
                    transform_type_idx = continuous_transform.transform_types.index(transform_type)
                    
                    # Apply transformation with random severity
                    transformed_img, _ = continuous_transform.apply_transforms(
                        images[i], 
                        transform_type=transform_type,
                        severity=np.random.uniform(0.0, 1.0),
                        return_params=True
                    )
                    
                    # Normalize after transformation
                    transformed_img = normalize(transformed_img)
                    
                    transformed_images.append(transformed_img)
                    transform_labels_list.append(transform_type_idx)
                
                transformed_images = torch.stack(transformed_images)
                transform_labels = torch.tensor(transform_labels_list, device=device)
                
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
        
        # Create ContinuousTransforms for BlendedTTT3fc training
        continuous_transform = ContinuousTransforms(severity=0.5)
        
        # Get normalization transform
        normalize = get_cifar10_normalize()
        
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
                
                # Apply continuous transformations
                transformed_images = []
                transform_labels_list = []
                
                for i in range(batch_size):
                    # Randomly choose transformation type
                    transform_type = np.random.choice(continuous_transform.transform_types)
                    transform_type_idx = continuous_transform.transform_types.index(transform_type)
                    
                    # Apply transformation with random severity
                    transformed_img, _ = continuous_transform.apply_transforms(
                        images[i], 
                        transform_type=transform_type,
                        severity=np.random.uniform(0.0, 1.0),
                        return_params=True
                    )
                    
                    # Normalize after transformation
                    transformed_img = normalize(transformed_img)
                    
                    transformed_images.append(transformed_img)
                    transform_labels_list.append(transform_type_idx)
                
                transformed_images = torch.stack(transformed_images)
                transform_labels = torch.tensor(transform_labels_list, device=device)
                
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


def train_healer_model(train_loader, val_loader):
    """Train Transformation Healer model on CIFAR-10 with ContinuousTransforms"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if model already exists
    healer_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_healer", "best_model.pt")
    if os.path.exists(healer_model_path):
        print(f"✓ Healer model already exists at {healer_model_path}, skipping training")
        return None
    
    print("\n🚀 Training Healer model on CIFAR-10...")
    
    # Create healer model
    healer_model = TransformationHealerCIFAR10(
        img_size=IMG_SIZE,
        patch_size=4,
        in_chans=3,
        embed_dim=384,
        depth=6,
        head_dim=64
    ).to(device)
    
    save_dir_healer = os.path.join(CHECKPOINT_PATH, "bestmodel_healer")
    os.makedirs(save_dir_healer, exist_ok=True)
    
    # Training setup
    optimizer = optim.AdamW(healer_model.parameters(), lr=0.0005)
    healer_loss_fn = HealerLossCIFAR10()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Create ContinuousTransforms for healer training
    continuous_transform = ContinuousTransforms(severity=0.5)
    
    # Get normalization transform
    normalize = get_cifar10_normalize()
    
    best_val_loss = float('inf')
    epochs = 50
    patience = 5
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        healer_model.train()
        train_loss = 0.0
        train_type_correct = 0
        train_total = 0
        
        for images, _ in tqdm(train_loader, desc=f"Healer Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Apply continuous transformations and store parameters
            transformed_images = []
            true_params = {
                'transform_type': [],
                'severity': [],
                'rotation_angle': [],
                'noise_std': [],
                'affine_params': []
            }
            
            for i in range(batch_size):
                # Randomly choose transformation type
                transform_type = np.random.choice(continuous_transform.transform_types)
                transform_type_idx = continuous_transform.transform_types.index(transform_type)
                
                # Apply transformation with random severity
                severity = np.random.uniform(0.0, 1.0)
                transformed_img, params = continuous_transform.apply_transforms(
                    images[i], 
                    transform_type=transform_type,
                    severity=severity,
                    return_params=True
                )
                
                # Normalize after transformation
                transformed_img = normalize(transformed_img)
                
                transformed_images.append(transformed_img)
                true_params['transform_type'].append(transform_type_idx)
                true_params['severity'].append(severity)
                
                # Store specific parameters based on transform type
                if transform_type == 'gaussian_noise':
                    true_params['noise_std'].append(torch.tensor(params.get('std', 0.0)))
                    true_params['rotation_angle'].append(torch.tensor(0.0))
                    true_params['affine_params'].append(torch.zeros(4))
                elif transform_type == 'rotation':
                    true_params['rotation_angle'].append(torch.tensor(params.get('angle', 0.0)))
                    true_params['noise_std'].append(torch.tensor(0.0))
                    true_params['affine_params'].append(torch.zeros(4))
                elif transform_type == 'affine':
                    affine_p = torch.tensor([
                        params.get('translate_x', 0.0),
                        params.get('translate_y', 0.0),
                        params.get('shear_x', 0.0),
                        params.get('shear_y', 0.0)
                    ])
                    true_params['affine_params'].append(affine_p)
                    true_params['rotation_angle'].append(torch.tensor(0.0))
                    true_params['noise_std'].append(torch.tensor(0.0))
                else:  # no_transform
                    true_params['rotation_angle'].append(torch.tensor(0.0))
                    true_params['noise_std'].append(torch.tensor(0.0))
                    true_params['affine_params'].append(torch.zeros(4))
            
            # Convert to tensors
            transformed_images = torch.stack(transformed_images)
            true_params['transform_type'] = torch.tensor(true_params['transform_type'], device=device)
            true_params['severity'] = torch.tensor(true_params['severity'], device=device, dtype=torch.float32)
            true_params['rotation_angle'] = torch.stack(true_params['rotation_angle']).to(device)
            true_params['noise_std'] = torch.stack(true_params['noise_std']).to(device)
            true_params['affine_params'] = torch.stack(true_params['affine_params']).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = healer_model(transformed_images)
            
            # Calculate loss
            loss, loss_dict = healer_loss_fn(predictions, true_params)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy for transform type prediction
            _, predicted_type = torch.max(predictions['transform_type_logits'], 1)
            train_total += true_params['transform_type'].size(0)
            train_type_correct += (predicted_type == true_params['transform_type']).sum().item()
        
        # Validation
        healer_model.eval()
        val_loss = 0.0
        val_type_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                batch_size = images.size(0)
                
                # Apply transformations for validation
                transformed_images = []
                true_params = {
                    'transform_type': [],
                    'severity': [],
                    'rotation_angle': [],
                    'noise_std': [],
                    'affine_params': []
                }
                
                for i in range(batch_size):
                    transform_type = np.random.choice(continuous_transform.transform_types)
                    transform_type_idx = continuous_transform.transform_types.index(transform_type)
                    
                    severity = 0.5  # Fixed severity for validation
                    transformed_img, params = continuous_transform.apply_transforms(
                        images[i], 
                        transform_type=transform_type,
                        severity=severity,
                        return_params=True
                    )
                    
                    # Normalize after transformation
                    transformed_img = normalize(transformed_img)
                    
                    transformed_images.append(transformed_img)
                    true_params['transform_type'].append(transform_type_idx)
                    true_params['severity'].append(severity)
                    
                    # Store parameters (same logic as training)
                    if transform_type == 'gaussian_noise':
                        true_params['noise_std'].append(torch.tensor(params.get('std', 0.0)))
                        true_params['rotation_angle'].append(torch.tensor(0.0))
                        true_params['affine_params'].append(torch.zeros(4))
                    elif transform_type == 'rotation':
                        true_params['rotation_angle'].append(torch.tensor(params.get('angle', 0.0)))
                        true_params['noise_std'].append(torch.tensor(0.0))
                        true_params['affine_params'].append(torch.zeros(4))
                    elif transform_type == 'affine':
                        affine_p = torch.tensor([
                            params.get('translate_x', 0.0),
                            params.get('translate_y', 0.0),
                            params.get('shear_x', 0.0),
                            params.get('shear_y', 0.0)
                        ])
                        true_params['affine_params'].append(affine_p)
                        true_params['rotation_angle'].append(torch.tensor(0.0))
                        true_params['noise_std'].append(torch.tensor(0.0))
                    else:
                        true_params['rotation_angle'].append(torch.tensor(0.0))
                        true_params['noise_std'].append(torch.tensor(0.0))
                        true_params['affine_params'].append(torch.zeros(4))
                
                # Convert to tensors
                transformed_images = torch.stack(transformed_images)
                true_params['transform_type'] = torch.tensor(true_params['transform_type'], device=device)
                true_params['severity'] = torch.tensor(true_params['severity'], device=device, dtype=torch.float32)
                true_params['rotation_angle'] = torch.stack(true_params['rotation_angle']).to(device)
                true_params['noise_std'] = torch.stack(true_params['noise_std']).to(device)
                true_params['affine_params'] = torch.stack(true_params['affine_params']).to(device)
                
                predictions = healer_model(transformed_images)
                loss, _ = healer_loss_fn(predictions, true_params)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted_type = torch.max(predictions['transform_type_logits'], 1)
                val_total += true_params['transform_type'].size(0)
                val_type_correct += (predicted_type == true_params['transform_type']).sum().item()
        
        val_loss /= len(val_loader)
        train_type_acc = train_type_correct / train_total
        val_type_acc = val_type_correct / val_total
        
        print(f"Healer Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Type Acc: {train_type_acc:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Type Acc: {val_type_acc:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': healer_model.state_dict(),
                'val_loss': val_loss,
                'val_type_acc': val_type_acc,
            }, os.path.join(save_dir_healer, "best_model.pt"))
            print(f"  ✅ New best model saved with val_loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")
            
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"  🛑 Early stopping triggered after {epoch+1} epochs")
            print(f"  Best validation loss: {best_val_loss:.4f}")
            break
    
    return healer_model


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
        ("Healer", "bestmodel_healer", "healer"),
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
            model = BlendedTTTCIFAR10(img_size=IMG_SIZE, patch_size=4, embed_dim=384, depth=8, num_classes=NUM_CLASSES)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif model_type == "ttt3fc":
            base_model = create_vit_model(
                img_size=IMG_SIZE, patch_size=4, in_chans=3, num_classes=NUM_CLASSES,
                embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
            )
            model = TestTimeTrainer3fc(base_model=base_model, img_size=IMG_SIZE, patch_size=4, embed_dim=384, num_classes=NUM_CLASSES)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif model_type == "blended3fc":
            model = BlendedTTT3fcCIFAR10(img_size=IMG_SIZE, patch_size=4, embed_dim=384, depth=8, num_classes=NUM_CLASSES)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif model_type == "healer":
            model = TransformationHealerCIFAR10(IMG_SIZE, 4, 3, 384, 6, 64)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(device)
        model.eval()
        
        # Evaluate
        if model_type == "healer":
            # Healer is evaluated differently - it predicts transformation types
            print(f"   Note: Healer model predicts transformation types, not classes.")
            print(f"   See healer training logs for transformation prediction accuracy.")
            continue
        
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


def load_base_model_for_ttt(device):
    """Load the base model for TTT/TTT3fc models"""
    base_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_main/best_model.pt")
    base_model = create_vit_model(
        img_size=IMG_SIZE, patch_size=4, in_chans=3, num_classes=NUM_CLASSES,
        embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
    )
    if os.path.exists(base_model_path):
        checkpoint = torch.load(base_model_path, map_location=device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
    return base_model.to(device)


def evaluate_models_with_transforms(val_loader, severities=[0.0, 0.3, 0.5, 0.7, 1.0]):
    """Evaluate all models with continuous transformations at different severities"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    continuous_transform = ContinuousTransforms(severity=0.5)
    
    # Get normalization transform
    normalize = get_cifar10_normalize()
    
    print("\n" + "="*80)
    print("📊 EVALUATING MODELS WITH CONTINUOUS TRANSFORMATIONS")
    print("="*80)
    print(f"Severities to test: {severities}")
    
    # Model configurations
    model_configs = [
        ("Main ViT", "bestmodel_main", "vit"),
        ("Robust ViT", "bestmodel_robust", "vit"),
        ("ResNet18 Baseline", "bestmodel_resnet18_baseline", "resnet"),
        ("ResNet18 Pretrained", "bestmodel_resnet18_pretrained", "resnet"),
        ("TTT", "bestmodel_ttt", "ttt"),
        ("BlendedTTT", "bestmodel_blended", "blended"),
        ("TTT3fc", "bestmodel_ttt3fc", "ttt3fc"),
        ("Healer", "bestmodel_healer", "healer"),
        ("BlendedTTT3fc", "bestmodel_blended3fc", "blended3fc"),
    ]
    
    all_results = {}
    
    for model_name, model_dir, model_type in model_configs:
        model_path = os.path.join(CHECKPOINT_PATH, model_dir, "best_model.pt")
        
        if not os.path.exists(model_path):
            print(f"⏭️  Skipping {model_name}: Model not found")
            continue
            
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Load model
        if model_type == "vit":
            model = create_vit_model(
                img_size=IMG_SIZE, patch_size=4, in_chans=3, num_classes=NUM_CLASSES,
                embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
            )
        elif model_type == "resnet":
            if "pretrained" in model_dir:
                import torchvision.models as models
                model = models.resnet18(pretrained=False)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
                model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            else:
                model = SimpleResNet18(num_classes=NUM_CLASSES)
        elif model_type == "ttt":
            base_model = load_base_model_for_ttt(device)
            model = TestTimeTrainer(base_model, IMG_SIZE, 4, 384)
        elif model_type == "ttt3fc":
            base_model = load_base_model_for_ttt(device)
            model = TestTimeTrainer3fc(base_model, IMG_SIZE, 4, 384, num_classes=NUM_CLASSES)
        elif model_type == "blended":
            model = BlendedTTTCIFAR10(IMG_SIZE, 4, 384, 8, NUM_CLASSES)
        elif model_type == "blended3fc":
            model = BlendedTTT3fcCIFAR10(IMG_SIZE, 4, 384, 8, num_classes=NUM_CLASSES)
        elif model_type == "healer":
            model = TransformationHealerCIFAR10(IMG_SIZE, 4, 3, 384, 6, 64)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Skip healer for classification evaluation
        if model_type == "healer":
            continue
        
        model_results = {}
        
        # Evaluate at each severity
        for severity in severities:
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Severity {severity}"):
                    images, labels = images.to(device), labels.to(device)
                    batch_size = images.size(0)
                    
                    if severity == 0.0:
                        # Clean images - all models need normalization since we're using val_loader_no_norm
                        transformed_images = []
                        for i in range(batch_size):
                            transformed_images.append(normalize(images[i]))
                        transformed_images = torch.stack(transformed_images)
                    else:
                        # Apply transformations
                        transformed_images = []
                        for i in range(batch_size):
                            # Randomly choose transformation
                            transform_type = np.random.choice(continuous_transform.transform_types[1:])  # Skip 'no_transform'
                            transformed_img, _ = continuous_transform.apply_transforms(
                                images[i],
                                transform_type=transform_type,
                                severity=severity,
                                return_params=True
                            )
                            # Normalize after transformation for all models since we're using val_loader_no_norm
                            transformed_img = normalize(transformed_img)
                            transformed_images.append(transformed_img)
                        transformed_images = torch.stack(transformed_images)
                    
                    # Get predictions
                    if model_type in ["ttt", "blended", "ttt3fc", "blended3fc"]:
                        outputs, _ = model(transformed_images)
                    else:
                        outputs = model(transformed_images)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            model_results[severity] = accuracy
            print(f"  Severity {severity}: {accuracy:.4f}")
        
        all_results[model_name] = model_results
    
    # Print summary table
    print("\n" + "="*100)
    print("📊 TRANSFORMATION ROBUSTNESS SUMMARY")
    print("="*100)
    print(f"{'Model':<25}", end="")
    for sev in severities:
        print(f"{'Sev ' + str(sev):>10}", end="")
    print(f"{'Avg Drop':>12}")
    print("-" * 100)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<25}", end="")
        for sev in severities:
            if sev in results:
                print(f"{results[sev]:>10.4f}", end="")
            else:
                print(f"{'--':>10}", end="")
        # Calculate average drop from clean
        if 0.0 in results and len(results) > 1:
            clean_acc = results[0.0]
            avg_drop = np.mean([clean_acc - acc for sev, acc in results.items() if sev > 0])
            print(f"{avg_drop:>12.4f}")
        else:
            print(f"{'--':>12}")
    
    return all_results


def apply_inverse_transform(transformed_img, transform_params, device):
    """Apply inverse transformation based on predicted parameters"""
    img = transformed_img.clone()
    
    if transform_params['transform_type'] == 'gaussian_noise':
        # For noise, we can't perfectly remove it, but we can try denoising
        # Simple approach: slight gaussian blur to reduce noise
        # Note: In practice, you'd use a proper denoising method
        return img  # Return as-is for now
        
    elif transform_params['transform_type'] == 'rotation':
        # Apply counter-rotation
        angle = -transform_params['rotation_angle']
        # Convert to PIL, rotate, and back to tensor
        img_pil = transforms.ToPILImage()(img.cpu())
        img_pil = transforms.functional.rotate(img_pil, angle)
        img = transforms.ToTensor()(img_pil).to(device)
        
    elif transform_params['transform_type'] == 'affine':
        # Apply inverse affine transformation
        # Calculate inverse affine matrix
        translate_x = -transform_params['translate_x']
        translate_y = -transform_params['translate_y']
        shear_x = -transform_params['shear_x']
        shear_y = -transform_params['shear_y']
        
        # Convert to PIL and apply inverse affine
        img_pil = transforms.ToPILImage()(img.cpu())
        img_pil = transforms.functional.affine(
            img_pil,
            angle=0,
            translate=(translate_x * img_pil.width, translate_y * img_pil.height),
            scale=1.0,
            shear=(shear_x, shear_y)
        )
        img = transforms.ToTensor()(img_pil).to(device)
    
    return img


def evaluate_vit_with_healer_guidance(val_loader_no_norm, healer_model, vit_model, model_name="ViT"):
    """Evaluate ViT model using healer-guided inverse transforms"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    healer_model.eval()
    vit_model.eval()
    
    continuous_transform = ContinuousTransforms(severity=1.0)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    severities = [0.3, 0.5, 0.7, 1.0]
    results = {}
    
    print(f"\n🔍 Evaluating {model_name} with Healer Guidance...")
    
    for severity in severities:
        correct = 0
        correct_with_healer = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader_no_norm, desc=f"Severity {severity}")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                batch_size = images.size(0)
                
                for i in range(batch_size):
                    # Apply random transformation
                    transform_type = np.random.choice(continuous_transform.transform_types[1:])
                    transformed_img, true_params = continuous_transform.apply_transforms(
                        images[i],
                        transform_type=transform_type,
                        severity=severity,
                        return_params=True
                    )
                    
                    # Get healer predictions on transformed image
                    healer_input = normalize(transformed_img).unsqueeze(0)
                    healer_output = healer_model(healer_input)
                    
                    # Get predicted transform type
                    _, predicted_type_idx = torch.max(healer_output['transform_type_logits'], 1)
                    transform_types = ['no_transform', 'gaussian_noise', 'rotation', 'affine']
                    predicted_type = transform_types[predicted_type_idx.item()]
                    
                    # Create predicted parameters dictionary
                    # Extract affine parameters from the affine_params tensor
                    affine_params = healer_output['affine_params'][0]  # Shape: [4]
                    
                    predicted_params = {
                        'transform_type': predicted_type,
                        'rotation_angle': healer_output['rotation_angle'][0].item(),
                        'translate_x': affine_params[0].item() * 0.1,  # Scale back to original range
                        'translate_y': affine_params[1].item() * 0.1,
                        'shear_x': affine_params[2].item() * 15.0,
                        'shear_y': affine_params[3].item() * 15.0
                    }
                    
                    # Apply inverse transform using healer predictions
                    healed_img = apply_inverse_transform(transformed_img, predicted_params, device)
                    
                    # Evaluate on original transformed image
                    transformed_input = normalize(transformed_img).unsqueeze(0)
                    output = vit_model(transformed_input)
                    _, predicted = torch.max(output, 1)
                    if predicted.item() == labels[i].item():
                        correct += 1
                    
                    # Evaluate on healer-corrected image
                    healed_input = normalize(healed_img).unsqueeze(0)
                    output_healed = vit_model(healed_input)
                    _, predicted_healed = torch.max(output_healed, 1)
                    if predicted_healed.item() == labels[i].item():
                        correct_with_healer += 1
                    
                    total += 1
                
                # Update progress bar
                if total > 0:
                    pbar.set_postfix({
                        'acc': f'{correct/total:.3f}',
                        'healed_acc': f'{correct_with_healer/total:.3f}'
                    })
        
        accuracy = correct / total
        healed_accuracy = correct_with_healer / total
        improvement = healed_accuracy - accuracy
        
        results[severity] = {
            'original': accuracy,
            'healed': healed_accuracy,
            'improvement': improvement
        }
        
        print(f"  Severity {severity}: Original: {accuracy:.4f}, Healed: {healed_accuracy:.4f}, Improvement: {improvement:+.4f}")
    
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


def create_transformation_robustness_plots(transform_results):
    """Create plots showing model robustness to transformations"""
    save_dir = os.path.join(CHECKPOINT_PATH, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract severities from first model's results
    if not transform_results:
        return
    
    first_model_results = next(iter(transform_results.values()))
    severities = sorted(list(first_model_results.keys()))
    
    plt.figure(figsize=(14, 8))
    
    # Plot lines for each model
    for model_name, results in transform_results.items():
        accuracies = [results.get(sev, 0) for sev in severities]
        plt.plot(severities, accuracies, 'o-', linewidth=2, markersize=8, label=model_name)
    
    plt.xlabel('Transformation Severity', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Model Robustness to Continuous Transformations', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.xlim(-0.05, max(severities) + 0.05)
    plt.ylim(0, 1.05)
    
    # Add percentage labels
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'transformation_robustness.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Transformation robustness plot saved to: {plot_path}")
    plt.close()
    
    # Create a heatmap of model performance across severities
    plt.figure(figsize=(12, 8))
    
    model_names = list(transform_results.keys())
    performance_matrix = []
    
    for model_name in model_names:
        results = transform_results[model_name]
        row = [results.get(sev, 0) for sev in severities]
        performance_matrix.append(row)
    
    performance_matrix = np.array(performance_matrix)
    
    plt.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Accuracy')
    
    # Set ticks and labels
    plt.xticks(range(len(severities)), [f'{s:.1f}' for s in severities])
    plt.yticks(range(len(model_names)), model_names)
    
    plt.xlabel('Transformation Severity', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    plt.title('Model Performance Heatmap Across Transformation Severities', fontsize=16)
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(severities)):
            text = plt.text(j, i, f'{performance_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, 'transformation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"📊 Transformation heatmap saved to: {heatmap_path}")
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
    parser.add_argument("--train_healer", action="store_true", help="Train Healer model")
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
                args.train_blended, args.train_healer, args.train_all, args.evaluate, args.visualize,
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
    # Load data without normalization for TTT and Blended models
    train_loader_no_norm, val_loader_no_norm = load_cifar10_data_no_norm()
    
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
            train_ttt_models(train_loader_no_norm, val_loader_no_norm)
    
    if not args.retrain and not args.skip_training and (args.train_all or args.train_blended):
        blended_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_blended", "best_model.pt")
        blended3fc_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_blended3fc", "best_model.pt")
        
        if os.path.exists(blended_model_path) and os.path.exists(blended3fc_model_path):
            print(f"\n✓ Blended models already exist:")
            print(f"  - Blended: {blended_model_path}")
            print(f"  - Blended3fc: {blended3fc_model_path}")
        else:
            print("\n=== TRAINING BLENDED MODELS ===")
            train_blended_models(train_loader_no_norm, val_loader_no_norm)
    
    if not args.retrain and not args.skip_training and (args.train_all or args.train_healer):
        healer_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_healer", "best_model.pt")
        
        if os.path.exists(healer_model_path):
            print(f"\n✓ Healer model already exists at {healer_model_path}")
        else:
            print("\n=== TRAINING HEALER MODEL ===")
            train_healer_model(train_loader_no_norm, val_loader_no_norm)
    
    # Evaluation phase (skip if --skip_evaluation is set)
    if not args.skip_evaluation and (args.evaluate or args.train_all):
        print("\n=== EVALUATION PHASE ===")
        
        # First evaluate on clean data
        print("\n📋 Part 1: Clean Data Evaluation")
        results = evaluate_all_models(val_loader)
        
        # Then evaluate with transformations
        print("\n📋 Part 2: Transformation Robustness Evaluation")
        transform_results = evaluate_models_with_transforms(
            val_loader_no_norm, 
            severities=[0.0, 0.3, 0.5, 0.7, 1.0]
        )
        
        # Evaluate ViT models with healer guidance
        print("\n📋 Part 3: Healer-Guided Inverse Transform Evaluation")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load healer model
        healer_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_healer", "best_model.pt")
        if os.path.exists(healer_model_path):
            healer_model = TransformationHealerCIFAR10(IMG_SIZE, 4, 3, 384, 6, 64).to(device)
            checkpoint = torch.load(healer_model_path, map_location=device)
            healer_model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded Healer model")
            
            # Evaluate Main ViT with healer
            main_vit_path = os.path.join(CHECKPOINT_PATH, "bestmodel_main", "best_model.pt")
            if os.path.exists(main_vit_path):
                main_vit = create_vit_model(
                    img_size=IMG_SIZE, patch_size=4, in_chans=3, num_classes=NUM_CLASSES,
                    embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
                ).to(device)
                checkpoint = torch.load(main_vit_path, map_location=device)
                main_vit.load_state_dict(checkpoint['model_state_dict'])
                
                main_vit_healer_results = evaluate_vit_with_healer_guidance(
                    val_loader_no_norm, healer_model, main_vit, "Main ViT"
                )
            
            # Evaluate Robust ViT with healer
            robust_vit_path = os.path.join(CHECKPOINT_PATH, "bestmodel_robust", "best_model.pt")
            if os.path.exists(robust_vit_path):
                robust_vit = create_vit_model(
                    img_size=IMG_SIZE, patch_size=4, in_chans=3, num_classes=NUM_CLASSES,
                    embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
                ).to(device)
                checkpoint = torch.load(robust_vit_path, map_location=device)
                robust_vit.load_state_dict(checkpoint['model_state_dict'])
                
                robust_vit_healer_results = evaluate_vit_with_healer_guidance(
                    val_loader_no_norm, healer_model, robust_vit, "Robust ViT"
                )
                
            # Print summary of healer improvements
            print("\n" + "="*80)
            print("📊 HEALER-GUIDED IMPROVEMENT SUMMARY")
            print("="*80)
            if 'main_vit_healer_results' in locals():
                print("\nMain ViT Improvements:")
                for sev, res in main_vit_healer_results.items():
                    print(f"  Severity {sev}: {res['improvement']:+.2%}")
                    
            if 'robust_vit_healer_results' in locals():
                print("\nRobust ViT Improvements:")
                for sev, res in robust_vit_healer_results.items():
                    print(f"  Severity {sev}: {res['improvement']:+.2%}")
        else:
            print("⚠️ Healer model not found. Skipping healer-guided evaluation.")
        
        if args.visualize or args.train_all:
            print("\n=== CREATING VISUALIZATIONS ===")
            create_performance_plots(results)
            # Create transformation robustness plots
            create_transformation_robustness_plots(transform_results)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📁 All models and results saved in: {CHECKPOINT_PATH}")
    print(f"📊 Visualizations saved in: {os.path.join(CHECKPOINT_PATH, 'visualizations')}")


if __name__ == "__main__":
    main()
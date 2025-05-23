#!/usr/bin/env python3
"""
Minimal training script with batch size 3, 2 epochs, very small dataset
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import experimental models
from experimental_vit import create_experimental_vit

# Configuration
BATCH_SIZE = 3
EPOCHS = 2
TRAIN_SAMPLES = 6  # 2 batches
VAL_SAMPLES = 3    # 1 batch
LEARNING_RATE = 0.001
ARCHITECTURE = 'mixed'  # Using Mixed architecture (Fourier + Mamba)

print("🚀 Minimal Training Configuration")
print("="*50)
print(f"Architecture: {ARCHITECTURE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Train samples: {TRAIN_SAMPLES}")
print(f"Val samples: {VAL_SAMPLES}")
print(f"Learning rate: {LEARNING_RATE}")
print("="*50)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Using device: {device}")

# Data preparation
print("\n📊 Preparing dataset...")
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use CIFAR-10 for quick testing
train_dataset = datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)
val_dataset = datasets.CIFAR10(
    root='./data', 
    train=False, 
    transform=transform
)

# Create small subsets
train_subset = Subset(train_dataset, list(range(TRAIN_SAMPLES)))
val_subset = Subset(val_dataset, list(range(VAL_SAMPLES)))

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Dataset ready: {TRAIN_SAMPLES} train, {VAL_SAMPLES} val samples")

# Create model
print(f"\n🏗️  Creating {ARCHITECTURE} model...")
model = create_experimental_vit(
    img_size=32,
    patch_size=8,
    in_chans=3,
    num_classes=10,
    embed_dim=64,    # Small model
    depth=4,         # Few layers
    num_heads=4,
    attention_type=ARCHITECTURE
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model created with {total_params:,} parameters")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
print(f"\n🚀 Starting training for {EPOCHS} epochs...")
print("-"*50)

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
              f"Loss={loss.item():.4f}, "
              f"Acc={100.*train_correct/train_total:.1f}%")
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100. * train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100. * val_correct / val_total
    
    print(f"\n📊 Epoch {epoch+1} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.1f}%")
    print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.1f}%")
    print("-"*50)

print("\n✅ Training completed!")
print(f"Final results: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")
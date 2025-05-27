import os
import torch
import wandb
import numpy as np
import shutil
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from copy import deepcopy

# Import our custom ViT model
from transformer_utils import set_seed, LayerNorm, Mlp, TransformerTrunk
from vit_implementation import create_vit_model, PatchEmbed, VisionTransformer
from new_new import * 

MAX_ROTATION = 360.0 


def train_main_model_robust(dataset_path="tiny-imagenet-200", severity=0.3, model_dir="./"):
    """
    Train the main classification model with data augmentation using continuous transforms
    for improved robustness against distributional shifts.
    
    Args:
        dataset_path: Path to the Tiny ImageNet dataset
        severity: Severity of transformations during training (0.0-1.0)
    
    Returns:
        trained_model: The trained robust ViT model
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create checkpoint directories
    model_dir_path = Path(model_dir)
    checkpoints_dir = model_dir_path / "checkpoints_robust"
    best_model_dir = model_dir_path / "bestmodel_robust"
    
    # Create directories if they don't exist
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize our custom model
    vit_model = create_vit_model(
        img_size=64,        # Tiny ImageNet is 64x64
        patch_size=8,       # Smaller patch size for smaller images
        in_chans=3,
        num_classes=200,    # Tiny ImageNet has 200 classes
        embed_dim=384,      # Reduced embedding dimension
        depth=8,            # Reduced depth for faster training
        head_dim=64,
        mlp_ratio=4.0,
        use_resnet_stem=True
    )
    
    # Wrap it with loss calculation for compatibility
    model = CustomModelWithLoss(vit_model)
    model.to(device)
    
    # Find optimal batch size for the GPU
    batch_size = find_optimal_batch_size(model, img_size=64, starting_batch_size=128, device=device)
    
    # Define image transformations with normalization
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create continuous transforms for training data
    ood_transform_train = ContinuousTransforms(severity=severity)
    
    # Dataset and DataLoader - now with OOD transforms for training
    train_dataset = TinyImageNetDataset(
        dataset_path, "train", transform_train, ood_transform=ood_transform_train
    )
    # Validation set without OOD transforms
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Custom collate function for OOD transformed data
    def collate_ood(batch):
        orig_imgs, trans_imgs, labels, _ = zip(*batch)
        
        # Stack everything into tensors
        orig_tensor = torch.stack(orig_imgs)
        trans_tensor = torch.stack(trans_imgs)
        labels_tensor = torch.tensor(labels)
        
        # Return only transformed images and labels (original images not needed for training)
        return trans_tensor, labels_tensor
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_ood
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Training parameters
    num_epochs = 50
    learning_rate = 1e-4
    warmup_steps = 1000
    patience = 3
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    # Learning rate scheduler with linear warmup and cosine decay
    def get_lr(step, total_steps, warmup_steps, base_lr):
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)
        else:
            decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
            return base_lr * 0.5 * (1 + np.cos(np.pi * decay_ratio))
    
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: get_lr(step, total_steps, warmup_steps, 1.0)
    )
    
    # Logging with wandb
    wandb.config.update({
        "model": "robust_classification",
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "transform_severity": severity
    }, allow_val_change=True)
    
    # Initialize early stopping variables
    best_val_acc = 0
    early_stop_counter = 0
    best_epoch = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        # Training step - now using transformed images
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Free up GPU memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Validation step
        val_loss, val_acc = validate_main_model(model, val_loader, device)
        
        # Log metrics
        wandb.log({
            "robust/epoch": epoch + 1,
            "robust/train_loss": train_loss,
            "robust/train_accuracy": train_acc,
            "robust/val_loss": val_loss,
            "robust/val_accuracy": val_acc,
            "robust/learning_rate": scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            
            # Save checkpoint
            checkpoint_path = checkpoints_dir / f"model_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            
            # Save best model
            best_model_path = best_model_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, best_model_path)
            
            # Don't save to wandb when using external directories
            # wandb.save(str(checkpoint_path))
            # wandb.save(str(best_model_path))
            
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
            
            # Track the best epoch for later reference
            best_epoch = epoch + 1
        else:
            early_stop_counter += 1
            
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Clean up old checkpoints except the best one
    print("Cleaning up checkpoints to save disk space...")
    for checkpoint_file in checkpoints_dir.glob("*.pt"):
        if f"model_epoch{best_epoch}.pt" != checkpoint_file.name:
            checkpoint_file.unlink()
            print(f"Deleted {checkpoint_file}")
    
    print(f"Robust training completed. Best model saved at: {best_model_dir / 'best_model.pt'}")
    
    # Load and return the best model
    checkpoint = torch.load(best_model_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.vit_model  # Return the actual ViT model without the loss wrapper
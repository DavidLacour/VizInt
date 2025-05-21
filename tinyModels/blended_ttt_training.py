
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

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from new_new import ContinuousTransforms, TinyImageNetDataset

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

def train_blended_ttt_model(base_model, dataset_path="tiny-imagenet-200"):
    """
    Train the BlendedTTT model on the Tiny ImageNet dataset
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create checkpoint directories
    checkpoints_dir = Path("checkpoints_blended")
    best_model_dir = Path("bestmodel_blended")
    
    # Create directories if they don't exist
    checkpoints_dir.mkdir(exist_ok=True)
    best_model_dir.mkdir(exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize BlendedTTT model
    blended_model = BlendedTTT(base_model)
    blended_model.to(device)
    
    # Define image transformations
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create continuous transforms for OOD
    ood_transform = ContinuousTransforms(severity=1.0)
    
    # Dataset and DataLoader
    train_dataset = TinyImageNetDataset(
        dataset_path, "train", transform_train, ood_transform=ood_transform
    )
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # In debug mode, use small subsamples for quick testing
    if DEBUG:
        # Create a small subset of the data
        train_indices = list(range(10))  # Just 10 training samples
        val_indices = list(range(10))    # Just 10 validation samples
        
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        print(f"DEBUG MODE: Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # Set batch size and epochs based on debug mode
    if DEBUG:
        batch_size = 1
        num_epochs = 1
        print("DEBUG MODE: Using batch size of 1 and 1 epoch")
    else:
        batch_size = 64
        num_epochs = 30
    
    # Custom collate function for the blended model
    def blended_collate(batch):
        orig_imgs, trans_imgs, labels, params = zip(*batch)
        
        # Proper collation
        orig_tensor = torch.stack(orig_imgs)
        trans_tensor = torch.stack(trans_imgs)
        labels_tensor = torch.tensor(labels)
        
        # Process transform params
        transform_type_indices = []
        severity_values = []
        noise_std_values = []
        rotation_angle_values = []
        translate_x_values = []
        translate_y_values = []
        shear_x_values = []
        shear_y_values = []
        
        for p in params:
            transform_type_map = {
                'no_transform': 0,
                'gaussian_noise': 1,
                'rotation': 2,
                'affine': 3
            }
            transform_type = p.get('transform_type', 'no_transform')
            transform_type_indices.append(transform_type_map.get(transform_type, 0))
            
            severity_values.append(p.get('severity', 1.0))
            noise_std_values.append(p.get('noise_std', 0.0))
            rotation_angle_values.append(p.get('rotation_angle', 0.0))
            translate_x_values.append(p.get('translate_x', 0.0))
            translate_y_values.append(p.get('translate_y', 0.0))
            shear_x_values.append(p.get('shear_x', 0.0))
            shear_y_values.append(p.get('shear_y', 0.0))
        
        # Create transform targets dict
        transform_targets = {
            'transform_type_idx': torch.tensor(transform_type_indices),
            'severity': torch.tensor(severity_values).float().unsqueeze(1),
            'noise_std': torch.tensor(noise_std_values).float().unsqueeze(1),
            'rotation_angle': torch.tensor(rotation_angle_values).float().unsqueeze(1),
            'translate_x': torch.tensor(translate_x_values).float().unsqueeze(1),
            'translate_y': torch.tensor(translate_y_values).float().unsqueeze(1),
            'shear_x': torch.tensor(shear_x_values).float().unsqueeze(1),
            'shear_y': torch.tensor(shear_y_values).float().unsqueeze(1)
        }
        
        return orig_tensor, trans_tensor, labels_tensor, transform_targets
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0 if DEBUG else 4,
        pin_memory=True,
        collate_fn=blended_collate
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0 if DEBUG else 4,
        pin_memory=True
    )
    
    # Loss functions
    class_loss_fn = nn.CrossEntropyLoss()
    
    # Helper function to compute auxiliary loss
    def compute_aux_loss(predictions, targets):
        # Transform type classification loss
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        
        transform_type_loss = ce_loss(
            predictions['transform_type_logits'], 
            targets['transform_type_idx']
        )
        
        # Get masks for each transform type
        transform_types = targets['transform_type_idx']
        noise_mask = (transform_types == 1)  # noise is index 1
        rot_mask = (transform_types == 2)   # rotation is index 2
        affine_mask = (transform_types == 3)  # affine is index 3
        
        # Initialize parameter losses with zero tensors
        severity_noise_loss = torch.tensor(0.0, device=predictions['severity_noise'].device)
        severity_rotation_loss = torch.tensor(0.0, device=predictions['severity_rotation'].device)
        severity_affine_loss = torch.tensor(0.0, device=predictions['severity_affine'].device)
        rotation_loss = torch.tensor(0.0, device=predictions['rotation_angle'].device)
        noise_loss = torch.tensor(0.0, device=predictions['noise_std'].device)
        affine_loss = torch.tensor(0.0, device=predictions['translate_x'].device)
        
        # Compute losses only if we have samples of that type
        if noise_mask.sum() > 0:
            severity_noise_loss = mse_loss(
                predictions['severity_noise'][noise_mask],
                targets['severity'][noise_mask]
            )
            noise_loss = mse_loss(
                predictions['noise_std'][noise_mask], 
                targets['noise_std'][noise_mask]
            )
        
        if rot_mask.sum() > 0:
            severity_rotation_loss = mse_loss(
                predictions['severity_rotation'][rot_mask],
                targets['severity'][rot_mask]
            )
            rotation_loss = mse_loss(
                predictions['rotation_angle'][rot_mask], 
                targets['rotation_angle'][rot_mask]
            )
            
        if affine_mask.sum() > 0:
            severity_affine_loss = mse_loss(
                predictions['severity_affine'][affine_mask],
                targets['severity'][affine_mask]
            )
            
            # Combined MSE loss for all affine parameters
            translate_x_loss = mse_loss(
                predictions['translate_x'][affine_mask], 
                targets['translate_x'][affine_mask]
            )
            translate_y_loss = mse_loss(
                predictions['translate_y'][affine_mask], 
                targets['translate_y'][affine_mask]
            )
            shear_x_loss = mse_loss(
                predictions['shear_x'][affine_mask], 
                targets['shear_x'][affine_mask]
            )
            shear_y_loss = mse_loss(
                predictions['shear_y'][affine_mask], 
                targets['shear_y'][affine_mask]
            )
            
            # Average all affine parameter losses
            affine_loss = (translate_x_loss + translate_y_loss + shear_x_loss + shear_y_loss) / 4.0
        
        # Combine all severity losses
        severity_loss = (severity_noise_loss + severity_rotation_loss + severity_affine_loss) / 3.0
        
        # Combine losses
        total_loss = (
            transform_type_loss + 
            0.5 * severity_loss + 
            0.3 * (rotation_loss + noise_loss) +
            0.2 * affine_loss
        )
        
        return total_loss
    
    # Training parameters
    learning_rate = 1e-4
    warmup_steps = 500 if not DEBUG else 10
    
    # Optimizer
    optimizer = torch.optim.AdamW(blended_model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    # Learning rate scheduler
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
        "model": "blended_ttt",
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "aux_loss_weight": 0.05,  # 5% for auxiliary task
        "debug_mode": DEBUG
    }, allow_val_change=True)
    
    # Initialize best validation accuracy
    best_val_acc = 0
    
    # Training loop
    for epoch in range(num_epochs):
        blended_model.train()
        train_loss = 0
        train_class_loss = 0
        train_aux_loss = 0
        class_correct = 0
        total_samples = 0
        
        # Training step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (clean_images, transformed_images, labels, transform_targets) in enumerate(progress_bar):
            # Move tensors to device
            clean_images = clean_images.to(device)
            transformed_images = transformed_images.to(device)
            labels = labels.to(device)
            
            # Move transform_targets to device
            for key in transform_targets:
                transform_targets[key] = transform_targets[key].to(device)
            
            # Forward pass on transformed images
            logits, aux_outputs = blended_model(transformed_images)
            
            # Calculate losses
            class_loss = class_loss_fn(logits, labels)
            aux_loss = compute_aux_loss(aux_outputs, transform_targets)
            
            # Combined loss with 95% classification, 5% auxiliary
            loss = 0.95 * class_loss + 0.05 * aux_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(blended_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            train_class_loss += class_loss.item()
            train_aux_loss += aux_loss.item()
            
            # Calculate classification accuracy
            class_preds = torch.argmax(logits, dim=1)
            class_correct += (class_preds == labels).sum().item()
            total_samples += len(labels)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # In debug mode, break after first batch
            if DEBUG and batch_idx >= 0:
                print("DEBUG MODE: Breaking after first batch")
                break
        
        # Calculate training metrics
        num_batches = min(1, len(train_loader)) if DEBUG else len(train_loader)
        train_loss /= max(1, num_batches)
        train_class_loss /= max(1, num_batches)
        train_aux_loss /= max(1, num_batches)
        train_class_acc = class_correct / max(1, total_samples)
        
        # Validation step - standard validation on clean data with only class output
        blended_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                # In debug mode, only evaluate on one batch
                if DEBUG and batch_idx > 0:
                    break
                    
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass - only care about classification for validation
                logits, _ = blended_model(images)
                val_loss += class_loss_fn(logits, labels).item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Avoid division by zero
        val_loss /= max(1, len(val_loader) if not DEBUG else 1)
        val_acc = val_correct / max(1, val_total)
        
        # Log metrics
        wandb.log({
            "blended/epoch": epoch + 1,
            "blended/train_loss": train_loss,
            "blended/train_class_loss": train_class_loss,
            "blended/train_aux_loss": train_aux_loss,
            "blended/train_class_accuracy": train_class_acc,
            "blended/val_loss": val_loss,
            "blended/val_accuracy": val_acc
        })
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Class Acc: {train_class_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save checkpoint if it's the best model or in debug mode
        if val_acc > best_val_acc or DEBUG:
            best_val_acc = val_acc
            
            # Save best model
            best_model_path = best_model_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': blended_model.state_dict(),
                'val_acc': val_acc,
            }, best_model_path)
            
            if not DEBUG:  # Only print this in non-debug mode
                print(f"Saved best model with validation accuracy: {val_acc:.4f}")
            else:
                print(f"DEBUG MODE: Saved model checkpoint regardless of accuracy")
    
    # Make sure at least one best model has been saved
    best_model_path = best_model_dir / "best_model.pt"
    if not best_model_path.exists():
        print("WARNING: No best model was saved. Saving current model as best.")
        torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': blended_model.state_dict(),
            'val_acc': 0.0,  # Default accuracy
        }, best_model_path)
    
    print(f"BlendedTTT model training completed. Best model saved at: {best_model_dir / 'best_model.pt'}")
    
    # Load and return the best model
    checkpoint = torch.load(best_model_path)
    blended_model.load_state_dict(checkpoint['model_state_dict'])
    
    return blended_model
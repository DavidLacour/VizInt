import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy

from vit_implementation import PatchEmbed
from transformer_utils import LayerNorm, TransformerTrunk, set_seed

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




class TestTimeTrainer(nn.Module):
    """
    Standard Test-Time Training (TTT) Model for adapting to test distributions.
    
    This model uses a self-supervised auxiliary task (predicting transformations)
    to adapt a pre-trained model at test time without requiring labels.
    """
    def __init__(
        self, 
        base_model, 
        img_size=64, 
        patch_size=8, 
        embed_dim=384,
        adaptation_steps=10, 
        adaptation_lr=1e-4
    ):
        super().__init__()
        self.base_model = base_model
        self.adaptation_steps = adaptation_steps
        self.adaptation_lr = adaptation_lr
        
        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Patch embedding for the self-supervised task
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            use_resnet_stem=True
        )
        
        # Learnable cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Lightweight transformer for the adaptation task
        self.transformer = TransformerTrunk(
            dim=embed_dim,
            depth=2,  # Shallow depth for efficiency during test-time
            head_dim=64,
            mlp_ratio=4.0,
            use_bias=False
        )
        
        # Normalization layer
        self.norm = LayerNorm(embed_dim, bias=False)
        
        # Self-supervised task heads
        # Predict the transformation type
        self.transform_head = nn.Linear(embed_dim, 4)  # no_transform, gaussian_noise, rotation, affine
        
        # Store the original base model to reset when needed
        self.original_base_model = deepcopy(base_model)
    
    def forward_features(self, x):
        B = x.shape[0]
        
        # Extract patches
        x = self.patch_embed(x)
        
        # Add cls token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed.expand(B, -1, -1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Apply normalization
        x = self.norm(x)
        
        return x
    
    def forward(self, x, aux_only=False):
        # Extract features
        features = self.forward_features(x)
        
        # Get CLS token
        cls_features = features[:, 0]
        
        # Predict transformation
        transform_logits = self.transform_head(cls_features)
        
        if aux_only:
            return transform_logits
        
        # Run the base model for classification
        with torch.no_grad():
            logits = self.base_model(x)
        
        return logits, transform_logits
    
    def adapt(self, x, transform_labels=None, reset=False):
        """
        Adapt the model to a new test instance using the self-supervised task.
        
        Args:
            x: Input image or batch of images
            transform_labels: Optional transformation labels for supervised adaptation
            reset: Whether to reset the base model to its original state
        
        Returns:
            adapted_logits: Classification logits after adaptation
        """
        # Reset the base model if requested
        if reset:
            self.base_model.load_state_dict(self.original_base_model.state_dict())
        
        # Store original parameters for selective adaptation
        original_params = {name: param.clone() for name, param in self.base_model.named_parameters()}
        
        # Enable gradients for the base model during adaptation
        for param in self.base_model.parameters():
            param.requires_grad = True
        
        # Create a temporary optimizer for adaptation
        optimizer = torch.optim.Adam(
            list(self.parameters()) + list(self.base_model.parameters()),
            lr=self.adaptation_lr
        )
        
        # Adaptation loop
        self.train()
        self.base_model.train()
        
        for _ in range(self.adaptation_steps):
            # Forward pass
            logits, transform_logits = self(x)
            
            # Self-supervised loss based on predicting transformations
            if transform_labels is not None:
                # If labels are provided, use supervised loss
                aux_loss = F.cross_entropy(transform_logits, transform_labels)
            else:
                # Otherwise use self-supervised loss (entropy minimization)
                probs = F.softmax(transform_logits, dim=1)
                aux_loss = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            aux_loss.backward()
            optimizer.step()
        
        # Evaluate with adapted model
        self.eval()
        self.base_model.eval()
        with torch.no_grad():
            adapted_logits = self.base_model(x)
        
        # Restore gradients state
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        return adapted_logits
    
    def adapt_batch(self, dataloader, reset=True):
        """
        Adapt the model to a batch of test data.
        
        Args:
            dataloader: DataLoader containing test data
            reset: Whether to reset the model before adaptation
        
        Returns:
            predictions: Dictionary of predictions for all samples
        """
        if reset:
            self.base_model.load_state_dict(self.original_base_model.state_dict())
        
        # Collect all predictions
        all_predictions = []
        all_labels = []
        
        # Process each batch
        for batch in dataloader:
            # For standard dataset
            if len(batch) == 2:
                images, labels = batch
                transformed_images = images
                transform_labels = None
            # For OOD dataset
            elif len(batch) == 4:
                orig_images, transformed_images, labels, params = batch
                # Extract transform types if available
                transform_labels = [
                    {'no_transform': 0, 'gaussian_noise': 1, 'rotation': 2, 'affine': 3}
                    .get(p.get('transform_type', 'no_transform'), 0)
                    for p in params
                ]
                transform_labels = torch.tensor(transform_labels, device=transformed_images.device)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            # Adapt to this batch
            adapted_logits = self.adapt(transformed_images, transform_labels, reset=False)
            
            # Store predictions
            predictions = torch.argmax(adapted_logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
        
        return {
            'predictions': all_predictions,
            'labels': all_labels,
            'accuracy': accuracy
        }


def train_ttt_model(dataset_path, base_model=None, severity=0.5, epochs=10):
    """
    Train the TTT model on the given dataset.
    
    Args:
        dataset_path: Path to the dataset
        base_model: Pre-trained base model (if None, a new one will be loaded)
        severity: Severity of transformations for training
        epochs: Number of training epochs
        
    Returns:
        ttt_model: Trained TTT model
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create checkpoint directories
    checkpoints_dir = Path("checkpoints_ttt")
    best_model_dir = Path("bestmodel_ttt")
    
    # Create directories if they don't exist
    checkpoints_dir.mkdir(exist_ok=True)
    best_model_dir.mkdir(exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load or create base model
    if base_model is None:
        # Try to load the main model
        base_model_path = "bestmodel_main/best_model.pt"
        if os.path.exists(base_model_path):
            from vit_implementation import create_vit_model
            base_model = create_vit_model(
                img_size=64, patch_size=8, in_chans=3, num_classes=200,
                embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
            )
            
            checkpoint = torch.load(base_model_path, map_location=device)
            
            # Check if model wrapped in CustomModelWithLoss
            if 'vit_model' in checkpoint['model_state_dict']:
                # Create a new state dict with the correct keys
                new_state_dict = {}
                for key, value in checkpoint['model_state_dict'].items():
                    if key.startswith("vit_model."):
                        new_key = key[len("vit_model."):]
                        new_state_dict[new_key] = value
                
                base_model.load_state_dict(new_state_dict)
            else:
                base_model.load_state_dict(checkpoint['model_state_dict'])
                
            base_model = base_model.to(device)
            base_model.eval()
        else:
            raise ValueError("Base model not found. Please train the main model first.")
    
    # Initialize the TTT model
    ttt_model = TestTimeTrainer(
        base_model=base_model,
        img_size=64,
        patch_size=8,
        embed_dim=384,
        adaptation_steps=5,  # Fewer steps during training
        adaptation_lr=1e-4
    )
    ttt_model = ttt_model.to(device)
    
    # Define image transformations
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create continuous transforms for OOD
    from new_new import ContinuousTransforms, TinyImageNetDataset, find_optimal_batch_size
    ood_transform = ContinuousTransforms(severity=severity)
    
    # Dataset and DataLoader with OOD transforms
    train_dataset = TinyImageNetDataset(
        dataset_path, "train", transform_train, ood_transform=ood_transform
    )
    
    # Create a validation set (20% of training data)
    dataset_size = len(train_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"Training set size for TTT: {train_size}")
    print(f"Validation set size for TTT: {val_size}")
    
    # Simplified collate function
    def collate_fn(batch):
        orig_imgs, trans_imgs, labels, params = zip(*batch)
        
        orig_tensor = torch.stack(orig_imgs)
        trans_tensor = torch.stack(trans_imgs)
        labels_tensor = torch.tensor(labels)
        
        # Extract transform types
        transform_types = []
        for p in params:
            if isinstance(p, dict) and 'transform_type' in p:
                t_type = p['transform_type']
                if t_type == 'no_transform':
                    transform_types.append(0)
                elif t_type == 'gaussian_noise':
                    transform_types.append(1)
                elif t_type == 'rotation':
                    transform_types.append(2)
                elif t_type == 'affine':
                    transform_types.append(3)
                else:
                    transform_types.append(0)
            else:
                transform_types.append(0)
        
        transform_types_tensor = torch.tensor(transform_types)
        
        # Keep params as a list of dictionaries
        return orig_tensor, trans_tensor, labels_tensor, transform_types_tensor
    
    # Determine batch size and data loading parameters
    batch_size = 50  # min(64, find_optimal_batch_size(ttt_model, img_size=64, starting_batch_size=64, device=device))
    
    # DataLoaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Training parameters
    learning_rate = 1e-4
    patience = 3
    
    # Optimizer
    optimizer = torch.optim.AdamW(ttt_model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training phase
        ttt_model.train()
        train_loss = 0
        transform_type_acc = 0
        total_samples = 0
        
        # Training step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (orig_images, transformed_images, labels, transform_types) in enumerate(progress_bar):
            # Move to device
            orig_images = orig_images.to(device)
            transformed_images = transformed_images.to(device)
            transform_types = transform_types.to(device)
            
            # Forward pass (only for self-supervised task)
            transform_logits = ttt_model(transformed_images, aux_only=True)
            
            # Calculate loss
            loss = F.cross_entropy(transform_logits, transform_types)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ttt_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Calculate transform type accuracy
            pred_types = torch.argmax(transform_logits, dim=1)
            transform_type_acc += (pred_types == transform_types).sum().item()
            total_samples += len(pred_types)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Update scheduler
        scheduler.step()
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        transform_type_acc /= total_samples
        
        # Validation phase
        ttt_model.eval()
        val_loss = 0
        val_transform_type_acc = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (orig_images, transformed_images, labels, transform_types) in enumerate(val_loader):
                # Move to device
                orig_images = orig_images.to(device)
                transformed_images = transformed_images.to(device)
                transform_types = transform_types.to(device)
                
                # Forward pass (only for self-supervised task)
                transform_logits = ttt_model(transformed_images, aux_only=True)
                
                # Calculate loss
                loss = F.cross_entropy(transform_logits, transform_types)
                
                # Update metrics
                val_loss += loss.item()
                
                # Calculate transform type accuracy
                pred_types = torch.argmax(transform_logits, dim=1)
                val_transform_type_acc += (pred_types == transform_types).sum().item()
                val_total_samples += len(pred_types)
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_transform_type_acc /= val_total_samples
        
        # Log metrics with wandb if available
        try:
            import wandb
            wandb.log({
                "ttt/epoch": epoch + 1,
                "ttt/train_loss": train_loss,
                "ttt/val_loss": val_loss,
                "ttt/train_transform_type_accuracy": transform_type_acc,
                "ttt/val_transform_type_accuracy": val_transform_type_acc,
                "ttt/learning_rate": scheduler.get_last_lr()[0]
            })
        except:
            pass
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Transform Type Acc - Train: {transform_type_acc:.4f}, Val: {val_transform_type_acc:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # Save checkpoint
            checkpoint_path = checkpoints_dir / f"model_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': ttt_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            
            # Save best model
            best_model_path = best_model_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': ttt_model.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
            
            print(f"Saved best model with validation loss: {val_loss:.4f}")
            
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
    
    print(f"TTT model training completed. Best model saved at: {best_model_dir / 'best_model.pt'}")
    
    # Load and return the best model
    checkpoint = torch.load(best_model_dir / "best_model.pt")
    ttt_model.load_state_dict(checkpoint['model_state_dict'])
    
    return ttt_model

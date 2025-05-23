"""
Enhanced training system with robust early stopping and automatic checkpoint cleanup.
Only keeps the best model and deletes all intermediate checkpoints.
"""

import torch
import torch.nn as nn
from pathlib import Path
import shutil
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import wandb

class EarlyStoppingTrainer:
    """
    Enhanced trainer with robust early stopping and checkpoint management.
    Automatically cleans up all checkpoints except the best model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        patience: int = 5,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        save_dir: str = "checkpoints",
        model_name: str = "model",
        cleanup_checkpoints: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the early stopping trainer.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to use for training
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights at the end
            save_dir: Directory to save checkpoints
            model_name: Name for the model files
            cleanup_checkpoints: Whether to delete intermediate checkpoints
            verbose: Whether to print progress
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # Checkpoint management
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.cleanup_checkpoints = cleanup_checkpoints
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        # Checkpoint files tracking
        self.checkpoint_files = []
        self.best_checkpoint_file = None
        
        # Move model to device
        self.model.to(self.device)
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool = False) -> str:
        """Save model checkpoint"""
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
        }
        
        if self.scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Create checkpoint filename
        if is_best:
            checkpoint_file = self.save_dir / f"{self.model_name}_best.pt"
            self.best_checkpoint_file = checkpoint_file
        else:
            checkpoint_file = self.save_dir / f"{self.model_name}_epoch_{epoch:03d}.pt"
            self.checkpoint_files.append(checkpoint_file)
        
        torch.save(checkpoint_data, checkpoint_file)
        
        if self.verbose and is_best:
            print(f"💾 Saved best model: {checkpoint_file}")
        
        return str(checkpoint_file)
    
    def cleanup_intermediate_checkpoints(self):
        """Delete all intermediate checkpoint files, keeping only the best one"""
        if not self.cleanup_checkpoints:
            return
        
        deleted_count = 0
        for checkpoint_file in self.checkpoint_files:
            if checkpoint_file.exists() and checkpoint_file != self.best_checkpoint_file:
                try:
                    checkpoint_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Warning: Could not delete {checkpoint_file}: {e}")
        
        if self.verbose and deleted_count > 0:
            print(f"🗑️  Cleaned up {deleted_count} intermediate checkpoints")
        
        # Clear the list
        self.checkpoint_files = []
    
    def check_early_stopping(self, val_loss: float, val_acc: float, epoch: int) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            val_loss: Current validation loss
            val_acc: Current validation accuracy
            epoch: Current epoch
            
        Returns:
            True if training should stop, False otherwise
        """
        # Check if this is the best model so far
        # We use validation accuracy as primary metric, loss as secondary
        is_best = False
        
        if val_acc > self.best_val_acc + self.min_delta:
            # Significant improvement in accuracy
            is_best = True
        elif abs(val_acc - self.best_val_acc) <= self.min_delta and val_loss < self.best_val_loss - self.min_delta:
            # Similar accuracy but better loss
            is_best = True
        
        if is_best:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.wait_count = 0
            
            # Save best weights if requested
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            # Save best checkpoint
            self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
            
            if self.verbose:
                print(f"✨ New best model! Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            
            return False
        else:
            self.wait_count += 1
            
            # Save regular checkpoint (will be cleaned up later)
            self.save_checkpoint(epoch, val_loss, val_acc, is_best=False)
            
            if self.verbose:
                print(f"⏱️  No improvement for {self.wait_count}/{self.patience} epochs")
            
            if self.wait_count >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"🛑 Early stopping triggered at epoch {epoch}")
                    print(f"   Best epoch: {self.best_epoch}")
                    print(f"   Best val acc: {self.best_val_acc:.4f}")
                    print(f"   Best val loss: {self.best_val_loss:.4f}")
                return True
            
            return False
    
    def restore_best_model(self):
        """Restore the best model weights"""
        if self.restore_best_weights and self.best_weights is not None:
            # Move weights back to device and load
            best_weights_device = {k: v.to(self.device) for k, v in self.best_weights.items()}
            self.model.load_state_dict(best_weights_device)
            if self.verbose:
                print(f"🔄 Restored best model weights from epoch {self.best_epoch}")
    
    def train_epoch(self, epoch: int, criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.verbose)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Handle different batch formats
            if len(batch) == 2:
                inputs, labels = batch
            elif len(batch) == 4:  # OOD dataset format
                _, inputs, labels, _ = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]  # For models that return (logits, aux_outputs)
            else:
                logits = outputs
            
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy_score(all_labels[-len(preds):], preds):.4f}"
            })
        
        train_loss /= len(self.train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        
        return train_loss, train_acc
    
    def validate_epoch(self, epoch: int, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle different batch formats
                if len(batch) == 2:
                    inputs, labels = batch
                elif len(batch) == 4:  # OOD dataset format
                    _, inputs, labels, _ = batch
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # For models that return (logits, aux_outputs)
                else:
                    logits = outputs
                
                loss = criterion(logits, labels)
                
                # Update metrics
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
        
        val_loss /= len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        return val_loss, val_acc
    
    def train(
        self,
        epochs: int,
        criterion: nn.Module,
        log_wandb: bool = True,
        log_prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Train the model with early stopping.
        
        Args:
            epochs: Maximum number of epochs
            criterion: Loss function
            log_wandb: Whether to log to Weights & Biases
            log_prefix: Prefix for wandb logs
            
        Returns:
            Dictionary with training results
        """
        if self.verbose:
            print(f"🚀 Starting training with early stopping (patience={self.patience})")
            print(f"   Max epochs: {epochs}")
            print(f"   Device: {self.device}")
            print(f"   Cleanup checkpoints: {self.cleanup_checkpoints}")
        
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
        
        try:
            for epoch in range(1, epochs + 1):
                # Training phase
                train_loss, train_acc = self.train_epoch(epoch, criterion)
                
                # Validation phase
                val_loss, val_acc = self.validate_epoch(epoch, criterion)
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Store history
                training_history['train_loss'].append(train_loss)
                training_history['train_acc'].append(train_acc)
                training_history['val_loss'].append(val_loss)
                training_history['val_acc'].append(val_acc)
                training_history['epochs'].append(epoch)
                
                # Log to wandb
                if log_wandb:
                    log_dict = {
                        f"{log_prefix}epoch": epoch,
                        f"{log_prefix}train_loss": train_loss,
                        f"{log_prefix}train_acc": train_acc,
                        f"{log_prefix}val_loss": val_loss,
                        f"{log_prefix}val_acc": val_acc,
                    }
                    if self.scheduler is not None:
                        log_dict[f"{log_prefix}learning_rate"] = self.scheduler.get_last_lr()[0]
                    
                    try:
                        wandb.log(log_dict)
                    except:
                        pass  # Continue if wandb logging fails
                
                # Print progress
                if self.verbose:
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler else "N/A"
                    print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {lr}")
                
                # Check early stopping
                if self.check_early_stopping(val_loss, val_acc, epoch):
                    break
        
        except KeyboardInterrupt:
            if self.verbose:
                print(f"\n⚠️  Training interrupted by user at epoch {epoch}")
        
        # Restore best model
        self.restore_best_model()
        
        # Cleanup intermediate checkpoints
        self.cleanup_intermediate_checkpoints()
        
        # Final results
        results = {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'stopped_epoch': self.stopped_epoch if self.stopped_epoch > 0 else epoch,
            'total_epochs': epoch,
            'early_stopped': self.stopped_epoch > 0,
            'training_history': training_history,
            'best_model_path': str(self.best_checkpoint_file) if self.best_checkpoint_file else None
        }
        
        if self.verbose:
            print(f"\n🎉 Training completed!")
            print(f"   Total epochs: {results['total_epochs']}")
            print(f"   Best epoch: {results['best_epoch']}")
            print(f"   Best val accuracy: {results['best_val_acc']:.4f}")
            print(f"   Best val loss: {results['best_val_loss']:.4f}")
            print(f"   Early stopped: {'Yes' if results['early_stopped'] else 'No'}")
            if self.best_checkpoint_file:
                print(f"   Best model saved: {self.best_checkpoint_file}")
        
        return results

class EarlyStoppingConfig:
    """Configuration for early stopping"""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        monitor: str = 'val_acc',  # 'val_acc' or 'val_loss'
        mode: str = 'max',  # 'max' for accuracy, 'min' for loss
        restore_best_weights: bool = True,
        cleanup_checkpoints: bool = True,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.cleanup_checkpoints = cleanup_checkpoints
        self.verbose = verbose

def get_early_stopping_config(model_type: str, backbone_name: str) -> EarlyStoppingConfig:
    """Get early stopping configuration based on model type and backbone"""
    
    # Default configurations
    configs = {
        'classification': EarlyStoppingConfig(
            patience=5,
            min_delta=1e-4,
            monitor='val_acc',
            mode='max'
        ),
        'healer': EarlyStoppingConfig(
            patience=3,
            min_delta=1e-5,
            monitor='val_loss',
            mode='min'
        ),
        'ttt': EarlyStoppingConfig(
            patience=3,
            min_delta=1e-4,
            monitor='val_loss',
            mode='min'
        ),
        'blended_ttt': EarlyStoppingConfig(
            patience=4,
            min_delta=1e-4,
            monitor='val_acc',
            mode='max'
        )
    }
    
    # Backbone-specific adjustments
    backbone_adjustments = {
        'vit_base': {'patience': 7},  # Larger models might need more patience
        'swin_small': {'patience': 6},
        'vgg16': {'patience': 4},  # CNN models converge faster
        'resnet18': {'patience': 4},
        'resnet50': {'patience': 5}
    }
    
    # Get base config
    config = configs.get(model_type, configs['classification'])
    
    # Apply backbone adjustments
    if backbone_name in backbone_adjustments:
        adjustments = backbone_adjustments[backbone_name]
        for key, value in adjustments.items():
            setattr(config, key, value)
    
    return config
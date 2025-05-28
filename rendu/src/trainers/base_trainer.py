"""
Base trainer class for all training procedures
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
import time


class BaseTrainer(ABC):
    """Abstract base class for all trainers"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: str = 'cuda',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize base trainer
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to train on
            logger: Logger instance
        """
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.early_stop_counter = 0
        
        # Setup from config
        self._setup_from_config()
        
    def _setup_from_config(self):
        """Setup trainer from configuration"""
        train_config = self.config.get('training', {})
        
        # Early stopping
        self.early_stopping_enabled = train_config.get('early_stopping', {}).get('enabled', True)
        self.early_stopping_patience = train_config.get('early_stopping', {}).get('patience', 5)
        
        # Learning rate and optimization
        self.learning_rate = train_config.get('learning_rate', 0.001)
        self.weight_decay = train_config.get('weight_decay', 0.0)
        
        # Training parameters
        self.max_epochs = train_config.get('epochs', 100)
        self.gradient_clip_val = train_config.get('gradient_clip_val', None)
        
    @abstractmethod
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for training"""
        pass
    
    @abstractmethod
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        pass
    
    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Single training step
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary with 'loss' and any other metrics
        """
        pass
    
    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Single validation step
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary with metrics
        """
        pass
    
    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            
        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        epoch_metrics = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.training_step(batch, batch_idx)
            
            # Backward pass
            loss = outputs['loss']
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            optimizer.step()
            
            # Update metrics
            for key, value in outputs.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({k: f"{v:.4f}" for k, v in outputs.items()})
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of average metrics
        """
        self.model.eval()
        epoch_metrics = {}
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs} [Val]")
            
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = self._batch_to_device(batch)
                
                # Forward pass
                outputs = self.validation_step(batch, batch_idx)
                
                # Update metrics
                for key, value in outputs.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value.item()
                
                # Update progress bar
                pbar.set_postfix({k: f"{v:.4f}" for k, v in outputs.items()})
        
        # Average metrics
        num_batches = len(val_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_metrics
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              checkpoint_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        # Create optimizer and scheduler
        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(optimizer)
        
        # Training history
        history = {
            'train': {},
            'val': {},
            'best_epoch': 0,
            'best_metric': None
        }
        
        # Training loop
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, optimizer)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics.get('loss', val_metrics.get('accuracy', 0)))
                else:
                    scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_metrics(epoch, train_metrics, val_metrics, epoch_time)
            
            # Update history
            for key, value in train_metrics.items():
                if key not in history['train']:
                    history['train'][key] = []
                history['train'][key].append(value)
                
            for key, value in val_metrics.items():
                if key not in history['val']:
                    history['val'][key] = []
                history['val'][key].append(value)
            
            # Check for improvement and save checkpoint
            improved = self._check_improvement(val_metrics)
            
            if improved and checkpoint_dir is not None:
                self._save_checkpoint(checkpoint_dir, epoch, optimizer, val_metrics)
                history['best_epoch'] = epoch
                history['best_metric'] = self.best_metric
            
            # Early stopping
            if self.early_stopping_enabled:
                if not improved:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                else:
                    self.early_stop_counter = 0
        
        return history
    
    def _batch_to_device(self, batch: Any) -> Any:
        """Move batch to device"""
        if isinstance(batch, (list, tuple)):
            return [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
        elif isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch
    
    def _check_improvement(self, metrics: Dict[str, float]) -> bool:
        """Check if metrics improved"""
        # Override this method for custom improvement checking
        metric_name = 'accuracy' if 'accuracy' in metrics else 'loss'
        current_metric = metrics[metric_name]
        
        if self.best_metric is None:
            self.best_metric = current_metric
            return True
        
        if metric_name == 'accuracy':
            improved = current_metric > self.best_metric
        else:  # loss
            improved = current_metric < self.best_metric
            
        if improved:
            self.best_metric = current_metric
            
        return improved
    
    def _save_checkpoint(self, checkpoint_dir: Path, epoch: int, 
                        optimizer: torch.optim.Optimizer, metrics: Dict[str, float]):
        """Save checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "best_model.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float], epoch_time: float):
        """Log epoch metrics"""
        log_str = f"Epoch {epoch + 1}/{self.max_epochs} - "
        log_str += f"Time: {epoch_time:.1f}s - "
        
        # Train metrics
        train_str = ", ".join([f"train_{k}: {v:.4f}" for k, v in train_metrics.items()])
        log_str += train_str + " - "
        
        # Val metrics
        val_str = ", ".join([f"val_{k}: {v:.4f}" for k, v in val_metrics.items()])
        log_str += val_str
        
        self.logger.info(log_str)
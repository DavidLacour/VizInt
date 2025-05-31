"""
Trainer for image corrector models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

from models.pretrained_correctors import PretrainedUNetCorrector, ImageToImageTransformer, HybridCorrector
from data.continuous_transforms import ContinuousTransforms


class CorrectorTrainer:
    """Trainer for image correction models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize corrector trainer
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.num_epochs = config.get('num_epochs', 50)
        
        self.model_type = config.get('model_type', 'unet')  # 'unet', 'transformer', 'hybrid'
        self.loss_type = config.get('loss_type', 'l1')  # 'l1', 'l2', 'perceptual', 'combined'
        self.transform_types = config.get('transform_types', ['gaussian_noise', 'rotation', 'affine'])
        self.severity_range = config.get('severity_range', [0.1, 0.8])
        
        self.l1_weight = config.get('l1_weight', 1.0)
        self.l2_weight = config.get('l2_weight', 0.5)
        self.perceptual_weight = config.get('perceptual_weight', 0.1)
        
        self.transforms = ContinuousTransforms(severity=1.0)
        
        if 'perceptual' in self.loss_type:
            self._init_perceptual_loss()
    
    def _init_perceptual_loss(self):
        """Initialize perceptual loss using pre-trained VGG"""
        import torchvision.models as models
        
        vgg = models.vgg16(pretrained=True).features[:16]  # Up to conv3_3
        self.perceptual_net = vgg.eval()
        for param in self.perceptual_net.parameters():
            param.requires_grad = False
        
        if torch.cuda.is_available():
            self.perceptual_net = self.perceptual_net.cuda()
    
    def _perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss using VGG features
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            
        Returns:
            Perceptual loss
        """
        # Normalize to ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (torch.clamp(pred, -1, 1) + 1) / 2
        target_norm = (torch.clamp(target, -1, 1) + 1) / 2
        
        pred_norm = (pred_norm - mean) / std
        target_norm = (target_norm - mean) / std
        
        # Resize if necessary (VGG expects at least 32x32)
        if pred_norm.shape[-1] < 32:
            pred_norm = torch.nn.functional.interpolate(pred_norm, size=(32, 32), mode='bilinear')
            target_norm = torch.nn.functional.interpolate(target_norm, size=(32, 32), mode='bilinear')
        
        pred_features = self.perceptual_net(pred_norm)
        target_features = self.perceptual_net(target_norm)
        
        return torch.nn.functional.mse_loss(pred_features, target_features)
    
    def _calculate_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate correction loss
        
        Args:
            pred: Predicted corrected image [B, C, H, W]
            target: Ground truth clean image [B, C, H, W]
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0
        
        if self.loss_type in ['l1', 'combined']:
            l1_loss = torch.nn.functional.l1_loss(pred, target)
            losses['l1'] = l1_loss
            total_loss += self.l1_weight * l1_loss
        
        if self.loss_type in ['l2', 'combined']:
            l2_loss = torch.nn.functional.mse_loss(pred, target)
            losses['l2'] = l2_loss
            total_loss += self.l2_weight * l2_loss
        
        if self.loss_type in ['perceptual', 'combined']:
            if hasattr(self, 'perceptual_net'):
                perceptual_loss = self._perceptual_loss(pred, target)
                losses['perceptual'] = perceptual_loss
                total_loss += self.perceptual_weight * perceptual_loss
        
        losses['total'] = total_loss
        return losses
    
    def _generate_training_batch(self, clean_images: torch.Tensor) -> tuple:
        """
        Generate training batch with random transformations
        
        Args:
            clean_images: Clean images [B, C, H, W]
            
        Returns:
            Tuple of (corrupted_images, clean_images)
        """
        batch_size = clean_images.shape[0]
        corrupted_images = []
        
        for i in range(batch_size):
            transform_type = np.random.choice(self.transform_types)
            severity = np.random.uniform(self.severity_range[0], self.severity_range[1])
            
            clean_img = clean_images[i]
            
            corrupted_img = self.transforms.apply_transforms(
                clean_img, 
                transform_type=transform_type, 
                severity=severity
            )
            
            corrupted_images.append(corrupted_img)
        
        corrupted_batch = torch.stack(corrupted_images)
        return corrupted_batch, clean_images
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Train one epoch
        
        Args:
            model: Model to train
            dataloader: Training dataloader
            optimizer: Optimizer
            
        Returns:
            Dictionary of average losses
        """
        model.train()
        epoch_losses = {}
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (clean_images, _) in enumerate(progress_bar):
            clean_images = clean_images.to(self.device)
            
            corrupted_images, targets = self._generate_training_batch(clean_images)
            corrupted_images = corrupted_images.to(self.device)
            
            optimizer.zero_grad()
            predictions = model(corrupted_images)
            
            losses = self._calculate_loss(predictions, targets)
            
            losses['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0
                epoch_losses[key] += value.item()
            
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'l1': f"{losses.get('l1', torch.tensor(0)).item():.4f}"
            })
        
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate one epoch
        
        Args:
            model: Model to validate
            dataloader: Validation dataloader
            
        Returns:
            Dictionary of validation metrics
        """
        model.eval()
        epoch_losses = {}
        num_batches = 0
        psnr_scores = []
        
        with torch.no_grad():
            for clean_images, _ in tqdm(dataloader, desc="Validation"):
                clean_images = clean_images.to(self.device)
                
                corrupted_images, targets = self._generate_training_batch(clean_images)
                corrupted_images = corrupted_images.to(self.device)
                
                predictions = model(corrupted_images)
                
                losses = self._calculate_loss(predictions, targets)
                
                mse = torch.nn.functional.mse_loss(predictions, targets)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                psnr_scores.append(psnr.item())
                
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value.item()
                
                num_batches += 1
        
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        epoch_losses['psnr'] = np.mean(psnr_scores)
        
        return epoch_losses
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None, 
              save_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Full training loop
        
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        model = model.to(self.device)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs
        )
        
        history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_losses = self.train_epoch(model, train_loader, optimizer)
            history['train_losses'].append(train_losses)
            
            # Validate
            if val_loader is not None:
                val_losses = self.validate_epoch(model, val_loader)
                history['val_losses'].append(val_losses)
                
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    patience_counter = 0
                    if save_dir:
                        save_path = save_dir / "best_model.pt"
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'val_losses': val_losses,
                            'train_losses': train_losses
                        }, save_path)
                        self.logger.info(f"Saved checkpoint to {save_path}")
                else:
                    patience_counter += 1
                
                self.logger.info(
                    f"Train Loss: {train_losses['total']:.4f}, "
                    f"Val Loss: {val_losses['total']:.4f}, "
                    f"PSNR: {val_losses['psnr']:.2f}dB, "
                    f"Patience: {patience_counter}/{patience}"
                )
                
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                self.logger.info(f"Train Loss: {train_losses['total']:.4f}")
            
            scheduler.step()
            history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            # Save intermediate checkpoint
            if save_dir and (epoch + 1) % 10 == 0:
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_losses': train_losses
                }, checkpoint_path)
        
        return history
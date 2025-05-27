"""
Trainer for BlendedTTT models
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
from src.trainers.base_trainer import BaseTrainer


class BlendedTrainer(BaseTrainer):
    """Trainer for BlendedTTT and BlendedTTT3fc models"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Initialize Blended trainer
        
        Args:
            model: BlendedTTT model to train
            config: Training configuration
            device: Device to train on
        """
        super().__init__(model, config, device)
        self.criterion = nn.CrossEntropyLoss()
        self.aux_criterion = nn.CrossEntropyLoss()
        
        # Get auxiliary loss weight
        self.aux_loss_weight = config.get('training', {}).get('aux_loss_weight', 0.5)
        
        # Import transformation module
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from new_new import ContinuousTransforms
        
        # Get dataset config to determine normalization
        dataset_name = config.get('dataset_name', 'tinyimagenet')
        if dataset_name == 'cifar10':
            self.normalize = torch.nn.Sequential(
                torch.nn.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                 std=[0.2023, 0.1994, 0.2010])
            )
        else:
            self.normalize = torch.nn.Sequential(
                torch.nn.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
            )
        
        self.continuous_transform = ContinuousTransforms(severity=0.5)
        
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for Blended training"""
        # Blended models use slightly higher learning rate
        lr = self.config.get('training', {}).get('learning_rate', 0.0005)
        return optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.weight_decay
        )
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=5,
            factor=0.5
        )
    
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Single training step for BlendedTTT"""
        images, labels = batch
        batch_size = images.size(0)
        
        # Apply transformations
        transformed_images = []
        transform_labels = []
        
        for i in range(batch_size):
            # Randomly choose transformation type
            transform_type = np.random.choice(self.continuous_transform.transform_types)
            transform_type_idx = self.continuous_transform.transform_types.index(transform_type)
            
            # Apply transformation with random severity
            severity = np.random.uniform(0.0, 1.0)
            transformed_img, _ = self.continuous_transform.apply_transforms_unnormalized(
                images[i], 
                transform_type=transform_type,
                severity=severity,
                return_params=True
            )
            
            # Normalize after transformation
            transformed_img = self.normalize(transformed_img)
            
            transformed_images.append(transformed_img)
            transform_labels.append(transform_type_idx)
        
        transformed_images = torch.stack(transformed_images)
        transform_labels = torch.tensor(transform_labels, device=self.device)
        
        # Forward pass
        class_logits, aux_outputs = self.model(transformed_images)
        
        # Classification loss
        cls_loss = self.criterion(class_logits, labels)
        
        # Auxiliary loss (transformation prediction)
        aux_loss = self.aux_criterion(aux_outputs['transform_type_logits'], transform_labels)
        
        # Combined loss
        loss = cls_loss + self.aux_loss_weight * aux_loss
        
        # Calculate accuracies
        _, predicted = torch.max(class_logits, 1)
        cls_accuracy = (predicted == labels).float().mean()
        
        _, aux_predicted = torch.max(aux_outputs['transform_type_logits'], 1)
        aux_accuracy = (aux_predicted == transform_labels).float().mean()
        
        return {
            'loss': loss,
            'cls_loss': cls_loss,
            'aux_loss': aux_loss,
            'accuracy': cls_accuracy,
            'aux_accuracy': aux_accuracy
        }
    
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Single validation step for BlendedTTT"""
        images, labels = batch
        
        # For validation, use clean images
        # Forward pass
        class_logits, _ = self.model(images)
        
        # Compute loss
        loss = self.criterion(class_logits, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(class_logits, 1)
        accuracy = (predicted == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def _check_improvement(self, metrics: Dict[str, float]) -> bool:
        """Check if validation metrics improved"""
        # For Blended models, we check accuracy improvement
        current_metric = metrics.get('accuracy', 0)
        
        if self.best_metric is None:
            self.best_metric = current_metric
            return True
        
        improved = current_metric > self.best_metric
        if improved:
            self.best_metric = current_metric
            
        return improved
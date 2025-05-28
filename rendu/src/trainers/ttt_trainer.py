"""
Trainer for TTT (Test-Time Training) models
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Any, Optional
from trainers.base_trainer import BaseTrainer


class TTTTrainer(BaseTrainer):
    """Trainer for TTT and TTT3fc models"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Initialize TTT trainer
        
        Args:
            model: TTT model to train
            config: Training configuration
            device: Device to train on
        """
        super().__init__(model, config, device)
        self.criterion = nn.CrossEntropyLoss()
        
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from src.data.continuous_transforms import ContinuousTransforms
        
        dataset_name = config.get('dataset_name', 'tinyimagenet')
        if dataset_name == 'cifar10':
            self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                                std=[0.2023, 0.1994, 0.2010])
        else:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
        
        self.continuous_transform = ContinuousTransforms(severity=0.5)
        
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for TTT training"""
        lr = self.config.get('training', {}).get('learning_rate', 0.0001)
        return optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.weight_decay
        )
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
    
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Single training step for TTT"""
        images, _ = batch  # We don't use labels for TTT training
        batch_size = images.size(0)
        
        # Apply transformations and create transformation labels
        transformed_images = []
        transform_labels = []
        
        for i in range(batch_size):
            transform_type = np.random.choice(self.continuous_transform.transform_types)
            transform_type_idx = self.continuous_transform.transform_types.index(transform_type)
            
            severity = np.random.uniform(0.0, 1.0)
            transformed_img, _ = self.continuous_transform.apply_transforms_unnormalized(
                images[i], 
                transform_type=transform_type,
                severity=severity,
                return_params=True
            )
            
            transformed_img = self.normalize(transformed_img)
            
            transformed_images.append(transformed_img)
            transform_labels.append(transform_type_idx)
        
        transformed_images = torch.stack(transformed_images)
        transform_labels = torch.tensor(transform_labels, device=self.device)
        
        logits, aux_outputs = self.model(transformed_images)
        transform_logits = aux_outputs['transform_predictions']
        
        # Compute loss - use only the transformation prediction part
        loss = self.criterion(transform_logits[:, :self.model.num_transforms], transform_labels)
        
        # Calculate accuracy
        _, predicted = torch.max(transform_logits[:, :self.model.num_transforms], 1)
        accuracy = (predicted == transform_labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Single validation step for TTT"""
        images, _ = batch
        batch_size = images.size(0)
        
        # Apply transformations with fixed severity for validation
        transformed_images = []
        transform_labels = []
        
        for i in range(batch_size):
            # Randomly choose transformation type
            transform_type = np.random.choice(self.continuous_transform.transform_types)
            transform_type_idx = self.continuous_transform.transform_types.index(transform_type)
            
            # Apply transformation with fixed severity
            transformed_img, _ = self.continuous_transform.apply_transforms_unnormalized(
                images[i], 
                transform_type=transform_type,
                severity=0.5,  # Fixed severity for validation
                return_params=True
            )
            
            transformed_img = self.normalize(transformed_img)
            
            transformed_images.append(transformed_img)
            transform_labels.append(transform_type_idx)
        
        transformed_images = torch.stack(transformed_images)
        transform_labels = torch.tensor(transform_labels, device=self.device)
        
    
        logits, aux_outputs = self.model(transformed_images)
        transform_logits = aux_outputs['transform_predictions']
        
        loss = self.criterion(transform_logits[:, :self.model.num_transforms], transform_labels)
        
        _, predicted = torch.max(transform_logits[:, :self.model.num_transforms], 1)
        accuracy = (predicted == transform_labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
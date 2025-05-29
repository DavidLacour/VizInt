"""
Trainer for standard classification models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from trainers.base_trainer import BaseTrainer
from data.continuous_transforms import ContinuousTransforms

class ClassificationTrainer(BaseTrainer):
    """Trainer for standard classification models"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: str = 'cuda',
                 robust_training: bool = False):
        """
        Initialize classification trainer
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to train on
            robust_training: Whether to use robust training with transformations
        """
        super().__init__(model, config, device)
        self.robust_training = robust_training
        self.criterion = nn.CrossEntropyLoss()
        
        if self.robust_training:
          
            
            robust_config = config.get('training', {}).get('robust', {})
            self.transform_severity = robust_config.get('severity', 0.5)
            self.apply_probability = robust_config.get('apply_probability', 0.5)
            self.continuous_transform = ContinuousTransforms(severity=self.transform_severity)
            
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for training"""
        optimizer_type = self.config.get('training', {}).get('optimizer', 'AdamW')
        
        if optimizer_type == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        scheduler_config = self.config.get('training', {}).get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'CosineAnnealingLR')
        
        if scheduler_type == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', self.max_epochs)
            )
        elif scheduler_type == 'StepLR':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                patience=scheduler_config.get('patience', 5),
                factor=scheduler_config.get('factor', 0.5)
            )
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Single training step"""
        images, labels = batch
        
        if self.robust_training and hasattr(self, 'continuous_transform'):
            images = self._apply_robust_transformations(images)
        outputs = self.model(images)
        
        #crash if the trainer is not used for classification model only 
        assert(not isinstance(outputs, tuple))
       
        logits = outputs
        
        loss = self.criterion(logits, labels)
        
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Single validation step"""
        images, labels = batch
        
        outputs = self.model(images)
        
        # Handle models that return tuples (like TTTWrapper)
        if isinstance(outputs, tuple):
            logits = outputs[0]
            aux_outputs = outputs[1] if len(outputs) > 1 else {}
        else:
            logits = outputs
            aux_outputs = {}
        
        loss = self.criterion(logits, labels)
        
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def _apply_robust_transformations(self, images: torch.Tensor) -> torch.Tensor:
        """Apply robust training transformations"""
        import numpy as np
        
        batch_size = images.size(0)
        transformed_images = []
        
        for i in range(batch_size):
            if np.random.rand() > self.apply_probability:
                transformed_images.append(images[i])
            else:
                transform_type = np.random.choice(self.continuous_transform.transform_types[1:])  
                severity = np.random.uniform(0.0, self.transform_severity)
                transformed_img, _ = self.continuous_transform.apply_transforms_unnormalized(
                    images[i], 
                    transform_type=transform_type,
                    severity=severity,
                    return_params=True
                )
                transformed_images.append(transformed_img)
        
        return torch.stack(transformed_images)
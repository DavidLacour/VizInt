"""
Trainer for Healer models with transformation prediction
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np
from trainers.base_trainer import BaseTrainer
from data.continuous_transforms import ContinuousTransforms


class HealerTrainer(BaseTrainer):
    """Trainer for Healer models with multi-task learning"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Initialize Healer trainer
        
        Args:
            model: Healer model to train
            config: Training configuration
            device: Device to train on
        """
        super().__init__(model, config, device)
        
        # Loss functions (no classification needed!)
        self.transform_type_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()
        
        # Training configuration
        train_config = config.get('training', {})
        healer_config = train_config.get('healer', {})
        
        # Loss weights (only for transformation prediction)
        self.transform_type_weight = healer_config.get('transform_type_weight', 1.0)
        self.transform_params_weight = healer_config.get('transform_params_weight', 0.5)
        
        # Transformation settings
        self.transform_severity = healer_config.get('transform_severity', 0.5)
        self.transform_probability = healer_config.get('transform_probability', 0.8)
        
        # Create continuous transforms
        self.continuous_transforms = ContinuousTransforms(severity=self.transform_severity)
        self.num_transform_types = len(self.continuous_transforms.transform_types)
        
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
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def apply_transformations_with_labels(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply transformations to images and return transformation labels
        
        Args:
            images: Input images [B, C, H, W]
            labels: Class labels [B]
            
        Returns:
            Tuple of:
                - Transformed images [B, C, H, W]
                - Class labels [B]
                - Dictionary of transformation ground truth
        """
        batch_size = images.size(0)
        transformed_images = []
        transform_types = []
        rotation_angles = []
        noise_stds = []
        translate_xs = []
        translate_ys = []
        shear_xs = []
        shear_ys = []
        
        for i in range(batch_size):
            if np.random.rand() < self.transform_probability:
                # Choose random transformation type (excluding 'none')
                transform_idx = np.random.randint(1, self.num_transform_types)
                transform_type = self.continuous_transforms.transform_types[transform_idx]
                
                # Random severity
                severity = np.random.uniform(0.1, self.transform_severity)
                
                # Apply transformation and get parameters
                transformed_img, params = self.continuous_transforms.apply_transforms_unnormalized(
                    images[i], 
                    transform_type=transform_type,
                    severity=severity,
                    return_params=True
                )
                transformed_images.append(transformed_img)
                transform_types.append(transform_idx)
                
                # Extract transformation parameters
                rotation_angles.append(params.get('angle', 0.0))
                noise_stds.append(params.get('noise_std', 0.0))
                translate_xs.append(params.get('translate_x', 0.0))
                translate_ys.append(params.get('translate_y', 0.0))
                shear_xs.append(params.get('shear_x', 0.0))
                shear_ys.append(params.get('shear_y', 0.0))
            else:
                # No transformation
                transformed_images.append(images[i])
                transform_types.append(0)  # 'none' type
                rotation_angles.append(0.0)
                noise_stds.append(0.0)
                translate_xs.append(0.0)
                translate_ys.append(0.0)
                shear_xs.append(0.0)
                shear_ys.append(0.0)
        
        # Stack into tensors
        transformed_images = torch.stack(transformed_images)
        transform_labels = {
            'transform_type': torch.tensor(transform_types, dtype=torch.long, device=self.device),
            'rotation_angle': torch.tensor(rotation_angles, dtype=torch.float32, device=self.device).unsqueeze(1),
            'noise_std': torch.tensor(noise_stds, dtype=torch.float32, device=self.device).unsqueeze(1),
            'translate_x': torch.tensor(translate_xs, dtype=torch.float32, device=self.device).unsqueeze(1),
            'translate_y': torch.tensor(translate_ys, dtype=torch.float32, device=self.device).unsqueeze(1),
            'shear_x': torch.tensor(shear_xs, dtype=torch.float32, device=self.device).unsqueeze(1),
            'shear_y': torch.tensor(shear_ys, dtype=torch.float32, device=self.device).unsqueeze(1)
        }
        
        return transformed_images, labels, transform_labels
    
    def compute_losses(self, predictions: Dict[str, torch.Tensor], 
                      transform_labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute transformation prediction losses
        
        Args:
            predictions: Model predictions for transformations
            transform_labels: Ground truth transformation parameters
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Transform type prediction loss (categorical)
        if 'transform_type_logits' in predictions:
            losses['transform_type'] = self.transform_type_criterion(
                predictions['transform_type_logits'], 
                transform_labels['transform_type']
            )
        
        # Regression losses for transformation parameters
        if 'rotation_angle' in predictions:
            losses['rotation'] = self.regression_criterion(
                predictions['rotation_angle'], 
                transform_labels['rotation_angle']
            )
        
        if 'noise_std' in predictions:
            losses['noise'] = self.regression_criterion(
                predictions['noise_std'], 
                transform_labels['noise_std']
            )
        
        if 'translate_x' in predictions and 'translate_y' in predictions:
            losses['translation'] = self.regression_criterion(
                predictions['translate_x'], 
                transform_labels['translate_x']
            ) + self.regression_criterion(
                predictions['translate_y'], 
                transform_labels['translate_y']
            )
        
        if 'shear_x' in predictions and 'shear_y' in predictions:
            losses['shear'] = self.regression_criterion(
                predictions['shear_x'], 
                transform_labels['shear_x']
            ) + self.regression_criterion(
                predictions['shear_y'], 
                transform_labels['shear_y']
            )
        
        # Combine losses
        total_loss = torch.tensor(0.0, device=self.device)
        
        if 'transform_type' in losses:
            total_loss = total_loss + self.transform_type_weight * losses['transform_type']
        
        # Add regression losses
        for key in ['rotation', 'noise', 'translation', 'shear']:
            if key in losses:
                total_loss = total_loss + self.transform_params_weight * losses[key]
        
        losses['total'] = total_loss
        return losses
    
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Single training step for transformation prediction"""
        images, labels = batch
        
        # Apply transformations and get ground truth
        transformed_images, labels, transform_labels = self.apply_transformations_with_labels(images, labels)
        
        # Forward pass - model returns transformation predictions only
        outputs = self.model(transformed_images, training_mode=True)
        
        # Compute losses (no class labels needed)
        losses = self.compute_losses(outputs, transform_labels)
        
        # Compute metrics
        metrics = {'loss': losses['total']}
        
        # Transform type accuracy
        if 'transform_type_logits' in outputs:
            _, predicted_transforms = torch.max(outputs['transform_type_logits'], 1)
            transform_accuracy = (predicted_transforms == transform_labels['transform_type']).float().mean()
            metrics['transform_acc'] = transform_accuracy
        
        # Add individual losses to metrics
        for key, value in losses.items():
            if key != 'total':
                metrics[f'loss_{key}'] = value
        
        return metrics
    
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Single validation step"""
        images, labels = batch
        
        # Apply transformations for validation too
        transformed_images, labels, transform_labels = self.apply_transformations_with_labels(images, labels)
        
        # Forward pass
        outputs = self.model(transformed_images, training_mode=True)
        
        # Compute losses (no class labels needed)
        losses = self.compute_losses(outputs, transform_labels)
        
        # Compute metrics
        metrics = {'loss': losses['total']}
        
        # Transform type accuracy
        if 'transform_type_logits' in outputs:
            _, predicted_transforms = torch.max(outputs['transform_type_logits'], 1)
            transform_accuracy = (predicted_transforms == transform_labels['transform_type']).float().mean()
            metrics['transform_acc'] = transform_accuracy
        
        return metrics
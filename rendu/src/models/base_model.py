"""
Base model class that all models should inherit from
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import logging


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Output tensor
        """
        pass
    
    def save_checkpoint(self, path: Path, epoch: int, optimizer: Optional[torch.optim.Optimizer] = None, 
                       metrics: Optional[Dict[str, float]] = None):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer state to save
            metrics: Additional metrics to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        if metrics is not None:
            checkpoint['metrics'] = metrics
            
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: Path, config: Optional[Dict[str, Any]] = None, 
                       device: str = 'cpu') -> Tuple['BaseModel', Dict[str, Any]]:
        """
        Load model from checkpoint
        
        Args:
            path: Path to checkpoint
            config: Override configuration
            device: Device to load model on
            
        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Use saved config if not provided
        if config is None:
            config = checkpoint.get('config', {})
            
        # Create model instance
        model = cls(config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        return model, checkpoint
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        Get number of parameters in model
        
        Args:
            trainable_only: Count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def freeze_parameters(self, param_names: Optional[list] = None):
        """
        Freeze model parameters
        
        Args:
            param_names: List of parameter names to freeze. If None, freeze all.
        """
        if param_names is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if any(pname in name for pname in param_names):
                    param.requires_grad = False
                    
    def unfreeze_parameters(self, param_names: Optional[list] = None):
        """
        Unfreeze model parameters
        
        Args:
            param_names: List of parameter names to unfreeze. If None, unfreeze all.
        """
        if param_names is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if any(pname in name for pname in param_names):
                    param.requires_grad = True


class ClassificationModel(BaseModel):
    """Base class for classification models"""
    
    def __init__(self, config: Dict[str, Any], num_classes: int):
        """
        Initialize classification model
        
        Args:
            config: Model configuration
            num_classes: Number of output classes
        """
        super().__init__(config)
        self.num_classes = num_classes
    
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (with no_grad and eval mode)
        
        Args:
            x: Input tensor
            
        Returns:
            Predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class TransformationAwareModel(ClassificationModel):
    """Base class for models that are aware of transformations"""
    
    def __init__(self, config: Dict[str, Any], num_classes: int, num_transforms: int):
        """
        Initialize transformation-aware model
        
        Args:
            config: Model configuration
            num_classes: Number of output classes
            num_transforms: Number of transformation types
        """
        super().__init__(config, num_classes)
        self.num_transforms = num_transforms
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass returning class logits and auxiliary outputs
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (class_logits, auxiliary_outputs)
        """
        pass
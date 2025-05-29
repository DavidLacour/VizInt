"""
Model factory for creating models based on configuration
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import logging

from models.base_model import BaseModel, ClassificationModel, TransformationAwareModel
from config.config_loader import ConfigLoader


from .vanilla_vit import VanillaViT, VanillaViTRobust
from .resnet import ResNetBaseline, ResNetPretrained
from .ttt import TTT, TTT3fc
from .blended_training import BlendedTraining
from .blended_training_3fc import BlendedTraining3fc
from .healer import Healer
from .wrappers import BlendedWrapper, TTTWrapper, HealerWrapper
from .pretrained_correctors import PretrainedUNetCorrector, ImageToImageTransformer, HybridCorrector
from .corrector_wrappers import (UNetCorrectorWrapper, TransformerCorrectorWrapper, 
                                HybridCorrectorWrapper, create_corrector_wrapper)


class ModelFactory:
    """Factory class for creating models"""
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize model factory
        
        Args:
            config_loader: Configuration loader instance
        """
        self.config = config_loader
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def create_model(self, 
                    model_type: str, 
                    dataset_name: str,
                    base_model: Optional[nn.Module] = None,
                    **kwargs) -> nn.Module:
        """
        Create a model based on type and dataset
        
        Args:
            model_type: Type of model to create
            dataset_name: Name of dataset
            base_model: Base model for TTT models
            **kwargs: Additional model-specific arguments
            
        Returns:
            Created model
        """
        dataset_config = self.config.get_dataset_config(dataset_name)
     
        if model_type.startswith('vanilla_vit'):
            config_key = 'vanilla_vit'
        elif model_type.startswith('blended_training'):
            config_key = 'blended_training'
        elif model_type.endswith('_robust'):
            config_key = model_type.replace('_robust', '')
        else:
            config_key = model_type
        
        model_config = self.config.get_model_config(config_key)
        
        num_classes = dataset_config['num_classes']
        img_size = dataset_config['img_size']
        
        model_config = model_config.copy()
        model_config['num_classes'] = num_classes
        model_config['img_size'] = img_size
        
        if model_type in ['vanilla_vit', 'main']:
            return VanillaViT(model_config)
            
        elif model_type in ['vanilla_vit_robust', 'main_robust']:
            return VanillaViTRobust(model_config)
            
        elif model_type in ['resnet', 'resnet_baseline', 'baseline']:
            return ResNetBaseline(model_config)
            
        elif model_type in ['resnet_pretrained', 'pretrained']:
            return ResNetPretrained(model_config)
            
        elif model_type == 'healer':
            return Healer(model_config)
            
        elif model_type in ['ttt', 'ttt_robust']:
            return TTT(model_config, base_model)
            
        elif model_type in ['ttt3fc', 'ttt3fc_robust']:
            return TTT3fc(model_config, base_model)
            
        elif model_type in ['blended_training', 'blended']:
            return BlendedTraining(model_config)
            
        elif model_type in ['blended_training_3fc', 'blended3fc']:
            return BlendedTraining3fc(model_config)
            
        elif model_type == 'blended_resnet18':
            # Create ResNet18 backbone
            backbone_config = model_config.copy()
            backbone_config['model_type'] = 'resnet18'
            backbone = ResNetBaseline(backbone_config)
            # Get feature dimension from ResNet18 (512 for resnet18)
            feature_dim = 512
            return BlendedWrapper(backbone, model_config, feature_dim)
            
        elif model_type == 'ttt_resnet18':
            # Create ResNet18 backbone
            backbone_config = model_config.copy()
            backbone_config['model_type'] = 'resnet18'
            backbone = ResNetBaseline(backbone_config)
            # Get feature dimension from ResNet18 (512 for resnet18)
            feature_dim = 512
            return TTTWrapper(backbone, model_config, feature_dim)
            
        elif model_type == 'blended_resnet50':
            # Create ResNet50 backbone
            backbone_config = model_config.copy()
            backbone_config['model_type'] = 'resnet50'
            backbone = ResNetPretrained(backbone_config)
            # Get feature dimension from ResNet50 (2048 for resnet50)
            feature_dim = 2048
            return BlendedWrapper(backbone, model_config, feature_dim)
            
        elif model_type == 'ttt_resnet50':
            # Create ResNet50 backbone
            backbone_config = model_config.copy()
            backbone_config['model_type'] = 'resnet50'
            backbone = ResNetPretrained(backbone_config)
            # Get feature dimension from ResNet50 (2048 for resnet50)
            feature_dim = 2048
            return TTTWrapper(backbone, model_config, feature_dim)
            
        elif model_type == 'healer_resnet18':
            # Create ResNet18 backbone
            backbone_config = model_config.copy()
            backbone_config['model_type'] = 'resnet18'
            backbone = ResNetBaseline(backbone_config)
            # Get feature dimension from ResNet18 (512 for resnet18)
            feature_dim = 512
            return HealerWrapper(backbone, model_config, feature_dim)
            
        elif model_type == 'healer_resnet50':
            # Create ResNet50 backbone
            backbone_config = model_config.copy()
            backbone_config['model_type'] = 'resnet50'
            backbone = ResNetPretrained(backbone_config)
            # Get feature dimension from ResNet50 (2048 for resnet50)
            feature_dim = 2048
            return HealerWrapper(backbone, model_config, feature_dim)
            
        # Corrector models for standalone training
        elif model_type == 'unet_corrector':
            return PretrainedUNetCorrector(model_config)
            
        elif model_type == 'transformer_corrector':
            return ImageToImageTransformer(model_config)
            
        elif model_type == 'hybrid_corrector':
            return HybridCorrector(model_config)
            
        # Corrector + classifier combinations
        elif model_type == 'unet_resnet18':
            backbone = ResNetBaseline(model_config)
            return UNetCorrectorWrapper(backbone, model_config)
            
        elif model_type == 'unet_resnet50':
            backbone = ResNetPretrained(model_config)
            return UNetCorrectorWrapper(backbone, model_config)
            
        elif model_type == 'unet_vit':
            backbone = VanillaViT(model_config)
            return UNetCorrectorWrapper(backbone, model_config)
            
        elif model_type == 'transformer_resnet18':
            backbone = ResNetBaseline(model_config)
            return TransformerCorrectorWrapper(backbone, model_config)
            
        elif model_type == 'transformer_resnet50':
            backbone = ResNetPretrained(model_config)
            return TransformerCorrectorWrapper(backbone, model_config)
            
        elif model_type == 'transformer_vit':
            backbone = VanillaViT(model_config)
            return TransformerCorrectorWrapper(backbone, model_config)
            
        elif model_type == 'hybrid_resnet18':
            backbone = ResNetBaseline(model_config)
            return HybridCorrectorWrapper(backbone, model_config)
            
        elif model_type == 'hybrid_resnet50':
            backbone = ResNetPretrained(model_config)
            return HybridCorrectorWrapper(backbone, model_config)
            
        elif model_type == 'hybrid_vit':
            backbone = VanillaViT(model_config)
            return HybridCorrectorWrapper(backbone, model_config)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    
    def load_model_from_checkpoint(self, 
                                  checkpoint_path: Path,
                                  model_type: str,
                                  dataset_name: str,
                                  base_model: Optional[nn.Module] = None,
                                  device: str = 'cpu') -> nn.Module:
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            model_type: Type of model
            dataset_name: Name of dataset
            base_model: Base model for TTT models
            device: Device to load on
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        if model_type in ['ttt', 'ttt3fc', 'ttt_robust', 'ttt3fc_robust']:
            # Check if the saved model has a base_model in the state dict
            # This indicates it was saved with a base model included
            has_saved_base_model = any(k.startswith('base_model.') for k in state_dict.keys())
            
            self.logger.info(f"Loading {model_type}: has_saved_base_model={has_saved_base_model}")
            
            if has_saved_base_model:
                # The saved model has a VanillaViT base model
                # We need to create the appropriate base model first
                base_model_keys = [k for k in state_dict.keys() if k.startswith('base_model.')]
                
                # Check if it's a VanillaViT based on the presence of cls_token
                if 'base_model.cls_token' in state_dict:
                    self.logger.info(f"Detected VanillaViT base model in saved {model_type}")
                    
                    # Create a VanillaViT base model
                    from .vanilla_vit import VanillaViT
                    dataset_config = self.config.get_dataset_config(dataset_name)
                    base_config = self.config.get_model_config('vanilla_vit').copy()
                    base_config['num_classes'] = dataset_config['num_classes']
                    base_config['img_size'] = dataset_config['img_size']
                    
                    # Create the base model
                    vit_base = VanillaViT(base_config)
                    
                    # Create TTT model with this base
                    model = self.create_model(model_type, dataset_name, base_model=vit_base)
                else:
                    # Unknown base model type, try without base model
                    self.logger.warning(f"Unknown base model type in {model_type}, creating without base model")
                    model = self.create_model(model_type, dataset_name, base_model=None)
            else:
                # The model was saved without base model, so we need to provide one
                self.logger.info(f"Creating {model_type} with external base model")
                model = self.create_model(model_type, dataset_name, base_model)
        elif model_type in ['blended_resnet18', 'ttt_resnet18', 'blended_resnet50', 'ttt_resnet50', 'healer_resnet18', 'healer_resnet50']:
            # Handle wrapped models
            self.logger.info(f"Loading wrapped model {model_type}")
            model = self.create_model(model_type, dataset_name)
        else:
            # Create model normally for non-TTT models
            model = self.create_model(model_type, dataset_name, base_model)
            
        # Load state dict
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        self.logger.info(f"Loaded {model_type} model from {checkpoint_path}")
        
        return model
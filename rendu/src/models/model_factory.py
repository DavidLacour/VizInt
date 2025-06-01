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
        elif model_type == 'resnet18_not_pretrained_robust':
            config_key = 'resnet'
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
            
        elif model_type == 'resnet18_not_pretrained_robust':
            return ResNetBaseline(model_config)
            
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
            backbone_config = model_config.copy()
            backbone_config['model_type'] = 'resnet18'
            backbone = ResNetBaseline(backbone_config)
            feature_dim = 512
            return BlendedWrapper(backbone, model_config, feature_dim)
            
        elif model_type == 'ttt_resnet18':
            backbone_config = model_config.copy()
            backbone_config['model_type'] = 'resnet18'
            backbone = ResNetBaseline(backbone_config)
            feature_dim = 512
            return TTTWrapper(backbone, model_config, feature_dim)
            
        
            
        elif model_type == 'healer_resnet18':
            backbone_config = model_config.copy()
            backbone_config['model_type'] = 'resnet18'
            backbone = ResNetBaseline(backbone_config)
            feature_dim = 512
            return HealerWrapper(backbone, model_config, feature_dim)
            
        elif model_type == 'unet_corrector':
            return PretrainedUNetCorrector(model_config)
            
        elif model_type == 'transformer_corrector':
            return ImageToImageTransformer(model_config)
            
        elif model_type == 'hybrid_corrector':
            return HybridCorrector(model_config)
            
        elif model_type in ['unet_resnet18', 'unet_vit', 'transformer_resnet18', 
                           'transformer_vit', 'hybrid_resnet18', 'hybrid_vit']:
            # Extract corrector type and backbone type
            if 'unet' in model_type:
                corrector_type = 'unet_corrector'
            elif 'transformer' in model_type:
                corrector_type = 'transformer_corrector'
            else:  # hybrid
                corrector_type = 'hybrid_corrector'
            
            if 'resnet18' in model_type:
                backbone_type = 'resnet'
                backbone_name = "ResNet18"
            else:  # vit
                backbone_type = 'vanilla_vit'
                backbone_name = "VanillaViT"
            
            # Check for pre-trained models
            checkpoint_dir = self.config.get_checkpoint_dir(dataset_name)
            corrector_checkpoint = checkpoint_dir / f"bestmodel_{corrector_type}" / "best_model.pt"
            backbone_checkpoint = checkpoint_dir / f"bestmodel_{backbone_type}" / "best_model.pt"
            
            missing_models = []
            if not corrector_checkpoint.exists():
                missing_models.append(f"{corrector_type} at {corrector_checkpoint}")
            if not backbone_checkpoint.exists():
                missing_models.append(f"{backbone_name} at {backbone_checkpoint}")
            
            if missing_models:
                error_msg = f"\nCannot create {model_type} - missing pre-trained models:\n"
                for model in missing_models:
                    error_msg += f"  - {model}\n"
                error_msg += f"\nPlease train the required models first before using {model_type}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Load pre-trained models
            self.logger.info(f"Loading pre-trained {backbone_name} from {backbone_checkpoint}")
            backbone = self.load_model_from_checkpoint(backbone_checkpoint, backbone_type, dataset_name)
            
            # Create appropriate wrapper with pre-trained corrector
            if 'unet' in model_type:
                return UNetCorrectorWrapper(backbone, model_config, corrector_checkpoint)
            elif 'transformer' in model_type:
                return TransformerCorrectorWrapper(backbone, model_config, corrector_checkpoint)
            else:  # hybrid
                return HybridCorrectorWrapper(backbone, model_config, corrector_checkpoint)
            
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        if model_type in ['ttt', 'ttt3fc', 'ttt_robust', 'ttt3fc_robust']:
            # Check if the saved model has a base_model in the state dict
            has_saved_base_model = any(k.startswith('base_model.') for k in state_dict.keys())
            
            self.logger.info(f"Loading {model_type}: has_saved_base_model={has_saved_base_model}")
            
            if has_saved_base_model:
                base_model_keys = [k for k in state_dict.keys() if k.startswith('base_model.')]
                
                # Check if it's a VanillaViT based on the presence of cls_token
                if 'base_model.cls_token' in state_dict:
                    self.logger.info(f"Detected VanillaViT base model in saved {model_type}")
                    
                    from .vanilla_vit import VanillaViT
                    dataset_config = self.config.get_dataset_config(dataset_name)
                    base_config = self.config.get_model_config('vanilla_vit').copy()
                    base_config['num_classes'] = dataset_config['num_classes']
                    base_config['img_size'] = dataset_config['img_size']
                    
                    vit_base = VanillaViT(base_config)
                    
                    model = self.create_model(model_type, dataset_name, base_model=vit_base)
                else:
                    self.logger.warning(f"Unknown base model type in {model_type}, creating without base model")
                    model = self.create_model(model_type, dataset_name, base_model=None)
            else:
                # The model was saved without base model, so we need to provide one
                self.logger.info(f"Creating {model_type} with external base model")
                model = self.create_model(model_type, dataset_name, base_model)
        elif model_type in ['blended_resnet18', 'ttt_resnet18', 'healer_resnet18']:
            self.logger.info(f"Loading wrapped model {model_type}")
            model = self.create_model(model_type, dataset_name)
        else:
            # Create model normally for non-TTT models
            model = self.create_model(model_type, dataset_name, base_model)
            
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        self.logger.info(f"Loaded {model_type} model from {checkpoint_path}")
        
        return model
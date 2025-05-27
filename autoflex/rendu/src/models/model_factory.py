"""
Model factory for creating models based on configuration
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import logging

# Add parent directory to path to import existing modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_model import BaseModel, ClassificationModel, TransformationAwareModel
from src.config.config_loader import ConfigLoader

# Import existing model implementations
from vit_implementation import create_vit_model
from baseline_models import SimpleResNet18
from ttt_model import TestTimeTrainer
from ttt3fc_model import TestTimeTrainer3fc
from blended_ttt_model import BlendedTTT
from blended_ttt3fc_model import BlendedTTT3fc
from blended_ttt_cifar10 import BlendedTTTCIFAR10
from blended_ttt3fc_cifar10 import BlendedTTT3fcCIFAR10
from cifar10_healer_additions import TransformationHealerCIFAR10
from new_new import TransformationHealer


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
        model_config = self.config.get_model_config(model_type.split('_')[0])  # Handle model_robust naming
        
        # Get dataset-specific parameters
        num_classes = dataset_config['num_classes']
        img_size = dataset_config['img_size']
        
        # Create model based on type
        if model_type in ['main', 'main_robust', 'vit']:
            return self._create_vit_model(dataset_name, num_classes, img_size, model_config)
            
        elif model_type in ['baseline', 'resnet18_baseline']:
            return self._create_resnet_baseline(num_classes)
            
        elif model_type in ['pretrained', 'resnet18_pretrained']:
            return self._create_resnet_pretrained(num_classes, img_size)
            
        elif model_type == 'healer':
            return self._create_healer_model(dataset_name, img_size, model_config)
            
        elif model_type in ['ttt', 'ttt_robust']:
            if base_model is None:
                raise ValueError("Base model required for TTT models")
            return self._create_ttt_model(base_model, img_size, model_config)
            
        elif model_type in ['ttt3fc', 'ttt3fc_robust']:
            if base_model is None:
                raise ValueError("Base model required for TTT3fc models")
            return self._create_ttt3fc_model(base_model, img_size, num_classes, model_config)
            
        elif model_type in ['blended', 'blended_robust']:
            return self._create_blended_model(dataset_name, img_size, num_classes, model_config)
            
        elif model_type in ['blended3fc', 'blended3fc_robust']:
            return self._create_blended3fc_model(dataset_name, img_size, num_classes, model_config)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_vit_model(self, dataset_name: str, num_classes: int, 
                         img_size: int, model_config: Dict[str, Any]) -> nn.Module:
        """Create Vision Transformer model"""
        patch_size = model_config['patch_size'].get(dataset_name.lower(), 8)
        
        return create_vit_model(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=model_config['embed_dim'],
            depth=model_config['depth'],
            head_dim=model_config['head_dim'],
            mlp_ratio=model_config['mlp_ratio'],
            use_resnet_stem=model_config['use_resnet_stem']
        )
    
    def _create_resnet_baseline(self, num_classes: int) -> nn.Module:
        """Create ResNet18 baseline model"""
        return SimpleResNet18(num_classes=num_classes)
    
    def _create_resnet_pretrained(self, num_classes: int, img_size: int) -> nn.Module:
        """Create pretrained ResNet18 model"""
        import torchvision.models as models
        
        model = models.resnet18(pretrained=True)
        
        # Adapt for smaller images if needed
        if img_size == 32:  # CIFAR-10
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        elif img_size == 64:  # TinyImageNet
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            with torch.no_grad():
                old_weight = old_conv.weight
                new_weight = old_weight[:, :, 2:5, 2:5].clone()
                model.conv1.weight.copy_(new_weight)
            model.maxpool = nn.Identity()
            
        # Change final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        return model
    
    def _create_healer_model(self, dataset_name: str, img_size: int, 
                           model_config: Dict[str, Any]) -> nn.Module:
        """Create healer model"""
        vit_config = self.config.get_model_config('vit')
        patch_size = vit_config['patch_size'].get(dataset_name.lower(), 8)
        
        if dataset_name.lower() == 'cifar10':
            return TransformationHealerCIFAR10(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=vit_config['embed_dim'],
                depth=model_config['depth'],
                head_dim=vit_config['head_dim']
            )
        else:
            return TransformationHealer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=vit_config['embed_dim'],
                depth=model_config['depth'],
                head_dim=vit_config['head_dim']
            )
    
    def _create_ttt_model(self, base_model: nn.Module, img_size: int, 
                         model_config: Dict[str, Any]) -> nn.Module:
        """Create TTT model"""
        vit_config = self.config.get_model_config('vit')
        
        return TestTimeTrainer(
            base_model=base_model,
            img_size=img_size,
            patch_size=vit_config['patch_size'].get('cifar10' if img_size == 32 else 'tinyimagenet', 8),
            embed_dim=vit_config['embed_dim']
        )
    
    def _create_ttt3fc_model(self, base_model: nn.Module, img_size: int, 
                            num_classes: int, model_config: Dict[str, Any]) -> nn.Module:
        """Create TTT3fc model"""
        vit_config = self.config.get_model_config('vit')
        
        return TestTimeTrainer3fc(
            base_model=base_model,
            img_size=img_size,
            patch_size=vit_config['patch_size'].get('cifar10' if img_size == 32 else 'tinyimagenet', 8),
            embed_dim=vit_config['embed_dim'],
            num_classes=num_classes
        )
    
    def _create_blended_model(self, dataset_name: str, img_size: int, 
                             num_classes: int, model_config: Dict[str, Any]) -> nn.Module:
        """Create BlendedTTT model"""
        vit_config = self.config.get_model_config('vit')
        patch_size = vit_config['patch_size'].get(dataset_name.lower(), 8)
        
        if dataset_name.lower() == 'cifar10':
            return BlendedTTTCIFAR10(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=vit_config['embed_dim'],
                depth=vit_config['depth'],
                num_classes=num_classes
            )
        else:
            return BlendedTTT(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=vit_config['embed_dim'],
                depth=vit_config['depth']
            )
    
    def _create_blended3fc_model(self, dataset_name: str, img_size: int, 
                                num_classes: int, model_config: Dict[str, Any]) -> nn.Module:
        """Create BlendedTTT3fc model"""
        vit_config = self.config.get_model_config('vit')
        patch_size = vit_config['patch_size'].get(dataset_name.lower(), 8)
        
        if dataset_name.lower() == 'cifar10':
            return BlendedTTT3fcCIFAR10(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=vit_config['embed_dim'],
                depth=vit_config['depth'],
                num_classes=num_classes
            )
        else:
            return BlendedTTT3fc(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=vit_config['embed_dim'],
                depth=vit_config['depth']
            )
    
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
        # Create model
        model = self.create_model(model_type, dataset_name, base_model)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Load state dict
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        self.logger.info(f"Loaded {model_type} model from {checkpoint_path}")
        
        return model
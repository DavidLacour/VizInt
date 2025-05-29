"""
Wrapper classes that combine trained correctors with classification backbones
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union, Tuple, Optional
from pathlib import Path

from .base_model import ClassificationModel
from .pretrained_correctors import PretrainedUNetCorrector, ImageToImageTransformer, HybridCorrector


class CorrectorWrapper(ClassificationModel):
    """
    Base wrapper that combines any corrector with any classification backbone
    """
    
    def __init__(self, 
                 corrector: nn.Module,
                 backbone: nn.Module,
                 config: Dict[str, Any]):
        """
        Initialize corrector wrapper
        
        Args:
            corrector: Pre-trained corrector model
            backbone: Classification backbone (e.g., ResNet, ViT)
            config: Configuration dictionary
        """
        num_classes = config['num_classes']
        super().__init__(config, num_classes)
        
        self.corrector = corrector
        self.backbone = backbone
        
        # Freeze corrector if specified
        if config.get('freeze_corrector', True):
            for param in self.corrector.parameters():
                param.requires_grad = False
        
        # Get backbone feature dimension
        if hasattr(backbone, 'feature_dim'):
            self.feature_dim = backbone.feature_dim
        elif hasattr(backbone, 'fc'):
            self.feature_dim = backbone.fc.in_features
        elif hasattr(backbone, 'classifier'):
            self.feature_dim = backbone.classifier.in_features
        else:
            # Try to infer from backbone
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, config['img_size'], config['img_size'])
                if hasattr(backbone, 'extract_features'):
                    dummy_output = backbone.extract_features(dummy_input)
                else:
                    dummy_output = backbone(dummy_input)
                
                if dummy_output.dim() > 2:
                    dummy_output = F.adaptive_avg_pool2d(dummy_output, 1).flatten(1)
                self.feature_dim = dummy_output.shape[1]
        
        # Replace backbone classifier if it exists
        if hasattr(backbone, 'fc'):
            backbone.fc = nn.Linear(self.feature_dim, num_classes)
        elif hasattr(backbone, 'classifier'):
            backbone.classifier = nn.Linear(self.feature_dim, num_classes)
        elif hasattr(backbone, 'head'):
            backbone.head = nn.Linear(self.feature_dim, num_classes)
    
    def correct_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply correction to input images
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Corrected tensor [B, C, H, W]
        """
        with torch.no_grad():
            return self.corrector.correct_image(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from corrected input
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Feature tensor [B, feature_dim]
        """
        # First correct the input
        corrected_x = self.correct_input(x)
        
        # Extract features using backbone
        if hasattr(self.backbone, 'extract_features'):
            features = self.backbone.extract_features(corrected_x)
        else:
            features = self.backbone(corrected_x)
        
        # Ensure features are 2D
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        return features
    
    def forward(self, x: torch.Tensor, return_corrected: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through corrector + backbone
        
        Args:
            x: Input tensor [B, C, H, W]
            return_corrected: Whether to return corrected images
            
        Returns:
            If return_corrected=False: Classification logits
            If return_corrected=True: Tuple of (logits, corrected_images)
        """
        # Correct input
        corrected_x = self.correct_input(x)
        
        # Extract features and classify
        features = self.extract_features(x)  # This will internally apply correction
        
        # Get logits from backbone
        if hasattr(self.backbone, 'classifier'):
            logits = self.backbone.classifier(features)
        elif hasattr(self.backbone, 'fc'):
            logits = self.backbone.fc(features)
        elif hasattr(self.backbone, 'head'):
            logits = self.backbone.head(features)
        else:
            # Fallback - backbone might return logits directly
            logits = self.backbone(corrected_x)
        
        if return_corrected:
            return logits, corrected_x
        return logits


class UNetCorrectorWrapper(CorrectorWrapper):
    """Wrapper specifically for UNet corrector + classification backbone"""
    
    def __init__(self, backbone: nn.Module, config: Dict[str, Any], corrector_checkpoint: Optional[Path] = None):
        """
        Initialize UNet corrector wrapper
        
        Args:
            backbone: Classification backbone
            config: Configuration dictionary
            corrector_checkpoint: Path to trained UNet corrector checkpoint
        """
        # Create UNet corrector
        corrector_config = {
            'img_size': config['img_size'],
            'in_channels': 3,
            'out_channels': 3,
            'use_residual': config.get('use_residual', True)
        }
        corrector = PretrainedUNetCorrector(corrector_config)
        
        # Load checkpoint if provided
        if corrector_checkpoint and corrector_checkpoint.exists():
            checkpoint = torch.load(corrector_checkpoint, map_location='cpu')
            corrector.load_state_dict(checkpoint['model_state_dict'])
        
        super().__init__(corrector, backbone, config)


class TransformerCorrectorWrapper(CorrectorWrapper):
    """Wrapper specifically for Transformer corrector + classification backbone"""
    
    def __init__(self, backbone: nn.Module, config: Dict[str, Any], corrector_checkpoint: Optional[Path] = None):
        """
        Initialize Transformer corrector wrapper
        
        Args:
            backbone: Classification backbone
            config: Configuration dictionary
            corrector_checkpoint: Path to trained Transformer corrector checkpoint
        """
        # Create Transformer corrector
        corrector_config = {
            'img_size': config['img_size'],
            'patch_size': config.get('corrector_patch_size', 8),
            'in_channels': 3,
            'out_channels': 3,
            'embed_dim': config.get('corrector_embed_dim', 768),
            'depth': config.get('corrector_depth', 12),
            'head_dim': config.get('corrector_head_dim', 64),
            'mlp_ratio': config.get('corrector_mlp_ratio', 4.0),
            'use_residual': config.get('use_residual', True)
        }
        corrector = ImageToImageTransformer(corrector_config)
        
        # Load checkpoint if provided
        if corrector_checkpoint and corrector_checkpoint.exists():
            checkpoint = torch.load(corrector_checkpoint, map_location='cpu')
            corrector.load_state_dict(checkpoint['model_state_dict'])
        
        super().__init__(corrector, backbone, config)


class HybridCorrectorWrapper(CorrectorWrapper):
    """Wrapper specifically for Hybrid corrector + classification backbone"""
    
    def __init__(self, backbone: nn.Module, config: Dict[str, Any], corrector_checkpoint: Optional[Path] = None):
        """
        Initialize Hybrid corrector wrapper
        
        Args:
            backbone: Classification backbone
            config: Configuration dictionary
            corrector_checkpoint: Path to trained Hybrid corrector checkpoint
        """
        # Create Hybrid corrector
        corrector_config = {
            'img_size': config['img_size'],
            'patch_size': config.get('corrector_patch_size', 8),
            'in_channels': 3,
            'out_channels': 3,
            'embed_dim': config.get('corrector_embed_dim', 384),
            'depth': config.get('corrector_depth', 6),
            'head_dim': config.get('corrector_head_dim', 64),
            'mlp_ratio': config.get('corrector_mlp_ratio', 4.0),
            'use_residual': config.get('use_residual', True),
            'use_transformer': config.get('use_transformer', True),
            'use_cnn': config.get('use_cnn', True),
            'fusion_weight': config.get('fusion_weight', 0.5)
        }
        corrector = HybridCorrector(corrector_config)
        
        # Load checkpoint if provided
        if corrector_checkpoint and corrector_checkpoint.exists():
            checkpoint = torch.load(corrector_checkpoint, map_location='cpu')
            corrector.load_state_dict(checkpoint['model_state_dict'])
        
        super().__init__(corrector, backbone, config)


class AdaptiveCorrectorWrapper(ClassificationModel):
    """
    Wrapper that can switch between different correctors based on input analysis
    """
    
    def __init__(self, 
                 correctors: Dict[str, nn.Module],
                 backbone: nn.Module,
                 config: Dict[str, Any]):
        """
        Initialize adaptive corrector wrapper
        
        Args:
            correctors: Dictionary of {name: corrector_model}
            backbone: Classification backbone
            config: Configuration dictionary
        """
        num_classes = config['num_classes']
        super().__init__(config, num_classes)
        
        self.correctors = nn.ModuleDict(correctors)
        self.backbone = backbone
        self.default_corrector = config.get('default_corrector', list(correctors.keys())[0])
        
        # Corruption detector (simple CNN to predict corruption type)
        self.corruption_detector = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, len(correctors))  # Predict which corrector to use
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config['img_size'], config['img_size'])
            if hasattr(backbone, 'extract_features'):
                dummy_output = backbone.extract_features(dummy_input)
            else:
                dummy_output = backbone(dummy_input)
            
            if dummy_output.dim() > 2:
                dummy_output = F.adaptive_avg_pool2d(dummy_output, 1).flatten(1)
            self.feature_dim = dummy_output.shape[1]
        
        # Replace backbone classifier
        if hasattr(backbone, 'fc'):
            backbone.fc = nn.Linear(self.feature_dim, num_classes)
        elif hasattr(backbone, 'classifier'):
            backbone.classifier = nn.Linear(self.feature_dim, num_classes)
    
    def select_corrector(self, x: torch.Tensor) -> str:
        """
        Select appropriate corrector based on input analysis
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Name of selected corrector
        """
        with torch.no_grad():
            # Use corruption detector to predict best corrector
            corruption_scores = self.corruption_detector(x)
            best_corrector_idx = torch.argmax(corruption_scores, dim=1)[0].item()
            corrector_names = list(self.correctors.keys())
            return corrector_names[best_corrector_idx]
    
    def correct_input(self, x: torch.Tensor, corrector_name: Optional[str] = None) -> torch.Tensor:
        """
        Apply correction using selected or specified corrector
        
        Args:
            x: Input tensor [B, C, H, W]
            corrector_name: Name of corrector to use (if None, auto-select)
            
        Returns:
            Corrected tensor [B, C, H, W]
        """
        if corrector_name is None:
            corrector_name = self.select_corrector(x)
        
        if corrector_name not in self.correctors:
            corrector_name = self.default_corrector
        
        corrector = self.correctors[corrector_name]
        return corrector.correct_image(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from corrected input"""
        corrected_x = self.correct_input(x)
        
        if hasattr(self.backbone, 'extract_features'):
            features = self.backbone.extract_features(corrected_x)
        else:
            features = self.backbone(corrected_x)
        
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        return features
    
    def forward(self, x: torch.Tensor, return_corrected: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through adaptive corrector + backbone"""
        corrected_x = self.correct_input(x)
        features = self.extract_features(x)
        
        if hasattr(self.backbone, 'classifier'):
            logits = self.backbone.classifier(features)
        elif hasattr(self.backbone, 'fc'):
            logits = self.backbone.fc(features)
        else:
            logits = self.backbone(corrected_x)
        
        if return_corrected:
            return logits, corrected_x
        return logits


def create_corrector_wrapper(corrector_type: str, 
                           backbone: nn.Module, 
                           config: Dict[str, Any],
                           checkpoint_path: Optional[Path] = None) -> CorrectorWrapper:
    """
    Factory function to create corrector wrappers
    
    Args:
        corrector_type: Type of corrector ('unet', 'transformer', 'hybrid')
        backbone: Classification backbone
        config: Configuration dictionary
        checkpoint_path: Path to corrector checkpoint
        
    Returns:
        Appropriate corrector wrapper
    """
    if corrector_type == 'unet':
        return UNetCorrectorWrapper(backbone, config, checkpoint_path)
    elif corrector_type == 'transformer':
        return TransformerCorrectorWrapper(backbone, config, checkpoint_path)
    elif corrector_type == 'hybrid':
        return HybridCorrectorWrapper(backbone, config, checkpoint_path)
    else:
        raise ValueError(f"Unknown corrector type: {corrector_type}")
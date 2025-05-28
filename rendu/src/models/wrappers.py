"""
Wrapper classes for Blended and TTT models that can work with any backbone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from copy import deepcopy
from .base_model import TransformationAwareModel


class BlendedWrapper(TransformationAwareModel):
    """
    Wrapper that adds transformation prediction and blending to any backbone
    
    This wrapper takes feature maps from any backbone and adds:
    1. Transformation prediction heads
    2. Feature fusion based on predicted transformations
    3. Joint training with auxiliary transformation loss
    """
    
    def __init__(self, 
                 backbone: nn.Module,
                 config: Dict[str, Any],
                 feature_dim: int):
        """
        Initialize BlendedWrapper
        
        Args:
            backbone: Any feature extraction backbone that outputs features
            config: Configuration dictionary containing:
                - num_classes: Number of output classes
                - num_transform_types: Number of transformation types (default: 4)
                - aux_loss_weight: Weight for auxiliary loss (default: 0.5)
                - dropout: Dropout rate (default: 0.1)
            feature_dim: Dimension of backbone output features
        """
        num_classes = config['num_classes']
        num_transforms = config.get('num_transform_types', 4)
        super().__init__(config, num_classes, num_transforms)
        
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.aux_loss_weight = config.get('aux_loss_weight', 0.5)
        self.dropout = config.get('dropout', 0.1)
        
        # Freeze backbone if specified
        if config.get('freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Transformation prediction heads
        self.transform_type_head = nn.Linear(feature_dim, num_transforms)
        self.rotation_head = nn.Linear(feature_dim, 1)
        self.noise_head = nn.Linear(feature_dim, 1)
        self.affine_params_head = nn.Linear(feature_dim, 4)  # tx, ty, shear_x, shear_y
        
        # Feature fusion network
        # Input: original features + transformation predictions
        fusion_input_dim = feature_dim + num_transforms + 6  # 6 = 1 rotation + 1 noise + 4 affine
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize wrapper-specific weights"""
        for head in [self.transform_type_head, self.rotation_head, 
                     self.noise_head, self.affine_params_head, self.classifier]:
            nn.init.normal_(head.weight, std=0.02)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using backbone
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        # Use backbone to extract features
        if hasattr(self.backbone, 'extract_features'):
            features = self.backbone.extract_features(x)
        else:
            features = self.backbone(x)
        
        # Ensure features are 2D [B, feature_dim]
        if features.dim() > 2:
            # Global average pooling if needed
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        return features
    
    def predict_transformations(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict transformation parameters from features
        
        Args:
            features: Feature tensor of shape (B, feature_dim)
            
        Returns:
            Dictionary containing transformation predictions
        """
        transform_type = self.transform_type_head(features)
        rotation = self.rotation_head(features)
        noise_level = self.noise_head(features)
        affine_params = self.affine_params_head(features)
        
        return {
            'transform_type': transform_type,
            'rotation': rotation,
            'noise_level': noise_level,
            'affine_params': affine_params
        }
    
    def forward(self, x: torch.Tensor, return_aux: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass of the model
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_aux: Whether to return auxiliary outputs
            
        Returns:
            If return_aux=False: just logits
            If return_aux=True: Tuple of (class_logits, auxiliary_outputs)
        """
        # Extract features from backbone
        features = self.extract_features(x)
        
        # Predict transformations
        transform_preds = self.predict_transformations(features)
        
        # Concatenate transformation predictions
        transform_features = torch.cat([
            transform_preds['transform_type'],
            transform_preds['rotation'],
            transform_preds['noise_level'],
            transform_preds['affine_params']
        ], dim=1)
        
        # Fuse features with transformation predictions
        combined_features = torch.cat([features, transform_features], dim=1)
        enhanced_features = self.feature_fusion(combined_features)
        
        # Residual connection
        enhanced_features = features + enhanced_features
        
        # Apply dropout
        enhanced_features = self.dropout_layer(enhanced_features)
        
        # Classification
        logits = self.classifier(enhanced_features)
        
        if return_aux:
            aux_outputs = {
                'transform_type': transform_preds['transform_type'],
                'rotation': transform_preds['rotation'],
                'noise_level': transform_preds['noise_level'],
                'affine_params': transform_preds['affine_params'],
                'features': enhanced_features
            }
            return logits, aux_outputs
        else:
            return logits


class TTTWrapper(TransformationAwareModel):
    """
    Wrapper that adds Test-Time Training capability to any backbone
    
    This wrapper takes a frozen backbone and adds:
    1. Adaptation layers that can be updated at test time
    2. Self-supervised transformation prediction task
    3. Test-time parameter adaptation
    """
    
    def __init__(self, 
                 backbone: nn.Module,
                 config: Dict[str, Any],
                 feature_dim: int):
        """
        Initialize TTTWrapper
        
        Args:
            backbone: Any feature extraction backbone (will be frozen)
            config: Configuration dictionary containing:
                - num_classes: Number of output classes
                - num_transform_types: Number of transformation types (default: 4)
                - ttt_layers: Number of adaptation layers (default: 2)
                - inner_steps: Number of inner loop steps (default: 5)
                - inner_lr: Learning rate for inner loop (default: 0.001)
            feature_dim: Dimension of backbone output features
        """
        num_classes = config['num_classes']
        num_transforms = config.get('num_transform_types', 4)
        super().__init__(config, num_classes, num_transforms)
        
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.ttt_layers = config.get('ttt_layers', 2)
        self.inner_steps = config.get('inner_steps', 5)
        self.inner_lr = config.get('inner_lr', 0.001)
        
        # Always freeze backbone for TTT
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Adaptation layers (these will be updated at test time)
        self.adaptation_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.LayerNorm(feature_dim)
            ) for _ in range(self.ttt_layers)
        ])
        
        # Transformation prediction head for self-supervised task
        self.transform_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_transforms + 5)  # type + params
        )
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize wrapper-specific weights"""
        for layer in self.adaptation_layers:
            if isinstance(layer[0], nn.Linear):
                nn.init.xavier_uniform_(layer[0].weight)
                nn.init.zeros_(layer[0].bias)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using backbone
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        # Use backbone to extract features
        with torch.no_grad():  # Backbone is always frozen
            if hasattr(self.backbone, 'extract_features'):
                features = self.backbone.extract_features(x)
            else:
                features = self.backbone(x)
        
        # Ensure features are 2D [B, feature_dim]
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        return features
    
    def adapt_parameters(self, x: torch.Tensor, transform_labels: torch.Tensor) -> Dict[str, float]:
        """
        Adapt model parameters using self-supervised task
        
        Args:
            x: Input tensor
            transform_labels: Ground truth transformation labels
            
        Returns:
            Dictionary of adaptation losses
        """
        # Create a copy of adaptation layers for inner loop
        adapted_layers = deepcopy(self.adaptation_layers)
        optimizer = torch.optim.SGD(adapted_layers.parameters(), lr=self.inner_lr)
        
        losses = []
        for step in range(self.inner_steps):
            # Forward pass with adapted layers
            features = self.extract_features(x)
            
            # Apply adaptation layers
            for layer in adapted_layers:
                features = features + layer(features)
            
            # Predict transformations
            transform_preds = self.transform_predictor(features)
            
            # Compute self-supervised loss
            loss = F.cross_entropy(transform_preds[:, :self.num_transforms], transform_labels)
            
            # Update adapted parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Update main model with adapted parameters
        with torch.no_grad():
            for main_layer, adapted_layer in zip(self.adaptation_layers, adapted_layers):
                main_layer.load_state_dict(adapted_layer.state_dict())
        
        return {'adaptation_loss': sum(losses) / len(losses)}
    
    def forward(self, x: torch.Tensor, 
                adapt: bool = False,
                transform_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the TTT model
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            adapt: Whether to perform test-time adaptation
            transform_labels: Ground truth transformation labels for adaptation
            
        Returns:
            Tuple of (class_logits, auxiliary_outputs)
        """
        # Adapt parameters if requested
        adaptation_info = {}
        if adapt and transform_labels is not None:
            adaptation_info = self.adapt_parameters(x, transform_labels)
        
        # Extract base features
        features = self.extract_features(x)
        
        # Apply adaptation layers
        adapted_features = features
        for layer in self.adaptation_layers:
            adapted_features = adapted_features + layer(adapted_features)
        
        # Get predictions
        logits = self.classifier(adapted_features)
        
        # Also get transformation predictions for auxiliary output
        transform_preds = self.transform_predictor(adapted_features)
        
        aux_outputs = {
            'transform_type': transform_preds[:, :self.num_transforms],
            'transform_params': transform_preds[:, self.num_transforms:],
            'features': adapted_features,
            **adaptation_info
        }
        
        return logits, aux_outputs


class HealerWrapper(nn.Module):
    """
    Wrapper that combines any backbone with Healer preprocessing
    """
    
    def __init__(self, backbone: nn.Module, config: Dict[str, Any], feature_dim: int):
        """
        Initialize HealerWrapper
        
        Args:
            backbone: Feature extraction backbone (e.g., ResNet, ViT)
            config: Configuration dictionary containing:
                - num_classes: Number of output classes
                - num_denoising_steps: Number of denoising steps (default: 3)
            feature_dim: Dimension of backbone output features
        """
        super().__init__()
        
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.config = config
        self.num_classes = config['num_classes']
        self.num_denoising_steps = config.get('num_denoising_steps', 3)
        
        # Import healer transforms
        from .healer_transforms import HealerTransforms
        self.healer_transforms = HealerTransforms
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, self.num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize wrapper-specific weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def heal_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply healer preprocessing to input
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Healed input tensor
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Apply multiple denoising steps
        healed_x = x.clone()
        
        for step in range(self.num_denoising_steps):
            # Apply Gaussian denoising (uses Wiener by default)
            healed_x = self.healer_transforms.apply_gaussian_denoising(
                healed_x, noise_std=0.1, device=device
            )
            
            # Apply batch correction for all transformation types
            # This will attempt to detect and correct common transformations
            try:
                healed_x = self.healer_transforms.apply_batch_correction(
                    healed_x, device=device
                )
            except Exception:
                # If batch correction fails, continue with just denoising
                pass
        
        return healed_x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using backbone after healing
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        # First heal the input
        healed_x = self.heal_input(x)
        
        # Extract features using backbone
        if hasattr(self.backbone, 'extract_features'):
            features = self.backbone.extract_features(healed_x)
        else:
            features = self.backbone(healed_x)
        
        # Ensure features are 2D [B, feature_dim]
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through healer + backbone + classifier
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Classification logits of shape (B, num_classes)
        """
        # Extract features (includes healing)
        features = self.extract_features(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
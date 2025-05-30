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
        
        if config.get('freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False
        
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
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        
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
        if hasattr(self.backbone, 'extract_features'):
            features = self.backbone.extract_features(x)
        else:
            features = self.backbone(x)
        
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
        features = self.extract_features(x)
        
        transform_preds = self.predict_transformations(features)
        
        transform_features = torch.cat([
            transform_preds['transform_type'],
            transform_preds['rotation'],
            transform_preds['noise_level'],
            transform_preds['affine_params']
        ], dim=1)
        
        combined_features = torch.cat([features, transform_features], dim=1)
        enhanced_features = self.feature_fusion(combined_features)
        
        enhanced_features = features + enhanced_features
        
        enhanced_features = self.dropout_layer(enhanced_features)
        
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
            features = self.extract_features(x)
        
            for layer in adapted_layers:
                features = features + layer(features)
            
            transform_preds = self.transform_predictor(features)
            
            loss = F.cross_entropy(transform_preds[:, :self.num_transforms], transform_labels)
            
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
        adaptation_info = {}
        if adapt and transform_labels is not None:
            adaptation_info = self.adapt_parameters(x, transform_labels)
        
        features = self.extract_features(x)
        
        adapted_features = features
        for layer in self.adaptation_layers:
            adapted_features = adapted_features + layer(adapted_features)
        
        logits = self.classifier(adapted_features)
        
        # Also get transformation predictions for auxiliary output
        transform_preds = self.transform_predictor(adapted_features)
        
        aux_outputs = {
            'transform_predictions': transform_preds,
            'transform_type': transform_preds[:, :self.num_transforms],
            'transform_params': transform_preds[:, self.num_transforms:],
            'features': adapted_features,
            **adaptation_info
        }
        
        return logits, aux_outputs


class HealerWrapper(nn.Module):
    """
    Wrapper that combines any backbone with Healer preprocessing using ResNet18 for transform prediction
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
        self.num_transforms = config.get('num_transform_types', 4)
        
        from .healer_transforms import HealerTransforms
        self.healer_transforms = HealerTransforms
        
        # Create ResNet18 for transform prediction
        import torchvision.models as models
        self.transform_predictor = models.resnet18(pretrained=False)
        
        resnet_feature_dim = self.transform_predictor.fc.in_features  # 512 for ResNet18
        
        self.transform_predictor.fc = nn.Identity()  # Remove original FC layer
        self.transform_type_head = nn.Linear(resnet_feature_dim, self.num_transforms)
        self.rotation_head = nn.Linear(resnet_feature_dim, 1)
        self.noise_head = nn.Linear(resnet_feature_dim, 1)
        self.affine_head = nn.Linear(resnet_feature_dim, 4)  # translate_x, translate_y, shear_x, shear_y
        
        self.classifier = nn.Linear(feature_dim, self.num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize wrapper-specific weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
        
        for head in [self.transform_type_head, self.rotation_head, self.noise_head, self.affine_head]:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
    
    def heal_input(self, x: torch.Tensor, return_transform_predictions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Apply healer preprocessing to input using ResNet18 to predict and correct transforms
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_transform_predictions: Whether to return transform predictions
            
        Returns:
            If return_transform_predictions=False: Healed input tensor
            If return_transform_predictions=True: Tuple of (healed_input, transform_predictions)
        """
        device = x.device
        batch_size = x.shape[0]
        
        with torch.no_grad():
            resnet_features = self.transform_predictor(x)
            
            transform_type_logits = self.transform_type_head(resnet_features)
            rotation_params = self.rotation_head(resnet_features)
            noise_params = self.noise_head(resnet_features)
            affine_params = self.affine_head(resnet_features)
            
            transform_type = torch.argmax(transform_type_logits, dim=1)
        
        healed_x = x.clone()
        
        # Apply Wiener denoising
        for i in range(batch_size):
            if transform_type[i] == 1:  # Gaussian noise detected
                noise_std = torch.sigmoid(noise_params[i]).item() * 0.5  # Scale to [0, 0.5]
            else:
                noise_std = 0.1  # Default noise level
            
            healed_x[i:i+1] = self.healer_transforms.apply_gaussian_denoising(
                healed_x[i:i+1], noise_std=noise_std, device=device
            )
        
        if return_transform_predictions:
            transform_predictions = {
                'transform_type_logits': transform_type_logits,
                'rotation_params': rotation_params,
                'noise_params': noise_params,
                'affine_params': affine_params
            }
            return healed_x, transform_predictions
        
        return healed_x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using backbone after healing
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        healed_x = self.heal_input(x, return_transform_predictions=False)
        
        if hasattr(self.backbone, 'extract_features'):
            features = self.backbone.extract_features(healed_x)
        else:
            features = self.backbone(healed_x)
        
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        return features
    
    def forward(self, x: torch.Tensor, return_aux: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through healer + backbone + classifier
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_aux: Whether to return auxiliary outputs (transform predictions)
            
        Returns:
            If return_aux=False: Classification logits of shape (B, num_classes)
            If return_aux=True: Tuple of (logits, auxiliary_outputs)
        """
        healed_x, transform_predictions = self.heal_input(x, return_transform_predictions=True)
        
        if hasattr(self.backbone, 'extract_features'):
            features = self.backbone.extract_features(healed_x)
        else:
            features = self.backbone(healed_x)
        
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        logits = self.classifier(features)
        
        if return_aux:
            aux_outputs = {
                'transform_type': transform_predictions['transform_type_logits'],
                'rotation': transform_predictions['rotation_params'],
                'noise_level': transform_predictions['noise_params'],
                'affine_params': transform_predictions['affine_params'],
                'features': features
            }
            return logits, aux_outputs
        
        return logits
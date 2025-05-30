"""
Test-Time Training (TTT) model implementations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from .base_model import TransformationAwareModel
import sys
from pathlib import Path
from copy import deepcopy

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.transformer_utils import LayerNorm, TransformerTrunk


class TTT(TransformationAwareModel):
    """
    Test-Time Training (TTT) model
    
    This model adapts at test time by updating its parameters based on
    self-supervised tasks performed on test samples.
    """
    
    def __init__(self, config: Dict[str, Any], base_model: Optional[nn.Module] = None):
        """
        Initialize TTT model
        
        Args:
            config: Model configuration containing:
                - num_classes: Number of output classes
                - img_size: Input image size
                - patch_size: Patch size
                - embed_dim: Embedding dimension
                - ttt_layers: Number of TTT adaptation layers
                - inner_steps: Number of inner loop steps for adaptation
                - inner_lr: Learning rate for inner loop
            base_model: Optional base model to wrap with TTT
        """
        num_classes = config['num_classes']
        num_transforms = config.get('num_transform_types', 4)
        super().__init__(config, num_classes, num_transforms)
        self.img_size = config['img_size']
        self.patch_size = config['patch_size']
        self.embed_dim = config['embed_dim']
        self.ttt_layers = config.get('ttt_layers', 2)
        self.inner_steps = config.get('inner_steps', 5)
        self.inner_lr = config.get('inner_lr', 0.001)
        
        if base_model is not None:
            self.base_model = base_model
            # Freeze base model parameters
            for param in self.base_model.parameters():
                param.requires_grad = False
        else:
            # Create a simple feature extractor if no base model provided
            self.base_model = self._create_feature_extractor()
        
        self.adaptation_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.LayerNorm(self.embed_dim)
            ) for _ in range(self.ttt_layers)
        ])
        
        self.transform_predictor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, self.num_transforms + 5)  # type + params
        )
        
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        
        self._init_weights()
        
    def _create_feature_extractor(self):
        """Create a simple feature extractor if no base model is provided"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, self.embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
    def _init_weights(self):
        """Initialize model weights"""
        for layer in self.adaptation_layers:
            if isinstance(layer[0], nn.Linear):
                nn.init.xavier_uniform_(layer[0].weight)
                nn.init.zeros_(layer[0].bias)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, embed_dim)
        """
        if hasattr(self.base_model, 'extract_features'):
            features = self.base_model.extract_features(x)
        else:
            features = self.base_model(x)
        
        if features.dim() > 2:
            features = features.mean(dim=list(range(2, features.dim())))
        
        if features.size(1) != self.embed_dim:
            # Project to correct dimension if needed
            if not hasattr(self, 'feature_proj'):
                self.feature_proj = nn.Linear(features.size(1), self.embed_dim).to(features.device)
            features = self.feature_proj(features)
        
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
        
        with torch.no_grad():
            for main_layer, adapted_layer in zip(self.adaptation_layers, adapted_layers):
                main_layer.load_state_dict(adapted_layer.state_dict())
        
        return {'adaptation_loss': sum(losses) / len(losses)}
    
    def forward(self, x: torch.Tensor, 
                adapt: bool = False,
                transform_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the model
        
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
        
        transform_preds = self.transform_predictor(adapted_features)
        
        aux_outputs = {
            'features': adapted_features,
            'transform_predictions': transform_preds,
            **adaptation_info
        }
        
        return logits, aux_outputs


class TTT3fc(TTT):
    """
    Test-Time Training model with 3 fully connected layers
    
    This extends the TTT model with additional FC layers for better adaptation.
    """
    
    def __init__(self, config: Dict[str, Any], base_model: Optional[nn.Module] = None):
        """
        Initialize TTT3fc model
        
        Args:
            config: Model configuration (extends TTT config with fc_layers)
            base_model: Optional base model to wrap with TTT
        """
        super().__init__(config, base_model)
        
        self.fc_layers = config.get('fc_layers', [512, 256, 128])
        
        fc_dims = [self.embed_dim] + self.fc_layers
        self.fc_blocks = nn.ModuleList()
        
        for i in range(len(self.fc_layers)):
            self.fc_blocks.append(nn.Sequential(
                nn.Linear(fc_dims[i], fc_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.1)),
                nn.LayerNorm(fc_dims[i+1])
            ))
        
        final_dim = self.fc_layers[-1]
        self.adaptation_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(final_dim, final_dim),
                nn.ReLU(),
                nn.LayerNorm(final_dim)
            ) for _ in range(self.ttt_layers)
        ])
        
        self.transform_predictor = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Linear(final_dim // 2, self.num_transforms + 5)
        )
        
        self.classifier = nn.Linear(final_dim, self.num_classes)
        
        self._init_weights()
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features through base model and FC layers
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, final_fc_dim)
        """
        features = super().extract_features(x)
        
        for fc_block in self.fc_blocks:
            features = fc_block(features)
        
        return features
"""
Blended Training model implementation (previously BlendedTTT)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union, Optional
from .base_model import TransformationAwareModel
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.transformer_utils import LayerNorm, TransformerTrunk, Mlp
from src.models.vit_implementation import PatchEmbed


class BlendedTraining(TransformationAwareModel):
    """
    Blended Training model (previously BlendedTTT)
    
    This model uses transformation predictions internally to enhance feature representations
    during both training and inference. It predicts transformation parameters and uses these
    predictions to improve classification accuracy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BlendedTraining model
        
        Args:
            config: Model configuration containing:
                - img_size: Input image size
                - patch_size: Patch size
                - num_classes: Number of output classes
                - embed_dim: Embedding dimension
                - depth: Number of transformer layers
                - head_dim: Dimension per attention head
                - mlp_ratio: MLP expansion ratio
                - aux_loss_weight: Weight for auxiliary transformation loss
                - dropout: Dropout rate
                - num_transform_types: Number of transformation types
        """
        num_classes = config['num_classes']
        num_transforms = config.get('num_transform_types', 4)  # none, noise, rotation, affine
        super().__init__(config, num_classes, num_transforms)
        
        # Extract configuration
        self.img_size = config['img_size']
        self.patch_size = config['patch_size']
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']
        self.head_dim = config['head_dim']
        self.mlp_ratio = config['mlp_ratio']
        self.aux_loss_weight = config.get('aux_loss_weight', 0.5)
        self.dropout = config.get('dropout', 0.1)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=self.embed_dim,
            use_resnet_stem=True
        )
        
        self.num_patches = self.patch_embed.num_patches
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, self.embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.blocks = TransformerTrunk(
            dim=self.embed_dim,
            depth=self.depth,
            head_dim=self.head_dim,
            mlp_ratio=self.mlp_ratio,
            use_bias=False
        ).blocks
        
        # Normalization layer
        self.norm = LayerNorm(self.embed_dim, bias=False)
        
        # Classification head
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        
        # Transformation prediction heads
        self.transform_type_head = nn.Linear(self.embed_dim, self.num_transforms)
        self.rotation_head = nn.Linear(self.embed_dim, 1)  # Rotation angle
        self.noise_head = nn.Linear(self.embed_dim, 1)     # Noise level
        self.affine_params_head = nn.Linear(self.embed_dim, 4)  # Translation (2) + shear (2)
        
        # Feature fusion layer for combining features with transformation predictions
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.embed_dim + self.num_transforms + 6, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize classification and transformation heads
        for head in [self.head, self.transform_type_head, self.rotation_head, 
                     self.noise_head, self.affine_params_head]:
            nn.init.normal_(head.weight, std=0.02)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
                
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply layer norm
        x = self.norm(x)
        
        # Extract CLS token representation
        features = x[:, 0]
        
        return features
    
    def predict_transformations(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict transformation parameters from features
        
        Args:
            features: Feature tensor of shape (B, embed_dim)
            
        Returns:
            Dictionary containing transformation predictions
        """
        # Predict transformation type
        transform_type = self.transform_type_head(features)
        
        # Predict transformation parameters
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
        # Extract base features
        features = self.extract_features(x)
        
        # Predict transformations
        transform_preds = self.predict_transformations(features)
        
        # Prepare transformation features for fusion
        transform_features = torch.cat([
            transform_preds['transform_type'],
            transform_preds['rotation'],
            transform_preds['noise_level'],
            transform_preds['affine_params']
        ], dim=1)
        
        # Combine features with transformation predictions
        combined_features = torch.cat([features, transform_features], dim=1)
        enhanced_features = self.feature_fusion(combined_features)
        
        # Apply residual connection
        enhanced_features = features + enhanced_features
        
        # Apply dropout
        enhanced_features = self.dropout_layer(enhanced_features)
        
        # Classification head
        logits = self.head(enhanced_features)
        
        if return_aux:
            # Prepare auxiliary outputs
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
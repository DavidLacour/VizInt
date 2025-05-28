"""
Blended Training 3FC model implementation (previously BlendedTTT3fc)
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Union
from .base_model import TransformationAwareModel
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.transformer_utils import LayerNorm, TransformerTrunk
from src.models.vit_implementation import PatchEmbed


class BlendedTraining3fc(TransformationAwareModel):
    """
    Blended Training model with 3 fully connected layers (previously BlendedTTT3fc)
    
    This model extends BlendedTraining by adding three fully connected layers
    for better feature processing and transformation prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BlendedTraining3fc model
        
        Args:
            config: Model configuration containing:
                - img_size: Input image size
                - patch_size: Patch size
                - num_classes: Number of output classes
                - embed_dim: Embedding dimension
                - depth: Number of transformer layers
                - head_dim: Dimension per attention head
                - mlp_ratio: MLP expansion ratio
                - fc_layers: List of dimensions for FC layers
                - aux_loss_weight: Weight for auxiliary transformation loss
                - dropout: Dropout rate
                - num_transform_types: Number of transformation types
        """
        num_classes = config['num_classes']
        num_transforms = config.get('num_transform_types', 4)  # none, noise, rotation, affine
        super().__init__(config, num_classes, num_transforms)
        
        self.img_size = config['img_size']
        self.patch_size = config['patch_size']
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']
        self.head_dim = config['head_dim']
        self.mlp_ratio = config['mlp_ratio']
        self.fc_layers = config.get('fc_layers', [512, 256, 128])
        self.aux_loss_weight = config.get('aux_loss_weight', 0.5)
        self.dropout = config.get('dropout', 0.1)
        
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=self.embed_dim,
            use_resnet_stem=True
        )
        
        self.num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, self.embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        
        
        self.blocks = TransformerTrunk(
            dim=self.embed_dim,
            depth=self.depth,
            head_dim=self.head_dim,
            mlp_ratio=self.mlp_ratio,
            use_bias=False
        ).blocks
        
        self.norm = LayerNorm(self.embed_dim, bias=False)
        
        # Triny extra Three fully connected layers to see if we to add distance to the feature map
        # we better augment the featuere map and pertub it less.
        fc_layers_dims = [self.embed_dim] + self.fc_layers
        self.fc_blocks = nn.ModuleList()
        
        for i in range(len(self.fc_layers)):
            self.fc_blocks.append(nn.Sequential(
                nn.Linear(fc_layers_dims[i], fc_layers_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.LayerNorm(fc_layers_dims[i+1])
            ))
        
        final_fc_dim = self.fc_layers[-1]
        
        self.head = nn.Linear(final_fc_dim, self.num_classes)
        
        self.transform_type_head = nn.Linear(final_fc_dim, self.num_transforms)
        self.rotation_head = nn.Linear(final_fc_dim, 1)
        self.noise_head = nn.Linear(final_fc_dim, 1)
        self.affine_params_head = nn.Linear(final_fc_dim, 4)
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(final_fc_dim + self.num_transforms + 6, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim)
        )
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for head in [self.head, self.transform_type_head, self.rotation_head, 
                     self.noise_head, self.affine_params_head]:
            nn.init.normal_(head.weight, std=0.02)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
                
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images through transformer and FC layers
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, final_fc_dim)
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        features = x[:, 0]
        
        for fc_block in self.fc_blocks:
            features = fc_block(features)
        
        return features
    
    def predict_transformations(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict transformation parameters from features
        
        Args:
            features: Feature tensor of shape (B, final_fc_dim)
            
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
        
        logits = self.head(enhanced_features)
        
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
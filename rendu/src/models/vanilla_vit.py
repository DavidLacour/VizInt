"""
Vanilla Vision Transformer model implementation
"""
import torch
import torch.nn as nn
from typing import Dict, Any
from .base_model import ClassificationModel
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.transformer_utils import LayerNorm, TransformerTrunk
from src.models.vit_implementation import PatchEmbed


class VanillaViT(ClassificationModel):
    """
    Vanilla Vision Transformer model (previously MainModel)
    
    This is the base ViT model used for image classification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VanillaViT model
        
        Args:
            config: Model configuration containing:
                - img_size: Input image size
                - patch_size: Patch size
                - num_classes: Number of output classes
                - embed_dim: Embedding dimension
                - depth: Number of transformer layers
                - head_dim: Dimension per attention head
                - mlp_ratio: MLP expansion ratio
                - use_resnet_stem: Whether to use ResNet-style stem
                - dropout: Dropout rate
        """
        num_classes = config['num_classes']
        super().__init__(config, num_classes)
        
        # Extract configuration
        self.img_size = config['img_size']
        self.patch_size = config['patch_size']
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']
        self.head_dim = config['head_dim']
        self.mlp_ratio = config['mlp_ratio']
        self.use_resnet_stem = config.get('use_resnet_stem', True)
        self.dropout = config.get('dropout', 0.1)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=self.embed_dim,
            use_resnet_stem=self.use_resnet_stem
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
        
        # Transformer trunk
        self.trunk = TransformerTrunk(
            dim=self.embed_dim,
            depth=self.depth,
            head_dim=self.head_dim,
            mlp_ratio=self.mlp_ratio,
            use_bias=False
        )
        
        # Layer normalization
        self.norm = LayerNorm(self.embed_dim, bias=False)
        
        # Classification head
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize patch embedding
        w = self.patch_embed.proj[0].weight.data if self.use_resnet_stem else self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize classification head
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Add CLS token: (B, num_patches, embed_dim) -> (B, 1 + num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer
        x = self.trunk(x)
        
        # Apply layer norm
        x = self.norm(x)
        
        # Extract CLS token representation
        features = x[:, 0]
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Extract features
        features = self.extract_features(x)
        
        # Apply dropout
        features = self.dropout_layer(features)
        
        # Classification head
        logits = self.head(features)
        
        return logits


class VanillaViTRobust(VanillaViT):
    """
    Robust version of VanillaViT (previously MainModel Robust)
    
    This model has the same architecture as VanillaViT but is designed
    to be trained with robust training techniques (e.g., with augmented data).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VanillaViTRobust model
        
        Args:
            config: Model configuration (same as VanillaViT)
        """
        super().__init__(config)
        # The architecture is identical to VanillaViT
        # The difference is in how it's trained (handled by the trainer)
"""
BlendedTTT model adapted for CIFAR-10 dataset (10 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_implementation import PatchEmbed
from transformer_utils import LayerNorm, TransformerTrunk


class BlendedTTTCIFAR10(nn.Module):
    """
    BlendedTTT models are standalone models that use transformation predictions 
    internally to get better feature maps. They don't do test time adaptation.
    
    This CIFAR-10 variant is optimized for 32x32 images with 10 classes.
    The model predicts transformation parameters and uses these predictions to
    enhance feature representations during both training and inference.
    """
    def __init__(self, img_size=32, patch_size=4, embed_dim=384, depth=8, num_classes=10):
        super().__init__()
        
        # Use the same patch embedding as the ViT model
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            use_resnet_stem=True
        )
        
        # Learnable cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.blocks = TransformerTrunk(
            dim=embed_dim,
            depth=depth,
            head_dim=64,
            mlp_ratio=4.0,
            use_bias=False
        ).blocks
        
        # Create normalization layers
        self.norm = LayerNorm(embed_dim)
        
        # Classification head for CIFAR-10
        self.head = nn.Linear(embed_dim, num_classes)  # num_classes for CIFAR-10
        
        # Create heads for different transformation parameters
        self.transform_type_head = nn.Linear(embed_dim, 4)  # 4 transform types (no transform, noise, rotation, affine)
        
        # Separate severity heads for each transform type
        self.severity_noise_head = nn.Linear(embed_dim, 1)   
        self.severity_rotation_head = nn.Linear(embed_dim, 1)      
        self.severity_affine_head = nn.Linear(embed_dim, 1)   
        
        # Specific parameter heads for each transform type
        self.rotation_head = nn.Linear(embed_dim, 1)        # Rotation angle
        self.noise_head = nn.Linear(embed_dim, 1)           # Noise std
        self.affine_head = nn.Linear(embed_dim, 4)          # Affine params
    
    def forward_features(self, x):
        B = x.shape[0]
        
        # Extract patches
        x = self.patch_embed(x)
        
        # Add cls token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Apply normalization
        x = self.norm(x)
        
        return x
    
    def forward(self, x, aux_only=False):
        # Extract features
        features = self.forward_features(x)
        
        # Get CLS token
        cls_features = features[:, 0]
        
        # For the auxiliary task, predict transformations
        aux_outputs = {
            'transform_type_logits': self.transform_type_head(cls_features),
            'severity_noise': torch.sigmoid(self.severity_noise_head(cls_features)),
            'severity_rotation': torch.sigmoid(self.severity_rotation_head(cls_features)),
            'severity_affine': torch.sigmoid(self.severity_affine_head(cls_features)),
            'rotation_angle': self.rotation_head(cls_features),
            'noise_std': torch.sigmoid(self.noise_head(cls_features)),  
            'affine_params': self.affine_head(cls_features)
        }
        
        if aux_only:
            return aux_outputs
        
        # Main classification task
        class_logits = self.head(cls_features)
        
        return class_logits, aux_outputs
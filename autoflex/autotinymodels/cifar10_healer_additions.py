"""
CIFAR-10 Healer Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer_utils import LayerNorm, TransformerTrunk
from vit_implementation import PatchEmbed


class TransformationHealerCIFAR10(nn.Module):
    """Transformation Healer model adapted for CIFAR-10 (32x32 images)"""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384, depth=6, head_dim=64):
        super().__init__()
        
        # Use the same patch embedding as our ViT model
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_resnet_stem=True
        )
        
        # Transformer backbone
        self.transformer = TransformerTrunk(
            dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            mlp_ratio=4.0,
            use_bias=False
        )
        
        # Heads for different transformation parameters
        self.transform_type_head = nn.Linear(embed_dim, 4)  # 4 types: no_transform, gaussian_noise, rotation, affine
        
        # Severity heads for each transform type
        self.severity_noise_head = nn.Linear(embed_dim, 1)   
        self.severity_rotation_head = nn.Linear(embed_dim, 1)
        self.severity_affine_head = nn.Linear(embed_dim, 1)
        
        # Specific parameter heads for each transform type
        self.rotation_head = nn.Linear(embed_dim, 1)        # Rotation angle
        self.noise_head = nn.Linear(embed_dim, 1)           # Noise std
        self.affine_head = nn.Linear(embed_dim, 4)          # Affine params: translate_x, translate_y, shear_x, shear_y
        
        # Learnable cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.norm = LayerNorm(embed_dim, bias=False)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use cls token for predictions
        x = self.norm(x)
        cls_output = x[:, 0]
        
        # Predict transformation type
        transform_type_logits = self.transform_type_head(cls_output)
        
        # Predict severity for each transform type
        severity_noise = torch.sigmoid(self.severity_noise_head(cls_output))
        severity_rotation = torch.sigmoid(self.severity_rotation_head(cls_output))
        severity_affine = torch.sigmoid(self.severity_affine_head(cls_output))
        
        # Predict specific parameters
        rotation_angle = self.rotation_head(cls_output)
        noise_std = torch.sigmoid(self.noise_head(cls_output))
        affine_params = torch.tanh(self.affine_head(cls_output))
        
        predictions = {
            'transform_type_logits': transform_type_logits,
            'severity_noise': severity_noise,
            'severity_rotation': severity_rotation,
            'severity_affine': severity_affine,
            'rotation_angle': rotation_angle,
            'noise_std': noise_std,
            'affine_params': affine_params
        }
        
        return predictions


class HealerLossCIFAR10(nn.Module):
    """Combined loss for training the CIFAR-10 healer model"""
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, true_params):
        """
        Calculate combined loss for healer predictions
        
        Args:
            predictions: Dict of model predictions
            true_params: Dict of true transformation parameters
        """
        # Transform type classification loss
        type_loss = self.ce_loss(
            predictions['transform_type_logits'], 
            true_params['transform_type']
        )
        
        # Severity prediction losses
        batch_size = predictions['severity_noise'].shape[0]
        severity_losses = []
        
        for i in range(batch_size):
            true_type = true_params['transform_type'][i].item()
            
            if true_type == 1:  # gaussian_noise
                severity_losses.append(
                    self.mse_loss(
                        predictions['severity_noise'][i], 
                        true_params['severity'][i].unsqueeze(0)
                    )
                )
            elif true_type == 2:  # rotation
                severity_losses.append(
                    self.mse_loss(
                        predictions['severity_rotation'][i], 
                        true_params['severity'][i].unsqueeze(0)
                    )
                )
            elif true_type == 3:  # affine
                severity_losses.append(
                    self.mse_loss(
                        predictions['severity_affine'][i], 
                        true_params['severity'][i].unsqueeze(0)
                    )
                )
        
        severity_loss = torch.stack(severity_losses).mean() if severity_losses else torch.tensor(0.0).to(predictions['severity_noise'].device)
        
        # Parameter prediction losses
        param_losses = []
        
        for i in range(batch_size):
            true_type = true_params['transform_type'][i].item()
            
            if true_type == 1 and 'noise_std' in true_params:  # gaussian_noise
                param_losses.append(
                    self.mse_loss(
                        predictions['noise_std'][i], 
                        true_params['noise_std'][i].unsqueeze(0)
                    )
                )
            elif true_type == 2 and 'rotation_angle' in true_params:  # rotation
                param_losses.append(
                    self.mse_loss(
                        predictions['rotation_angle'][i], 
                        true_params['rotation_angle'][i].unsqueeze(0)
                    )
                )
            elif true_type == 3 and 'affine_params' in true_params:  # affine
                param_losses.append(
                    self.mse_loss(
                        predictions['affine_params'][i], 
                        true_params['affine_params'][i]
                    )
                )
        
        param_loss = torch.stack(param_losses).mean() if param_losses else torch.tensor(0.0).to(predictions['severity_noise'].device)
        
        # Combine losses
        total_loss = type_loss + 0.5 * severity_loss + 0.5 * param_loss
        
        return total_loss, {
            'type_loss': type_loss.item(),
            'severity_loss': severity_loss.item() if isinstance(severity_loss, torch.Tensor) else 0.0,
            'param_loss': param_loss.item() if isinstance(param_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }
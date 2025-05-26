import os
import torch
import wandb
import numpy as np
import shutil
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from copy import deepcopy
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from new_new import ContinuousTransforms, TinyImageNetDataset

import os
import torch
import wandb
import numpy as np
import shutil
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from copy import deepcopy

# Import our custom ViT model
from transformer_utils import set_seed, LayerNorm, Mlp, TransformerTrunk
from vit_implementation import create_vit_model, PatchEmbed, VisionTransformer


MAX_ROTATION = 360.0 
MAX_STD_GAUSSIAN_NOISE = 0.5
MAX_TRANSLATION_AFFINE = 0.1
MAX_SHEAR_ANGLE = 15.0
DEBUG = False 


class MLP3Layer(nn.Module):
    """3-layer MLP with ReLU activations and dropout"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class BlendedTTT3fc(nn.Module):
    """
    BlendedTTT3fc models are standalone models that use transformation predictions 
    internally to get better feature maps. They don't do test time adaptation.
    
    This variant includes 3 fully connected layers before classification and 
    transform predictions, providing additional capacity for feature refinement.
    
    The model predicts transformation parameters and uses these predictions to
    enhance feature representations during both training and inference.
    """
    def __init__(self, img_size=64, patch_size=8, embed_dim=384, depth=8, hidden_dim=512, dropout_rate=0.1):
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
        
        # Transformer backbone
        self.blocks = TransformerTrunk(
                dim=embed_dim,
                depth=depth,
                head_dim=64,
                mlp_ratio=4.0,
                use_bias=False
            ).blocks
        
        # Normalization layer
        self.norm = LayerNorm(embed_dim, bias=False)
        
        # 3-layer MLP for classification head (instead of single linear layer)
        self.head = MLP3Layer(embed_dim, hidden_dim, 200, dropout_rate)  # 200 classes for Tiny ImageNet
        
        # 3-layer MLP for transform type prediction
        self.transform_type_head = MLP3Layer(embed_dim, hidden_dim, 4, dropout_rate)  # 4 transform types
        
        # 3-layer MLPs for severity heads for each transform type
        self.severity_noise_head = MLP3Layer(embed_dim, hidden_dim, 1, dropout_rate)   
        self.severity_rotation_head = MLP3Layer(embed_dim, hidden_dim, 1, dropout_rate)      
        self.severity_affine_head = MLP3Layer(embed_dim, hidden_dim, 1, dropout_rate)   
        
        # 3-layer MLPs for specific parameter heads for each transform type
        self.rotation_head = MLP3Layer(embed_dim, hidden_dim, 1, dropout_rate)        # Rotation angle
        self.noise_head = MLP3Layer(embed_dim, hidden_dim, 1, dropout_rate)           # Noise std
        self.affine_head = MLP3Layer(embed_dim, hidden_dim, 4, dropout_rate)          # Affine params
    
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
        
        # For the auxiliary task, predict transformations using 3-layer MLPs
        aux_outputs = {
            'transform_type_logits': self.transform_type_head(cls_features),
            'severity_noise': torch.sigmoid(self.severity_noise_head(cls_features)),
            'severity_rotation': torch.sigmoid(self.severity_rotation_head(cls_features)),
            'severity_affine': torch.sigmoid(self.severity_affine_head(cls_features)),
            'rotation_angle': torch.tanh(self.rotation_head(cls_features)) * MAX_ROTATION,
            'noise_std': torch.sigmoid(self.noise_head(cls_features)) * MAX_STD_GAUSSIAN_NOISE,
            'translate_x': torch.tanh(self.affine_head(cls_features)[:, 0:1]) * MAX_TRANSLATION_AFFINE,
            'translate_y': torch.tanh(self.affine_head(cls_features)[:, 1:2]) * MAX_TRANSLATION_AFFINE,
            'shear_x': torch.tanh(self.affine_head(cls_features)[:, 2:3]) * MAX_SHEAR_ANGLE,
            'shear_y': torch.tanh(self.affine_head(cls_features)[:, 3:4]) * MAX_SHEAR_ANGLE
        }
        
        if aux_only:
            return aux_outputs
        
        # For classification, use CLS token with 3-layer MLP
        logits = self.head(cls_features)
        
        return logits, aux_outputs
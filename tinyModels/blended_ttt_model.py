
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

class BlendedTTT(nn.Module):
    def __init__(self, base_model, img_size=64, patch_size=8, embed_dim=384,depth=8):
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
        
        # Use only 4 transformer blocks
        # If base_model has a transformer attribute with blocks, use those
        if hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'blocks'):
            # Use first 4 blocks of the base model
            self.blocks = nn.ModuleList(base_model.transformer.blocks[:4])
        else:
            # Otherwise create a new transformer trunk with 4 blocks
            self.blocks = TransformerTrunk(
                dim=embed_dim,
                depth=depth,  # 
                head_dim=64,
                mlp_ratio=4.0,
                use_bias=False
            ).blocks
        
        # Normalization layer
        self.norm = LayerNorm(embed_dim, bias=False)
        
        # Classification head
        self.head = nn.Linear(embed_dim, 200)  # 200 classes for Tiny ImageNet
        
        # Create heads for different transformation parameters
        self.transform_type_head = nn.Linear(embed_dim, 4)  # 3 transform types (no transform,noise, rotation, affine)
        
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
            'rotation_angle': torch.tanh(self.rotation_head(cls_features)) * MAX_ROTATION,
            'noise_std': torch.sigmoid(self.noise_head(cls_features)) * MAX_STD_GAUSSIAN_NOISE,
            'translate_x': torch.tanh(self.affine_head(cls_features)[:, 0:1]) * MAX_TRANSLATION_AFFINE,
            'translate_y': torch.tanh(self.affine_head(cls_features)[:, 1:2]) * MAX_TRANSLATION_AFFINE,
            'shear_x': torch.tanh(self.affine_head(cls_features)[:, 2:3]) * MAX_SHEAR_ANGLE,
            'shear_y': torch.tanh(self.affine_head(cls_features)[:, 3:4]) * MAX_SHEAR_ANGLE
        }
        
        if aux_only:
            return aux_outputs
        
        # For classification, use CLS token
        logits = self.head(cls_features)
        
        return logits, aux_outputs
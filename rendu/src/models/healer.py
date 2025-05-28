"""
Unified Healer model implementation
"""
import torch
import torch.nn as nn
from typing import Dict, Any
from .base_model import TransformationAwareModel
import sys
from pathlib import Path
from torchvision import transforms

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.transformer_utils import LayerNorm, TransformerTrunk
from src.models.vit_implementation import PatchEmbed


class Healer(TransformationAwareModel):
    """
    Unified Healer model for both CIFAR-10 and TinyImageNet
    
    The Healer model is designed to predict and correct transformations in images,
    working alongside main models to improve their robustness.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Healer model
        
        Args:
            config: Model configuration containing:
                - img_size: Input image size
                - patch_size: Patch size
                - num_classes: Number of output classes (not used for healer)
                - embed_dim: Embedding dimension
                - depth: Number of transformer layers
                - head_dim: Dimension per attention head
                - mlp_ratio: MLP expansion ratio
                - num_denoising_steps: Number of denoising steps
                - dropout: Dropout rate
        """
        # Healer doesn't do classification, but we keep the interface consistent
        num_classes = config.get('num_classes', 0)
        num_transforms = config.get('num_transform_types', 4)
        super().__init__(config, num_classes, num_transforms)
        
        # Extract configuration
        self.img_size = config['img_size']
        self.patch_size = config['patch_size']
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']
        self.head_dim = config['head_dim']
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.num_denoising_steps = config.get('num_denoising_steps', 3)
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
        
        # Transformation prediction heads
        self.transform_type_head = nn.Linear(self.embed_dim, self.num_transforms)
        self.rotation_head = nn.Linear(self.embed_dim, 1)
        self.noise_head = nn.Linear(self.embed_dim, 1)
        self.affine_head = nn.Linear(self.embed_dim, 4)  # translate_x, translate_y, shear_x, shear_y
        
        # Remove the complex healing network and output projection
        # We'll use direct inverse transformations instead
        
        # Classification head (for consistency with training framework)
        self.classification_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else None
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize transformation heads
        for head in [self.transform_type_head, self.rotation_head, self.noise_head, self.affine_head]:
            nn.init.normal_(head.weight, std=0.02)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        
        # Initialize classification head if present
        if self.classification_head is not None:
            nn.init.normal_(self.classification_head.weight, std=0.02)
            if self.classification_head.bias is not None:
                nn.init.zeros_(self.classification_head.bias)
    
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
        
        # Return both CLS token and patch features
        return x
    
    def predict_transformations(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict transformation parameters from features
        
        Args:
            features: Feature tensor of shape (B, 1 + num_patches, embed_dim)
            
        Returns:
            Dictionary containing transformation predictions
        """
        # Use CLS token for predictions
        cls_features = features[:, 0]
        
        # Predict transformation type
        transform_type_logits = self.transform_type_head(cls_features)
        
        # Predict transformation parameters
        rotation_angle = torch.tanh(self.rotation_head(cls_features)) * 180.0
        noise_std = torch.sigmoid(self.noise_head(cls_features)) * 0.5
        
        # Affine transformation parameters
        affine_params = self.affine_head(cls_features)  # [B, 4]
        translate_x = torch.tanh(affine_params[:, 0:1]) * 0.1
        translate_y = torch.tanh(affine_params[:, 1:2]) * 0.1
        shear_x = torch.tanh(affine_params[:, 2:3]) * 15.0
        shear_y = torch.tanh(affine_params[:, 3:4]) * 15.0
        
        return {
            'transform_type_logits': transform_type_logits,
            'rotation_angle': rotation_angle,
            'noise_std': noise_std,
            'translate_x': translate_x,
            'translate_y': translate_y,
            'shear_x': shear_x,
            'shear_y': shear_y
        }
    
    def apply_correction(self, transformed_images: torch.Tensor, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply inverse transformations to correct distorted images based on healer predictions.
        
        Args:
            transformed_images: Tensor of transformed images [B, C, H, W]
            predictions: Dictionary of transformation predictions from the healer model
                
        Returns:
            corrected_images: Tensor of corrected images [B, C, H, W]
        """
        device = transformed_images.device
        batch_size = transformed_images.shape[0]
        
        # Get the predicted transform types
        transform_type_logits = predictions['transform_type_logits']
        transform_types = torch.argmax(transform_type_logits, dim=1)  # [B]
        
        # Initialize corrected images as a clone of transformed images
        corrected_images = transformed_images.clone()
        
        # Process each image in the batch
        for i in range(batch_size):
            img = transformed_images[i].unsqueeze(0)  # [1, C, H, W]
            t_type = transform_types[i].item()
            
            # No transform (type 0) - keep the image as is
            if t_type == 0:
                continue
                
            # Gaussian noise (type 1) - apply denoising
            elif t_type == 1:
                noise_std = predictions['noise_std'][i].item()
                # Simple denoising by smoothing
                if noise_std > 0.01:
                    # Apply a small blur to reduce noise
                    img_cpu = img.cpu()
                    to_pil = transforms.ToPILImage()
                    to_tensor = transforms.ToTensor()
                    pil_img = to_pil(img_cpu.squeeze(0))
                    
                    #is the better way to deal with gaussian noise ?
                    # Adjust blur size based on noise level
                    blur_radius = max(1, int(min(2.0, noise_std * 4.0)))
                    if blur_radius % 2 == 0:  # Ensure odd number
                        blur_radius += 1
                    denoised_img = transforms.functional.gaussian_blur(pil_img, blur_radius)
                    corrected_img = to_tensor(denoised_img).unsqueeze(0).to(device)
                    corrected_images[i] = corrected_img.squeeze(0)
            
            # Rotation (type 2) - apply inverse rotation
            elif t_type == 2:
                angle = predictions['rotation_angle'][i].item()
                # Apply inverse rotation (negative angle)
                img_cpu = img.cpu()
                to_pil = transforms.ToPILImage()
                to_tensor = transforms.ToTensor()
                pil_img = to_pil(img_cpu.squeeze(0))
                
                # Apply inverse rotation
                rotated_img = transforms.functional.rotate(pil_img, -angle)
                corrected_img = to_tensor(rotated_img).unsqueeze(0).to(device)
                corrected_images[i] = corrected_img.squeeze(0)
            
            # Affine transform (type 3) - apply inverse affine transform
            elif t_type == 3:
                translate_x = predictions['translate_x'][i].item()
                translate_y = predictions['translate_y'][i].item()
                shear_x = predictions['shear_x'][i].item()
                shear_y = predictions['shear_y'][i].item()
                
                # Convert to CPU for PIL operations
                img_cpu = img.cpu()
                to_pil = transforms.ToPILImage()
                to_tensor = transforms.ToTensor()
                pil_img = to_pil(img_cpu.squeeze(0))
                
                # Get image size for translation calculation
                width, height = pil_img.size
                translate_pixels = (-translate_x * width, -translate_y * height)  # Negative for inverse
                
                # Apply inverse affine transformation
                affine_img = transforms.functional.affine(
                    pil_img, 
                    angle=0.0,
                    translate=translate_pixels,
                    scale=1.0,
                    shear=[-shear_x, -shear_y]  # Negative for inverse
                )
                corrected_img = to_tensor(affine_img).unsqueeze(0).to(device)
                corrected_images[i] = corrected_img.squeeze(0)
        
        return corrected_images
    
    
    def forward(self, x: torch.Tensor, 
                return_reconstruction: bool = False,
                return_logits: bool = True) -> Any:
        """
        Forward pass of the healer model
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_reconstruction: Whether to return reconstructed image
            return_logits: Whether to return classification logits
            
        Returns:
            If return_logits is True and classification_head exists: returns logits
            Otherwise: returns tuple of (predictions, None) for compatibility
        """
        # Extract features
        features = self.extract_features(x)
        
        # For classification mode (training)
        if return_logits and self.classification_head is not None:
            # Use CLS token for classification
            cls_features = features[:, 0]
            logits = self.classification_head(cls_features)
            return logits
        
        # For healing mode (inference/evaluation)
        # Predict transformations
        transform_preds = self.predict_transformations(features)
        
        # Return predictions and None for compatibility with existing interface
        return transform_preds, None
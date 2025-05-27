"""
CIFAR-10 Healer Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
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
    
    def apply_correction(self, transformed_images, predictions):
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
                # Simple denoising by smoothing (can be improved with better methods)
                if noise_std > 0.01:  # Only apply if significant noise is detected
                    # Apply a small blur to reduce noise
                    img_cpu = img.cpu()
                    to_pil = transforms.ToPILImage()
                    to_tensor = transforms.ToTensor()
                    pil_img = to_pil(img_cpu.squeeze(0))
                    
                    # Adjust blur size based on noise level
                    blur_radius = max(1, int(min(2.0, noise_std * 4.0)))
                    if blur_radius % 2 == 0:  # Ensure odd number for kernel size
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
                affine_params = predictions['affine_params'][i]
                translate_x = affine_params[0].item()
                translate_y = affine_params[1].item()
                shear_x = affine_params[2].item()
                shear_y = affine_params[3].item()
                
                # Convert to CPU for PIL operations
                img_cpu = img.cpu()
                to_pil = transforms.ToPILImage()
                to_tensor = transforms.ToTensor()
                pil_img = to_pil(img_cpu.squeeze(0))
                
                # Get image size for translation calculation
                width, height = pil_img.size
                translate_pixels = (-translate_x * width, -translate_y * height)  # Note the negative sign for inverse
                
                # Apply inverse affine transformation
                # For inverse shear, we apply the negative values
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
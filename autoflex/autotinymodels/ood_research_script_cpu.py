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

# Initialize wandb
wandb.init(project="vit-tiny-imagenet-ood", name="transform-healing-ttt")

# Define continuous transformations for OOD testing
class ContinuousTransforms:
    def __init__(self, severity=1.0):
        self.severity = severity
        self.transform_types = ['gaussian_noise', 'rotation', 'h_flip', 'affine']
        self.transform_probs = [0.7, 0.7, 0.5, 0.7]  # Adjusted weights
        
    def apply_transforms(self, img, transform_type=None, severity=None, return_params=False):
        """
        Apply a continuous transformation to the image on CPU
        
        Args:
            img: The input tensor image [C, H, W]
            transform_type: If None, randomly choose from transform_types
            severity: If None, use self.severity, otherwise use specified severity
            return_params: If True, return the transformation parameters
            
        Returns:
            transformed_img: The transformed image
            transform_params: Dictionary of transformation parameters (if return_params=True)
        """
        # Store original device but move image to CPU for transforms
        original_device = img.device
        img = img.cpu()  # Move to CPU before transformations
        
        if severity is None:
            severity = self.severity
            
        if transform_type is None:
            # Choose a transform type based on probabilities
            transform_type = random.choices(
                self.transform_types, 
                weights=self.transform_probs, 
                k=1
            )[0]
        
        # Initialize all possible parameters with default values
        transform_params = {
            'transform_type': transform_type,
            'severity': severity,
            'noise_std': 0.0,
            'rotation_angle': 0.0,
            'h_flip': 0.0,
            'translate_x': 0.0,
            'translate_y': 0.0,
            'shear_x': 0.0,
            'shear_y': 0.0
        }
        
        # Apply the selected transformation on CPU
        if transform_type == 'gaussian_noise':
            # Gaussian noise with continuous severity
            std = severity * 0.5  # Scale severity to a reasonable std range
            noise = torch.randn_like(img) * std
            transformed_img = img + noise
            transformed_img = torch.clamp(transformed_img, 0, 1)
            transform_params['noise_std'] = std
            
        elif transform_type == 'rotation':
            # Rotation with continuous angle
            max_angle = 30.0 * severity  # Scale severity to angle in degrees
            angle = random.uniform(-max_angle, max_angle)
            
            # Convert to PIL for rotation
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            pil_img = to_pil(img)
            rotated_img = transforms.functional.rotate(pil_img, angle)
            transformed_img = to_tensor(rotated_img)
            transform_params['rotation_angle'] = angle
            
        elif transform_type == 'h_flip':
            # Horizontal flip - simple binary transformation
            # Convert to PIL for flip
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            pil_img = to_pil(img)
            flipped_img = transforms.functional.hflip(pil_img)
            transformed_img = to_tensor(flipped_img)
            transform_params['h_flip'] = 1.0  # Binary flag: 1.0 means flipped
            
        elif transform_type == 'affine':
            # Affine transformation with translation and shear
            # Scale the severity to control the magnitude of transformation
            max_translate = 0.1 * severity  # Maximum translation as fraction of image size
            max_shear = 15.0 * severity     # Maximum shear angle in degrees
            
            # Generate random translation and shear parameters
            translate_x = random.uniform(-max_translate, max_translate)
            translate_y = random.uniform(-max_translate, max_translate)
            shear_x = random.uniform(-max_shear, max_shear)
            shear_y = random.uniform(-max_shear, max_shear)
            
            # Convert to PIL for affine transform
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            pil_img = to_pil(img)
            
            # Get image size for translation calculation
            width, height = pil_img.size
            translate_pixels = (translate_x * width, translate_y * height)
            
            # Apply affine transformation
            affine_img = transforms.functional.affine(
                pil_img, 
                angle=0.0,  # No rotation here (we have separate rotation transform)
                translate=translate_pixels,
                scale=1.0,   # No scaling
                shear=[shear_x, shear_y]
            )
            transformed_img = to_tensor(affine_img)
            
            # Store parameters
            transform_params['translate_x'] = translate_x
            transform_params['translate_y'] = translate_y
            transform_params['shear_x'] = shear_x
            transform_params['shear_y'] = shear_y
        
        # Return result, moving back to original device if needed
        if return_params:
            return transformed_img.to(original_device), transform_params
        else:
            return transformed_img.to(original_device)


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, ood_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.ood_transform = ood_transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # Load class mapping from wnids.txt
        with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
            wnids = [line.strip() for line in f]
        
        # Create mapping from WordNet IDs to indices
        for idx, wnid in enumerate(wnids):
            self.class_to_idx[wnid] = idx
        
        # Load dataset based on split
        if split == 'train':
            # For training set - images are in train/<wnid>/images/<wnid>_<num>.JPEG
            for class_id, wnid in enumerate(wnids):
                img_dir = os.path.join(root_dir, 'train', wnid, 'images')
                if os.path.isdir(img_dir):
                    for img_file in os.listdir(img_dir):
                        if img_file.endswith('.JPEG'):
                            self.image_paths.append(os.path.join(img_dir, img_file))
                            self.labels.append(class_id)
        elif split == 'val':
            # For validation set - need to parse val_annotations.txt
            val_annotations_file = os.path.join(root_dir, 'val', 'val_annotations.txt')
            with open(val_annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    img_file, wnid = parts[0], parts[1]
                    if wnid in self.class_to_idx:
                        self.image_paths.append(os.path.join(root_dir, 'val', 'images', img_file))
                        self.labels.append(self.class_to_idx[wnid])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and convert image
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image and the same label if image is corrupted
            image = Image.new('RGB', (64, 64), color='black')
        
        # For OOD testing, we need to separate the transforms
        if self.ood_transform:
            # Convert to tensor first but don't normalize yet
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(image)
            
            # Apply OOD transform to the unnormalized tensor
            # Now the CPU transform will be used
            transformed_tensor, transform_params = self.ood_transform.apply_transforms(
                img_tensor, return_params=True
            )
            
            # Now apply normalization to both original and transformed images
            if self.transform:
                # Extract the normalization transform
                normalize_transform = None
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        normalize_transform = t
                        break
                
                if normalize_transform:
                    # Apply just the normalization
                    normalized_img = normalize_transform(img_tensor)
                    normalized_transformed = normalize_transform(transformed_tensor)
                    return normalized_img, normalized_transformed, label, transform_params
            
            # If no normalization found or no transform provided
            return img_tensor, transformed_tensor, label, transform_params
        
        # Apply standard transformations for normal training/validation
        if self.transform:
            image = self.transform(image)
        
        return image, label


# TransformationHealer - a model that predicts and corrects transformations
class TransformationHealer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=384, depth=6, head_dim=64):
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
        self.transform_type_head = nn.Linear(embed_dim, 4)  # 4 transform types (noise, rotation, h_flip, affine)
        self.severity_head = nn.Linear(embed_dim, 1)        # Scalar severity
        
        # Specific parameter heads for each transform type
        self.rotation_head = nn.Linear(embed_dim, 1)        # Rotation angle
        self.noise_head = nn.Linear(embed_dim, 1)           # Noise std
        self.h_flip_head = nn.Linear(embed_dim, 1)          # Horizontal flip flag
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
        
        # Extract patches
        x = self.patch_embed(x)
        
        # Add cls token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed.expand(B, -1, -1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Get cls token output
        x = self.norm(x[:, 0])
        
        # Predict transform type probabilities
        transform_type_logits = self.transform_type_head(x)
        
        # Predict various parameters
        severity = torch.sigmoid(self.severity_head(x))  # 0-1 range
        rotation_angle = torch.tanh(self.rotation_head(x)) * 30.0  # -30 to 30 degrees
        noise_std = torch.sigmoid(self.noise_head(x)) * 0.5  # 0-0.5 range
        
        # Horizontal flip prediction (binary)
        h_flip = torch.sigmoid(self.h_flip_head(x))  # 0-1 range, >0.5 means flipped
        
        # Affine transformation parameters
        affine_params = self.affine_head(x)
        translate_x = torch.tanh(affine_params[:, 0]) * 0.1  # -0.1 to 0.1 range (10% of image size)
        translate_y = torch.tanh(affine_params[:, 1]) * 0.1  # -0.1 to 0.1 range
        shear_x = torch.tanh(affine_params[:, 2]) * 15.0  # -15 to 15 degrees
        shear_y = torch.tanh(affine_params[:, 3]) * 15.0  # -15 to 15 degrees
        
        return {
            'transform_type_logits': transform_type_logits,
            'severity': severity,
            'rotation_angle': rotation_angle,
            'noise_std': noise_std,
            'h_flip': h_flip,
            'translate_x': translate_x,
            'translate_y': translate_y,
            'shear_x': shear_x,
            'shear_y': shear_y
        }
    
    def apply_correction(self, images, predictions):
        """
        Apply inverse transformations to correct the distorted images
        
        Args:
            images: Batch of distorted images [B, C, H, W]
            predictions: Dictionary of transformation predictions
            
        Returns:
            corrected_images: Batch of corrected images [B, C, H, W]
        """
        # Store original device
        original_device = images.device
        
        # Move predictions to CPU for processing
        transform_types = torch.argmax(predictions['transform_type_logits'], dim=1).cpu()
        
        # Copy predictions to CPU
        cpu_predictions = {
            'transform_type_logits': predictions['transform_type_logits'].cpu(),
            'severity': predictions['severity'].cpu(),
            'rotation_angle': predictions['rotation_angle'].cpu(),
            'noise_std': predictions['noise_std'].cpu(),
            'h_flip': predictions['h_flip'].cpu(),
            'translate_x': predictions['translate_x'].cpu(),
            'translate_y': predictions['translate_y'].cpu(),
            'shear_x': predictions['shear_x'].cpu(),
            'shear_y': predictions['shear_y'].cpu(),
        }
        
        transform_map = {
            0: 'gaussian_noise',
            1: 'rotation',
            2: 'h_flip',
            3: 'affine'
        }
        
        corrected_images = []
        
        # Process each image on CPU
        for i, img in enumerate(images):
            img = img.cpu()  # Move image to CPU for processing
            transform_type = transform_map[transform_types[i].item()]
            
            # Apply inverse transformation based on predicted type
            if transform_type == 'gaussian_noise':
                # For noise, we can't perfectly recover, but we can apply a denoising filter
                # Here we use a simple Gaussian blur as denoising
                std = cpu_predictions['noise_std'][i].item()
                if std > 0.1:  # Only denoise if significant noise detected
                    # Convert to PIL for blur
                    to_pil = transforms.ToPILImage()
                    to_tensor = transforms.ToTensor()
                    pil_img = to_pil(img)
                    # Apply mild blur to reduce noise
                    kernel_size = 3
                    sigma = std * 1.5
                    denoised_img = transforms.functional.gaussian_blur(pil_img, kernel_size, sigma)
                    corrected_img = to_tensor(denoised_img)
                else:
                    corrected_img = img
                    
            elif transform_type == 'rotation':
                # For rotation, apply negative of predicted angle
                angle = -cpu_predictions['rotation_angle'][i].item()
                to_pil = transforms.ToPILImage()
                to_tensor = transforms.ToTensor()
                pil_img = to_pil(img)
                rotated_img = transforms.functional.rotate(pil_img, angle)
                corrected_img = to_tensor(rotated_img)
                
            elif transform_type == 'h_flip':
                # For horizontal flip, just flip again to invert
                is_flipped = cpu_predictions['h_flip'][i].item() > 0.5
                if is_flipped:
                    to_pil = transforms.ToPILImage()
                    to_tensor = transforms.ToTensor()
                    pil_img = to_pil(img)
                    unflipped_img = transforms.functional.hflip(pil_img)
                    corrected_img = to_tensor(unflipped_img)
                else:
                    corrected_img = img
                    
            elif transform_type == 'affine':
                # For affine transformation, apply the inverse transformation
                translate_x = -cpu_predictions['translate_x'][i].item()  # Negative of predicted translation
                translate_y = -cpu_predictions['translate_y'][i].item() 
                shear_x = -cpu_predictions['shear_x'][i].item()  # Negative of predicted shear
                shear_y = -cpu_predictions['shear_y'][i].item()
                
                to_pil = transforms.ToPILImage()
                to_tensor = transforms.ToTensor()
                pil_img = to_pil(img)
                
                # Get image size for translation calculation
                width, height = pil_img.size
                translate_pixels = (translate_x * width, translate_y * height)
                
                # Apply inverse affine transformation
                corrected_pil = transforms.functional.affine(
                    pil_img, 
                    angle=0.0,
                    translate=translate_pixels,
                    scale=1.0,
                    shear=[shear_x, shear_y]
                )
                corrected_img = to_tensor(corrected_pil)
            
            corrected_images.append(corrected_img)
            
        # Stack and move back to original device
        return torch.stack(corrected_images).to(original_device)


# Loss function for the healer model
class HealerLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predictions, targets):
        """
        Calculate the combined loss for the healer model
        
        Args:
            predictions: Dictionary of model predictions
            targets: Dictionary of target values
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Transform type classification loss
        transform_type_loss = self.ce_loss(
            predictions['transform_type_logits'], 
            targets['transform_type_idx']
        )
        
        # Get the actual transform type for each sample
        batch_size = predictions['transform_type_logits'].shape[0]
        transform_types = targets['transform_type_idx']
        
        # Initialize parameter losses
        severity_loss = torch.tensor(0.0, device=predictions['severity'].device)
        rotation_loss = torch.tensor(0.0, device=predictions['severity'].device)
        noise_loss = torch.tensor(0.0, device=predictions['severity'].device)
        h_flip_loss = torch.tensor(0.0, device=predictions['severity'].device)
        affine_loss = torch.tensor(0.0, device=predictions['severity'].device)
        
        # For each transform type, calculate parameter loss only for samples of that type
        # Gaussian noise (index 0)
        noise_mask = (transform_types == 0)
        if noise_mask.sum() > 0:
            noise_loss = self.mse_loss(
                predictions['noise_std'][noise_mask], 
                targets['noise_std'][noise_mask]
            )
            
        # Rotation (index 1)
        rot_mask = (transform_types == 1)
        if rot_mask.sum() > 0:
            rotation_loss = self.mse_loss(
                predictions['rotation_angle'][rot_mask], 
                targets['rotation_angle'][rot_mask]
            )
            
        # Horizontal flip (index 2) - binary classification
        h_flip_mask = (transform_types == 2)
        if h_flip_mask.sum() > 0:
            h_flip_loss = self.bce_loss(
                predictions['h_flip'][h_flip_mask], 
                targets['h_flip'][h_flip_mask]
            )
            
        # Affine (index 3) - multiple regression parameters
        affine_mask = (transform_types == 3)
        if affine_mask.sum() > 0:
            # Combined MSE loss for all affine parameters
            translate_x_loss = self.mse_loss(
                predictions['translate_x'][affine_mask], 
                targets['translate_x'][affine_mask]
            )
            translate_y_loss = self.mse_loss(
                predictions['translate_y'][affine_mask], 
                targets['translate_y'][affine_mask]
            )
            shear_x_loss = self.mse_loss(
                predictions['shear_x'][affine_mask], 
                targets['shear_x'][affine_mask]
            )
            shear_y_loss = self.mse_loss(
                predictions['shear_y'][affine_mask], 
                targets['shear_y'][affine_mask]
            )
            
            # Average all affine parameter losses
            affine_loss = (translate_x_loss + translate_y_loss + shear_x_loss + shear_y_loss) / 4.0
            
        # Severity loss for all samples
        severity_loss = self.mse_loss(predictions['severity'], targets['severity'])
        
        # Combine losses
        total_loss = (
            transform_type_loss + 
            0.5 * severity_loss + 
            0.3 * (rotation_loss + noise_loss + h_flip_loss) +
            0.2 * affine_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'transform_type': transform_type_loss.item(),
            'severity': severity_loss.item(),
            'rotation': rotation_loss.item(),
            'noise': noise_loss.item(),
            'h_flip': h_flip_loss.item(),
            'affine': affine_loss.item()
        }
        
        return total_loss, loss_dict


# TTT (Test-Time Training) model - adapts during inference using continuous transformations
class TTTModel(nn.Module):
    def __init__(self, base_model, img_size=64, patch_size=8, embed_dim=384):
        super().__init__()
        self.base_model = base_model
        
        # Create heads for different transformation parameters, similar to the healer model
        # but for self-supervised learning during test time
        self.transform_type_head = nn.Linear(embed_dim, 4)  # 4 transform types (noise, rotation, h_flip, affine)
        self.severity_head = nn.Linear(embed_dim, 1)        # Scalar severity
        
        # Specific parameter heads for each transform type
        self.rotation_head = nn.Linear(embed_dim, 1)        # Rotation angle
        self.noise_head = nn.Linear(embed_dim, 1)           # Noise std
        self.h_flip_head = nn.Linear(embed_dim, 1)          # Horizontal flip flag
        self.affine_head = nn.Linear(embed_dim, 4)          # Affine params: translate_x, translate_y, shear_x, shear_y
        
        # Initialize the continuous transform generator
        self.transforms = ContinuousTransforms(severity=1.0)
        
    def forward(self, x, labels=None, aux_only=False, aux_predictions=None):
        # Extract features from the base model
        features = self.base_model.forward_features(x)
        cls_features = features[:, 0]  # Use CLS token features
        
        # For the auxiliary task, predict transformations
        aux_outputs = {
            'transform_type_logits': self.transform_type_head(cls_features),
            'severity': torch.sigmoid(self.severity_head(cls_features)),
            'rotation_angle': torch.tanh(self.rotation_head(cls_features)) * 30.0,
            'noise_std': torch.sigmoid(self.noise_head(cls_features)) * 0.5,
            'h_flip': torch.sigmoid(self.h_flip_head(cls_features)),
            'translate_x': torch.tanh(self.affine_head(cls_features)[:, 0:1]) * 0.1,
            'translate_y': torch.tanh(self.affine_head(cls_features)[:, 1:2]) * 0.1,
            'shear_x': torch.tanh(self.affine_head(cls_features)[:, 2:3]) * 15.0,
            'shear_y': torch.tanh(self.affine_head(cls_features)[:, 3:4]) * 15.0
        }
        
        if aux_only:
            # For test-time training, only return auxiliary task outputs
            return aux_outputs
        
        # For main task, use the base model's classification head
        logits = self.base_model.forward_head(features)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
        # Return combined outputs
        result = type('OutputsWithLoss', (), {
            'loss': loss,
            'logits': logits,
            'aux_outputs': aux_outputs
        })
        
        return result
    
    def compute_aux_loss(self, predictions, targets):
        """Compute loss for the self-supervised auxiliary task"""
        # Transform type classification loss
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()
        
        transform_type_loss = ce_loss(
            predictions['transform_type_logits'], 
            targets['transform_type_idx']
        )
        
        # Get masks for each transform type
        transform_types = targets['transform_type_idx']
        noise_mask = (transform_types == 0)
        rot_mask = (transform_types == 1)
        h_flip_mask = (transform_types == 2)
        affine_mask = (transform_types == 3)
        
        # Initialize parameter losses
        severity_loss = mse_loss(predictions['severity'], targets['severity'])
        
        # Calculate specific losses only for samples with that transformation
        # Initialize with zero tensors
        rotation_loss = torch.tensor(0.0, device=predictions['severity'].device)
        noise_loss = torch.tensor(0.0, device=predictions['severity'].device)
        h_flip_loss = torch.tensor(0.0, device=predictions['severity'].device)
        affine_loss = torch.tensor(0.0, device=predictions['severity'].device)
        
        # Compute losses only if we have samples of that type
        if noise_mask.sum() > 0:
            noise_loss = mse_loss(
                predictions['noise_std'][noise_mask], 
                targets['noise_std'][noise_mask]
            )
        
        if rot_mask.sum() > 0:
            rotation_loss = mse_loss(
                predictions['rotation_angle'][rot_mask], 
                targets['rotation_angle'][rot_mask]
            )
        
        if h_flip_mask.sum() > 0:
            h_flip_loss = bce_loss(
                predictions['h_flip'][h_flip_mask], 
                targets['h_flip'][h_flip_mask]
            )
            
        if affine_mask.sum() > 0:
            # Combined MSE loss for all affine parameters
            translate_x_loss = mse_loss(
                predictions['translate_x'][affine_mask], 
                targets['translate_x'][affine_mask]
            )
            translate_y_loss = mse_loss(
                predictions['translate_y'][affine_mask], 
                targets['translate_y'][affine_mask]
            )
            shear_x_loss = mse_loss(
                predictions['shear_x'][affine_mask], 
                targets['shear_x'][affine_mask]
            )
            shear_y_loss = mse_loss(
                predictions['shear_y'][affine_mask], 
                targets['shear_y'][affine_mask]
            )
            
            # Average all affine parameter losses
            affine_loss = (translate_x_loss + translate_y_loss + shear_x_loss + shear_y_loss) / 4.0
        
        # Combine all losses
        total_loss = (
            transform_type_loss + 
            0.5 * severity_loss + 
            0.3 * (rotation_loss + noise_loss + h_flip_loss) +
            0.2 * affine_loss
        )
        
        return total_loss
    
    def test_time_adapt(self, x, num_steps=10, lr=0.001):
        """
        Adapt the model at test time using self-supervision on the input image
        with continuous transformations
        
        Args:
            x: Input image [B, C, H, W]
            num_steps: Number of adaptation steps
            lr: Learning rate for adaptation
            
        Returns:
            adapted_model: Adapted copy of the model
        """
        # Track the device of input images
        device = x.device
        
        # Create a copy of the model for adaptation
        adapted_model = deepcopy(self)
        adapted_model.train()
        
        # Only update the auxiliary task heads and last few layers of the model
        params_to_update = []
        params_to_update += list(adapted_model.transform_type_head.parameters())
        params_to_update += list(adapted_model.severity_head.parameters())
        params_to_update += list(adapted_model.rotation_head.parameters())
        params_to_update += list(adapted_model.noise_head.parameters())
        params_to_update += list(adapted_model.h_flip_head.parameters())
        params_to_update += list(adapted_model.affine_head.parameters())
        params_to_update += list(adapted_model.base_model.transformer.blocks[-2:].parameters())
        
        # Set up optimizer
        optimizer = torch.optim.SGD(params_to_update, lr=lr)
        
        # Adapt the model for several steps
        for _ in range(num_steps):
            # Generate self-supervised samples with various transformations
            transformed_images = []
            transform_targets = {
                'transform_type_idx': [],
                'severity': [],
                'rotation_angle': [],
                'noise_std': [],
                'h_flip': [],
                'translate_x': [],
                'translate_y': [],
                'shear_x': [],
                'shear_y': []
            }
            
            # Apply different transformations to each image
            for img in x:
                # Move to CPU for transformations
                img_cpu = img.cpu()
                
                # Apply all transform types to each image
                for t_idx, t_type in enumerate(['gaussian_noise', 'rotation', 'h_flip', 'affine']):
                    # Perform transformation on CPU
                    t_img, t_params = self.transforms.apply_transforms(
                        img_cpu, transform_type=t_type, return_params=True
                    )
                    # Move back to the correct device
                    t_img = t_img.to(device)
                            
                    # Add transformed image
                    transformed_images.append(t_img)
                    
                    # Add transformation parameters to targets
                    transform_targets['transform_type_idx'].append(t_idx)
                    transform_targets['severity'].append(t_params['severity'])
                    transform_targets['rotation_angle'].append(t_params['rotation_angle'])
                    transform_targets['noise_std'].append(t_params['noise_std'])
                    transform_targets['h_flip'].append(t_params['h_flip'])
                    transform_targets['translate_x'].append(t_params['translate_x'])
                    transform_targets['translate_y'].append(t_params['translate_y'])
                    transform_targets['shear_x'].append(t_params['shear_x'])
                    transform_targets['shear_y'].append(t_params['shear_y'])
            
            # Convert lists to tensors and move to device
            transformed_images = torch.stack(transformed_images).to(device)
            
            for key in transform_targets:
                if key == 'transform_type_idx':
                    transform_targets[key] = torch.tensor(transform_targets[key], dtype=torch.long, device=device)
                else:
                    transform_targets[key] = torch.tensor(transform_targets[key], dtype=torch.float32, device=device).unsqueeze(1)
            
            # Forward pass - compute auxiliary task predictions only
            aux_outputs = adapted_model(transformed_images, aux_only=True)
            
            # Compute loss
            loss = adapted_model.compute_aux_loss(aux_outputs, transform_targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
            optimizer.step()
        
        # Switch back to eval mode
        adapted_model.eval()
        
        return adapted_model


# Custom loss wrapper for compatibility with the training script
class CustomModelWithLoss(torch.nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, pixel_values, labels=None):
        # Forward pass through ViT model
        logits = self.vit_model(pixel_values)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return type('OutputsWithLoss', (), {
            'loss': loss,
            'logits': logits
        })


def find_optimal_batch_size(model, img_size, starting_batch_size=2000, device=None):
    """
    Find the optimal batch size that fits in GPU memory with precision.
    
    Args:
        model: The model to be used in training
        img_size: Size of input images
        starting_batch_size: Initial batch size to try
        device: Device to use (if None, will be determined automatically)
        
    Returns:
        optimal_batch_size: The largest batch size that fits in memory
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU with default batch size.")
        return starting_batch_size
    
    model = model.to(device)
    
    # If we're on a CPU, just return the starting batch size
    if device.type == 'cpu':
        return starting_batch_size
    
    # Phase 1: Binary search to quickly find a working batch size
    print("Phase 1: Finding an initial working batch size...")
    upper_bound = starting_batch_size
    lower_bound = 1
    current_batch_size = upper_bound
    
    # Test if starting batch size works
    try:
        # Generate random test data
        dummy_inputs = torch.randn(current_batch_size, 3, img_size, img_size, device=device)
        
        # Try a forward and backward pass
        outputs = model(dummy_inputs)
        
        # If we have a wrapped model with loss, we need a dummy label
        if hasattr(model, 'loss_fn'):
            dummy_labels = torch.randint(0, 200, (current_batch_size,), device=device)
            outputs = model(pixel_values=dummy_inputs, labels=dummy_labels)
            loss = outputs.loss
            loss.backward()
        else:
            # For regular models, just do a backward pass on output sum
            dummy_loss = outputs.sum()
            dummy_loss.backward()
        
        # Clean up
        torch.cuda.empty_cache()
        
        # If we get here, the starting batch size worked!
        print(f"Starting batch size {starting_batch_size} fits in memory!")
    except torch.cuda.OutOfMemoryError:
        # Starting batch size is too large, go into binary search
        upper_bound = starting_batch_size
        while upper_bound > lower_bound:
            # Try the middle point
            current_batch_size = (upper_bound + lower_bound) // 2
            print(f"Testing batch size: {current_batch_size}...")
            
            try:
                # Generate random test data
                dummy_inputs = torch.randn(current_batch_size, 3, img_size, img_size, device=device)
                
                # Try a forward and backward pass
                outputs = model(dummy_inputs)
                
                # If we have a wrapped model with loss, we need a dummy label
                if hasattr(model, 'loss_fn'):
                    dummy_labels = torch.randint(0, 200, (current_batch_size,), device=device)
                    outputs = model(pixel_values=dummy_inputs, labels=dummy_labels)
                    loss = outputs.loss
                    loss.backward()
                else:
                    # For regular models, just do a backward pass on output sum
                    dummy_loss = outputs.sum()
                    dummy_loss.backward()
                
                # Clean up
                torch.cuda.empty_cache()
                
                # If we get here, it fits - try a larger batch size
                lower_bound = current_batch_size
            except torch.cuda.OutOfMemoryError:
                # Too big, free memory and try a smaller batch size
                torch.cuda.empty_cache()
                upper_bound = current_batch_size - 1
    
    # Phase 2: Linear search to find the exact maximum batch size
    print(f"Phase 2: Refining batch size around {current_batch_size}...")
    
    # Start with the batch size that worked from Phase 1
    working_batch_size = current_batch_size
    
    # Try increasing in small steps until we hit OOM
    increment = 16  # Start with bigger steps
    
    while increment >= 1:
        test_batch_size = working_batch_size + increment
        print(f"Testing batch size: {test_batch_size}...")
        
        try:
            # Generate random test data
            dummy_inputs = torch.randn(test_batch_size, 3, img_size, img_size, device=device)
            
            # Try a forward and backward pass
            outputs = model(dummy_inputs)
            
            # If we have a wrapped model with loss, we need a dummy label
            if hasattr(model, 'loss_fn'):
                dummy_labels = torch.randint(0, 200, (test_batch_size,), device=device)
                outputs = model(pixel_values=dummy_inputs, labels=dummy_labels)
                loss = outputs.loss
                loss.backward()
            else:
                # For regular models, just do a backward pass on output sum
                dummy_loss = outputs.sum()
                dummy_loss.backward()
            
            # Clean up
            torch.cuda.empty_cache()
            
            # If we get here, it worked - update the working batch size
            working_batch_size = test_batch_size
        except torch.cuda.OutOfMemoryError:
            # Too big, reduce the increment
            torch.cuda.empty_cache()
            increment = increment // 2
    
    # The working batch size is now the maximum that fits
    print(f"Found optimal batch size: {working_batch_size}")
    
    # For safety, return a much smaller batch size to allow for memory fluctuations
    # and transformation operations during training (25% of maximum)
    safe_batch_size = max(1, int(working_batch_size *  0.95))
    print(f"Using safe batch size: {safe_batch_size}")  # More conservative for CPU transforms
    
    return safe_batch_size


def train_main_model(dataset_path="tiny-imagenet-200"):
    """
    Train the main classification model on the Tiny ImageNet dataset
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create checkpoint directories
    checkpoints_dir = Path("checkpoints_main")
    best_model_dir = Path("bestmodel_main")
    
    # Create directories if they don't exist
    checkpoints_dir.mkdir(exist_ok=True)
    best_model_dir.mkdir(exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize our custom model
    vit_model = create_vit_model(
        img_size=64,        # Tiny ImageNet is 64x64
        patch_size=8,       # Smaller patch size for smaller images
        in_chans=3,
        num_classes=200,    # Tiny ImageNet has 200 classes
        embed_dim=384,      # Reduced embedding dimension
        depth=8,            # Reduced depth for faster training
        head_dim=64,
        mlp_ratio=4.0,
        use_resnet_stem=True
    )
    
    # Wrap it with loss calculation for compatibility
    model = CustomModelWithLoss(vit_model)
    model.to(device)
    
    # Find optimal batch size for the GPU
    batch_size = find_optimal_batch_size(model, img_size=64, starting_batch_size=128, device=device)
    
    # Define image transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset and DataLoader
    train_dataset = TinyImageNetDataset(dataset_path, "train", transform_train)
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Training parameters
    num_epochs = 1
    learning_rate = 1e-4
    warmup_steps = 1000
    patience = 7
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    # Learning rate scheduler with linear warmup and cosine decay
    def get_lr(step, total_steps, warmup_steps, base_lr):
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)
        else:
            decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
            return base_lr * 0.5 * (1 + np.cos(np.pi * decay_ratio))
    
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: get_lr(step, total_steps, warmup_steps, 1.0)
    )
    
    # Logging with wandb
    wandb.config.update({
        "model": "main_classification",
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps
    }, allow_val_change=True)
    
    # Initialize early stopping variables
    best_val_acc = 0
    early_stop_counter = 0
    best_epoch = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        # Training step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Free up GPU memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Validation step
        val_loss, val_acc = validate_main_model(model, val_loader, device)
        
        # Log metrics
        wandb.log({
            "main/epoch": epoch + 1,
            "main/train_loss": train_loss,
            "main/train_accuracy": train_acc,
            "main/val_loss": val_loss,
            "main/val_accuracy": val_acc,
            "main/learning_rate": scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            
            # Save checkpoint
            checkpoint_path = checkpoints_dir / f"model_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            
            # Save best model
            best_model_path = best_model_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, best_model_path)
            
            # Log to wandb
            wandb.save(str(checkpoint_path))
            wandb.save(str(best_model_path))
            
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
            
            # Track the best epoch for later reference
            best_epoch = epoch + 1
        else:
            early_stop_counter += 1
            
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Clean up old checkpoints except the best one
    print("Cleaning up checkpoints to save disk space...")
    for checkpoint_file in checkpoints_dir.glob("*.pt"):
        if f"model_epoch{best_epoch}.pt" != checkpoint_file.name:
            checkpoint_file.unlink()
            print(f"Deleted {checkpoint_file}")
    
    print(f"Training completed. Best model saved at: {best_model_dir / 'best_model.pt'}")
    
    # Load and return the best model
    checkpoint = torch.load(best_model_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.vit_model  # Return the actual ViT model without the loss wrapper


def validate_main_model(model, loader, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            
            # Update metrics
            val_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    val_loss /= len(loader)
    val_acc = accuracy_score(all_labels, all_preds)
    
    return val_loss, val_acc


def train_healer_model(dataset_path="tiny-imagenet-200", severity=1.0):
    """
    Train the transformation healer model
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create checkpoint directories
    checkpoints_dir = Path("checkpoints_healer")
    best_model_dir = Path("bestmodel_healer")
    
    # Create directories if they don't exist
    checkpoints_dir.mkdir(exist_ok=True)
    best_model_dir.mkdir(exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the healer model
    healer_model = TransformationHealer(
        img_size=64,
        patch_size=8,
        in_chans=3,
        embed_dim=384,
        depth=6,
        head_dim=64
    )
    healer_model.to(device)
    
    # Loss function
    healer_loss = HealerLoss()
    
    # Define image transformations
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create continuous transforms for OOD
    ood_transform = ContinuousTransforms(severity=severity)
    
    # Dataset and DataLoader with OOD transforms
    train_dataset = TinyImageNetDataset(
        dataset_path, "train", transform_train, ood_transform=ood_transform
    )
    
    print(f"Training set size for healer: {len(train_dataset)}")
    
    # Debug function for collate
    def debug_collate(batch):
        orig_imgs, trans_imgs, labels, params = zip(*batch)
        # Print type and structure of params for debugging
        print(f"Type of params: {type(params)}")
        print(f"Length of params: {len(params)}")
        print(f"Type of first item in params: {type(params[0])}")
        if isinstance(params[0], dict):
            print(f"Keys in first param: {params[0].keys()}")
        
        # Proper collation
        orig_tensor = torch.stack(orig_imgs)
        trans_tensor = torch.stack(trans_imgs)
        labels_tensor = torch.tensor(labels)
        # Keep params as a list of dictionaries
        
        return orig_tensor, trans_tensor, labels_tensor, params
    
    # Start with a small batch size for debugging, it will automatically be reduced for memory limits
    batch_size = 4  # Reduced batch size for debugging
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # No parallelism for debugging 
        pin_memory=True,
        collate_fn=debug_collate
    )
    
    # Training parameters
    num_epochs = 1  # Changed to 1 for debugging
    learning_rate = 5e-5
    warmup_steps = 500
    
    # Optimizer
    optimizer = torch.optim.AdamW(healer_model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler with linear warmup and cosine decay
    def get_lr(step, total_steps, warmup_steps, base_lr):
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)
        else:
            decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
            return base_lr * 0.5 * (1 + np.cos(np.pi * decay_ratio))
    
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: get_lr(step, total_steps, warmup_steps, 1.0)
    )
    
    # Logging with wandb
    wandb.config.update({
        "model": "transformation_healer",
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "ood_severity": severity
    }, allow_val_change=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        healer_model.train()
        train_loss = 0
        transform_type_acc = 0
        total_samples = 0
        
        # Training step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (orig_images, transformed_images, labels, transform_params) in enumerate(progress_bar):
            # Move to device after transformations have been applied
            orig_images = orig_images.to(device)
            transformed_images = transformed_images.to(device)
            
            # Debug output for the first batch
            if batch_idx == 0:
                print("\nFirst batch transform_params structure:")
                for i, tp in enumerate(transform_params):
                    print(f"Item {i}: {type(tp)}")
                    if hasattr(tp, "keys"):
                        print(f"  Keys: {tp.keys()}")
                    else:
                        print(f"  Value: {tp}")
            
            # Convert transform_params to target tensors
            targets = {}
            
            # Create target tensors for each transformation type
            transform_type_mapping = {
                'gaussian_noise': 0,
                'rotation': 1,
                'h_flip': 2,
                'affine': 3
            }
            
            # Extract parameters from dict to tensors - with more error checking
            try:
                # Initialize tensors
                transform_types = []
                severity_values = []
                noise_std_values = []
                rotation_angle_values = []
                h_flip_values = []
                translate_x_values = []
                translate_y_values = []
                shear_x_values = []
                shear_y_values = []
                
                for params in transform_params:
                    # Check if params is a dictionary
                    if not isinstance(params, dict):
                        print(f"Warning: params is not a dictionary, it's {type(params)}")
                        if isinstance(params, tuple) and len(params) >= 2 and isinstance(params[0], str):
                            # Assume the first element is transform_type and second is severity
                            transform_type = params[0]
                            severity = params[1] if len(params) > 1 else 1.0
                            
                            transform_types.append(transform_type_mapping.get(transform_type, 0))
                            severity_values.append(severity)
                            # Default values for other parameters
                            noise_std_values.append(0.0)
                            rotation_angle_values.append(0.0)
                            h_flip_values.append(0.0)
                            translate_x_values.append(0.0)
                            translate_y_values.append(0.0)
                            shear_x_values.append(0.0)
                            shear_y_values.append(0.0)
                            
                            continue
                    
                    # Normal dictionary case
                    transform_type = params.get('transform_type', 'gaussian_noise')
                    transform_types.append(transform_type_mapping.get(transform_type, 0))
                    
                    severity_values.append(params.get('severity', 1.0))
                    noise_std_values.append(params.get('noise_std', 0.0))
                    rotation_angle_values.append(params.get('rotation_angle', 0.0))
                    h_flip_values.append(params.get('h_flip', 0.0))
                    translate_x_values.append(params.get('translate_x', 0.0))
                    translate_y_values.append(params.get('translate_y', 0.0))
                    shear_x_values.append(params.get('shear_x', 0.0))
                    shear_y_values.append(params.get('shear_y', 0.0))
                
                # Convert lists to tensors
                targets['transform_type_idx'] = torch.tensor(transform_types, device=device)
                targets['severity'] = torch.tensor(severity_values, device=device).unsqueeze(1)
                targets['noise_std'] = torch.tensor(noise_std_values, device=device).unsqueeze(1)
                targets['rotation_angle'] = torch.tensor(rotation_angle_values, device=device).unsqueeze(1)
                targets['h_flip'] = torch.tensor(h_flip_values, device=device).unsqueeze(1)
                targets['translate_x'] = torch.tensor(translate_x_values, device=device).unsqueeze(1)
                targets['translate_y'] = torch.tensor(translate_y_values, device=device).unsqueeze(1)
                targets['shear_x'] = torch.tensor(shear_x_values, device=device).unsqueeze(1)
                targets['shear_y'] = torch.tensor(shear_y_values, device=device).unsqueeze(1)
                
            except Exception as e:
                print(f"Error processing parameters: {e}")
                print(f"transform_params type: {type(transform_params)}")
                print(f"First few transform_params: {transform_params[:min(5, len(transform_params))]}")
                raise
            
            # Forward pass
            predictions = healer_model(transformed_images)
            
            # Calculate loss
            loss, loss_dict = healer_loss(predictions, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(healer_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Calculate transform type accuracy
            pred_types = torch.argmax(predictions['transform_type_logits'], dim=1)
            transform_type_acc += (pred_types == targets['transform_type_idx']).sum().item()
            total_samples += len(pred_types)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Break after a few batches for debugging
            if batch_idx >= 5:
                print("Breaking after 5 batches for debugging")
                break
        
        # Calculate training metrics
        train_loss /= min(5, len(train_loader))
        transform_type_acc /= total_samples
        
        # Log metrics
        wandb.log({
            "healer/epoch": epoch + 1,
            "healer/train_loss": train_loss,
            "healer/transform_type_accuracy": transform_type_acc,
            **{f"healer/loss_{k}": v for k, v in loss_dict.items()},
            "healer/learning_rate": scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Transform Type Acc: {transform_type_acc:.4f}")
        
        # Save checkpoint
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            
            checkpoint_path = checkpoints_dir / f"model_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': healer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            
            # Save best model
            best_model_path = best_model_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': healer_model.state_dict(),
                'train_loss': train_loss,
            }, best_model_path)
            
            # Log to wandb
            wandb.save(str(checkpoint_path))
            wandb.save(str(best_model_path))
            
            print(f"Saved best model with training loss: {train_loss:.4f}")
    
    print(f"Healer model training completed. Best model saved at: {best_model_dir / 'best_model.pt'}")
    
    # Load and return the best model
    checkpoint = torch.load(best_model_dir / "best_model.pt")
    healer_model.load_state_dict(checkpoint['model_state_dict'])
    
    return healer_model


def train_ttt_model(base_model, dataset_path="tiny-imagenet-200"):
    """
    Train the Test-Time Training (TTT) model with continuous transformations
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create checkpoint directories
    checkpoints_dir = Path("checkpoints_ttt")
    best_model_dir = Path("bestmodel_ttt")
    
    # Create directories if they don't exist
    checkpoints_dir.mkdir(exist_ok=True)
    best_model_dir.mkdir(exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize TTT model with the base model
    ttt_model = TTTModel(base_model)
    ttt_model.to(device)
    
    # Define image transformations
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create continuous transforms for training - will operate on CPU
    continuous_transform = ContinuousTransforms(severity=1.0)
    
    # Dataset and DataLoader
    train_dataset = TinyImageNetDataset(dataset_path, "train", transform_train)
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform_train)
    
    print(f"Training set size for TTT: {len(train_dataset)}")
    
    # Reduce batch size to account for CPU transforms and memory usage
    batch_size = 64  # Use a smaller batch size to accommodate CPU transforms
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Training parameters
    num_epochs = 1
    learning_rate = 1e-4
    
    # Optimizer - only update the auxiliary heads and the last few layers
    params_to_update = []
    params_to_update += list(ttt_model.transform_type_head.parameters())
    params_to_update += list(ttt_model.severity_head.parameters())
    params_to_update += list(ttt_model.rotation_head.parameters())
    params_to_update += list(ttt_model.noise_head.parameters())
    params_to_update += list(ttt_model.h_flip_head.parameters())
    params_to_update += list(ttt_model.affine_head.parameters())
    params_to_update += list(ttt_model.base_model.transformer.blocks[-2:].parameters())
    
    optimizer = torch.optim.AdamW(params_to_update, lr=learning_rate, weight_decay=0.01)
    
    # Loss function
    class_loss_fn = nn.CrossEntropyLoss()
    
    # Logging with wandb
    wandb.config.update({
        "model": "ttt_model_continuous",
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size
    }, allow_val_change=True)
    
    # Initialize best validation accuracy
    best_val_acc = 0
    
    # Training loop
    for epoch in range(num_epochs):
        ttt_model.train()
        train_loss = 0
        train_class_loss = 0
        train_aux_loss = 0
        class_correct = 0
        total_samples = 0
        
        # Training step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Apply different transformations to create self-supervised data
            transformed_images = []
            transform_targets = {
                'transform_type_idx': [],
                'severity': [],
                'rotation_angle': [],
                'noise_std': [],
                'h_flip': [],
                'translate_x': [],
                'translate_y': [],
                'shear_x': [],
                'shear_y': []
            }
            
            # Apply different transformations to each image in the batch on CPU
            for img in images:
                # Move to CPU for transformations
                img_cpu = img.cpu()
                
                # Apply 4 different transformations to each image (one of each type)
                for t_idx, t_type in enumerate(['gaussian_noise', 'rotation', 'h_flip', 'affine']):
                    # Apply transformation and get parameters on CPU
                    t_img, t_params = continuous_transform.apply_transforms(
                        img_cpu, transform_type=t_type, return_params=True
                    )
                    # Move transformed image back to device
                    t_img = t_img.to(device)
                    
                    # Add transformed image
                    transformed_images.append(t_img)
                    
                    # Add transformation parameters to targets
                    transform_targets['transform_type_idx'].append(t_idx)
                    transform_targets['severity'].append(t_params['severity'])
                    
                    # Add default values for all parameters (will be overwritten for relevant ones)
                    transform_targets['rotation_angle'].append(0.0)
                    transform_targets['noise_std'].append(0.0)
                    transform_targets['h_flip'].append(0.0)
                    transform_targets['translate_x'].append(0.0)
                    transform_targets['translate_y'].append(0.0)
                    transform_targets['shear_x'].append(0.0)
                    transform_targets['shear_y'].append(0.0)
                    
                    # Set the specific parameter for this transformation
                    if t_type == 'gaussian_noise':
                        transform_targets['noise_std'][-1] = t_params['noise_std']
                    elif t_type == 'rotation':
                        transform_targets['rotation_angle'][-1] = t_params['rotation_angle']
                    elif t_type == 'h_flip':
                        transform_targets['h_flip'][-1] = t_params['h_flip']
                    elif t_type == 'affine':
                        transform_targets['translate_x'][-1] = t_params['translate_x']
                        transform_targets['translate_y'][-1] = t_params['translate_y']
                        transform_targets['shear_x'][-1] = t_params['shear_x']
                        transform_targets['shear_y'][-1] = t_params['shear_y']
            
            # Convert lists to tensors and move to device
            transformed_images = torch.stack(transformed_images).to(device)
            
            for key in transform_targets:
                if key == 'transform_type_idx':
                    transform_targets[key] = torch.tensor(transform_targets[key], dtype=torch.long, device=device)
                else:
                    transform_targets[key] = torch.tensor(transform_targets[key], dtype=torch.float32, device=device).unsqueeze(1)
            
            # Repeat labels for the transformations (4 transformations per image)
            repeated_labels = labels.repeat_interleave(4)
            
            # Forward pass
            outputs = ttt_model(transformed_images, repeated_labels)
            
            # Calculate classification loss
            class_loss = outputs.loss
            
            # Calculate auxiliary transformation prediction loss
            aux_loss = ttt_model.compute_aux_loss(outputs.aux_outputs, transform_targets)
            
            # Combined loss
            loss = class_loss + aux_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_class_loss += class_loss.item()
            train_aux_loss += aux_loss.item()
            
            # Calculate classification accuracy
            class_preds = torch.argmax(outputs.logits, dim=1)
            class_correct += (class_preds == repeated_labels).sum().item()
            
            total_samples += len(transformed_images)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Free up memory
            torch.cuda.empty_cache()
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_class_loss /= len(train_loader)
        train_aux_loss /= len(train_loader)
        train_class_acc = class_correct / total_samples
        
        # Validation step
        val_acc = evaluate_ttt_model(ttt_model, val_loader, device)
        
        # Log metrics
        wandb.log({
            "ttt/epoch": epoch + 1,
            "ttt/train_loss": train_loss,
            "ttt/train_class_loss": train_class_loss,
            "ttt/train_aux_loss": train_aux_loss,
            "ttt/train_class_accuracy": train_class_acc,
            "ttt/val_accuracy": val_acc
        })
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Class Acc: {train_class_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save checkpoint if it's the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save best model
            best_model_path = best_model_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': ttt_model.state_dict(),
                'val_acc': val_acc,
            }, best_model_path)
            
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
    
    print(f"TTT model training completed. Best model saved at: {best_model_dir / 'best_model.pt'}")
    
    # Load and return the best model
    checkpoint = torch.load(best_model_dir / "best_model.pt")
    ttt_model.load_state_dict(checkpoint['model_state_dict'])
    
    return ttt_model


def evaluate_ttt_model(model, loader, device):
    """
    Evaluate the TTT model on a dataset
    """
    model.eval()
    correct = 0
    total = 0
    
    # For debugging, limit evaluation to 5 batches
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= 5:  # Limit to 5 batches for debugging
                break
                
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Avoid division by zero
    if total == 0:
        return 0.0
        
    return correct / total


def evaluate_models(main_model, healer_model, ttt_model, dataset_path="tiny-imagenet-200", severity=0.0):
    """
    Comprehensive evaluation of models on both clean and transformed data.
    
    Args:
        main_model: The base classification model
        healer_model: The transformation healing model
        ttt_model: The test-time training model
        dataset_path: Path to the dataset
        severity: Severity of transformations (0.0 means no transformations)
        
    Returns:
        results: Dictionary containing all evaluation metrics
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define image transformations
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # For severity 0, we don't apply transformations
    if severity > 0:
        # Dataset with transformations
        ood_transform = ContinuousTransforms(severity=severity)
        val_dataset = TinyImageNetDataset(
            dataset_path, "val", transform_val, ood_transform=ood_transform
        )
    else:
        # Dataset without transformations (clean data)
        val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
    
    batch_size = 64  # Reduced for CPU transforms
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Ensure models are in eval mode
    main_model.eval()
    if healer_model is not None:
        healer_model.eval()
    if ttt_model is not None:
        ttt_model.eval()
    
    # Metrics
    results = {
        'main': {'correct': 0, 'total': 0},
        'healer': {'correct': 0, 'total': 0} if healer_model is not None else None,
        'ttt': {'correct': 0, 'total': 0} if ttt_model is not None else None,
    }
    
    # For transformed data, track per-transformation metrics
    transform_types = ['gaussian_noise', 'rotation', 'h_flip', 'affine']
    if severity > 0:
        for model_name in ['main', 'healer', 'ttt']:
            if results[model_name] is not None:
                results[model_name]['per_transform'] = {t: {'correct': 0, 'total': 0} for t in transform_types}
    
    # Helper function to determine transform type by examining parameter values
    def determine_transform_type(params):
        """
        Determine transform type by examining parameter values directly
        """
        try:
            # Handle different possible formats
            if isinstance(params, dict):
                # Check each transform parameter to determine which was applied
                if params.get('noise_std', 0.0) > 0.01:
                    return 'gaussian_noise'
                elif abs(params.get('rotation_angle', 0.0)) > 0.01:
                    return 'rotation'
                elif params.get('h_flip', 0.0) > 0.5:  # h_flip is binary (0 or 1)
                    return 'h_flip'
                elif (abs(params.get('translate_x', 0.0)) > 0.001 or 
                      abs(params.get('translate_y', 0.0)) > 0.001 or
                      abs(params.get('shear_x', 0.0)) > 0.001 or
                      abs(params.get('shear_y', 0.0)) > 0.001):
                    return 'affine'
                # Fallback to the explicit transform_type if available
                elif 'transform_type' in params:
                    t_type = params['transform_type']
                    if isinstance(t_type, str):
                        if 'noise' in t_type.lower() or 'gaussian' in t_type.lower():
                            return 'gaussian_noise'
                        elif 'rot' in t_type.lower():
                            return 'rotation'
                        elif 'flip' in t_type.lower():
                            return 'h_flip'
                        elif 'affine' in t_type.lower() or 'trans' in t_type.lower():
                            return 'affine'
                
                # If we still can't determine, use a default
                return 'gaussian_noise'
            
            # Handle string case (unlikely but for completeness)
            elif isinstance(params, str):
                t_type = params.lower()
                if 'noise' in t_type or 'gaussian' in t_type:
                    return 'gaussian_noise'
                elif 'rot' in t_type:
                    return 'rotation'
                elif 'flip' in t_type or 'mirror' in t_type:
                    return 'h_flip'
                elif 'affine' in t_type or 'trans' in t_type:
                    return 'affine'
                return 'gaussian_noise'  # Default
            
            # Handle unexpected formats
            else:
                print(f"Warning: Unexpected params format: {type(params)}")
                return 'gaussian_noise'  # Default fallback
                
        except Exception as e:
            print(f"Error determining transform type: {e}")
            return 'gaussian_noise'  # Safe fallback
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i < 500 : 
                if severity > 0:
                    # Unpack batch with transformations
                    orig_images, transformed_images, labels, transform_params = batch
                    orig_images = orig_images.to(device)
                    transformed_images = transformed_images.to(device)
                    labels = labels.to(device)
                    
                    # 1. Evaluate main model on transformed images
                    main_outputs = main_model(transformed_images)
                    main_preds = torch.argmax(main_outputs, dim=1)
                    results['main']['correct'] += (main_preds == labels).sum().item()
                    results['main']['total'] += labels.size(0)
                    
                    # Track per-transformation accuracy for main model
                    for i, params in enumerate(transform_params):
                        # Use the helper function to determine transform type based on parameter values
                        t_type = determine_transform_type(params)
                        
                        results['main']['per_transform'][t_type]['total'] += 1
                        if main_preds[i] == labels[i]:
                            results['main']['per_transform'][t_type]['correct'] += 1
                    
                    # 2. Evaluate healer model if provided
                    if healer_model is not None:
                        # Predict transformations
                        healer_predictions = healer_model(transformed_images)
                        
                        # Apply inverse transformations
                        corrected_images = healer_model.apply_correction(transformed_images, healer_predictions)
                        
                        # Run corrected images through main model
                        healer_outputs = main_model(corrected_images)
                        healer_preds = torch.argmax(healer_outputs, dim=1)
                        results['healer']['correct'] += (healer_preds == labels).sum().item()
                        results['healer']['total'] += labels.size(0)
                        
                        # Track per-transformation accuracy for healer
                        for i, params in enumerate(transform_params):
                            t_type = determine_transform_type(params)
                            
                            results['healer']['per_transform'][t_type]['total'] += 1
                            if healer_preds[i] == labels[i]:
                                results['healer']['per_transform'][t_type]['correct'] += 1
                    
                    # 3. Evaluate TTT model if provided
                    if ttt_model is not None:
                        # Create copy and adapt it at test time
                        with torch.enable_grad():
                            adapted_model = ttt_model.test_time_adapt(transformed_images, num_steps=5)
                        
                        # Run adapted model on transformed images
                        ttt_outputs = adapted_model(transformed_images)
                        ttt_preds = torch.argmax(ttt_outputs.logits, dim=1)
                        results['ttt']['correct'] += (ttt_preds == labels).sum().item()
                        results['ttt']['total'] += labels.size(0)
                        
                        # Track per-transformation accuracy for TTT
                        for i, params in enumerate(transform_params):
                            t_type = determine_transform_type(params)
                            
                            results['ttt']['per_transform'][t_type]['total'] += 1
                            if ttt_preds[i] == labels[i]:
                                results['ttt']['per_transform'][t_type]['correct'] += 1
                else:
                    # Clean data evaluation (no transformations)
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Main model on clean images
                    main_outputs = main_model(images)
                    main_preds = torch.argmax(main_outputs, dim=1)
                    results['main']['correct'] += (main_preds == labels).sum().item()
                    results['main']['total'] += labels.size(0)
                    
                    # For clean data, healer should not be used (it would add noise)
                    # But we can evaluate TTT on clean data
                    if ttt_model is not None:
                        # Just use TTT without adaptation since there's no distribution shift
                        ttt_outputs = ttt_model(images)
                        ttt_preds = torch.argmax(ttt_outputs.logits, dim=1)
                        results['ttt']['correct'] += (ttt_preds == labels).sum().item()
                        results['ttt']['total'] += labels.size(0)
        
        # Calculate overall accuracies
        for model_name in ['main', 'healer', 'ttt']:
            if results[model_name] is not None and results[model_name]['total'] > 0:
                results[model_name]['accuracy'] = results[model_name]['correct'] / results[model_name]['total']
                
                # Calculate per-transformation accuracies
                if severity > 0 and 'per_transform' in results[model_name]:
                    results[model_name]['per_transform_acc'] = {}
                    for t_type in transform_types:
                        t_total = results[model_name]['per_transform'][t_type]['total']
                        if t_total > 0:
                            results[model_name]['per_transform_acc'][t_type] = (
                                results[model_name]['per_transform'][t_type]['correct'] / t_total
                            )
                        else:
                            results[model_name]['per_transform_acc'][t_type] = 0.0
        
        return results


def evaluate_full_pipeline(main_model, healer_model, ttt_model, dataset_path="tiny-imagenet-200", severities=[0.3]):
    """
    Comprehensive evaluation across multiple severities including clean data.
    
    Args:
        main_model: The base classification model
        healer_model: The transformation healing model
        ttt_model: The test-time training model
        dataset_path: Path to the dataset
        severities: List of severity levels to evaluate
        
    Returns:
        all_results: Dictionary containing all evaluation results
    """
    all_results = {}
    
    # First evaluate on clean data
    print("\nEvaluating on clean data (no transformations)...")
    clean_results = evaluate_models(main_model, healer_model, ttt_model, dataset_path, severity=0.0)
    all_results[0.0] = clean_results
    
    # Print clean data results
    print(f"Clean Data Accuracy:")
    print(f"  Main Model: {clean_results['main']['accuracy']:.4f}")
    if clean_results['ttt'] is not None:
        print(f"  TTT Model: {clean_results['ttt']['accuracy']:.4f}")
    
    # Then evaluate on transformed data at different severities
    for severity in severities:
        if severity == 0.0:
            continue  # Skip, already evaluated
            
        print(f"\nEvaluating with severity {severity}...")
        ood_results = evaluate_models(main_model, healer_model, ttt_model, dataset_path, severity=severity)
        all_results[severity] = ood_results
        
        # Print OOD results
        print(f"OOD Accuracy (Severity {severity}):")
        print(f"  Main Model: {ood_results['main']['accuracy']:.4f}")
        if ood_results['healer'] is not None:
            print(f"  Healer Model: {ood_results['healer']['accuracy']:.4f}")
        if ood_results['ttt'] is not None:
            print(f"  TTT Model: {ood_results['ttt']['accuracy']:.4f}")
            
        # Calculate and print robustness metrics (drop compared to clean data)
        if severity > 0.0:
            main_drop = clean_results['main']['accuracy'] - ood_results['main']['accuracy']
            print(f"\nAccuracy Drop from Clean Data:")
            print(f"  Main Model: {main_drop:.4f} ({main_drop/clean_results['main']['accuracy']*100:.1f}%)")
            
            if ood_results['healer'] is not None:
                healer_drop = clean_results['main']['accuracy'] - ood_results['healer']['accuracy']
                print(f"  Healer Model: {healer_drop:.4f} ({healer_drop/clean_results['main']['accuracy']*100:.1f}%)")
                
            if ood_results['ttt'] is not None and clean_results['ttt'] is not None:
                ttt_drop = clean_results['ttt']['accuracy'] - ood_results['ttt']['accuracy']
                print(f"  TTT Model: {ttt_drop:.4f} ({ttt_drop/clean_results['ttt']['accuracy']*100:.1f}%)")
        
        # Print per-transformation accuracies
        if severity > 0.0:
            print("\nPer-Transformation Accuracy:")
            for t_type in ['gaussian_noise', 'rotation', 'h_flip', 'affine']:
                print(f"  {t_type.upper()}:")
                print(f"    Main: {ood_results['main']['per_transform_acc'][t_type]:.4f}")
                if ood_results['healer'] is not None:
                    print(f"    Healer: {ood_results['healer']['per_transform_acc'][t_type]:.4f}")
                if ood_results['ttt'] is not None:
                    print(f"    TTT: {ood_results['ttt']['per_transform_acc'][t_type]:.4f}")
    
    # Log comprehensive results to wandb
    log_results_to_wandb(all_results)
    
    return all_results


def log_results_to_wandb(all_results):
    """Log comprehensive results to wandb with detailed tables and charts."""
    # Overall accuracy across severities
    severity_data = []
    transform_data = {t: [] for t in ['gaussian_noise', 'rotation', 'h_flip', 'affine']}
    
    for severity, results in all_results.items():
        # Skip missing or malformed results
        if not isinstance(results, dict) or 'main' not in results:
            continue
            
        # Overall accuracy row
        row = [severity, results['main']['accuracy']]
        
        if severity > 0.0:
            if results['healer'] is not None:
                row.append(results['healer']['accuracy'])
            else:
                row.append(None)
        else:
            row.append(None)  # Healer not applicable to clean data
            
        if results['ttt'] is not None:
            row.append(results['ttt']['accuracy'])
        else:
            row.append(None)
            
        # Add robustness metrics (only for OOD data)
        if severity > 0.0 and 0.0 in all_results:
            clean_acc = all_results[0.0]['main']['accuracy']
            main_drop = clean_acc - results['main']['accuracy'] 
            row.append(main_drop)
            
            if results['healer'] is not None:
                healer_drop = clean_acc - results['healer']['accuracy']
                row.append(healer_drop)
            else:
                row.append(None)
                
            if results['ttt'] is not None and all_results[0.0]['ttt'] is not None:
                ttt_drop = all_results[0.0]['ttt']['accuracy'] - results['ttt']['accuracy']
                row.append(ttt_drop)
            else:
                row.append(None)
        else:
            # For clean data, no drop
            row.extend([0.0, None, 0.0])
            
        severity_data.append(row)
        
        # Per-transformation data
        if severity > 0.0 and 'per_transform_acc' in results['main']:
            for t_type in transform_data.keys():
                if t_type in results['main']['per_transform_acc']:
                    t_row = [
                        severity,
                        t_type,
                        results['main']['per_transform_acc'][t_type]
                    ]
                    
                    if results['healer'] is not None and 'per_transform_acc' in results['healer']:
                        t_row.append(results['healer']['per_transform_acc'][t_type])
                    else:
                        t_row.append(None)
                        
                    if results['ttt'] is not None and 'per_transform_acc' in results['ttt']:
                        t_row.append(results['ttt']['per_transform_acc'][t_type])
                    else:
                        t_row.append(None)
                        
                    transform_data[t_type].append(t_row)
    
    # Log overall accuracy table
    columns = [
        "Severity", "Main Acc", "Healer Acc", "TTT Acc", 
        "Main Drop", "Healer Drop", "TTT Drop"
    ]
    wandb.log({"overall_accuracy": wandb.Table(data=severity_data, columns=columns)})
    
    # Log per-transformation tables
    for t_type, rows in transform_data.items():
        if rows:  # Only log if we have data
            t_columns = ["Severity", "Transform", "Main Acc", "Healer Acc", "TTT Acc"]
            wandb.log({f"transform_{t_type}": wandb.Table(data=rows, columns=t_columns)})
    
    # Create line chart for accuracy vs severity
    accs_by_severity = {
        'severity': [],
        'main': [],
        'healer': [],
        'ttt': []
    }
    
    for severity, results in sorted(all_results.items()):
        accs_by_severity['severity'].append(severity)
        accs_by_severity['main'].append(results['main']['accuracy'])
        
        # In the log_results_to_wandb function, around line 2162, change:
        # In the log_results_to_wandb function, around line 2162
        if results['healer'] is not None and 'accuracy' in results['healer']:
            accs_by_severity['healer'].append(results['healer']['accuracy'])
        elif severity > 0.0:
            accs_by_severity['healer'].append(0)  # Placeholder for OOD data
        else:
            accs_by_severity['healer'].append(None)  # Placeholder for clean data
            
        if results['ttt'] is not None:
            accs_by_severity['ttt'].append(results['ttt']['accuracy'])
        else:
            accs_by_severity['ttt'].append(0)  # Placeholder
    
    # Log chart data
    for i, severity in enumerate(accs_by_severity['severity']):
        wandb.log({
            "chart/severity": severity,
            "chart/main_acc": accs_by_severity['main'][i],
            "chart/healer_acc": accs_by_severity['healer'][i] if i < len(accs_by_severity['healer']) else None,
            "chart/ttt_acc": accs_by_severity['ttt'][i]
        })

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Dataset path
    dataset_path = "tiny-imagenet-200"
    
    # Step 1: Check for existing main classification model or train a new one
    main_model_path = "bestmodel_main/best_model.pt"
    if os.path.exists(main_model_path):
        print(f"Found existing main classification model at {main_model_path}")
        # Initialize the model with the same architecture
        base_model = create_vit_model(
            img_size=64,
            patch_size=8,
            in_chans=3,
            num_classes=200,
            embed_dim=384,
            depth=8,
            head_dim=64,
            mlp_ratio=4.0,
            use_resnet_stem=True
        )
        # Load the saved weights - need to strip the "vit_model." prefix
        checkpoint = torch.load(main_model_path)
        
        # Create a new state dict with the correct keys
        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith("vit_model."):
                new_key = key[len("vit_model."):]
                new_state_dict[new_key] = value
        
        # Load the adjusted state dict
        base_model.load_state_dict(new_state_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = base_model.to(device)
        base_model.eval()
        print(f"Loaded model with validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    else:
        print("=== Training Main Classification Model ===")
        base_model = train_main_model(dataset_path)
    
    # Step 2: Check for existing healer model or train a new one
    healer_model_path = "bestmodel_healer/best_model.pt"
    if os.path.exists(healer_model_path):
        print(f"Found existing healer model at {healer_model_path}")
        # Initialize the model with the same architecture
        healer_model = TransformationHealer(
            img_size=64,
            patch_size=8,
            in_chans=3,
            embed_dim=384,
            depth=6,
            head_dim=64
        )
        # Load the saved weights
        checkpoint = torch.load(healer_model_path)
        healer_model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        healer_model = healer_model.to(device)
        healer_model.eval()
        print(f"Loaded healer model with training loss: {checkpoint.get('train_loss', 'N/A')}")
    else:
        print("\n=== Training Transformation Healer Model ===")
        healer_model = train_healer_model(dataset_path, severity=1.0)
    
    # Step 3: Check for existing TTT model or train a new one
    ttt_model_path = "bestmodel_ttt/best_model.pt"
    if os.path.exists(ttt_model_path):
        print(f"Found existing TTT model at {ttt_model_path}")
        # Initialize the model with the base model
        ttt_model = TTTModel(base_model)
        # Load the saved weights
        checkpoint = torch.load(ttt_model_path)
        ttt_model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ttt_model = ttt_model.to(device)
        ttt_model.eval()
        print(f"Loaded TTT model with validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    else:
        print("\n=== Training Test-Time Training (TTT) Model ===")
        ttt_model = train_ttt_model(base_model, dataset_path)
    
    # Step 4: Comprehensive evaluation with and without OOD transformations
    print("\n=== Comprehensive Evaluation ===")
    severities = [ 0.2]  # Including 0.0 for clean data
    all_results = evaluate_full_pipeline(
        base_model, healer_model, ttt_model, dataset_path, severities
    )
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
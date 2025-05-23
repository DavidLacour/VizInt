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

# Initialize wandb
wandb.init(project="vit-tiny-imagenet-ood", name="transform-healing-gpu-optimized")

# Define GPU-optimized continuous transformations for OOD testing
class ContinuousTransforms:
    def __init__(self, severity=1.0):
        self.severity = severity
        self.transform_types = ['no_transform', 'gaussian_noise', 'rotation', 'affine']
        self.transform_probs = [0.3, 1.0, 1.0, 1.0]  # Adjusted weights
        
    def apply_transforms(self, img, transform_type=None, severity=None, return_params=False):
        """
        Apply a continuous transformation to the image - keeping operations on the same device as input
        
        Args:
            img: The input tensor image [C, H, W]
            transform_type: If None, randomly choose from transform_types
            severity: If None, use self.severity, otherwise use specified severity
            return_params: If True, return the transformation parameters
            
        Returns:
            transformed_img: The transformed image
            transform_params: Dictionary of transformation parameters (if return_params=True)
        """
        # Keep track of the original device
        device = img.device
        
        if severity is None:
            severity = self.severity
            
        if transform_type is None:
            # Choose a transform type based on probabilities
            transform_type = random.choices(
                self.transform_types, 
                weights=self.transform_probs, 
                k=1
            )[0]
        
        # Initialize parameters with default values
        transform_params = {
            'transform_type': transform_type,
            'severity': severity,
            'noise_std': 0.0,
            'rotation_angle': 0.0,
            'translate_x': 0.0,
            'translate_y': 0.0,
            'shear_x': 0.0,
            'shear_y': 0.0
        }
        
        # OPTIMIZED: Apply transforms on the original device when possible
        if transform_type == 'gaussian_noise':
            # Gaussian noise can be applied directly on GPU
            std = severity * MAX_STD_GAUSSIAN_NOISE
            noise = torch.randn_like(img, device=device) * std
            transformed_img = img + noise
            transformed_img = torch.clamp(transformed_img, 0, 1)
            transform_params['noise_std'] = std
            
        elif transform_type == 'rotation':
            # For rotation, we still need PIL, so transfer to CPU temporarily
            max_angle = MAX_ROTATION * severity
            angle = random.uniform(-max_angle, max_angle)
            
            # Note: We must use CPU for PIL operations
            img_cpu = img.cpu()
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            pil_img = to_pil(img_cpu)
            rotated_img = transforms.functional.rotate(pil_img, angle)
            transformed_img = to_tensor(rotated_img).to(device)  # Transfer back to original device
            transform_params['rotation_angle'] = angle
            
        elif transform_type == 'affine':
            # Affine transformation also needs PIL
            max_translate = MAX_TRANSLATION_AFFINE * severity
            max_shear = MAX_SHEAR_ANGLE * severity
            
            translate_x = random.uniform(-max_translate, max_translate)
            translate_y = random.uniform(-max_translate, max_translate)
            shear_x = random.uniform(-max_shear, max_shear)
            shear_y = random.uniform(-max_shear, max_shear)
            
            # Note: We must use CPU for PIL operations
            img_cpu = img.cpu()
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            pil_img = to_pil(img_cpu)
            
            # Get image size for translation calculation
            width, height = pil_img.size
            translate_pixels = (translate_x * width, translate_y * height)
            
            # Apply affine transformation
            affine_img = transforms.functional.affine(
                pil_img, 
                angle=0.0,
                translate=translate_pixels,
                scale=1.0,
                shear=[shear_x, shear_y]
            )
            transformed_img = to_tensor(affine_img).to(device)  # Transfer back to original device
            
            transform_params['translate_x'] = translate_x
            transform_params['translate_y'] = translate_y
            transform_params['shear_x'] = shear_x
            transform_params['shear_y'] = shear_y
        
        else:  # 'no_transform' case
            transformed_img = img.clone()
        
        if return_params:
            return transformed_img, transform_params
        else:
            return transformed_img


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
        self.transform_type_head = nn.Linear(embed_dim, 4)
        
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
                    
                    # Adjust blur size based on noise level and ensure it's an odd integer
                    blur_radius = min(2.0, noise_std * 4.0)
                    # Convert to odd integer (gaussian_blur requires odd kernel size)
                    kernel_size = max(3, int(blur_radius * 2) + 1)  # Ensure odd and at least 3
                    if kernel_size % 2 == 0:
                        kernel_size += 1  # Make sure it's odd
                    
                    denoised_img = transforms.functional.gaussian_blur(pil_img, kernel_size)
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
        
        # Predict transform-specific severities - these already have shape [B, 1] from the linear layer
        severity_noise = torch.sigmoid(self.severity_noise_head(x))
        severity_rotation = torch.sigmoid(self.severity_rotation_head(x))
        severity_affine = torch.sigmoid(self.severity_affine_head(x))
        
        # Predict various parameters - these already have shape [B, 1] from the linear layer
        rotation_angle = torch.tanh(self.rotation_head(x)) * 180.0
        noise_std = torch.sigmoid(self.noise_head(x)) * 0.5
        
        # Affine transformation parameters - fix the shape issue by using proper slicing
        affine_params = self.affine_head(x)  # [B, 4]
        translate_x = torch.tanh(affine_params[:, 0:1]) * 0.1  # Use [:, 0:1] instead of [:, 0] to keep dimensions
        translate_y = torch.tanh(affine_params[:, 1:2]) * 0.1  # Use [:, 1:2] instead of [:, 1]
        shear_x = torch.tanh(affine_params[:, 2:3]) * 15.0     # Use [:, 2:3] instead of [:, 2]
        shear_y = torch.tanh(affine_params[:, 3:4]) * 15.0     # Use [:, 3:4] instead of [:, 3]
        
        return {
            'transform_type_logits': transform_type_logits,
            'severity_noise': severity_noise,
            'severity_rotation': severity_rotation,
            'severity_affine': severity_affine,
            'rotation_angle': rotation_angle,
            'noise_std': noise_std,
            'translate_x': translate_x,
            'translate_y': translate_y,
            'shear_x': shear_x,
            'shear_y': shear_y
        }

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
        
        # Initialize parameter losses with zero tensors
        severity_noise_loss = torch.tensor(0.0, device=predictions['severity_noise'].device)
        severity_rotation_loss = torch.tensor(0.0, device=predictions['severity_rotation'].device)
        severity_affine_loss = torch.tensor(0.0, device=predictions['severity_affine'].device)
        rotation_loss = torch.tensor(0.0, device=predictions['rotation_angle'].device)
        noise_loss = torch.tensor(0.0, device=predictions['noise_std'].device)
        affine_loss = torch.tensor(0.0, device=predictions['translate_x'].device)
        
        # For each transform type, calculate parameter loss only for samples of that type
        # Gaussian noise (index 1)
        noise_mask = (transform_types == 1)
        if noise_mask.sum() > 0:
            severity_noise_loss = self.mse_loss(
                predictions['severity_noise'][noise_mask], 
                targets['severity'][noise_mask]
            )
            noise_loss = self.mse_loss(
                predictions['noise_std'][noise_mask], 
                targets['noise_std'][noise_mask]
            )
            
        # Rotation (index 2)
        rot_mask = (transform_types == 2)
        if rot_mask.sum() > 0:
            severity_rotation_loss = self.mse_loss(
                predictions['severity_rotation'][rot_mask], 
                targets['severity'][rot_mask]
            )
            rotation_loss = self.mse_loss(
                predictions['rotation_angle'][rot_mask], 
                targets['rotation_angle'][rot_mask]
            )
            
        # Affine (index 3) - multiple regression parameters
        affine_mask = (transform_types == 3)
        if affine_mask.sum() > 0:
            severity_affine_loss = self.mse_loss(
                predictions['severity_affine'][affine_mask], 
                targets['severity'][affine_mask]
            )
            
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
        
        # Combine all severity losses
        severity_loss = (severity_noise_loss + severity_rotation_loss + severity_affine_loss) / 3.0
        
        # Combine losses with weights
        total_loss = (
            transform_type_loss + 
            0.5 * severity_loss + 
            0.3 * (rotation_loss + noise_loss) +
            0.2 * affine_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'transform_type': transform_type_loss.item(),
            'severity_noise': severity_noise_loss.item(),
            'severity_rotation': severity_rotation_loss.item(),
            'severity_affine': severity_affine_loss.item(),
            'rotation': rotation_loss.item(),
            'noise': noise_loss.item(),
            'affine': affine_loss.item()
        }
        
        return total_loss, loss_dict


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
    
    # For safety, return a slightly smaller batch size to allow for memory fluctuations
    # Less conservative reduction factor since we improved GPU usage
    safe_batch_size = max(1, int(working_batch_size * 0.97))
    print(f"Using safe batch size: {safe_batch_size}")
    
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
    batch_size =  find_optimal_batch_size(model, img_size=64, starting_batch_size=128, device=device)
    
    # Define image transformations
    transform_train = transforms.Compose([
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
    
    # OPTIMIZED: Use more workers for data loading on GPU systems
    num_workers = 8 if torch.cuda.is_available() else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    # Training parameters
    num_epochs = 50
    learning_rate = 1e-4
    warmup_steps = 1000
    patience = 5
    
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
    
    # Create a validation set (20% of training data)
    dataset_size = len(train_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"Training set size for healer: {train_size}")
    print(f"Validation set size for healer: {val_size}")
    
    # Simplified collate function
    def collate_fn(batch):
        orig_imgs, trans_imgs, labels, params = zip(*batch)
        
        orig_tensor = torch.stack(orig_imgs)
        trans_tensor = torch.stack(trans_imgs)
        labels_tensor = torch.tensor(labels)
        
        # Keep params as a list of dictionaries
        return orig_tensor, trans_tensor, labels_tensor, params
    
    # Determine batch size and data loading parameters
    batch_size = 50 #find_optimal_batch_size(healer_model, img_size=64, starting_batch_size=128, device=device)
    
    # OPTIMIZED: Use more workers for GPU systems, but pin_memory=False during transformation phase
    # to avoid extra memory usage in CPU-GPU transfer
    num_workers = 6 if torch.cuda.is_available() else 2
    pin_memory = False  # Set to False to avoid extra memory usage when transferring between CPU and GPU
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    # Training parameters
    num_epochs = 15
    learning_rate = 5e-5
    warmup_steps = 500
    patience = 3
    
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
        "model": "transformation_healer_gpu",
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "ood_severity": severity,
        "early_stopping_patience": patience
    }, allow_val_change=True)
    
    # Training loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_epoch = 0
    
    # Helper function to extract parameters
    def extract_transform_params(transform_params):
        """
        Simplified parameter extraction
        """
        transform_type_mapping = {
            'no_transform': 0,
            'no_transfrom': 0,  # Handle typo in original code
            'gaussian_noise': 1,
            'rotation': 2,
            'affine': 3
        }
        
        # Initialize lists
        transform_types = []
        severity_values = []
        noise_std_values = []
        rotation_angle_values = []
        translate_x_values = []
        translate_y_values = []
        shear_x_values = []
        shear_y_values = []
        
        for params in transform_params:
            # Extract transform type - handle different possible formats
            if isinstance(params, dict):
                transform_type = params.get('transform_type', 'no_transform')
                transform_types.append(transform_type_mapping.get(transform_type, 0))
                
                severity_values.append(float(params.get('severity', 1.0)))
                noise_std_values.append(float(params.get('noise_std', 0.0)))
                rotation_angle_values.append(float(params.get('rotation_angle', 0.0)))
                translate_x_values.append(float(params.get('translate_x', 0.0)))
                translate_y_values.append(float(params.get('translate_y', 0.0)))
                shear_x_values.append(float(params.get('shear_x', 0.0)))
                shear_y_values.append(float(params.get('shear_y', 0.0)))
            else:
                # Handle string or other unexpected formats
                transform_type = 'no_transform'
                if isinstance(params, str):
                    transform_type = params
                elif isinstance(params, tuple) and len(params) > 0:
                    transform_type = params[0] if isinstance(params[0], str) else 'no_transform'
                    
                transform_types.append(transform_type_mapping.get(transform_type, 0))
                severity_values.append(1.0)
                noise_std_values.append(0.0)
                rotation_angle_values.append(0.0)
                translate_x_values.append(0.0)
                translate_y_values.append(0.0)
                shear_x_values.append(0.0)
                shear_y_values.append(0.0)
        
        # Create target dictionary
        targets = {
            'transform_type_idx': torch.tensor(transform_types),
            'severity': torch.tensor(severity_values).unsqueeze(1),
            'noise_std': torch.tensor(noise_std_values).unsqueeze(1),
            'rotation_angle': torch.tensor(rotation_angle_values).unsqueeze(1),
            'translate_x': torch.tensor(translate_x_values).unsqueeze(1),
            'translate_y': torch.tensor(translate_y_values).unsqueeze(1),
            'shear_x': torch.tensor(shear_x_values).unsqueeze(1),
            'shear_y': torch.tensor(shear_y_values).unsqueeze(1)
        }
        
        return targets
    
    for epoch in range(num_epochs):
        # Training phase
        healer_model.train()
        train_loss = 0
        transform_type_acc = 0
        total_samples = 0
        
        # Training step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (orig_images, transformed_images, labels, transform_params) in enumerate(progress_bar):
            # Move to device
            orig_images = orig_images.to(device)
            transformed_images = transformed_images.to(device)
            
            # Extract and prepare target tensors
            targets = extract_transform_params(transform_params)
            targets = {k: v.to(device) for k, v in targets.items()}
            
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
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        transform_type_acc /= total_samples
        
        # Validation phase
        healer_model.eval()
        val_loss = 0
        val_transform_type_acc = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (orig_images, transformed_images, labels, transform_params) in enumerate(val_loader):
                # Move to device
                orig_images = orig_images.to(device)
                transformed_images = transformed_images.to(device)
                
                # Extract and prepare target tensors
                targets = extract_transform_params(transform_params)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                # Forward pass
                predictions = healer_model(transformed_images)
                
                # Calculate loss
                loss, loss_dict = healer_loss(predictions, targets)
                
                # Update metrics
                val_loss += loss.item()
                
                # Calculate transform type accuracy
                pred_types = torch.argmax(predictions['transform_type_logits'], dim=1)
                val_transform_type_acc += (pred_types == targets['transform_type_idx']).sum().item()
                val_total_samples += len(pred_types)
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_transform_type_acc /= val_total_samples
        
        # Log metrics
        wandb.log({
            "healer/epoch": epoch + 1,
            "healer/train_loss": train_loss,
            "healer/val_loss": val_loss,
            "healer/train_transform_type_accuracy": transform_type_acc,
            "healer/val_transform_type_accuracy": val_transform_type_acc,
            **{f"healer/loss_{k}": v for k, v in loss_dict.items()},
            "healer/learning_rate": scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Transform Type Acc - Train: {transform_type_acc:.4f}, Val: {val_transform_type_acc:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # Save checkpoint
            checkpoint_path = checkpoints_dir / f"model_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': healer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            
            # Save best model
            best_model_path = best_model_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': healer_model.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
            
            # Log to wandb
            wandb.save(str(checkpoint_path))
            wandb.save(str(best_model_path))
            
            print(f"Saved best model with validation loss: {val_loss:.4f}")
            
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
    
    print(f"Healer model training completed. Best model saved at: {best_model_dir / 'best_model.pt'}")
    
    # Load and return the best model
    checkpoint = torch.load(best_model_dir / "best_model.pt")
    healer_model.load_state_dict(checkpoint['model_state_dict'])
    
    return healer_model


def evaluate_models(main_model, healer_model, dataset_path="tiny-imagenet-200", severity=0.0):
    """
    Streamlined evaluation of models on both clean and transformed data.
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
    
    # Increase batch size for evaluation
    batch_size = 128 if torch.cuda.is_available() else 64
    
    # Simplified collate function for OOD dataset
    def collate_fn(batch):
        if severity > 0:
            orig_imgs, trans_imgs, labels, params = zip(*batch)
            return torch.stack(orig_imgs), torch.stack(trans_imgs), torch.tensor(labels), params
        else:
            images, labels = zip(*batch)
            return torch.stack(images), torch.tensor(labels)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_fn if severity > 0 else None
    )
    
    # Ensure models are in eval mode
    main_model.eval()
    if healer_model is not None:
        healer_model.eval()
    
    # Metrics
    results = {
        'main': {'correct': 0, 'total': 0},
        'healer': {'correct': 0, 'total': 0} if healer_model is not None else None,
    }
    
    # For transformed data, track per-transformation metrics
    transform_types = ['no_transform', 'gaussian_noise', 'rotation', 'affine']
    if severity > 0:
        for model_name in ['main', 'healer']:
            if results[model_name] is not None:
                results[model_name]['per_transform'] = {t: {'correct': 0, 'total': 0} for t in transform_types}
    
    # Helper function to determine transform type consistently
    def get_transform_type(params):
        if isinstance(params, dict):
            t_type = params.get('transform_type', '').lower()
            if 'no_' in t_type or t_type == '':
                return 'no_transform'
            elif 'noise' in t_type or 'gaussian' in t_type:
                return 'gaussian_noise'
            elif 'rot' in t_type:
                return 'rotation'
            elif 'affine' in t_type or 'trans' in t_type:
                return 'affine'
            
            # Also check parameters
            if params.get('noise_std', 0.0) > 0.01:
                return 'gaussian_noise'
            elif abs(params.get('rotation_angle', 0.0)) > 0.1:
                return 'rotation'
            elif (abs(params.get('translate_x', 0.0)) > 0.001 or 
                  abs(params.get('translate_y', 0.0)) > 0.001 or
                  abs(params.get('shear_x', 0.0)) > 0.001 or
                  abs(params.get('shear_y', 0.0)) > 0.001):
                return 'affine'
            
            return 'no_transform'
        elif isinstance(params, str):
            t_type = params.lower()
            if 'no_' in t_type:
                return 'no_transform'
            elif 'noise' in t_type or 'gaussian' in t_type:
                return 'gaussian_noise'
            elif 'rot' in t_type:
                return 'rotation'
            elif 'affine' in t_type or 'trans' in t_type:
                return 'affine'
            return 'no_transform'
        else:
            return 'no_transform'
    
    with torch.no_grad():
        if severity > 0:
            # Evaluation with transformations
            for batch in tqdm(val_loader, desc=f"Evaluating models (severity {severity})"):
                orig_images, transformed_images, labels, transform_params = batch
                orig_images = orig_images.to(device)
                transformed_images = transformed_images.to(device)
                labels = labels.to(device)
                
                # 1. Evaluate main model on transformed images
                main_outputs = main_model(transformed_images)
                main_preds = torch.argmax(main_outputs, dim=1)
                results['main']['correct'] += (main_preds == labels).sum().item()
                results['main']['total'] += labels.size(0)
                
                # Track per-transformation accuracy
                for i, params in enumerate(transform_params):
                    t_type = get_transform_type(params)
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
                        t_type = get_transform_type(params)
                        results['healer']['per_transform'][t_type]['total'] += 1
                        if healer_preds[i] == labels[i]:
                            results['healer']['per_transform'][t_type]['correct'] += 1
        else:
            # Clean data evaluation (no transformations)
            for images, labels in tqdm(val_loader, desc="Evaluating models (clean data)"):
                images = images.to(device)
                labels = labels.to(device)
                
                # Main model on clean images
                main_outputs = main_model(images)
                main_preds = torch.argmax(main_outputs, dim=1)
                results['main']['correct'] += (main_preds == labels).sum().item()
                results['main']['total'] += labels.size(0)
    
    # Calculate overall accuracies
    for model_name in ['main', 'healer']:
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

'''
def evaluate_full_pipeline(main_model, healer_model, dataset_path="tiny-imagenet-200", severities=[0.3]):
    """
    Comprehensive evaluation across multiple severities including clean data.
    
    Args:
        main_model: The base classification model
        healer_model: The transformation healing model
        dataset_path: Path to the dataset
        severities: List of severity levels to evaluate
        
    Returns:
        all_results: Dictionary containing all evaluation results
    """
    all_results = {}
    
    # First evaluate on clean data
    print("\nEvaluating on clean data (no transformations)...")
    clean_results = evaluate_models(main_model, healer_model, dataset_path, severity=0.0)
    all_results[0.0] = clean_results
    
    # Print clean data results
    print(f"Clean Data Accuracy:")
    print(f"  Main Model: {clean_results['main']['accuracy']:.4f}")
    if clean_results['healer'] is not None:
        print(f"  Healer Model: {clean_results['healer']['accuracy']:.4f}")
    
    # Then evaluate on transformed data at different severities
    for severity in severities:
        if severity == 0.0:
            continue  # Skip, already evaluated
            
        print(f"\nEvaluating with severity {severity}...")
        ood_results = evaluate_models(main_model, healer_model, dataset_path, severity=severity)
        all_results[severity] = ood_results
        
        # Print OOD results
        print(f"OOD Accuracy (Severity {severity}):")
        print(f"  Main Model: {ood_results['main']['accuracy']:.4f}")
        if ood_results['healer'] is not None:
            print(f"  Healer Model: {ood_results['healer']['accuracy']:.4f}")
            
        # Calculate and print robustness metrics (drop compared to clean data)
        if severity > 0.0:
            main_drop = clean_results['main']['accuracy'] - ood_results['main']['accuracy']
            print(f"\nAccuracy Drop from Clean Data:")
            print(f"  Main Model: {main_drop:.4f} ({main_drop/clean_results['main']['accuracy']*100:.1f}%)")
            
            if ood_results['healer'] is not None:
                healer_drop = clean_results['main']['accuracy'] - ood_results['healer']['accuracy']
                print(f"  Healer Model: {healer_drop:.4f} ({healer_drop/clean_results['main']['accuracy']*100:.1f}%)")
        
        # Print per-transformation accuracies
        if severity > 0.0:
            print("\nPer-Transformation Accuracy:")
            for t_type in ['no_transform', 'gaussian_noise', 'rotation', 'affine']:
                print(f"  {t_type.upper()}:")
                print(f"    Main: {ood_results['main']['per_transform_acc'][t_type]:.4f}")
                if ood_results['healer'] is not None:
                    print(f"    Healer: {ood_results['healer']['per_transform_acc'][t_type]:.4f}")
    
    # Log comprehensive results to wandb
    log_wandb_results(all_results)
    
    return all_results
'''

def log_wandb_results(all_results):
    """
    Log evaluation results to Weights & Biases
    """
    # Log clean data results
    if 0.0 in all_results:
        clean_results = all_results[0.0]
        clean_acc = {
            "eval/clean_accuracy": clean_results['main']['accuracy']
        }
        if clean_results['healer'] is not None:
            clean_acc["eval/clean_healer_accuracy"] = clean_results['healer']['accuracy']
        wandb.log(clean_acc)
    
    # Log OOD results
    for severity, results in all_results.items():
        if severity == 0.0:
            continue  # Skip clean results, already logged
        
        # Main metrics
        ood_metrics = {
            f"eval/ood_s{severity}_accuracy": results['main']['accuracy'],
        }
        
        if results['healer'] is not None:
            ood_metrics[f"eval/ood_s{severity}_healer_accuracy"] = results['healer']['accuracy']
        
        # Per-transformation metrics
        if 'per_transform_acc' in results['main']:
            for t_type, acc in results['main']['per_transform_acc'].items():
                ood_metrics[f"eval/ood_s{severity}_{t_type}_accuracy"] = acc
                
                if results['healer'] is not None and 'per_transform_acc' in results['healer']:
                    ood_metrics[f"eval/ood_s{severity}_{t_type}_healer_accuracy"] = (
                        results['healer']['per_transform_acc'][t_type]
                    )
        
        # Log all metrics together
        wandb.log(ood_metrics)

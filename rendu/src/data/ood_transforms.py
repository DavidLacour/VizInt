"""
Out-of-Distribution (OOD) transforms for robust evaluation
Contains funky and extreme transformations that go beyond typical augmentation
"""
import torch
import random
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


class OODTransforms:
    """
    Extreme and funky transformations for OOD evaluation
    These are designed to be more challenging than typical augmentations
    """
    
    def __init__(self, severity=1.0):
        self.severity = severity
        self.transform_types = [
            'extreme_noise',
            'color_inversion', 
            'pixelation',
            'extreme_rotation',
            'channel_shuffle',
            'extreme_blur',
            'posterization',
            'solarization',
            'jpeg_compression',
            'random_masking'
        ]
    
    def apply_ood_transform(self, img, transform_type=None, severity=None, return_params=False):
        """
        Apply an extreme OOD transformation to the image
        
        Args:
            img: Input tensor image [C, H, W] in range [0, 1]
            transform_type: If None, randomly choose from transform_types
            severity: If None, use self.severity, otherwise use specified severity
            return_params: If True, return the transformation parameters
            
        Returns:
            transformed_img: The transformed image
            transform_params: Dictionary of transformation parameters (if return_params=True)
        """
        device = img.device
        
        if severity is None:
            severity = self.severity
            
        if transform_type is None:
            transform_type = random.choice(self.transform_types)
        
        transform_params = {
            'transform_type': transform_type,
            'severity': severity
        }
        
        # Apply the specific transformation
        if transform_type == 'extreme_noise':
            transformed_img = self._extreme_noise(img, severity)
            
        elif transform_type == 'color_inversion':
            transformed_img = self._color_inversion(img, severity)
            
        elif transform_type == 'pixelation':
            transformed_img = self._pixelation(img, severity)
            
        elif transform_type == 'extreme_rotation':
            transformed_img = self._extreme_rotation(img, severity)
            
        elif transform_type == 'channel_shuffle':
            transformed_img = self._channel_shuffle(img, severity)
            
        elif transform_type == 'extreme_blur':
            transformed_img = self._extreme_blur(img, severity)
            
        elif transform_type == 'posterization':
            transformed_img = self._posterization(img, severity)
            
        elif transform_type == 'solarization':
            transformed_img = self._solarization(img, severity)
            
        elif transform_type == 'jpeg_compression':
            transformed_img = self._jpeg_compression(img, severity)
            
        elif transform_type == 'random_masking':
            transformed_img = self._random_masking(img, severity)
            
        else:
            transformed_img = img.clone()
        
        if return_params:
            return transformed_img, transform_params
        else:
            return transformed_img
    
    def _extreme_noise(self, img, severity):
        """Add extreme Gaussian noise"""
        max_std = 1.0  # Very high noise
        std = severity * max_std
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, 0, 1)
    
    def _color_inversion(self, img, severity):
        """Invert colors with varying intensity"""
        inversion_factor = severity
        return torch.clamp(img * (1 - inversion_factor) + (1 - img) * inversion_factor, 0, 1)
    
    def _pixelation(self, img, severity):
        """Apply pixelation effect"""
        # Reduce resolution and upscale back
        _, h, w = img.shape
        min_size = max(4, int(h * (1 - severity) + 4 * severity))  # Down to 4x4 at max severity
        
        # Downsample
        img_small = F.interpolate(img.unsqueeze(0), size=(min_size, min_size), mode='nearest')
        # Upsample back
        return F.interpolate(img_small, size=(h, w), mode='nearest').squeeze(0)
    
    def _extreme_rotation(self, img, severity):
        """Apply extreme rotation (up to 180 degrees)"""
        max_angle = 180.0 * severity
        angle = random.uniform(-max_angle, max_angle)
        
        # Convert to PIL for rotation
        img_cpu = img.cpu()
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        pil_img = to_pil(img_cpu)
        rotated_img = transforms.functional.rotate(pil_img, angle)
        return to_tensor(rotated_img).to(img.device)
    
    def _channel_shuffle(self, img, severity):
        """Shuffle color channels"""
        if img.shape[0] == 3 and random.random() < severity:
            # Shuffle RGB channels
            perm = torch.randperm(3)
            return img[perm]
        return img
    
    def _extreme_blur(self, img, severity):
        """Apply extreme Gaussian blur"""
        max_sigma = 5.0 * severity  # Very strong blur
        sigma = random.uniform(0.1, max_sigma)
        
        # Create Gaussian kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply blur using conv2d
        img_expanded = img.unsqueeze(0)  # Add batch dimension
        
        # Create 2D Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # Create 2D kernel
        kernel_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        kernel_2d = kernel_2d.expand(img.shape[0], 1, kernel_size, kernel_size).to(img.device)
        
        # Apply convolution with padding
        padding = kernel_size // 2
        blurred = F.conv2d(img_expanded, kernel_2d, padding=padding, groups=img.shape[0])
        
        return blurred.squeeze(0)
    
    def _posterization(self, img, severity):
        """Reduce color depth (posterization)"""
        # Reduce bits per channel
        max_reduction = 6  # Reduce from 8 bits to 2 bits at max severity
        bits_to_remove = int(severity * max_reduction)
        bits_to_keep = 8 - bits_to_remove
        
        # Convert to int, apply bit mask, convert back
        img_int = (img * 255).long()
        mask = (0xFF << bits_to_remove) & 0xFF
        img_posterized = (img_int & mask).float() / 255.0
        
        return torch.clamp(img_posterized, 0, 1)
    
    def _solarization(self, img, severity):
        """Apply solarization effect"""
        threshold = 1.0 - severity  # Lower threshold = more solarization
        mask = img > threshold
        solarized = img.clone()
        solarized[mask] = 1.0 - solarized[mask]
        return solarized
    
    def _jpeg_compression(self, img, severity):
        """Simulate JPEG compression artifacts"""
        # This is a simplified version - in practice you'd use actual JPEG encoding/decoding
        # We'll simulate by adding quantization noise and blocking artifacts
        
        # Block size for DCT-like effect
        block_size = 8
        
        # Add quantization noise
        quantization_factor = severity * 0.1
        noise = (torch.rand_like(img) - 0.5) * quantization_factor
        
        # Create blocking artifacts by averaging within blocks
        _, h, w = img.shape
        compressed = img.clone()
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = compressed[:, i:i+block_size, j:j+block_size]
                if block.numel() > 0:
                    # Average within block (simplified compression)
                    block_mean = block.mean(dim=(1, 2), keepdim=True)
                    # Blend with original based on severity
                    compressed[:, i:i+block_size, j:j+block_size] = (
                        block * (1 - severity * 0.5) + 
                        block_mean * (severity * 0.5)
                    )
        
        return torch.clamp(compressed + noise, 0, 1)
    
    def _random_masking(self, img, severity):
        """Apply random rectangular masks"""
        masked_img = img.clone()
        
        # Number of masks based on severity
        num_masks = int(severity * 10)  # Up to 10 masks
        
        _, h, w = img.shape
        
        for _ in range(num_masks):
            # Random mask size (up to 1/4 of image)
            mask_h = random.randint(1, h // 4)
            mask_w = random.randint(1, w // 4)
            
            # Random position
            start_h = random.randint(0, h - mask_h)
            start_w = random.randint(0, w - mask_w)
            
            # Random mask value (black, white, or random color)
            mask_type = random.choice(['black', 'white', 'random'])
            if mask_type == 'black':
                mask_value = 0.0
            elif mask_type == 'white':
                mask_value = 1.0
            else:
                mask_value = torch.rand(3, 1, 1).to(img.device)
            
            masked_img[:, start_h:start_h+mask_h, start_w:start_w+mask_w] = mask_value
        
        return masked_img
    
    def apply_random_ood_transform(self, img, severity=None):
        """Apply a random OOD transformation"""
        return self.apply_ood_transform(img, transform_type=None, severity=severity)
    
    def get_transform_types(self):
        """Get list of available transform types"""
        return self.transform_types.copy()
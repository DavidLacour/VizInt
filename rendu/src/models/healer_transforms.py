"""
Static healer transformation functions that can be called independently
"""
import torch
from torchvision import transforms
from typing import Dict, Optional, Tuple
import numpy as np
import torch.nn.functional as F
from scipy import ndimage
import cv2


class HealerTransforms:
    """Static methods for applying healer corrections to transformed images"""
    
    @staticmethod
    def apply_wiener_denoising(image: torch.Tensor,
                             noise_std: float,
                             method: str = 'wiener',
                             device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Apply advanced denoising using Wiener filter or other methods
        
        Args:
            image: Input image tensor [C, H, W] or [B, C, H, W]
            noise_std: Estimated noise standard deviation
            method: Denoising method ('wiener', 'bilateral', 'nlm', 'gaussian')
            device: Device to return tensor on (defaults to input device)
            
        Returns:
            Denoised image tensor
        """
        if device is None:
            device = image.device
            
        # Handle batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = image.shape[0]
        denoised_images = []
        
        for i in range(batch_size):
            img = image[i]
            
            if noise_std > 0.01:
                # Convert to numpy for processing
                img_np = img.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
                img_np = np.clip(img_np, 0, 1)
                
                if method == 'wiener':
                    denoised_np = HealerTransforms._wiener_filter(img_np, noise_std)
                elif method == 'bilateral':
                    denoised_np = HealerTransforms._bilateral_filter(img_np, noise_std)
                elif method == 'nlm':
                    denoised_np = HealerTransforms._nlm_filter(img_np, noise_std)
                elif method == 'bm3d':
                    denoised_np = HealerTransforms._bm3d_inspired_filter(img_np, noise_std)
                else:  # fallback to Gaussian blur
                    denoised_np = HealerTransforms._gaussian_filter(img_np, noise_std)
                
                denoised_tensor = torch.from_numpy(
                    denoised_np.transpose(2, 0, 1)
                ).float().to(device)
                denoised_images.append(denoised_tensor)
            else:
                # No significant noise, keep original
                denoised_images.append(img)
        
        result = torch.stack(denoised_images)
        return result.squeeze(0) if squeeze_output else result
    
    @staticmethod
    def _wiener_filter(image_np: np.ndarray, noise_std: float) -> np.ndarray:
        """
        Apply improved Wiener filter for noise reduction
        
        Args:
            image_np: Image as numpy array [H, W, C]
            noise_std: Noise standard deviation
            
        Returns:
            Filtered image
        """
        output = np.zeros_like(image_np)
        
        for c in range(image_np.shape[2]):
            channel = image_np[:, :, c]
            
            img_fft = np.fft.fft2(channel)
            power_spectrum = np.abs(img_fft) ** 2
            
            # Improved noise power estimation
            # Account for the fact that noise power is distributed across all frequencies
            h, w = channel.shape
            noise_variance = noise_std ** 2
            noise_power_spectrum = noise_variance * h * w 
            
            freq_y = np.fft.fftfreq(h).reshape(-1, 1)
            freq_x = np.fft.fftfreq(w).reshape(1, -1)
            freq_radius = np.sqrt(freq_y**2 + freq_x**2)
            
            from scipy.ndimage import median_filter
            
            # Use median filter to be robust to outliers
            local_power = median_filter(power_spectrum, size=5)
            signal_power = np.maximum(local_power - noise_power_spectrum, 
                                    power_spectrum * 0.01)  # Keep at least 1% to avoid zeros
            
            regularization = 1.0 + 10.0 * freq_radius
            
            wiener_gain = signal_power / (signal_power + noise_power_spectrum * regularization + 1e-10)
            
            # Use sigmoid-like function for smooth cutoff
            transition_freq = 0.3
            smooth_factor = 1.0 / (1.0 + np.exp(20 * (freq_radius - transition_freq)))
            wiener_gain = wiener_gain * smooth_factor + (1 - smooth_factor) * wiener_gain * 0.1
            
            filtered_fft = img_fft * wiener_gain
            filtered = np.real(np.fft.ifft2(filtered_fft))
            
            # Post-processing: slight sharpening to compensate for smoothing
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(filtered, sigma=0.5)
            sharpened = filtered + 0.1 * (filtered - blurred)

            output[:, :, c] = np.clip(sharpened, 0, 1)
        
        return output
    
    @staticmethod
    def _bilateral_filter(image_np: np.ndarray, noise_std: float) -> np.ndarray:
        """Apply bilateral filter for edge-preserving denoising"""
        img_uint8 = (image_np * 255).astype(np.uint8)
        
        d = max(5, int(noise_std * 20))  # Diameter
        sigma_color = max(10, noise_std * 200)
        sigma_space = max(10, noise_std * 200)
        
        denoised = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
        
        return denoised.astype(np.float32) / 255.0
    
    @staticmethod
    def _nlm_filter(image_np: np.ndarray, noise_std: float) -> np.ndarray:
        """Apply Non-Local Means denoising"""
        img_uint8 = (image_np * 255).astype(np.uint8)
        
        h = max(3, noise_std * 30)  # Filter strength
        template_window_size = 7
        search_window_size = 21
        
        if len(img_uint8.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                img_uint8, None, h, h, template_window_size, search_window_size
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                img_uint8, None, h, template_window_size, search_window_size
            )
        
        return denoised.astype(np.float32) / 255.0
    
    @staticmethod
    def _bm3d_inspired_filter(image_np: np.ndarray, noise_std: float) -> np.ndarray:
        """
        Simplified BM3D-inspired denoising using patch similarity
        """
        output = np.zeros_like(image_np)
        
        patch_size = 8
        search_window = 21
        h_filtering = noise_std * 10 
        
        if image_np.shape[2] == 3:
            gray = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
        else:
            gray = image_np[:, :, 0]
        
        h, w = gray.shape
        half_patch = patch_size // 2
        half_search = search_window // 2
        
        for c in range(image_np.shape[2]):
            channel = image_np[:, :, c]
            filtered_channel = np.zeros_like(channel)
            weight_sum = np.zeros_like(channel)
            
            padded = np.pad(channel, 
                          ((half_search + half_patch, half_search + half_patch),
                           (half_search + half_patch, half_search + half_patch)), 
                          mode='reflect')
            gray_padded = np.pad(gray,
                               ((half_search + half_patch, half_search + half_patch),
                                (half_search + half_patch, half_search + half_patch)),
                               mode='reflect')
            
            for i in range(half_search + half_patch, h + half_search + half_patch):
                for j in range(half_search + half_patch, w + half_search + half_patch):
                    ref_patch = gray_padded[i-half_patch:i+half_patch+1, 
                                          j-half_patch:j+half_patch+1]
                    

                    for di in range(-half_search, half_search + 1):
                        for dj in range(-half_search, half_search + 1):
                            comp_patch = gray_padded[i+di-half_patch:i+di+half_patch+1,
                                                   j+dj-half_patch:j+dj+half_patch+1]
                            
                            dist = np.sum((ref_patch - comp_patch) ** 2) / (patch_size ** 2)
                            weight = np.exp(-dist / (h_filtering ** 2))
                            
                            filtered_channel[i-half_search-half_patch, j-half_search-half_patch] += \
                                weight * padded[i+di, j+dj]
                            weight_sum[i-half_search-half_patch, j-half_search-half_patch] += weight
            
            output[:, :, c] = filtered_channel / (weight_sum + 1e-10)
        
        return np.clip(output, 0, 1)
    
    @staticmethod
    def _gaussian_filter(image_np: np.ndarray, noise_std: float) -> np.ndarray:
        """Apply Gaussian filter (original method)"""
        sigma = max(0.5, min(2.0, noise_std * 4.0))
        
        output = np.zeros_like(image_np)
        for c in range(image_np.shape[2]):
            output[:, :, c] = ndimage.gaussian_filter(image_np[:, :, c], sigma=sigma)
        
        return np.clip(output, 0, 1)
    
    @staticmethod
    def apply_gaussian_denoising(image: torch.Tensor, 
                               noise_std: float,
                               device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Apply denoising to remove Gaussian noise from an image
        (Kept for backward compatibility - redirects to wiener method)
        
        Args:
            image: Input image tensor [C, H, W] or [B, C, H, W]
            noise_std: Estimated noise standard deviation
            device: Device to return tensor on (defaults to input device)
            
        Returns:
            Denoised image tensor
        """
        return HealerTransforms.apply_wiener_denoising(
            image, noise_std, method='wiener', device=device
        )
    
    @staticmethod
    def apply_inverse_rotation(image: torch.Tensor,
                              rotation_angle: float,
                              device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Apply inverse rotation to correct a rotated image
        
        Args:
            image: Input image tensor [C, H, W] or [B, C, H, W]
            rotation_angle: Rotation angle in degrees (will be negated)
            device: Device to return tensor on (defaults to input device)
            
        Returns:
            Corrected image tensor
        """
        if device is None:
            device = image.device
            
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = image.shape[0]
        corrected_images = []
        
        for i in range(batch_size):
            img = image[i]
            
            img_cpu = img.cpu()
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            pil_img = to_pil(img_cpu)
            
            rotated_img = transforms.functional.rotate(pil_img, -rotation_angle)
            corrected_tensor = to_tensor(rotated_img).to(device)
            corrected_images.append(corrected_tensor)
        
        result = torch.stack(corrected_images)
        return result.squeeze(0) if squeeze_output else result
    
    @staticmethod
    def apply_inverse_affine(image: torch.Tensor,
                           translate_x: float,
                           translate_y: float,
                           shear_x: float,
                           shear_y: float,
                           device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Apply inverse affine transformation to correct transformed image
        
        Args:
            image: Input image tensor [C, H, W] or [B, C, H, W]
            translate_x: X translation factor (fraction of width)
            translate_y: Y translation factor (fraction of height)
            shear_x: X shear angle in degrees
            shear_y: Y shear angle in degrees
            device: Device to return tensor on (defaults to input device)
            
        Returns:
            Corrected image tensor
        """
        if device is None:
            device = image.device
            
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = image.shape[0]
        corrected_images = []
        
        for i in range(batch_size):
            img = image[i]
            
            img_cpu = img.cpu()
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            pil_img = to_pil(img_cpu)
            
            width, height = pil_img.size
            translate_pixels = (-translate_x * width, -translate_y * height)
            
            affine_img = transforms.functional.affine(
                pil_img,
                angle=0.0,
                translate=translate_pixels,
                scale=1.0,
                shear=[-shear_x, -shear_y]  # Negative for inverse
            )
            corrected_tensor = to_tensor(affine_img).to(device)
            corrected_images.append(corrected_tensor)
        
        result = torch.stack(corrected_images)
        return result.squeeze(0) if squeeze_output else result
    
    @staticmethod
    def apply_correction_by_type(image: torch.Tensor,
                               transform_type: int,
                               transform_params: Dict[str, float],
                               device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Apply correction based on transformation type
        
        Args:
            image: Input image tensor [C, H, W] or [B, C, H, W]
            transform_type: Type of transformation (0: none, 1: noise, 2: rotation, 3: affine)
            transform_params: Dictionary containing transformation parameters
            device: Device to return tensor on
            
        Returns:
            Corrected image tensor
        """
        if transform_type == 0:
            # No transformation
            return image
        elif transform_type == 1:
            # Gaussian noise
            noise_std = transform_params.get('noise_std', 0.0)
            return HealerTransforms.apply_gaussian_denoising(image, noise_std, device)
        elif transform_type == 2:
            # Rotation
            rotation_angle = transform_params.get('rotation_angle', 0.0)
            return HealerTransforms.apply_inverse_rotation(image, rotation_angle, device)
        elif transform_type == 3:
            # Affine transformation
            translate_x = transform_params.get('translate_x', 0.0)
            translate_y = transform_params.get('translate_y', 0.0)
            shear_x = transform_params.get('shear_x', 0.0)
            shear_y = transform_params.get('shear_y', 0.0)
            return HealerTransforms.apply_inverse_affine(
                image, translate_x, translate_y, shear_x, shear_y, device
            )
        else:
            # Unknown transformation type
            return image
    
    @staticmethod
    def apply_batch_correction(images: torch.Tensor,
                             predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply corrections to a batch of images based on predictions
        
        Args:
            images: Batch of images [B, C, H, W]
            predictions: Dictionary of predictions from healer model containing:
                - transform_type_logits: [B, num_transforms]
                - rotation_angle: [B, 1]
                - noise_std: [B, 1]
                - translate_x: [B, 1]
                - translate_y: [B, 1]
                - shear_x: [B, 1]
                - shear_y: [B, 1]
                
        Returns:
            Corrected batch of images [B, C, H, W]
        """
        device = images.device
        batch_size = images.shape[0]
        
        transform_type_logits = predictions['transform_type_logits']
        transform_types = torch.argmax(transform_type_logits, dim=1)  # [B]
        
        corrected_images = images.clone()
        
        for i in range(batch_size):
            t_type = transform_types[i].item()
            params = {
                'noise_std': predictions['noise_std'][i].item(),
                'rotation_angle': predictions['rotation_angle'][i].item(),
                'translate_x': predictions['translate_x'][i].item(),
                'translate_y': predictions['translate_y'][i].item(),
                'shear_x': predictions['shear_x'][i].item(),
                'shear_y': predictions['shear_y'][i].item()
            }
            
            corrected_images[i] = HealerTransforms.apply_correction_by_type(
                images[i], t_type, params, device
            )
        
        return corrected_images
    
    @staticmethod
    def create_mock_predictions(transform_type: str,
                              **kwargs) -> Dict[str, torch.Tensor]:
        """
        Create mock predictions for testing specific transformations
        
        Args:
            transform_type: Type of transformation ('none', 'gaussian_noise', 'rotation', 'affine')
            **kwargs: Transformation parameters (e.g., noise_std=0.1, rotation_angle=45)
            
        Returns:
            Dictionary of predictions in the expected format
        """
        type_map = {
            'none': 0,
            'no_transform': 0,
            'gaussian_noise': 1,
            'noise': 1,
            'rotation': 2,
            'rotate': 2,
            'affine': 3,
            'affine_transform': 3
        }
        
        t_type = type_map.get(transform_type.lower(), 0)
        
        # Create one-hot transform type logits
        transform_logits = torch.zeros(1, 4)
        transform_logits[0, t_type] = 10.0  # High confidence
        
        predictions = {
            'transform_type_logits': transform_logits,
            'noise_std': torch.tensor([[kwargs.get('noise_std', 0.0)]]),
            'rotation_angle': torch.tensor([[kwargs.get('rotation_angle', 0.0)]]),
            'translate_x': torch.tensor([[kwargs.get('translate_x', 0.0)]]),
            'translate_y': torch.tensor([[kwargs.get('translate_y', 0.0)]]),
            'shear_x': torch.tensor([[kwargs.get('shear_x', 0.0)]]),
            'shear_y': torch.tensor([[kwargs.get('shear_y', 0.0)]])
        }
        
        return predictions


# Example usage functions
def demo_gaussian_denoising(image: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    """Demo: Remove Gaussian noise from image"""
    return HealerTransforms.apply_gaussian_denoising(image, noise_std)


def demo_rotation_correction(image: torch.Tensor, angle: float = 45.0) -> torch.Tensor:
    """Demo: Correct rotation in image"""
    return HealerTransforms.apply_inverse_rotation(image, angle)


def demo_affine_correction(image: torch.Tensor, 
                         tx: float = 0.1, ty: float = 0.1,
                         sx: float = 15.0, sy: float = 15.0) -> torch.Tensor:
    """Demo: Correct affine transformation"""
    return HealerTransforms.apply_inverse_affine(image, tx, ty, sx, sy)


def demo_auto_correction(image: torch.Tensor, transform_type: str, **params) -> torch.Tensor:
    """Demo: Automatically correct based on transformation type"""
    predictions = HealerTransforms.create_mock_predictions(transform_type, **params)
    return HealerTransforms.apply_batch_correction(image.unsqueeze(0), predictions).squeeze(0)
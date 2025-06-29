import torch
import random
from torchvision import transforms


MAX_ROTATION = 360.0 
MAX_STD_GAUSSIAN_NOISE = 0.5
MAX_TRANSLATION_AFFINE = 0.1
MAX_SHEAR_ANGLE = 15.0


class ContinuousTransforms:
    def __init__(self, severity=1.0):
        self.severity = severity
        self.transform_types = ['no_transform', 'gaussian_noise', 'rotation', 'affine']
        self.transform_probs = [1.0, 1.0, 1.0, 1.0]  # TODO config
        
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
        device = img.device
        
        if severity is None:
            severity = self.severity
            
        if transform_type is None:
            transform_type = random.choices(
                self.transform_types, 
                weights=self.transform_probs, 
                k=1
            )[0]
        
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
        
        if transform_type == 'gaussian_noise':
            std = severity * MAX_STD_GAUSSIAN_NOISE
            noise = torch.randn_like(img, device=device) * std
            transformed_img = img + noise
            transformed_img = torch.clamp(transformed_img, 0, 1)
            transform_params['noise_std'] = std
            
        elif transform_type == 'rotation':
            max_angle = MAX_ROTATION * severity
            angle = random.uniform(-max_angle, max_angle)
            
            # Ensure image is in valid range before PIL conversion
            img_clamped = torch.clamp(img, 0, 1)
            img_cpu = img_clamped.cpu()
            
            # Use proper mode for PIL conversion
            to_pil = transforms.ToPILImage(mode='RGB')
            to_tensor = transforms.ToTensor()
            
            pil_img = to_pil(img_cpu)
            rotated_img = transforms.functional.rotate(pil_img, angle)
            transformed_img = to_tensor(rotated_img).to(device)  
            transform_params['rotation_angle'] = angle
            
        elif transform_type == 'affine':
            max_translate = MAX_TRANSLATION_AFFINE * severity
            max_shear = MAX_SHEAR_ANGLE * severity
            
            translate_x = random.uniform(-max_translate, max_translate)
            translate_y = random.uniform(-max_translate, max_translate)
            shear_x = random.uniform(-max_shear, max_shear)
            shear_y = random.uniform(-max_shear, max_shear)
            
            # Ensure image is in valid range before PIL conversion
            img_clamped = torch.clamp(img, 0, 1)
            img_cpu = img_clamped.cpu()
            
            # Use proper mode for PIL conversion
            to_pil = transforms.ToPILImage(mode='RGB')
            to_tensor = transforms.ToTensor()
            
            pil_img = to_pil(img_cpu)
            
            width, height = pil_img.size
            translate_pixels = (translate_x * width, translate_y * height)
            
            affine_img = transforms.functional.affine(
                pil_img, 
                angle=0.0,
                translate=translate_pixels,
                scale=1.0,
                shear=[shear_x, shear_y]
            )
            transformed_img = to_tensor(affine_img).to(device)  
            
            transform_params['translate_x'] = translate_x
            transform_params['translate_y'] = translate_y
            transform_params['shear_x'] = shear_x
            transform_params['shear_y'] = shear_y
        
        else: 
            transformed_img = img.clone()
        
        if return_params:
            return transformed_img, transform_params
        else:
            return transformed_img
    
    def apply_transforms_unnormalized(self, img, transform_type=None, severity=None, return_params=False):
        """
        Apply a continuous transformation to the image without normalization/clamping
        
        Args:
            img: The input tensor image [C, H, W]
            transform_type: If None, randomly choose from transform_types
            severity: If None, use self.severity, otherwise use specified severity
            return_params: If True, return the transformation parameters
            
        Returns:
            transformed_img: The transformed image (without clamping for noise, but clamped for PIL operations)
            transform_params: Dictionary of transformation parameters (if return_params=True)
        """
        device = img.device
        
        if severity is None:
            severity = self.severity
            
        if transform_type is None:
            transform_type = random.choices(
                self.transform_types, 
                weights=self.transform_probs, 
                k=1
            )[0]
        
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
        
        if transform_type == 'gaussian_noise':
            std = severity * MAX_STD_GAUSSIAN_NOISE
            noise = torch.randn_like(img, device=device) * std
            transformed_img = img + noise
            # NO CLAMPING - values can go outside [0,1] range
            transform_params['noise_std'] = std
            
        elif transform_type == 'rotation':
            max_angle = MAX_ROTATION * severity
            angle = random.uniform(-max_angle, max_angle)
            
            # Must clamp for PIL conversion even in unnormalized version
            img_clamped = torch.clamp(img, 0, 1)
            img_cpu = img_clamped.cpu()
            
            to_pil = transforms.ToPILImage(mode='RGB')
            to_tensor = transforms.ToTensor()
            
            pil_img = to_pil(img_cpu)
            rotated_img = transforms.functional.rotate(pil_img, angle)
            transformed_img = to_tensor(rotated_img).to(device)  
            transform_params['rotation_angle'] = angle
            
        elif transform_type == 'affine':
            max_translate = MAX_TRANSLATION_AFFINE * severity
            max_shear = MAX_SHEAR_ANGLE * severity
            
            translate_x = random.uniform(-max_translate, max_translate)
            translate_y = random.uniform(-max_translate, max_translate)
            shear_x = random.uniform(-max_shear, max_shear)
            shear_y = random.uniform(-max_shear, max_shear)
            
            # Must clamp for PIL conversion even in unnormalized version
            img_clamped = torch.clamp(img, 0, 1)
            img_cpu = img_clamped.cpu()
            
            to_pil = transforms.ToPILImage(mode='RGB')
            to_tensor = transforms.ToTensor()
            
            pil_img = to_pil(img_cpu)
            
            width, height = pil_img.size
            translate_pixels = (translate_x * width, translate_y * height)
            
            affine_img = transforms.functional.affine(
                pil_img, 
                angle=0.0,
                translate=translate_pixels,
                scale=1.0,
                shear=[shear_x, shear_y]
            )
            transformed_img = to_tensor(affine_img).to(device)
            
            transform_params['translate_x'] = translate_x
            transform_params['translate_y'] = translate_y
            transform_params['shear_x'] = shear_x
            transform_params['shear_y'] = shear_y
        
        else: 
            transformed_img = img.clone()
        
        if return_params:
            return transformed_img, transform_params
        else:
            return transformed_img
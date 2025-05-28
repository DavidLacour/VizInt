# Static Healer Transforms Usage Guide

The `HealerTransforms` class provides static methods for applying healer corrections without needing a trained healer model. This is useful for testing, debugging, or applying known corrections.

## Quick Start

```python
from src.models.healer_transforms import HealerTransforms
import torch

# Load your image (C, H, W format)
image = torch.rand(3, 32, 32)  # Example image

# Apply specific corrections
denoised = HealerTransforms.apply_gaussian_denoising(image, noise_std=0.1)
unrotated = HealerTransforms.apply_inverse_rotation(image, rotation_angle=45.0)
unaffined = HealerTransforms.apply_inverse_affine(image, 
                                                  translate_x=0.1, 
                                                  translate_y=-0.1,
                                                  shear_x=15.0, 
                                                  shear_y=-10.0)
```

## Available Static Methods

### 1. Gaussian Denoising
```python
denoised = HealerTransforms.apply_gaussian_denoising(
    image,           # Input image [C,H,W] or [B,C,H,W]
    noise_std=0.1,   # Estimated noise standard deviation
    device=None      # Optional: specify device
)
```

### 2. Rotation Correction
```python
corrected = HealerTransforms.apply_inverse_rotation(
    image,              # Input image
    rotation_angle=30,  # Rotation angle in degrees
    device=None
)
```

### 3. Affine Transform Correction
```python
corrected = HealerTransforms.apply_inverse_affine(
    image,
    translate_x=0.1,   # Translation as fraction of width
    translate_y=-0.1,  # Translation as fraction of height  
    shear_x=15.0,      # Shear angle in degrees
    shear_y=-10.0,     # Shear angle in degrees
    device=None
)
```

### 4. Automatic Correction by Type
```python
# Transform types: 0=none, 1=noise, 2=rotation, 3=affine
params = {
    'noise_std': 0.1,
    'rotation_angle': 45.0,
    'translate_x': 0.1,
    'translate_y': -0.1,
    'shear_x': 15.0,
    'shear_y': -10.0
}

corrected = HealerTransforms.apply_correction_by_type(
    image,
    transform_type=2,  # Rotation
    transform_params=params
)
```

### 5. Batch Processing
```python
# Process multiple images with different corrections
batch_images = torch.rand(4, 3, 32, 32)  # Batch of 4 images

# Create predictions dictionary (mimics healer output)
predictions = {
    'transform_type_logits': torch.tensor([[10,0,0,0], [0,10,0,0], [0,0,10,0], [0,0,0,10]]),
    'noise_std': torch.tensor([[0.0], [0.2], [0.0], [0.0]]),
    'rotation_angle': torch.tensor([[0.0], [0.0], [30.0], [0.0]]),
    'translate_x': torch.tensor([[0.0], [0.0], [0.0], [0.1]]),
    'translate_y': torch.tensor([[0.0], [0.0], [0.0], [-0.1]]),
    'shear_x': torch.tensor([[0.0], [0.0], [0.0], [15.0]]),
    'shear_y': torch.tensor([[0.0], [0.0], [0.0], [-10.0]])
}

corrected_batch = HealerTransforms.apply_batch_correction(batch_images, predictions)
```

### 6. Mock Predictions Helper
```python
# Create mock predictions for testing
predictions = HealerTransforms.create_mock_predictions(
    'rotation',           # Transform type name
    rotation_angle=45.0   # Parameters
)

# Apply correction using mock predictions
corrected = HealerTransforms.apply_batch_correction(
    image.unsqueeze(0),  # Add batch dimension
    predictions
).squeeze(0)  # Remove batch dimension
```

## Complete Example

```python
import torch
from torchvision import transforms
from src.models.healer_transforms import HealerTransforms

# Load an image
image = ...  # Your image tensor [C, H, W]

# Example 1: Known gaussian noise
noisy_image = image + torch.randn_like(image) * 0.1
denoised = HealerTransforms.apply_gaussian_denoising(noisy_image, noise_std=0.1)

# Example 2: Known rotation
to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()
rotated = transforms.functional.rotate(to_pil(image), 30)
rotated_tensor = to_tensor(rotated)
corrected = HealerTransforms.apply_inverse_rotation(rotated_tensor, 30)

# Example 3: Using mock predictions (simulates healer model output)
predictions = HealerTransforms.create_mock_predictions(
    'affine',
    translate_x=0.1,
    translate_y=-0.1,
    shear_x=15,
    shear_y=-15
)
corrected = HealerTransforms.apply_batch_correction(
    image.unsqueeze(0), 
    predictions
).squeeze(0)
```

## Integration with Healer Model

The Healer model now uses these static methods internally:

```python
# Inside the Healer model
def apply_correction(self, transformed_images, predictions):
    return HealerTransforms.apply_batch_correction(transformed_images, predictions)
```

This means you can:
1. Use the static methods directly when you know the transformation parameters
2. Use the Healer model when you need to predict the transformation parameters
3. Test corrections without training a model
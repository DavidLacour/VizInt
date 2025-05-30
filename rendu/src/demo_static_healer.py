#!/usr/bin/env python3
"""
Demo script showing how to use static healer transformation functions
"""
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

sys.path.append(str(Path(__file__).parent.parent))

from src.models.healer_transforms import HealerTransforms
from src.data.continuous_transforms import ContinuousTransforms
from src.config.config_loader import ConfigLoader
from src.data.data_loader import DataLoaderFactory


def demo_static_corrections():
    """Demonstrate static healer corrections"""
    print("🎨 Static Healer Transforms Demo")
    print("=" * 50)
    
    config = ConfigLoader()
    data_factory = DataLoaderFactory(config)
    
    print("Loading CIFAR-10 dataset...")
    _, val_loader = data_factory.create_data_loaders(
        'cifar10', 
        with_normalization=False,
        with_augmentation=False
    )
    
    images, labels = next(iter(val_loader))
    original_image = images[0] 
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    show_image(axes[0, 0], original_image, "Original Image")
    
    print("\n1. Gaussian Noise Correction:")
    
    noise_std = 0.2
    noisy_image = original_image + torch.randn_like(original_image) * noise_std
    noisy_image = torch.clamp(noisy_image, 0, 1)
    
    show_image(axes[0, 1], noisy_image, f"Noisy (σ={noise_std})")
    
    denoised = HealerTransforms.apply_gaussian_denoising(noisy_image, noise_std)
    show_image(axes[0, 2], denoised, "Denoised (Static)")
    
    mock_predictions = HealerTransforms.create_mock_predictions('gaussian_noise', noise_std=noise_std)
    denoised_batch = HealerTransforms.apply_batch_correction(
        noisy_image.unsqueeze(0), mock_predictions
    ).squeeze(0)
    show_image(axes[0, 3], denoised_batch, "Denoised (Mock Pred)")
    
    print("2. Rotation Correction:")

    angle = 30.0
    transform_engine = ContinuousTransforms(severity=0.5)
    rotated_image = transforms.functional.rotate(
        transforms.ToPILImage()(original_image), angle
    )
    rotated_tensor = transforms.ToTensor()(rotated_image)
    
    show_image(axes[1, 0], original_image, "Original")
    show_image(axes[1, 1], rotated_tensor, f"Rotated ({angle}°)")
    
    corrected = HealerTransforms.apply_inverse_rotation(rotated_tensor, angle)
    show_image(axes[1, 2], corrected, "Corrected (Static)")
    
    mock_predictions = HealerTransforms.create_mock_predictions('rotation', rotation_angle=angle)
    corrected_batch = HealerTransforms.apply_batch_correction(
        rotated_tensor.unsqueeze(0), mock_predictions
    ).squeeze(0)
    show_image(axes[1, 3], corrected_batch, "Corrected (Mock Pred)")
    
    print("3. Affine Transform Correction:")

    tx, ty = 0.1, -0.1
    sx, sy = 15.0, -10.0
    
    pil_img = transforms.ToPILImage()(original_image)
    width, height = pil_img.size
    affine_img = transforms.functional.affine(
        pil_img,
        angle=0,
        translate=(tx * width, ty * height),
        scale=1.0,
        shear=[sx, sy]
    )
    affine_tensor = transforms.ToTensor()(affine_img)
    
    show_image(axes[2, 0], original_image, "Original")
    show_image(axes[2, 1], affine_tensor, f"Affine Transform")
    
    corrected = HealerTransforms.apply_inverse_affine(affine_tensor, tx, ty, sx, sy)
    show_image(axes[2, 2], corrected, "Corrected (Static)")
    
    mock_predictions = HealerTransforms.create_mock_predictions(
        'affine', translate_x=tx, translate_y=ty, shear_x=sx, shear_y=sy
    )
    corrected_batch = HealerTransforms.apply_batch_correction(
        affine_tensor.unsqueeze(0), mock_predictions
    ).squeeze(0)
    show_image(axes[2, 3], corrected_batch, "Corrected (Mock Pred)")
    
    plt.tight_layout()
    plt.suptitle("Static Healer Transforms Demo", fontsize=16, y=1.02)
    
    output_dir = Path("../../../visualizationsrendu/demos/")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "static_healer_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Demo saved to: {output_path}")
    plt.show()
    
    print("\n" + "="*50)
    print("Example Usage:")
    print("="*50)
    
    print("""
# Import the static transforms
from src.models.healer_transforms import HealerTransforms

# Method 1: Direct static method calls
denoised = HealerTransforms.apply_gaussian_denoising(noisy_image, noise_std=0.1)
corrected = HealerTransforms.apply_inverse_rotation(rotated_image, angle=45.0)
corrected = HealerTransforms.apply_inverse_affine(affine_image, tx=0.1, ty=0.1, sx=15, sy=15)

# Method 2: Using mock predictions (simulates healer output)
predictions = HealerTransforms.create_mock_predictions('rotation', rotation_angle=30.0)
corrected = HealerTransforms.apply_batch_correction(images_batch, predictions)

# Method 3: Apply by type with parameters
params = {'rotation_angle': 45.0}
corrected = HealerTransforms.apply_correction_by_type(image, transform_type=2, transform_params=params)
""")


def show_image(ax, tensor, title):
    """Display tensor as image"""
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')


if __name__ == '__main__':
    demo_static_corrections()
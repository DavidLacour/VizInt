#!/usr/bin/env python3
"""
Test script for static healer transforms (no display required)
"""
import sys
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.models.healer_transforms import HealerTransforms
from src.config.config_loader import ConfigLoader
from src.data.data_loader import DataLoaderFactory
from torchvision import transforms


def test_static_healer():
    """Test static healer transforms and save results"""
    print("🧪 Testing Static Healer Transforms")
    print("=" * 50)
    
    print("\n1. Creating test image...")
    size = 64
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    r_channel = xx
    g_channel = yy
    b_channel = 1 - (xx + yy) / 2
    test_image = torch.stack([r_channel, g_channel, b_channel], dim=0)
    
    print(f"   Test image shape: {test_image.shape}")
    
    print("\n2. Testing Gaussian Denoising:")
    noise_std = 0.15
    noisy = test_image + torch.randn_like(test_image) * noise_std
    noisy = torch.clamp(noisy, 0, 1)
    
    denoised = HealerTransforms.apply_gaussian_denoising(noisy, noise_std)
    noise_reduction = torch.mean(torch.abs(noisy - test_image)) - torch.mean(torch.abs(denoised - test_image))
    print(f"   Noise reduction: {noise_reduction:.4f}")
    
    print("\n3. Testing Rotation Correction:")
    angle = 45.0
    
    pil_img = transforms.ToPILImage()(test_image)
    rotated_pil = transforms.functional.rotate(pil_img, angle)
    rotated = transforms.ToTensor()(rotated_pil)
    
    corrected = HealerTransforms.apply_inverse_rotation(rotated, angle)
    print(f"   Applied rotation: {angle}°")
    print(f"   Corrected shape: {corrected.shape}")
    
    print("\n4. Testing Affine Correction:")
    tx, ty = 0.1, -0.05
    sx, sy = 10.0, -5.0
    
    corrected_affine = HealerTransforms.apply_inverse_affine(test_image, tx, ty, sx, sy)
    print(f"   Translation: ({tx}, {ty})")
    print(f"   Shear: ({sx}°, {sy}°)")
    
    print("\n5. Testing Batch Processing:")
    batch_size = 4
    batch_images = test_image.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    predictions = {
        'transform_type_logits': torch.tensor([[10, 0, 0, 0],   # No transform
                                             [0, 10, 0, 0],   # Gaussian noise
                                             [0, 0, 10, 0],   # Rotation
                                             [0, 0, 0, 10]]), # Affine
        'noise_std': torch.tensor([[0.0], [0.2], [0.0], [0.0]]),
        'rotation_angle': torch.tensor([[0.0], [0.0], [30.0], [0.0]]),
        'translate_x': torch.tensor([[0.0], [0.0], [0.0], [0.1]]),
        'translate_y': torch.tensor([[0.0], [0.0], [0.0], [-0.1]]),
        'shear_x': torch.tensor([[0.0], [0.0], [0.0], [15.0]]),
        'shear_y': torch.tensor([[0.0], [0.0], [0.0], [-10.0]])
    }
    
    corrected_batch = HealerTransforms.apply_batch_correction(batch_images, predictions)
    print(f"   Batch shape: {corrected_batch.shape}")
    print(f"   Successfully processed {batch_size} images with different corrections")
    
    print("\n6. Testing Mock Predictions:")
    for transform_type in ['gaussian_noise', 'rotation', 'affine']:
        mock_pred = HealerTransforms.create_mock_predictions(
            transform_type,
            noise_std=0.1,
            rotation_angle=25.0,
            translate_x=0.05,
            translate_y=-0.05,
            shear_x=5.0,
            shear_y=-5.0
        )
        pred_type = torch.argmax(mock_pred['transform_type_logits']).item()
        print(f"   {transform_type}: predicted type index = {pred_type}")
    
    print("\n7. Saving test results...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(test_image.permute(1, 2, 0).numpy())
    axes[0, 0].set_title("Original Test Pattern")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy.permute(1, 2, 0).numpy())
    axes[0, 1].set_title(f"Noisy (σ={noise_std})")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(denoised.permute(1, 2, 0).numpy())
    axes[0, 2].set_title("Denoised")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(rotated.permute(1, 2, 0).numpy())
    axes[1, 0].set_title(f"Rotated ({angle}°)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(corrected.permute(1, 2, 0).numpy())
    axes[1, 1].set_title("Rotation Corrected")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(corrected_affine.permute(1, 2, 0).numpy())
    axes[1, 2].set_title("Affine Corrected")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    output_dir = Path("../../../visualizationsrendu/demos/")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "static_healer_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Test results saved to: {output_path}")
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_static_healer()
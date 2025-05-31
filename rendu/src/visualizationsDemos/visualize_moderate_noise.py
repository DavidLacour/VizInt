#!/usr/bin/env python3
"""
Visualize denoising methods with moderate noise (σ=0.1) where bilateral excels
"""
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.models.healer_transforms import HealerTransforms
from src.config.config_loader import ConfigLoader
from src.data.data_loader import DataLoaderFactory


def visualize_moderate_noise():
    """Compare denoising at different noise levels"""
    config = ConfigLoader()
    data_factory = DataLoaderFactory(config)
    _, val_loader = data_factory.create_data_loaders('cifar10', with_normalization=False, with_augmentation=False)
    
    images, labels = next(iter(val_loader))
    original = images[0]
    
    noise_levels = [0.05, 0.1, 0.2, 0.3]
    methods = ['gaussian', 'bilateral', 'nlm', 'wiener']
    
    fig, axes = plt.subplots(len(noise_levels) + 1, len(methods) + 1, figsize=(20, 20))
    
    axes[0, 0].text(0.5, 0.5, 'Original', ha='center', va='center', fontsize=14, weight='bold')
    axes[0, 0].axis('off')
    
    for j, method in enumerate(methods):
        axes[0, j+1].text(0.5, 0.5, method.capitalize(), ha='center', va='center', fontsize=14, weight='bold')
        axes[0, j+1].axis('off')
    
    for i in range(1, len(noise_levels) + 1):
        show_image(axes[i, 0], original, f"σ = {noise_levels[i-1]}")
    
    # Process each noise level
    for i, noise_std in enumerate(noise_levels):
        noisy = original + torch.randn_like(original) * noise_std
        noisy = torch.clamp(noisy, 0, 1)
        
        for j, method in enumerate(methods):
            denoised = HealerTransforms.apply_wiener_denoising(noisy, noise_std, method=method)
            
            mse = torch.mean((original - denoised) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)) if mse > 0 else float('inf')
            
            show_image(axes[i+1, j+1], denoised, f"PSNR: {psnr:.1f}")
            
            if noise_std == 0.1 and method == 'bilateral':
                axes[i+1, j+1].patch.set_edgecolor('green')
                axes[i+1, j+1].patch.set_linewidth(3)
    
    plt.suptitle('Denoising Performance at Different Noise Levels', fontsize=16)
    plt.tight_layout()
    
    output_path = Path("../../../visualizationsrendu/demos/") / "noise_levels_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved to: {output_path}")
    
    create_zoomed_comparison(original, 0.1)


def create_zoomed_comparison(original, noise_std=0.1):
    """Create detailed comparison at moderate noise level"""
    noisy = original + torch.randn_like(original) * noise_std
    noisy = torch.clamp(noisy, 0, 1)
    
    methods = {
        'Original': original,
        'Noisy': noisy,
        'Gaussian': HealerTransforms.apply_wiener_denoising(noisy, noise_std, method='gaussian'),
        'Bilateral': HealerTransforms.apply_wiener_denoising(noisy, noise_std, method='bilateral'),
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    h, w = original.shape[1:]
    crop_size = 16
    start_h, start_w = h//2 - crop_size//2, w//2 - crop_size//2
    
    for idx, (name, img) in enumerate(methods.items()):
        ax = axes[0, idx]
        show_image(ax, img, name)
        
        rect = plt.Rectangle((start_w, start_h), crop_size, crop_size, 
                           fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        
        ax_zoom = axes[1, idx]
        zoom = img[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
        show_image(ax_zoom, zoom, f"{name} (Zoomed)")
        
        if name not in ['Original', 'Noisy']:
            mse = torch.mean((original - img) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            ax_zoom.text(0.5, -0.15, f"PSNR: {psnr:.1f} dB", 
                        transform=ax_zoom.transAxes, ha='center')
    
    plt.suptitle(f'Detailed Comparison at σ={noise_std} (Moderate Noise)', fontsize=14)
    plt.tight_layout()
    
    output_path = Path("../../../visualizationsrendu/demos/") / "zoomed_denoising_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Zoomed comparison saved to: {output_path}")


def show_image(ax, tensor, title):
    """Display tensor as image"""
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


if __name__ == '__main__':
    visualize_moderate_noise()
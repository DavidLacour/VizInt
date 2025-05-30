#!/usr/bin/env python3
"""
Healer Wiener Denoising Demo

This script specifically demonstrates the Wiener deconvolution method
used by the Healer model for removing Gaussian noise.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.healer_transforms import HealerTransforms
from data.data_loader import DataLoaderFactory
from config.config_loader import ConfigLoader
from utils.transformer_utils import set_seed


def visualize_wiener_denoising_process(image: torch.Tensor, 
                                      noise_std: float = 0.2,
                                      save_path: Path = None):
    """Visualize the Wiener denoising process step by step"""
    
    # Add Gaussian noise
    noise = torch.randn_like(image) * noise_std
    noisy_image = torch.clamp(image + noise, 0, 1)
    
    # Apply Wiener denoising
    denoised = HealerTransforms.apply_wiener_denoising(noisy_image, noise_std, method='wiener')
    
    # Calculate metrics
    mse_noisy = torch.mean((image - noisy_image) ** 2).item()
    mse_denoised = torch.mean((image - denoised) ** 2).item()
    psnr_noisy = 10 * np.log10(1.0 / mse_noisy)
    psnr_denoised = 10 * np.log10(1.0 / mse_denoised)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    img_orig = image.permute(1, 2, 0).cpu().numpy()
    axes[0, 0].imshow(np.clip(img_orig, 0, 1))
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # Noisy image
    img_noisy = noisy_image.permute(1, 2, 0).cpu().numpy()
    axes[0, 1].imshow(np.clip(img_noisy, 0, 1))
    axes[0, 1].set_title(f'Noisy Image (σ={noise_std})\nPSNR: {psnr_noisy:.2f} dB', fontsize=14)
    axes[0, 1].axis('off')
    
    # Denoised image
    img_denoised = denoised.permute(1, 2, 0).cpu().numpy()
    axes[0, 2].imshow(np.clip(img_denoised, 0, 1))
    axes[0, 2].set_title(f'Wiener Denoised\nPSNR: {psnr_denoised:.2f} dB', fontsize=14)
    axes[0, 2].axis('off')
    
    # Noise visualization
    noise_vis = (noisy_image - image).permute(1, 2, 0).cpu().numpy()
    axes[1, 0].imshow(noise_vis, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title('Added Noise', fontsize=14)
    axes[1, 0].axis('off')
    
    # Removed noise
    removed_noise = (noisy_image - denoised).permute(1, 2, 0).cpu().numpy()
    axes[1, 1].imshow(removed_noise, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title('Removed Noise', fontsize=14)
    axes[1, 1].axis('off')
    
    # Residual (difference from original)
    residual = (denoised - image).permute(1, 2, 0).cpu().numpy()
    axes[1, 2].imshow(residual, cmap='RdBu', vmin=-0.1, vmax=0.1)
    axes[1, 2].set_title('Residual Error', fontsize=14)
    axes[1, 2].axis('off')
    
    plt.suptitle('Healer Wiener Denoising Process', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Wiener denoising visualization to {save_path}")
    
    plt.show()
    
    # Print improvement metrics
    print(f"\nDenoising Metrics:")
    print(f"  Noise std: {noise_std}")
    print(f"  PSNR (noisy): {psnr_noisy:.2f} dB")
    print(f"  PSNR (denoised): {psnr_denoised:.2f} dB")
    print(f"  PSNR improvement: {psnr_denoised - psnr_noisy:.2f} dB")


def compare_noise_levels(image: torch.Tensor, save_path: Path = None):
    """Compare Wiener denoising performance across different noise levels"""
    
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.4]
    
    fig, axes = plt.subplots(3, len(noise_levels), figsize=(20, 12))
    
    psnr_noisy_list = []
    psnr_denoised_list = []
    
    for i, noise_std in enumerate(noise_levels):
        # Add noise
        noise = torch.randn_like(image) * noise_std
        noisy_image = torch.clamp(image + noise, 0, 1)
        
        # Apply Wiener denoising
        denoised = HealerTransforms.apply_wiener_denoising(noisy_image, noise_std)
        
        # Calculate PSNR
        mse_noisy = torch.mean((image - noisy_image) ** 2).item()
        mse_denoised = torch.mean((image - denoised) ** 2).item()
        psnr_noisy = 10 * np.log10(1.0 / mse_noisy) if mse_noisy > 0 else float('inf')
        psnr_denoised = 10 * np.log10(1.0 / mse_denoised) if mse_denoised > 0 else float('inf')
        
        psnr_noisy_list.append(psnr_noisy)
        psnr_denoised_list.append(psnr_denoised)
        
        # Display images
        img_orig = image.permute(1, 2, 0).cpu().numpy()
        img_noisy = noisy_image.permute(1, 2, 0).cpu().numpy()
        img_denoised = denoised.permute(1, 2, 0).cpu().numpy()
        
        axes[0, i].imshow(np.clip(img_orig, 0, 1))
        axes[0, i].set_title(f'σ = {noise_std}', fontsize=12)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(np.clip(img_noisy, 0, 1))
        axes[1, i].set_title(f'PSNR: {psnr_noisy:.1f} dB', fontsize=10)
        axes[1, i].axis('off')
        
        axes[2, i].imshow(np.clip(img_denoised, 0, 1))
        axes[2, i].set_title(f'PSNR: {psnr_denoised:.1f} dB', fontsize=10)
        axes[2, i].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.75, 'Original', rotation=90, va='center', fontsize=14, weight='bold')
    fig.text(0.02, 0.5, 'Noisy', rotation=90, va='center', fontsize=14, weight='bold')
    fig.text(0.02, 0.25, 'Denoised', rotation=90, va='center', fontsize=14, weight='bold')
    
    plt.suptitle('Wiener Denoising Performance Across Noise Levels', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved noise level comparison to {save_path}")
    
    plt.show()
    
    # Plot PSNR improvement graph
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(noise_levels, psnr_noisy_list, 'o-', label='Noisy', linewidth=2, markersize=8)
    ax.plot(noise_levels, psnr_denoised_list, 's-', label='Wiener Denoised', linewidth=2, markersize=8)
    ax.set_xlabel('Noise Standard Deviation', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR vs Noise Level', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        psnr_plot_path = save_path.parent / f"{save_path.stem}_psnr_plot.png"
        plt.savefig(psnr_plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved PSNR plot to {psnr_plot_path}")
    
    plt.show()


def main():
    """Run Wiener denoising demos"""
    set_seed(42)
    
    # Create output directory
    vis_dir = Path('./healer_wiener_visualizations')
    vis_dir.mkdir(exist_ok=True)
    
    print("Loading sample image...")
    config = ConfigLoader('config/cifar10_config.yaml')
    data_factory = DataLoaderFactory(config)
    _, val_loader = data_factory.get_dataloaders('cifar10', batch_size=1)
    
    # Get first image
    for images, _ in val_loader:
        image = images[0]
        break
    
    print("\n1. Visualizing Wiener denoising process...")
    visualize_wiener_denoising_process(
        image, 
        noise_std=0.2,
        save_path=vis_dir / 'wiener_denoising_process.png'
    )
    
    print("\n2. Comparing performance across noise levels...")
    compare_noise_levels(
        image,
        save_path=vis_dir / 'wiener_noise_levels_comparison.png'
    )
    
    print(f"\nAll visualizations saved to {vis_dir}/")
    print("\nWiener denoising demo completed!")


if __name__ == "__main__":
    main()
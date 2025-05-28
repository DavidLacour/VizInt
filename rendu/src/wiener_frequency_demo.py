#!/usr/bin/env python3
"""
Demonstrate Wiener filter's frequency selectivity advantage
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


def demonstrate_frequency_selectivity():
    """Show how Wiener filter selectively preserves frequencies"""
    print("🌊 Wiener Filter Frequency Selectivity Demo")
    print("=" * 60)
    
    # Create test image with different frequency components
    size = 128
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Create image with multiple frequency components
    # Low frequency: gradients
    low_freq = xx + yy
    
    # Medium frequency: stripes
    med_freq = np.sin(20 * np.pi * xx) * 0.3
    
    # High frequency: fine texture
    high_freq = np.sin(100 * np.pi * xx) * np.sin(100 * np.pi * yy) * 0.1
    
    # Combine
    test_image = low_freq + med_freq + high_freq
    test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
    
    # Convert to RGB tensor
    test_tensor = torch.tensor(test_image, dtype=torch.float32)
    test_tensor = test_tensor.unsqueeze(0).repeat(3, 1, 1)
    
    # Add noise
    noise_std = 0.15
    noisy = test_tensor + torch.randn_like(test_tensor) * noise_std
    noisy = torch.clamp(noisy, 0, 1)
    
    # Apply filters
    results = {
        'Original': test_tensor,
        'Noisy': noisy,
        'Gaussian': HealerTransforms.apply_wiener_denoising(noisy, noise_std, method='gaussian'),
        'Bilateral': HealerTransforms.apply_wiener_denoising(noisy, noise_std, method='bilateral'),
        'Wiener': HealerTransforms.apply_wiener_denoising(noisy, noise_std, method='wiener')
    }
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Top row: Images
    for idx, (name, img) in enumerate(results.items()):
        ax = plt.subplot(3, 5, idx + 1)
        show_image(ax, img, name)
    
    # Middle row: Frequency spectra
    for idx, (name, img) in enumerate(results.items()):
        ax = plt.subplot(3, 5, idx + 6)
        show_spectrum(ax, img, f"{name} Spectrum")
    
    # Bottom row: Frequency response curves
    ax_response = plt.subplot(3, 1, 3)
    plot_frequency_responses(ax_response, results)
    
    plt.tight_layout()
    
    # Save
    output_path = Path("../../../visualizationsrendu/demos/") / "wiener_frequency_selectivity.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to: {output_path}")
    
    # Create a second figure showing real image example
    create_real_image_demo()


def create_real_image_demo():
    """Show Wiener advantages on real CIFAR-10 image"""
    # Load real image
    config = ConfigLoader()
    data_factory = DataLoaderFactory(config)
    _, val_loader = data_factory.create_data_loaders('cifar10', with_normalization=False, with_augmentation=False)
    
    images, _ = next(iter(val_loader))
    original = images[0]
    
    # Add noise
    noise_std = 0.1
    noisy = original + torch.randn_like(original) * noise_std
    noisy = torch.clamp(noisy, 0, 1)
    
    # Apply methods
    methods = {
        'Original': original,
        'Noisy (σ=0.1)': noisy,
        'Wiener (Default)': HealerTransforms.apply_gaussian_denoising(noisy, noise_std),  # Uses Wiener
        'Gaussian': HealerTransforms.apply_wiener_denoising(noisy, noise_std, method='gaussian'),
        'Bilateral': HealerTransforms.apply_wiener_denoising(noisy, noise_std, method='bilateral')
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Show images
    for idx, (name, img) in enumerate(methods.items()):
        ax = axes[0, idx]
        show_image(ax, img, name)
        
        # Calculate metrics
        if 'Noisy' not in name and name != 'Original':
            psnr = calculate_psnr(original, img)
            ax.text(0.5, -0.1, f"PSNR: {psnr:.1f} dB", 
                   transform=ax.transAxes, ha='center')
    
    # Show difference maps
    for idx, (name, img) in enumerate(methods.items()):
        ax = axes[1, idx]
        if name == 'Original':
            ax.text(0.5, 0.5, 'Reference', ha='center', va='center', fontsize=14)
            ax.axis('off')
        else:
            diff = torch.abs(original - img).mean(0)
            im = ax.imshow(diff.numpy(), cmap='hot', vmin=0, vmax=0.2)
            ax.set_title(f"{name} Error", fontsize=10)
            ax.axis('off')
    
    plt.suptitle('Wiener Filter Advantages on Real Images', fontsize=16)
    plt.tight_layout()
    
    output_path = Path("../../../visualizationsrendu/demos/") / "wiener_real_image_advantage.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Real image demo saved to: {output_path}")


def show_image(ax, tensor, title):
    """Display image"""
    if tensor.dim() == 3:
        img = tensor.permute(1, 2, 0).numpy()
    else:
        img = tensor.numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img, cmap='gray' if tensor.dim() == 2 else None)
    ax.set_title(title, fontsize=12, weight='bold')
    ax.axis('off')


def show_spectrum(ax, tensor, title):
    """Show frequency spectrum"""
    # Convert to grayscale for visualization
    if tensor.dim() == 3:
        gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
    else:
        gray = tensor
    
    # FFT and shift
    fft = np.fft.fft2(gray.numpy())
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)
    
    im = ax.imshow(magnitude, cmap='hot')
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def plot_frequency_responses(ax, results):
    """Plot radial frequency responses"""
    colors = {'Original': 'black', 'Noisy': 'red', 'Gaussian': 'blue', 
              'Bilateral': 'green', 'Wiener': 'purple'}
    
    for name, img in results.items():
        # Convert to grayscale
        if img.dim() == 3:
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        else:
            gray = img
        
        # FFT
        fft = np.fft.fft2(gray.numpy())
        power = np.abs(fft) ** 2
        
        # Radial average
        h, w = gray.shape
        y, x = np.ogrid[:h, :w]
        center = (h//2, w//2)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        radial_prof = np.zeros(r.max() + 1)
        for i in range(r.max() + 1):
            mask = r == i
            if mask.any():
                radial_prof[i] = power[mask].mean()
        
        # Normalize and plot
        freqs = np.arange(len(radial_prof)) / len(radial_prof)
        ax.semilogy(freqs[:len(freqs)//2], radial_prof[:len(radial_prof)//2] / radial_prof[0], 
                   label=name, color=colors.get(name, 'gray'), linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Normalized Frequency', fontsize=12)
    ax.set_ylabel('Normalized Power', fontsize=12)
    ax.set_title('Frequency Response Comparison', fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    
    # Add annotations
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.05, 1e-3, 'Low Freq\n(Structure)', ha='center', fontsize=9)
    ax.text(0.2, 1e-3, 'Mid Freq\n(Details)', ha='center', fontsize=9)
    ax.text(0.4, 1e-3, 'High Freq\n(Noise)', ha='center', fontsize=9)


def calculate_psnr(original, denoised):
    """Calculate PSNR"""
    mse = torch.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


if __name__ == '__main__':
    demonstrate_frequency_selectivity()
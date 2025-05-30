#!/usr/bin/env python3
"""
Visualize all 4 denoising methods on a single image with 0.3 Gaussian noise
"""
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.models.healer_transforms import HealerTransforms
from src.config.config_loader import ConfigLoader
from src.data.data_loader import DataLoaderFactory


def visualize_denoising_methods():
    """Visualize all denoising methods on a single image"""
    print("🎨 Visualizing Denoising Methods on Single Image")
    print("=" * 60)
    
    config = ConfigLoader()
    data_factory = DataLoaderFactory(config)
    
    print("Loading CIFAR-10 dataset...")
    _, val_loader = data_factory.create_data_loaders(
        'cifar10', 
        with_normalization=False,
        with_augmentation=False
    )
    
    images, labels = next(iter(val_loader))
    original = images[0] 
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    label = class_names[labels[0].item()]
    
    noise_std = 0.3
    noise = torch.randn_like(original) * noise_std
    noisy = original + noise
    noisy = torch.clamp(noisy, 0, 1)
    
    actual_noise = noisy - original
    actual_noise_std = torch.std(actual_noise).item()
    noise_psnr = calculate_psnr(original, noisy)
    
    print(f"\nImage: {label}")
    print(f"Applied noise std: {noise_std}")
    print(f"Actual noise std: {actual_noise_std:.3f}")
    print(f"Noisy image PSNR: {noise_psnr:.2f} dB")
    
    methods = ['gaussian', 'bilateral', 'nlm', 'wiener']
    results = {}
    
    print("\nApplying denoising methods...")
    for method in methods:
        print(f"  - {method.capitalize()}...", end='', flush=True)
        start_time = time.time()
        
        denoised = HealerTransforms.apply_wiener_denoising(
            noisy, noise_std, method=method
        )
        
        elapsed = time.time() - start_time
        
        psnr = calculate_psnr(original, denoised)
        mse = torch.mean((original - denoised) ** 2).item()
        ssim = calculate_ssim(original, denoised)
        
        results[method] = {
            'image': denoised,
            'psnr': psnr,
            'mse': mse,
            'ssim': ssim,
            'time': elapsed
        }
        
        print(f" done ({elapsed*1000:.1f}ms)")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    show_image(axes[0, 0], original, f"Original\n{label}")
    show_image(axes[0, 1], noisy, f"Noisy (σ={noise_std})\nPSNR: {noise_psnr:.1f} dB")
    
    noise_viz = (actual_noise - actual_noise.min()) / (actual_noise.max() - actual_noise.min())
    show_image(axes[0, 2], noise_viz, f"Noise Pattern\nstd: {actual_noise_std:.3f}")
    
    # Show all 4 denoising methods
    # Gaussian
    result = results['gaussian']
    title = f"Gaussian (Original)\nPSNR: {result['psnr']:.1f} dB\nSSIM: {result['ssim']:.3f}\nTime: {result['time']*1000:.1f}ms"
    show_image(axes[1, 0], result['image'], title)
    
    # Bilateral
    result = results['bilateral']
    title = f"Bilateral (Recommended)\nPSNR: {result['psnr']:.1f} dB\nSSIM: {result['ssim']:.3f}\nTime: {result['time']*1000:.1f}ms"
    show_image(axes[1, 1], result['image'], title)
    axes[1, 1].patch.set_edgecolor('green')
    axes[1, 1].patch.set_linewidth(3)
    
    # NLM
    result = results['nlm']
    title = f"Non-Local Means\nPSNR: {result['psnr']:.1f} dB\nSSIM: {result['ssim']:.3f}\nTime: {result['time']*1000:.1f}ms"
    show_image(axes[1, 2], result['image'], title)
    
    # Add Wiener
    ax_inset = fig.add_axes([0.75, 0.62, 0.2, 0.2])  # [left, bottom, width, height]
    result = results['wiener']
    show_image(ax_inset, result['image'], f"Wiener\nPSNR: {result['psnr']:.1f}")
    ax_inset.patch.set_edgecolor('gray')
    ax_inset.patch.set_linewidth(2)
    
    plt.suptitle(f"Denoising Methods Comparison - Gaussian Noise (σ={noise_std})", fontsize=16)
    plt.tight_layout()
    
    output_dir = Path("../../../visualizationsrendu/demos/")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "single_image_denoising_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    create_detailed_comparison(original, noisy, results, noise_std, label)
    print("\n" + "="*80)
    print("📊 DENOISING RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<12} {'PSNR (dB)':<12} {'MSE':<12} {'SSIM':<12} {'Time (ms)':<12}")
    print("-"*80)
    
    for method in methods:
        r = results[method]
        print(f"{method:<12} {r['psnr']:<12.2f} {r['mse']:<12.6f} {r['ssim']:<12.3f} {r['time']*1000:<12.1f}")
    
    print("-"*80)
    print("Best PSNR:  ", max(methods, key=lambda m: results[m]['psnr']))
    print("Best SSIM:  ", max(methods, key=lambda m: results[m]['ssim']))
    print("Fastest:    ", min(methods, key=lambda m: results[m]['time']))
    
    plt.show()


def create_detailed_comparison(original, noisy, results, noise_std, label):
    """Create a detailed side-by-side comparison"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original, noisy, and two best methods
    axes[0, 0].imshow(original.permute(1, 2, 0).numpy())
    axes[0, 0].set_title(f"Original\n{label}", fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy.permute(1, 2, 0).numpy())
    axes[0, 1].set_title(f"Noisy (σ={noise_std})", fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(results['bilateral']['image'].permute(1, 2, 0).numpy())
    axes[0, 2].set_title("Bilateral (Recommended)\nEdge-preserving", fontsize=12, color='green')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(results['gaussian']['image'].permute(1, 2, 0).numpy())
    axes[0, 3].set_title("Gaussian (Original)\nSimple blur", fontsize=12)
    axes[0, 3].axis('off')
    
    # Difference maps
    for idx, method in enumerate(['gaussian', 'bilateral', 'nlm', 'wiener']):
        ax = axes[1, idx]
        
        diff = torch.abs(original - results[method]['image'])
        diff_normalized = diff / diff.max()
        
        im = ax.imshow(diff_normalized.mean(0).numpy(), cmap='hot', vmin=0, vmax=0.5)
        ax.set_title(f"{method.capitalize()}\nError Map", fontsize=10)
        ax.axis('off')
    
    plt.suptitle("Detailed Denoising Comparison", fontsize=14)
    plt.tight_layout()
    
    output_path = Path("../../../visualizationsrendu/demos/") / "detailed_denoising_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Detailed comparison saved to: {output_path}")
    plt.close()


def calculate_psnr(original, denoised):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = torch.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (simplified)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    if img1.dim() == 3 and img1.shape[0] == 3:
        img1_gray = 0.299 * img1[0] + 0.587 * img1[1] + 0.114 * img1[2]
        img2_gray = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]
    else:
        img1_gray = img1
        img2_gray = img2
    
    mu1 = img1_gray.mean()
    mu2 = img2_gray.mean()
    
    var1 = ((img1_gray - mu1) ** 2).mean()
    var2 = ((img2_gray - mu2) ** 2).mean()
    covar = ((img1_gray - mu1) * (img2_gray - mu2)).mean()
    
    numerator = (2 * mu1 * mu2 + C1) * (2 * covar + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2)
    
    return (numerator / denominator).item()


def show_image(ax, tensor, title):
    """Display tensor as image"""
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')


if __name__ == '__main__':
    visualize_denoising_methods()
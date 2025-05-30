#!/usr/bin/env python3
"""
Compare different denoising methods: Gaussian blur vs Wiener deconvolution
"""
import sys
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.healer_transforms import HealerTransforms
from src.config.config_loader import ConfigLoader
from src.data.data_loader import DataLoaderFactory


def compare_denoising_methods():
    """Compare different denoising methods on CIFAR-10 images"""
    print("🔬 Comparing Denoising Methods")
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
    
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.3]
    methods = ['gaussian', 'bilateral', 'nlm', 'wiener']
    
    num_images = 3
    fig = plt.figure(figsize=(20, 4 * num_images))
    
    # Process each test image
    for img_idx in range(num_images):
        original = images[img_idx]
        
        for noise_idx, noise_std in enumerate(noise_levels):
            noisy = original + torch.randn_like(original) * noise_std
            noisy = torch.clamp(noisy, 0, 1)
            
            noise_psnr = calculate_psnr(original, noisy)
            
            row = img_idx * len(noise_levels) + noise_idx
            
            ax = plt.subplot(num_images * len(noise_levels), len(methods) + 2, 
                           row * (len(methods) + 2) + 1)
            show_image(ax, original, "Original")
            
            ax = plt.subplot(num_images * len(noise_levels), len(methods) + 2, 
                           row * (len(methods) + 2) + 2)
            show_image(ax, noisy, f"Noisy\nσ={noise_std:.2f}\nPSNR={noise_psnr:.1f}")
            
            # Apply each method
            for method_idx, method in enumerate(methods):
                start_time = time.time()
                
                if method == 'gaussian':
                    denoised = HealerTransforms.apply_wiener_denoising(
                        noisy, noise_std, method='gaussian'
                    )
                else:
                    denoised = HealerTransforms.apply_wiener_denoising(
                        noisy, noise_std, method=method
                    )
                
                elapsed_time = time.time() - start_time
                
                psnr = calculate_psnr(original, denoised)
                ssim = calculate_ssim(original, denoised)
                
                ax = plt.subplot(num_images * len(noise_levels), len(methods) + 2, 
                               row * (len(methods) + 2) + 3 + method_idx)
                show_image(ax, denoised, 
                         f"{method.capitalize()}\nPSNR={psnr:.1f}\nSSIM={ssim:.3f}\nt={elapsed_time:.3f}s")
    
    plt.tight_layout()
    plt.suptitle("Denoising Methods Comparison", fontsize=16, y=1.001)
    
    output_dir = Path("../../../visualizationsrendu/demos/")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "denoising_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Comparison saved to: {output_path}")
    
    print("\n📊 Quantitative Comparison")
    print("=" * 60)
    run_quantitative_comparison()


def run_quantitative_comparison():
    """Run quantitative comparison on more images"""
    config = ConfigLoader()
    data_factory = DataLoaderFactory(config)
    _, val_loader = data_factory.create_data_loaders(
        'cifar10', 
        with_normalization=False,
        with_augmentation=False
    )
    
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    methods = ['gaussian', 'wiener', 'bilateral', 'nlm']
    num_test_images = 50
    
    results = {method: {noise: {'psnr': [], 'ssim': [], 'time': []} 
                       for noise in noise_levels} 
              for method in methods}
    
    all_images = []
    for batch_images, _ in val_loader:
        all_images.extend([img for img in batch_images])
        if len(all_images) >= num_test_images:
            break
    test_images = all_images[:num_test_images]
    
    print(f"Testing on {len(test_images)} images...")
    
    # Run tests
    for img_idx, original in enumerate(test_images):
        if img_idx % 10 == 0:
            print(f"Processing image {img_idx}/{len(test_images)}...")
        
        for noise_std in noise_levels:
            noisy = original + torch.randn_like(original) * noise_std
            noisy = torch.clamp(noisy, 0, 1)
            
            for method in methods:
                start_time = time.time()
                
                if method == 'gaussian':
                    denoised = HealerTransforms.apply_wiener_denoising(
                        noisy, noise_std, method='gaussian'
                    )
                else:
                    denoised = HealerTransforms.apply_wiener_denoising(
                        noisy, noise_std, method=method
                    )
                
                elapsed_time = time.time() - start_time
                
                psnr = calculate_psnr(original, denoised)
                ssim = calculate_ssim(original, denoised)
                
                results[method][noise_std]['psnr'].append(psnr)
                results[method][noise_std]['ssim'].append(ssim)
                results[method][noise_std]['time'].append(elapsed_time)
    
    print("\n📈 Average Results")
    print("-" * 80)
    print(f"{'Method':<12} {'Noise σ':<10} {'PSNR (dB)':<12} {'SSIM':<10} {'Time (ms)':<10}")
    print("-" * 80)
    
    for method in methods:
        for noise_std in noise_levels:
            avg_psnr = np.mean(results[method][noise_std]['psnr'])
            avg_ssim = np.mean(results[method][noise_std]['ssim'])
            avg_time = np.mean(results[method][noise_std]['time']) * 1000  # Convert to ms
            
            print(f"{method:<12} {noise_std:<10.2f} {avg_psnr:<12.2f} {avg_ssim:<10.3f} {avg_time:<10.1f}")
        print()
    
    create_summary_plots(results, noise_levels, methods)


def create_summary_plots(results, noise_levels, methods):
    """Create summary plots for the comparison"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # PSNR vs Noise Level
    for method in methods:
        psnr_means = [np.mean(results[method][noise]['psnr']) for noise in noise_levels]
        psnr_stds = [np.std(results[method][noise]['psnr']) for noise in noise_levels]
        ax1.errorbar(noise_levels, psnr_means, yerr=psnr_stds, 
                    label=method.capitalize(), marker='o', capsize=5)
    
    ax1.set_xlabel('Noise Standard Deviation')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR vs Noise Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SSIM vs Noise Level
    for method in methods:
        ssim_means = [np.mean(results[method][noise]['ssim']) for noise in noise_levels]
        ssim_stds = [np.std(results[method][noise]['ssim']) for noise in noise_levels]
        ax2.errorbar(noise_levels, ssim_means, yerr=ssim_stds, 
                    label=method.capitalize(), marker='o', capsize=5)
    
    ax2.set_xlabel('Noise Standard Deviation')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM vs Noise Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Processing Time
    avg_times = {}
    for method in methods:
        all_times = []
        for noise in noise_levels:
            all_times.extend(results[method][noise]['time'])
        avg_times[method] = np.mean(all_times) * 1000  # Convert to ms
    
    methods_cap = [m.capitalize() for m in methods]
    times = [avg_times[m] for m in methods]
    bars = ax3.bar(methods_cap, times)
    
    # Color bars
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax3.set_ylabel('Average Time (ms)')
    ax3.set_title('Processing Time Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_dir = Path("../../../visualizationsrendu/demos/")
    output_path = output_dir / "denoising_metrics_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Metrics comparison saved to: {output_path}")


def calculate_psnr(original, denoised):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = torch.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_ssim(img1, img2, window_size=11):
    """Calculate Structural Similarity Index (simplified version)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    if img1.dim() == 3 and img1.shape[0] == 3:
        img1 = 0.299 * img1[0] + 0.587 * img1[1] + 0.114 * img1[2]
        img2 = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]
    
    mu1 = img1.mean()
    mu2 = img2.mean()
    
    var1 = ((img1 - mu1) ** 2).mean()
    var2 = ((img2 - mu2) ** 2).mean()
    covar = ((img1 - mu1) * (img2 - mu2)).mean()
    
    numerator = (2 * mu1 * mu2 + C1) * (2 * covar + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2)
    
    return (numerator / denominator).item()


def show_image(ax, tensor, title):
    """Display tensor as image"""
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


if __name__ == '__main__':
    compare_denoising_methods()
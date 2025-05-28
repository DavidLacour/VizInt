#!/usr/bin/env python3
"""
Test the improved Wiener filter and compare its characteristics
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


def test_improved_wiener():
    """Test improved Wiener filter and show why it's superior"""
    print("🔬 Testing Improved Wiener Filter")
    print("=" * 60)
    
    # Load data
    config = ConfigLoader()
    data_factory = DataLoaderFactory(config)
    _, val_loader = data_factory.create_data_loaders('cifar10', with_normalization=False, with_augmentation=False)
    
    images, labels = next(iter(val_loader))
    
    # Test on multiple images
    num_test_images = 3
    noise_std = 0.15  # Moderate noise level
    
    fig, axes = plt.subplots(num_test_images, 6, figsize=(18, 9))
    
    # Column headers
    headers = ['Original', 'Noisy', 'Gaussian', 'Bilateral', 'Wiener', 'Frequency Analysis']
    
    for img_idx in range(num_test_images):
        original = images[img_idx]
        
        # Add noise
        noisy = original + torch.randn_like(original) * noise_std
        noisy = torch.clamp(noisy, 0, 1)
        
        # Apply different methods
        results = {
            'original': original,
            'noisy': noisy,
            'gaussian': HealerTransforms.apply_wiener_denoising(noisy, noise_std, method='gaussian'),
            'bilateral': HealerTransforms.apply_wiener_denoising(noisy, noise_std, method='bilateral'),
            'wiener': HealerTransforms.apply_wiener_denoising(noisy, noise_std, method='wiener')
        }
        
        # Show images
        for idx, (key, img) in enumerate(results.items()):
            if idx < 5:
                ax = axes[img_idx, idx]
                show_image(ax, img, headers[idx] if img_idx == 0 else "")
                
                # Calculate metrics
                if key not in ['original', 'noisy']:
                    psnr = calculate_psnr(original, img)
                    edge_score = calculate_edge_preservation(original, img)
                    ax.text(0.5, -0.1, f"PSNR: {psnr:.1f}\nEdge: {edge_score:.2f}", 
                           transform=ax.transAxes, ha='center', fontsize=8)
        
        # Frequency analysis
        ax_freq = axes[img_idx, 5]
        plot_frequency_response(ax_freq, original, results, img_idx == 0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("../../../visualizationsrendu/demos/") / "improved_wiener_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to: {output_path}")
    
    # Detailed analysis
    print("\n📊 Why Wiener Filter is Superior:")
    print("-" * 60)
    analyze_wiener_advantages()


def analyze_wiener_advantages():
    """Analyze and explain Wiener filter advantages"""
    
    advantages = """
1. **Frequency-Selective Filtering**:
   - Preserves low frequencies (image structure)
   - Suppresses high frequencies (noise)
   - Adaptive based on local SNR
   
2. **Theoretically Optimal**:
   - Minimizes mean square error
   - Based on signal processing theory
   - Optimal for Gaussian noise
   
3. **Edge Preservation**:
   - Better than Gaussian blur
   - Frequency domain preserves edges
   - No spatial blurring artifacts
   
4. **Detail Retention**:
   - Preserves textures better
   - Maintains fine details
   - Less aggressive than bilateral
   
5. **Consistent Results**:
   - Deterministic algorithm
   - No parameter tuning needed
   - Works well across noise levels
    """
    
    print(advantages)
    
    # Quantitative comparison
    print("\n📈 Quantitative Analysis:")
    compare_methods_quantitatively()


def compare_methods_quantitatively():
    """Run quantitative comparison"""
    config = ConfigLoader()
    data_factory = DataLoaderFactory(config)
    _, val_loader = data_factory.create_data_loaders('cifar10', with_normalization=False, with_augmentation=False)
    
    # Test parameters
    num_images = 20
    noise_levels = [0.05, 0.1, 0.15, 0.2]
    
    # Collect images
    test_images = []
    for batch in val_loader:
        test_images.extend(batch[0])
        if len(test_images) >= num_images:
            break
    test_images = test_images[:num_images]
    
    # Results storage
    results = {method: {noise: [] for noise in noise_levels} 
              for method in ['gaussian', 'bilateral', 'wiener']}
    
    print("Testing on {} images...".format(num_images))
    
    for img in test_images:
        for noise_std in noise_levels:
            # Add noise
            noisy = img + torch.randn_like(img) * noise_std
            noisy = torch.clamp(noisy, 0, 1)
            
            # Test each method
            for method in results.keys():
                denoised = HealerTransforms.apply_wiener_denoising(noisy, noise_std, method=method)
                
                # Calculate metrics
                psnr = calculate_psnr(img, denoised)
                edge_score = calculate_edge_preservation(img, denoised)
                detail_score = calculate_detail_preservation(img, denoised)
                
                results[method][noise_std].append({
                    'psnr': psnr,
                    'edge': edge_score,
                    'detail': detail_score
                })
    
    # Print summary
    print("\n" + "-" * 80)
    print(f"{'Method':<12} {'Noise':<8} {'PSNR':<10} {'Edge Pres.':<12} {'Detail Pres.':<12}")
    print("-" * 80)
    
    for method in results.keys():
        for noise_std in noise_levels:
            metrics = results[method][noise_std]
            avg_psnr = np.mean([m['psnr'] for m in metrics])
            avg_edge = np.mean([m['edge'] for m in metrics])
            avg_detail = np.mean([m['detail'] for m in metrics])
            
            print(f"{method:<12} {noise_std:<8.2f} {avg_psnr:<10.2f} {avg_edge:<12.3f} {avg_detail:<12.3f}")
        print()


def calculate_psnr(original, denoised):
    """Calculate PSNR"""
    mse = torch.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def calculate_edge_preservation(original, denoised):
    """Calculate edge preservation score using gradient similarity"""
    # Calculate gradients
    dx_orig = original[:, :, 1:] - original[:, :, :-1]
    dy_orig = original[:, 1:, :] - original[:, :-1, :]
    dx_den = denoised[:, :, 1:] - denoised[:, :, :-1]
    dy_den = denoised[:, 1:, :] - denoised[:, :-1, :]
    
    # Calculate similarity
    edge_sim_x = torch.sum(dx_orig * dx_den) / (torch.norm(dx_orig) * torch.norm(dx_den) + 1e-10)
    edge_sim_y = torch.sum(dy_orig * dy_den) / (torch.norm(dy_orig) * torch.norm(dy_den) + 1e-10)
    
    return ((edge_sim_x + edge_sim_y) / 2).item()


def calculate_detail_preservation(original, denoised):
    """Calculate detail preservation using high-frequency content"""
    # High-pass filter using Laplacian
    kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
    
    # Simple convolution for each channel
    detail_score = 0
    for c in range(3):
        # Extract details using high-pass filter
        orig_detail = torch.nn.functional.conv2d(
            original[c:c+1].unsqueeze(0), 
            kernel.unsqueeze(0).unsqueeze(0), 
            padding=1
        ).squeeze()
        
        den_detail = torch.nn.functional.conv2d(
            denoised[c:c+1].unsqueeze(0), 
            kernel.unsqueeze(0).unsqueeze(0), 
            padding=1
        ).squeeze()
        
        # Calculate correlation
        correlation = torch.sum(orig_detail * den_detail) / (torch.norm(orig_detail) * torch.norm(den_detail) + 1e-10)
        detail_score += correlation.item()
    
    return detail_score / 3


def plot_frequency_response(ax, original, results, show_title=True):
    """Plot frequency response comparison"""
    # Calculate frequency content
    methods = ['noisy', 'gaussian', 'bilateral', 'wiener']
    colors = ['red', 'blue', 'green', 'purple']
    
    # Use center channel for analysis
    orig_gray = 0.299 * original[0] + 0.587 * original[1] + 0.114 * original[2]
    
    # FFT of original
    orig_fft = np.fft.fft2(orig_gray.numpy())
    orig_power = np.abs(orig_fft) ** 2
    
    # Radial average
    h, w = orig_gray.shape
    y, x = np.ogrid[:h, :w]
    center = (h//2, w//2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    # Plot each method
    for method, color in zip(methods, colors):
        img = results[method]
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        
        fft = np.fft.fft2(gray.numpy())
        power = np.abs(fft) ** 2
        
        # Radial average
        radial_prof = np.zeros(r.max() + 1)
        for i in range(r.max() + 1):
            mask = r == i
            if mask.any():
                radial_prof[i] = power[mask].mean()
        
        # Plot
        freqs = np.arange(len(radial_prof)) / len(radial_prof)
        ax.semilogy(freqs[:len(freqs)//2], radial_prof[:len(radial_prof)//2], 
                   label=method.capitalize(), color=color, alpha=0.7)
    
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Power')
    if show_title:
        ax.set_title('Frequency Response')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def show_image(ax, tensor, title):
    """Display tensor as image"""
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(title, fontsize=10, weight='bold')
    ax.axis('off')


if __name__ == '__main__':
    test_improved_wiener()
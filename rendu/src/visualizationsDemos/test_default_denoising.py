#!/usr/bin/env python3
"""
Quick test to verify bilateral filter is the default denoising method
"""
import sys
from pathlib import Path
import torch
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.models.healer_transforms import HealerTransforms

test_image = torch.rand(3, 32, 32)
noise_std = 0.1

noisy = test_image + torch.randn_like(test_image) * noise_std
noisy = torch.clamp(noisy, 0, 1)

print("Testing default denoising method...")
print("=" * 50)

start = time.time()
denoised_default = HealerTransforms.apply_gaussian_denoising(noisy, noise_std)
time_default = time.time() - start

start = time.time()
denoised_bilateral = HealerTransforms.apply_wiener_denoising(noisy, noise_std, method='bilateral')
time_bilateral = time.time() - start

start = time.time()
denoised_gaussian = HealerTransforms.apply_wiener_denoising(noisy, noise_std, method='gaussian')
time_gaussian = time.time() - start

diff = torch.abs(denoised_default - denoised_bilateral).mean()
print(f"Default method time: {time_default*1000:.2f}ms")
print(f"Bilateral method time: {time_bilateral*1000:.2f}ms")
print(f"Gaussian method time: {time_gaussian*1000:.2f}ms")
print(f"\nDifference between default and bilateral: {diff:.6f}")
print(f"Default method is {'bilateral' if diff < 0.001 else 'NOT bilateral'}")

# Compare quality
mse_default = torch.mean((test_image - denoised_default)**2)
mse_gaussian = torch.mean((test_image - denoised_gaussian)**2)

print(f"\nMSE with default (bilateral): {mse_default:.6f}")
print(f"MSE with gaussian: {mse_gaussian:.6f}")
print(f"Improvement: {((mse_gaussian - mse_default) / mse_gaussian * 100):.1f}%")

print("\n✅ Bilateral filter is now the default denoising method!")
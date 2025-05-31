#!/usr/bin/env python3
"""
Corrector Models Visualization Demo

This script demonstrates the corrector models (UNet, Transformer, Hybrid)
and their ability to clean corrupted images before classification.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from typing import Tuple, List
from torchvision import transforms

sys.path.append(str(Path(__file__).parent))

from models.pretrained_correctors import PretrainedUNetCorrector, ImageToImageTransformer, HybridCorrector
from models.model_factory import ModelFactory
from data.continuous_transforms import ContinuousTransforms
from data.data_loader import DataLoaderFactory
from config.config_loader import ConfigLoader
from utils.transformer_utils import set_seed


def load_sample_image(config_path: str = 'config/cifar10_config.yaml') -> Tuple[torch.Tensor, int]:
    """Load a sample image from CIFAR-10 or TinyImageNet"""
    config = ConfigLoader(config_path)
    config.config['training']['batch_size'] = 1
    data_factory = DataLoaderFactory(config)
    
    dataset_name = 'cifar10' if 'cifar10' in config_path else 'tinyimagenet'
    _, val_loader = data_factory.create_data_loaders(dataset_name, with_augmentation=False)
    
    for batch in val_loader:
        if dataset_name == 'tinyimagenet' and len(batch) == 4:
            images, _, labels, _ = batch
        else:
            images, labels = batch
        return images[0], labels[0].item()


def apply_corruption(image: torch.Tensor, corruption_type: str = 'mixed') -> torch.Tensor:
    """Apply corruption to image using ContinuousTransforms and additional effects"""
    continuous_transform = ContinuousTransforms(severity=1.0)
    
    if corruption_type == 'gaussian_noise':
        # Use ContinuousTransforms for gaussian noise with severity 0.6 (0.6 * 0.5 = 0.3 std)
        corrupted = continuous_transform.apply_transforms(image, 'gaussian_noise', severity=0.6)
        
    elif corruption_type == 'rotation':
        # Use ContinuousTransforms for rotation with severity 0.125 (0.125 * 360 = 45 degrees max)
        corrupted = continuous_transform.apply_transforms(image, 'rotation', severity=0.125)
        
    elif corruption_type == 'blur':
        # Apply Gaussian blur (not in ContinuousTransforms)
        img_cpu = image.cpu()
        pil_img = transforms.ToPILImage()(img_cpu)
        blurred = transforms.functional.gaussian_blur(pil_img, kernel_size=7, sigma=2.0)
        corrupted = transforms.ToTensor()(blurred).to(image.device)
        
    elif corruption_type == 'mixed':
        # Apply multiple corruptions
        # First add noise using ContinuousTransforms
        corrupted = continuous_transform.apply_transforms(image, 'gaussian_noise', severity=0.3)
        
        # Then blur
        img_cpu = corrupted.cpu()
        pil_img = transforms.ToPILImage()(torch.clamp(img_cpu, 0, 1))
        blurred = transforms.functional.gaussian_blur(pil_img, kernel_size=5, sigma=1.0)
        corrupted = transforms.ToTensor()(blurred).to(image.device)
        
        # Finally add slight rotation using ContinuousTransforms
        corrupted = continuous_transform.apply_transforms(corrupted, 'rotation', severity=0.028)
        
    else:
        corrupted = image.clone()
    
    return torch.clamp(corrupted, 0, 1)


def visualize_corrector_comparison(image: torch.Tensor, 
                                 corrector_models: dict,
                                 corruption_type: str = 'gaussian_noise',
                                 save_path: Path = None):
    """Compare different corrector models on the same corruption"""
    
    # Apply corruption
    corrupted = apply_corruption(image, corruption_type)
    
    # Create visualization
    num_models = len(corrector_models) + 2  # +2 for original and corrupted
    fig, axes = plt.subplots(1, num_models, figsize=(4*num_models, 4))
    
    # Original image
    img_orig = image.permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(np.clip(img_orig, 0, 1))
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    
    # Corrupted image
    img_corrupt = corrupted.permute(1, 2, 0).cpu().numpy()
    axes[1].imshow(np.clip(img_corrupt, 0, 1))
    axes[1].set_title(f'Corrupted\n({corruption_type})', fontsize=14)
    axes[1].axis('off')
    
    # Apply each corrector
    for idx, (name, corrector) in enumerate(corrector_models.items()):
        corrector.eval()
        with torch.no_grad():
            # Add batch dimension
            corrected = corrector(corrupted.unsqueeze(0)).squeeze(0)
        
        img_corrected = corrected.permute(1, 2, 0).cpu().numpy()
        axes[idx + 2].imshow(np.clip(img_corrected, 0, 1))
        axes[idx + 2].set_title(f'{name}\nCorrected', fontsize=14)
        axes[idx + 2].axis('off')
    
    plt.suptitle(f'Corrector Models Comparison: {corruption_type.replace("_", " ").title()}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_severity_comparison(image: torch.Tensor,
                                corrector_model: torch.nn.Module,
                                model_name: str,
                                save_path: Path = None):
    """Show corrector performance across different severity levels"""
    
    severities = [0.1, 0.2, 0.3, 0.4, 0.5]
    fig, axes = plt.subplots(3, len(severities) + 1, figsize=(4*(len(severities)+1), 12))
    
    # Original in first column
    img_orig = image.permute(1, 2, 0).cpu().numpy()
    for i in range(3):
        axes[i, 0].imshow(np.clip(img_orig, 0, 1))
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=12)
        axes[i, 0].axis('off')
    
    corrector_model.eval()
    
    for col, severity in enumerate(severities):
        # Apply noise with varying severity
        noise = torch.randn_like(image) * severity
        corrupted = torch.clamp(image + noise, 0, 1)
        
        # Apply correction
        with torch.no_grad():
            corrected = corrector_model(corrupted.unsqueeze(0)).squeeze(0)
        
        # Calculate metrics
        mse_corrupted = torch.mean((image - corrupted) ** 2).item()
        mse_corrected = torch.mean((image - corrected) ** 2).item()
        psnr_corrupted = 10 * np.log10(1.0 / mse_corrupted) if mse_corrupted > 0 else float('inf')
        psnr_corrected = 10 * np.log10(1.0 / mse_corrected) if mse_corrected > 0 else float('inf')
        
        # Display corrupted
        img_corrupt = corrupted.permute(1, 2, 0).cpu().numpy()
        axes[1, col+1].imshow(np.clip(img_corrupt, 0, 1))
        axes[1, col+1].set_title(f'σ={severity}\nPSNR: {psnr_corrupted:.1f}', fontsize=12)
        axes[1, col+1].axis('off')
        
        # Display corrected
        img_corrected = corrected.permute(1, 2, 0).cpu().numpy()
        axes[2, col+1].imshow(np.clip(img_corrected, 0, 1))
        axes[2, col+1].set_title(f'Corrected\nPSNR: {psnr_corrected:.1f}', fontsize=12)
        axes[2, col+1].axis('off')
        
        # Empty top row
        axes[0, col+1].axis('off')
    
    plt.suptitle(f'{model_name} Corrector: Performance Across Noise Levels', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved severity comparison to {save_path}")
    
    plt.show()


def load_or_create_correctors(config_path: str, dataset_name: str, checkpoint_dir: Path = None):
    """Load pretrained correctors or create new ones"""
    config = ConfigLoader(config_path)
    model_factory = ModelFactory(config)
    
    correctors = {}
    
    # Try to load pretrained correctors
    corrector_types = ['unet_corrector', 'transformer_corrector', 'hybrid_corrector']
    
    for corrector_type in corrector_types:
        if checkpoint_dir:
            checkpoint_path = checkpoint_dir / f'best_{corrector_type}.pt'
            if checkpoint_path.exists():
                print(f"Loading pretrained {corrector_type} from {checkpoint_path}")
                corrector = model_factory.create_model(corrector_type, dataset_name)
                corrector.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
                correctors[corrector_type.replace('_corrector', '').upper()] = corrector
                continue
        
        # Create new corrector
        print(f"Creating new {corrector_type} (untrained)")
        corrector = model_factory.create_model(corrector_type, dataset_name)
        correctors[corrector_type.replace('_corrector', '').upper()] = corrector
    
    return correctors


def main():
    """Run corrector visualization demos"""
    parser = argparse.ArgumentParser(description='Corrector Models Demo')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'tinyimagenet'], 
                        default='cifar10', help='Dataset to use')
    args = parser.parse_args()
    
    set_seed(42)
    
    # Create output directory
    vis_dir = Path(f'./corrector_visualizations_{args.dataset}')
    vis_dir.mkdir(exist_ok=True)
    
    print(f"Loading sample image from {args.dataset.upper()}...")
    config_path = f'config/{args.dataset}_config.yaml'
    image, label = load_sample_image(config_path)
    
    # Check for pretrained correctors
    checkpoint_base = Path(f'../../../{args.dataset}checkpointsrendufunky4/correctors')
    if not checkpoint_base.exists():
        checkpoint_base = None
        print("No pretrained correctors found, using untrained models")
    
    print("\nLoading corrector models...")
    correctors = load_or_create_correctors(config_path, args.dataset, checkpoint_base)
    
    print("\n1. Comparing correctors on different corruption types...")
    corruption_types = ['gaussian_noise', 'rotation', 'blur', 'mixed']
    
    for corruption_type in corruption_types:
        print(f"\n   Processing {corruption_type}...")
        visualize_corrector_comparison(
            image,
            correctors,
            corruption_type,
            save_path=vis_dir / f'correctors_{corruption_type}_comparison.png'
        )
    
    print("\n2. Analyzing performance across severity levels...")
    for name, corrector in correctors.items():
        print(f"\n   Analyzing {name} corrector...")
        visualize_severity_comparison(
            image,
            corrector,
            name,
            save_path=vis_dir / f'{name.lower()}_severity_comparison.png'
        )
    
    print(f"\nAll visualizations saved to {vis_dir}/")
    print("\nCorrector demo completed!")


if __name__ == "__main__":
    main()
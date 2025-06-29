#!/usr/bin/env python3
"""
Healer Model Visualization Demos

This script demonstrates the Healer model's ability to correct various transformations:
- Gaussian noise removal using Wiener deconvolution
- Rotation correction
- Affine transformation correction
- Comparison of different denoising methods
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from torchvision import transforms

sys.path.append(str(Path(__file__).parent))

from models.healer_transforms import HealerTransforms
from data.continuous_transforms import ContinuousTransforms
from data.data_loader import DataLoaderFactory
from config.config_loader import ConfigLoader
from models.model_factory import ModelFactory
from utils.transformer_utils import set_seed


def load_sample_images(config_path: str = 'config/cifar10_config.yaml', 
                      num_images: int = 1) -> Tuple[torch.Tensor, List[int]]:
    """Load sample images from CIFAR-10 or TinyImageNet"""
    config = ConfigLoader(config_path)
    # Temporarily override batch size in config
    config.config['training']['batch_size'] = num_images
    data_factory = DataLoaderFactory(config)
    
    dataset_name = 'cifar10' if 'cifar10' in config_path else 'tinyimagenet'
    _, val_loader = data_factory.create_data_loaders(dataset_name, with_augmentation=False)
    
    for batch in val_loader:
        if dataset_name == 'tinyimagenet' and len(batch) == 4:
            # Handle TinyImageNet OOD loader format
            images, _, labels, _ = batch
        else:
            images, labels = batch
        return images, labels.tolist()


def apply_transformations(images: torch.Tensor, 
                         transform_type: str,
                         severity: float = 0.5) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """Apply transformations using ContinuousTransforms class"""
    transformed_images = []
    params_list = []
    
    # Create transform instance with the specified severity
    continuous_transform = ContinuousTransforms(severity=severity)
    
    for img in images:
        # Apply transformation and get parameters
        transformed, params = continuous_transform.apply_transforms(
            img, 
            transform_type=transform_type,
            severity=severity,
            return_params=True
        )
        transformed_images.append(transformed)
        
        # Extract relevant parameters based on transform type
        if transform_type == 'gaussian_noise':
            params_list.append({'noise_std': params['noise_std']})
        elif transform_type == 'rotation':
            params_list.append({'rotation_angle': params['rotation_angle']})
        elif transform_type == 'affine':
            params_list.append({
                'translate_x': params['translate_x'],
                'translate_y': params['translate_y'],
                'shear_x': params['shear_x'],
                'shear_y': params['shear_y']
            })
        else:
            params_list.append({})
    
    return torch.stack(transformed_images), params_list


def visualize_healer_corrections(images: torch.Tensor,
                               transform_type: str,
                               severity: float = 0.5,
                               save_path: Path = None):
    """Visualize Healer corrections for different transformations"""
    
    # Apply transformations with fixed parameters
    transformed_images, params_list = apply_transformations(images, transform_type, severity)
    
    # Create mock predictions for Healer with exact parameters
    batch_size = images.shape[0]
    
    # Map transform types
    type_map = {
        'gaussian_noise': 1,
        'rotation': 2, 
        'affine': 3,
        'no_transform': 0
    }
    t_type = type_map.get(transform_type, 0)
    
    # Create predictions for the batch with exact parameters from first image
    transform_logits = torch.zeros(batch_size, 4)
    transform_logits[:, t_type] = 10.0  # High confidence
    
    # Use exact parameters from the transformation
    params = params_list[0] if params_list else {}
    
    predictions = {
        'transform_type_logits': transform_logits,
        'noise_std': torch.full((batch_size, 1), params.get('noise_std', 0.0)),
        'rotation_angle': torch.full((batch_size, 1), params.get('rotation_angle', 0.0)),
        'translate_x': torch.full((batch_size, 1), params.get('translate_x', 0.0)),
        'translate_y': torch.full((batch_size, 1), params.get('translate_y', 0.0)),
        'shear_x': torch.full((batch_size, 1), params.get('shear_x', 0.0)),
        'shear_y': torch.full((batch_size, 1), params.get('shear_y', 0.0))
    }
    
    # Apply Healer corrections
    corrected_images = HealerTransforms.apply_batch_correction(transformed_images, predictions)
    
    # Ensure corrected images are in valid range
    corrected_images = torch.clamp(corrected_images, 0, 1)
    
    # Create visualization for single image
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    img_orig = images[0].permute(1, 2, 0).cpu().numpy()
    img_orig = np.clip(img_orig, 0, 1)
    axes[0].imshow(img_orig)
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    
    # Transformed image
    img_trans = transformed_images[0].permute(1, 2, 0).cpu().numpy()
    img_trans = np.clip(img_trans, 0, 1)
    axes[1].imshow(img_trans)
    axes[1].set_title(f'{transform_type.replace("_", " ").title()}\n(severity={severity})', fontsize=14)
    axes[1].axis('off')
    
    # Corrected image
    img_corr = corrected_images[0].permute(1, 2, 0).cpu().numpy()
    img_corr = np.clip(img_corr, 0, 1)
    axes[2].imshow(img_corr)
    axes[2].set_title('Healer Corrected', fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle(f'Healer Correction Demo: {transform_type.replace("_", " ").title()}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def compare_denoising_methods(image: torch.Tensor,
                            noise_std: float = 0.2,
                            save_path: Path = None):
    """Compare different denoising methods available in Healer"""
    
    # Add noise to image
    noise = torch.randn_like(image) * noise_std
    noisy_image = torch.clamp(image + noise, 0, 1)
    
    # Apply different denoising methods
    methods = ['wiener', 'bilateral', 'nlm', 'gaussian']
    denoised_images = {}
    
    for method in methods:
        denoised = HealerTransforms.apply_wiener_denoising(
            noisy_image, noise_std, method=method
        )
        denoised_images[method] = denoised
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original image
    img_orig = image.permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(np.clip(img_orig, 0, 1))
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')
    
    # Noisy image
    img_noisy = noisy_image.permute(1, 2, 0).cpu().numpy()
    axes[1].imshow(np.clip(img_noisy, 0, 1))
    axes[1].set_title(f'Noisy (σ={noise_std})', fontsize=12)
    axes[1].axis('off')
    
    # Denoised images
    for i, (method, denoised) in enumerate(denoised_images.items()):
        img_denoised = denoised.permute(1, 2, 0).cpu().numpy()
        axes[i+2].imshow(np.clip(img_denoised, 0, 1))
        axes[i+2].set_title(f'{method.upper()} Denoising', fontsize=12)
        axes[i+2].axis('off')
    
    # Hide the last subplot
    axes[-1].axis('off')
    
    plt.suptitle('Comparison of Denoising Methods in Healer', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved denoising comparison to {save_path}")
    
    plt.show()


def visualize_severity_levels(image: torch.Tensor,
                            transform_type: str,
                            severities: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
                            save_path: Path = None):
    """Visualize Healer corrections across different severity levels"""
    
    fig, axes = plt.subplots(3, len(severities) + 1, figsize=(4*(len(severities)+1), 12))
    
    # Original image in first column
    img_orig = image.permute(1, 2, 0).cpu().numpy()
    for i in range(3):
        axes[i, 0].imshow(np.clip(img_orig, 0, 1))
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=12)
        axes[i, 0].axis('off')
    
    # Process each severity level
    for col, severity in enumerate(severities):
        # Apply transformation with specified severity
        transformed, params_list = apply_transformations(image.unsqueeze(0), transform_type, severity)
        transformed = transformed.squeeze(0)
        params = params_list[0] if params_list else {}
        
        # Create predictions for single image
        type_map = {
            'gaussian_noise': 1,
            'rotation': 2, 
            'affine': 3,
            'no_transform': 0
        }
        t_type = type_map.get(transform_type, 0)
        
        transform_logits = torch.zeros(1, 4)
        transform_logits[0, t_type] = 10.0
        
        predictions = {
            'transform_type_logits': transform_logits,
            'noise_std': torch.tensor([[params.get('noise_std', 0.0)]]),
            'rotation_angle': torch.tensor([[params.get('rotation_angle', 0.0)]]),
            'translate_x': torch.tensor([[params.get('translate_x', 0.0)]]),
            'translate_y': torch.tensor([[params.get('translate_y', 0.0)]]),
            'shear_x': torch.tensor([[params.get('shear_x', 0.0)]]),
            'shear_y': torch.tensor([[params.get('shear_y', 0.0)]])
        }
        
        corrected = HealerTransforms.apply_batch_correction(
            transformed.unsqueeze(0), predictions
        ).squeeze(0)
        
        # Display transformed image
        img_trans = transformed.permute(1, 2, 0).cpu().numpy()
        axes[1, col+1].imshow(np.clip(img_trans, 0, 1))
        axes[1, col+1].set_title(f'Severity={severity}', fontsize=12)
        axes[1, col+1].axis('off')
        
        # Display corrected image
        img_corr = corrected.permute(1, 2, 0).cpu().numpy()
        axes[2, col+1].imshow(np.clip(img_corr, 0, 1))
        axes[2, col+1].axis('off')
        
        # Empty top row for spacing
        axes[0, col+1].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.75, 'Original', rotation=90, va='center', fontsize=14, weight='bold')
    fig.text(0.02, 0.5, 'Transformed', rotation=90, va='center', fontsize=14, weight='bold')
    fig.text(0.02, 0.25, 'Corrected', rotation=90, va='center', fontsize=14, weight='bold')
    
    plt.suptitle(f'Healer Corrections Across Severity Levels: {transform_type.replace("_", " ").title()}', 
                 fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved severity comparison to {save_path}")
    
    plt.show()


def demo_trained_healer(config_path: str = 'config/cifar10_config.yaml',
                       checkpoint_path: str = None,
                       save_dir: Path = None):
    """Demo using a trained Healer model"""
    
    config = ConfigLoader(config_path)
    model_factory = ModelFactory(config)
    
    dataset_name = 'cifar10' if 'cifar10' in config_path else 'tinyimagenet'
    
    # Load or create Healer model
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading trained Healer from {checkpoint_path}")
        healer = model_factory.load_model_from_checkpoint(
            Path(checkpoint_path), 'healer', dataset_name
        )
    else:
        print("Creating new Healer model (untrained)")
        healer = model_factory.create_model('healer', dataset_name)
    
    healer.eval()
    
    # Load sample images
    images, labels = load_sample_images(config_path, num_images=1)
    
    # Test each transformation type
    transform_types = ['gaussian_noise', 'rotation', 'affine']
    
    for transform_type in transform_types:
        # Apply transformation
        transformed, _ = apply_transformations(images, transform_type, severity=0.5)
        
        # Get Healer predictions
        with torch.no_grad():
            predictions = healer(transformed, training_mode=True)
        
        # Apply corrections
        corrected = HealerTransforms.apply_batch_correction(transformed, predictions)
        
        # Visualize results for single image
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original
        img_orig = images[0].permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(np.clip(img_orig, 0, 1))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Transformed
        img_trans = transformed[0].permute(1, 2, 0).cpu().numpy()
        axes[1].imshow(np.clip(img_trans, 0, 1))
        axes[1].set_title(f'{transform_type.replace("_", " ").title()}')
        axes[1].axis('off')
        
        # Corrected
        img_corr = corrected[0].permute(1, 2, 0).cpu().numpy()
        axes[2].imshow(np.clip(img_corr, 0, 1))
        axes[2].set_title('Healer Corrected')
        axes[2].axis('off')
        
        plt.suptitle(f'Trained Healer Demo: {transform_type.replace("_", " ").title()}', fontsize=16)
        plt.tight_layout()
        
        if save_dir:
            save_path = save_dir / f'trained_healer_{transform_type}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()


def main():
    """Run all Healer visualization demos"""
    import argparse
    parser = argparse.ArgumentParser(description='Healer Visualization Demos')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'tinyimagenet'], 
                        default='cifar10', help='Dataset to use')
    args = parser.parse_args()
    
    set_seed(42)
    
    # Create output directory
    vis_dir = Path(f'./healer_visualizations_{args.dataset}')
    vis_dir.mkdir(exist_ok=True)
    
    print(f"Loading sample image from {args.dataset.upper()}...")
    config_path = f'config/{args.dataset}_config.yaml'
    images, labels = load_sample_images(config_path, num_images=1)
    
    print("\n1. Demonstrating Healer corrections for different transformations...")
    for transform_type in ['gaussian_noise', 'rotation', 'affine']:
        print(f"\n   Processing {transform_type}...")
        visualize_healer_corrections(
            images, 
            transform_type, 
            severity=0.5,
            save_path=vis_dir / f'healer_{transform_type}_correction.png'
        )
    
    print("\n2. Comparing denoising methods...")
    compare_denoising_methods(
        images[0], 
        noise_std=0.2,
        save_path=vis_dir / f'denoising_methods_comparison.png'
    )
    
    print("\n3. Visualizing corrections across severity levels...")
    for transform_type in ['gaussian_noise', 'rotation', 'affine']:
        print(f"\n   Processing {transform_type} severity levels...")
        visualize_severity_levels(
            images[0],
            transform_type,
            severities=[0.1, 0.3, 0.5, 0.7, 0.9],
            save_path=vis_dir / f'healer_{transform_type}_severities.png'
        )
    
    print("\n4. Demo with trained Healer model...")
    # Check if trained model exists
    checkpoint_base = Path(f'./checkpoints/{args.dataset}/bestmodel_healer/best_model.pt')
    if checkpoint_base.exists():
        demo_trained_healer(
            config_path=config_path,
            checkpoint_path=str(checkpoint_base),
            save_dir=vis_dir
        )
    else:
        # Try alternative checkpoint location
        alt_checkpoint = Path(f'../../../{args.dataset}checkpointsrendufunky4/bestmodel_healer/best_model.pt')
        if alt_checkpoint.exists():
            demo_trained_healer(
                config_path=config_path,
                checkpoint_path=str(alt_checkpoint),
                save_dir=vis_dir
            )
        else:
            print(f"   No trained Healer model found for {args.dataset}. Train a Healer model first to see this demo.")
    
    print(f"\nAll visualizations saved to {vis_dir}/")
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
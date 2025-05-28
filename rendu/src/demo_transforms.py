#!/usr/bin/env python3
"""
Interactive demo for visualizing transformations and healer corrections
"""
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config_loader import ConfigLoader
from src.data.data_loader import DataLoaderFactory
from src.data.continuous_transforms import ContinuousTransforms
from src.models.model_factory import ModelFactory


def run_demo(dataset='cifar10', severity=0.5, debug=True, show_healer=True):
    """
    Run interactive demo
    
    Args:
        dataset: 'cifar10' or 'tinyimagenet'
        severity: Transformation severity (0.0 to 1.0)
        debug: Use debug mode
        show_healer: Show healer corrections
    """
    print(f"\n🎨 Transform & Healer Demo - {dataset.upper()}")
    print("=" * 50)
    
    # Load configuration
    config = ConfigLoader()
    device = torch.device(config.get_device())
    
    # Create factories
    data_factory = DataLoaderFactory(config)
    model_factory = ModelFactory(config)
    
    # Load data
    print("📊 Loading dataset...")
    _, val_loader = data_factory.create_data_loaders(
        dataset, 
        with_normalization=False,
        with_augmentation=False
    )
    
    # Load healer model
    healer_model = None
    if show_healer:
        print("🔮 Loading healer model...")
        checkpoint_dir = config.get_checkpoint_dir(dataset, use_debug_dir=debug)
        healer_path = checkpoint_dir / "bestmodel_healer" / "best_model.pt"
        
        if healer_path.exists():
            healer_model = model_factory.load_model_from_checkpoint(
                healer_path, 'healer', dataset, device=device
            )
            healer_model.eval()
            print("✅ Healer model loaded successfully!")
        else:
            print("❌ Healer model not found!")
            show_healer = False
    
    # Get normalization
    normalize = data_factory.get_normalization_transform(dataset)
    
    # Create transforms
    transform_engine = ContinuousTransforms(severity=severity)
    
    # Get sample images
    images, labels = next(iter(val_loader))
    
    # Dataset-specific denormalization
    if dataset == 'cifar10':
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        class_names = None  # Too many classes for TinyImageNet
    
    # Process first 3 images
    for img_idx in range(min(3, len(images))):
        print(f"\n📸 Processing image {img_idx + 1}...")
        
        original = images[img_idx]
        label = labels[img_idx].item()
        
        # Create visualization
        if show_healer and healer_model:
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.2)
        else:
            fig = plt.figure(figsize=(20, 5))
            gs = fig.add_gridspec(1, 5, hspace=0.3, wspace=0.2)
        
        # Original image
        ax = fig.add_subplot(gs[0, 0])
        show_tensor_image(ax, original, "Original")
        if class_names:
            ax.text(0.5, -0.15, f"Class: {class_names[label]}", 
                   transform=ax.transAxes, ha='center', fontsize=10)
        else:
            ax.text(0.5, -0.15, f"Class ID: {label}", 
                   transform=ax.transAxes, ha='center', fontsize=10)
        
        # Apply transformations
        transforms_to_show = [
            ('no_transform', 'No Transform'),
            ('gaussian_noise', 'Gaussian Noise'),
            ('rotation', 'Rotation'),
            ('affine', 'Affine Transform')
        ]
        
        for t_idx, (transform_type, display_name) in enumerate(transforms_to_show[1:], 1):
            # Apply transformation
            transformed, params = transform_engine.apply_transforms(
                original, 
                transform_type=transform_type,
                severity=severity,
                return_params=True
            )
            
            # Show transformed image
            ax = fig.add_subplot(gs[0, t_idx])
            show_tensor_image(ax, transformed, display_name)
            add_transform_info(ax, transform_type, params)
            
            # Apply healer if available
            if show_healer and healer_model:
                # Normalize for healer
                normalized = normalize(transformed).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # Get predictions
                    predictions, _ = healer_model(normalized, return_reconstruction=False, return_logits=False)
                    
                    # Apply correction
                    corrected = healer_model.apply_correction(normalized, predictions)
                    corrected = corrected[0].cpu()
                
                # Denormalize for display
                corrected_denorm = denormalize_tensor(corrected, mean, std)
                
                # Show corrected image
                ax = fig.add_subplot(gs[1, t_idx])
                show_tensor_image(ax, corrected_denorm, "Healer Corrected")
                
                # Add prediction info
                if 'transform_type_logits' in predictions:
                    pred_idx = torch.argmax(predictions['transform_type_logits'], dim=1).item()
                    pred_type = transform_engine.transform_types[pred_idx]
                    conf = torch.softmax(predictions['transform_type_logits'], dim=1)[0, pred_idx].item()
                    ax.text(0.5, -0.1, f"Predicted: {pred_type}\nConfidence: {conf:.2%}", 
                           transform=ax.transAxes, ha='center', fontsize=9,
                           color='green' if pred_type == transform_type else 'red')
        
        # Add reference image in bottom left if using healer
        if show_healer and healer_model:
            ax = fig.add_subplot(gs[1, 0])
            show_tensor_image(ax, original, "Original (Reference)")
        
        plt.suptitle(f"{dataset.upper()} Transformation Demo - Severity: {severity}", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    print("\n✨ Demo completed!")


def show_tensor_image(ax, tensor, title):
    """Display tensor as image"""
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(title, fontsize=12, pad=10)
    ax.axis('off')


def add_transform_info(ax, transform_type, params):
    """Add transformation parameter info"""
    info_text = ""
    if transform_type == 'gaussian_noise':
        info_text = f"σ = {params['noise_std']:.3f}"
    elif transform_type == 'rotation':
        info_text = f"θ = {params['rotation_angle']:.1f}°"
    elif transform_type == 'affine':
        info_text = f"tx={params['translate_x']:.2f}, ty={params['translate_y']:.2f}\nsx={params['shear_x']:.1f}°, sy={params['shear_y']:.1f}°"
    
    if info_text:
        ax.text(0.5, -0.1, info_text, transform=ax.transAxes, 
               ha='center', fontsize=9, color='blue')


def denormalize_tensor(tensor, mean, std):
    """Denormalize tensor for visualization"""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    denormalized = tensor * std_t + mean_t
    return torch.clamp(denormalized, 0, 1)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive transformation and healer demo')
    parser.add_argument('--dataset', choices=['cifar10', 'tinyimagenet'], 
                       default='cifar10', help='Dataset to use')
    parser.add_argument('--severity', type=float, default=0.5,
                       help='Transformation severity (0.0-1.0)')
    parser.add_argument('--no-healer', action='store_true',
                       help='Disable healer corrections')
    parser.add_argument('--no-debug', action='store_true',
                       help='Use full model checkpoints instead of debug')
    
    args = parser.parse_args()
    
    run_demo(
        dataset=args.dataset,
        severity=args.severity,
        debug=not args.no_debug,
        show_healer=not args.no_healer
    )
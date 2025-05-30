"""
Visualization script for viewing transformations and healer corrections
"""
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from src.config.config_loader import ConfigLoader
from src.data.data_loader import DataLoaderFactory
from src.data.continuous_transforms import ContinuousTransforms
from src.models.model_factory import ModelFactory


def visualize_transform_and_healing(args):
    """
    Main visualization function
    """
    config = ConfigLoader()
    device = torch.device(config.get_device())
    
    data_factory = DataLoaderFactory(config)
    model_factory = ModelFactory(config)
    
    print(f"Loading {args.dataset} dataset...")
    train_loader, val_loader = data_factory.create_data_loaders(
        args.dataset, 
        with_normalization=False,  # We'll normalize manually after transforms
        with_augmentation=False
    )
    
    data_loader = val_loader
    
    # Load healer model if requested
    healer_model = None
    if args.use_healer:
        checkpoint_dir = config.get_checkpoint_dir(args.dataset, use_debug_dir=args.debug)
        healer_path = checkpoint_dir / "bestmodel_healer" / "best_model.pt"
        
        if healer_path.exists():
            print(f"Loading healer model from {healer_path}")
            healer_model = model_factory.load_model_from_checkpoint(
                healer_path, 'healer', args.dataset, device=device
            )
            healer_model.eval()
        else:
            print(f"Healer model not found at {healer_path}")
            args.use_healer = False
    
    normalize = data_factory.get_normalization_transform(args.dataset)
    transforms_for_robustness = ContinuousTransforms(severity=args.severity)
    
    images_batch, labels_batch = next(iter(data_loader))
    
    num_images = min(args.num_images, len(images_batch))
    
    # Process each image
    for img_idx in range(num_images):
        original_img = images_batch[img_idx]
        label = labels_batch[img_idx].item()
        
        fig_rows = 2 if args.use_healer else 1
        fig, axes = plt.subplots(fig_rows, 4, figsize=(16, 4 * fig_rows))
        if fig_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Show original image
        ax = axes[0, 0]
        show_image(ax, original_img, f"Original\nLabel: {label}")
        
        # Apply different transformations
        transform_types = ['gaussian_noise', 'rotation', 'affine']
        
        for t_idx, transform_type in enumerate(transform_types):
            transformed_img, transform_params = transforms_for_robustness.apply_transforms(
                original_img, 
                transform_type=transform_type,
                severity=args.severity,
                return_params=True
            )
            
            # Show transformed image
            ax = axes[0, t_idx + 1]
            title = get_transform_title(transform_type, transform_params)
            show_image(ax, transformed_img, title)
            
            # Apply healer correction if available
            if args.use_healer and healer_model is not None:
                normalized_img = normalize(transformed_img)
                normalized_batch = normalized_img.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    predictions, _ = healer_model(normalized_batch, return_reconstruction=False, return_logits=False)
                    corrected_batch = healer_model.apply_correction(normalized_batch, predictions)
                    corrected_img = corrected_batch[0].cpu()
                
                denormalized_img = denormalize_image(corrected_img, args.dataset)
                
                ax = axes[1, t_idx + 1]
                
                if 'transform_type_logits' in predictions:
                    pred_transform_idx = torch.argmax(predictions['transform_type_logits'], dim=1).item()
                    pred_transform = transforms_for_robustness.transform_types[pred_transform_idx]
                    confidence = torch.softmax(predictions['transform_type_logits'], dim=1)[0, pred_transform_idx].item()
                    pred_text = f"Pred: {pred_transform}\n(conf: {confidence:.2f})"
                else:
                    pred_text = "Healer Corrected"
                
                show_image(ax, denormalized_img, f"Healer Corrected\n{pred_text}")
        
        # If using healer, show original in bottom left for reference
        if args.use_healer:
            ax = axes[1, 0]
            show_image(ax, original_img, "Original (Reference)")
        
        plt.suptitle(f"{args.dataset.upper()} - Image {img_idx + 1} - Severity: {args.severity}", fontsize=16)
        plt.tight_layout()
        
        if args.save_path:
            save_path = Path(args.save_path) / f"{args.dataset}_transform_demo_{img_idx + 1}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def show_image(ax, img_tensor, title):
    """Display image on axis"""
    if img_tensor.dim() == 3:
        img_np = img_tensor.permute(1, 2, 0).numpy()
    else:
        img_np = img_tensor.numpy()
    
    img_np = np.clip(img_np, 0, 1)
    
    ax.imshow(img_np)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def get_transform_title(transform_type, transform_params):
    """Generate descriptive title for transformation"""
    severity = transform_params.get('severity', 0)
    
    if transform_type == 'gaussian_noise':
        noise_std = transform_params.get('noise_std', 0)
        return f"Gaussian Noise\nstd={noise_std:.3f}"
    elif transform_type == 'rotation':
        angle = transform_params.get('rotation_angle', 0)
        return f"Rotation\nangle={angle:.1f}°"
    elif transform_type == 'affine':
        tx = transform_params.get('translate_x', 0)
        ty = transform_params.get('translate_y', 0)
        sx = transform_params.get('shear_x', 0)
        sy = transform_params.get('shear_y', 0)
        return f"Affine\ntx={tx:.2f}, ty={ty:.2f}\nsx={sx:.1f}, sy={sy:.1f}"
    else:
        return transform_type


def denormalize_image(img_tensor, dataset_name):
    """Denormalize image for visualization"""
    if dataset_name.lower() == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    else:  # tinyimagenet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    denormalized = img_tensor * std + mean
    return torch.clamp(denormalized, 0, 1)


def main():
    parser = argparse.ArgumentParser(description='Visualize transformations and healer corrections')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'tinyimagenet'], 
                       required=True, help='Dataset to use')
    parser.add_argument('--severity', type=float, default=0.5,
                       help='Severity of transformations (0.0 to 1.0)')
    parser.add_argument('--num_images', type=int, default=3,
                       help='Number of images to visualize')
    parser.add_argument('--use_healer', action='store_true',
                       help='Apply healer corrections to transformed images')
    parser.add_argument('--debug', action='store_true',
                       help='Use debug mode (loads from debug checkpoint directory)')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save visualizations (if not provided, displays on screen)')
    
    args = parser.parse_args()
    
    visualize_transform_and_healing(args)


if __name__ == '__main__':
    main()
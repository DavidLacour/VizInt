#!/usr/bin/env python3
"""
Test script to generate transformation visualizations
"""
import sys
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.config.config_loader import ConfigLoader
from src.data.data_loader import DataLoaderFactory
from src.data.continuous_transforms import ContinuousTransforms
from src.models.model_factory import ModelFactory


def test_visualizations():
    """Generate test visualizations"""
    print("🎨 Generating transformation visualizations...")
    
    dataset = 'cifar10'
    severity = 0.5
    use_debug = True
    
    config = ConfigLoader()
    device = torch.device(config.get_device())
    
    data_factory = DataLoaderFactory(config)
    model_factory = ModelFactory(config)
    
    print(f"Loading {dataset} dataset...")
    _, val_loader = data_factory.create_data_loaders(
        dataset, 
        with_normalization=False,
        with_augmentation=False
    )
    
    print("Loading healer model...")
    checkpoint_dir = config.get_checkpoint_dir(dataset, use_debug_dir=use_debug)
    healer_path = checkpoint_dir / "bestmodel_healer" / "best_model.pt"
    
    healer_model = None
    if healer_path.exists():
        healer_model = model_factory.load_model_from_checkpoint(
            healer_path, 'healer', dataset, device=device
        )
        healer_model.eval()
        print("✅ Healer model loaded!")
    else:
        print("❌ Healer model not found!")
    
    normalize = data_factory.get_normalization_transform(dataset)
    
    transform_engine = ContinuousTransforms(severity=severity)
    
    images, labels = next(iter(val_loader))
    
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\nGenerating visualization...")
    original = images[0]
    label = labels[0].item()
    
    fig = plt.figure(figsize=(20, 10))
    
    ax = plt.subplot(2, 5, 1)
    show_image(ax, original, f"Original\nClass: {class_names[label]}")
    
    transform_types = ['gaussian_noise', 'rotation', 'affine']
    
    for idx, t_type in enumerate(transform_types):
        transformed, params = transform_engine.apply_transforms(
            original, transform_type=t_type, severity=severity, return_params=True
        )
        
        ax = plt.subplot(2, 5, idx + 2)
        show_image(ax, transformed, get_transform_title(t_type, params))
        
        if healer_model:
            normalized = normalize(transformed).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions, _ = healer_model(normalized, return_reconstruction=False, return_logits=False)
                corrected = healer_model.apply_correction(normalized, predictions)
                corrected = corrected[0].cpu()
            
            corrected_denorm = denormalize_tensor(corrected, mean, std)
            
            ax = plt.subplot(2, 5, idx + 7)
            
            if 'transform_type_logits' in predictions:
                pred_idx = torch.argmax(predictions['transform_type_logits'], dim=1).item()
                pred_type = transform_engine.transform_types[pred_idx]
                conf = torch.softmax(predictions['transform_type_logits'], dim=1)[0, pred_idx].item()
                title = f"Healer Corrected\nPred: {pred_type} ({conf:.1%})"
            else:
                title = "Healer Corrected"
            
            show_image(ax, corrected_denorm, title)
    
    if healer_model:
        ax = plt.subplot(2, 5, 6)
        show_image(ax, original, "Original (Reference)")
    
    plt.suptitle(f"CIFAR-10 Transformation Demo - Severity: {severity}", fontsize=16)
    plt.tight_layout()
    
    output_path = Path("../../../visualizationsrendu/demos/")
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "transform_healer_demo.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    plt.close()


def show_image(ax, tensor, title):
    """Display tensor as image"""
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def get_transform_title(t_type, params):
    """Get descriptive title"""
    if t_type == 'gaussian_noise':
        return f"Gaussian Noise\nσ = {params['noise_std']:.3f}"
    elif t_type == 'rotation':
        return f"Rotation\nangle = {params['rotation_angle']:.1f}°"
    elif t_type == 'affine':
        return f"Affine Transform\ntx={params['translate_x']:.2f}, ty={params['translate_y']:.2f}"
    return t_type


def denormalize_tensor(tensor, mean, std):
    """Denormalize tensor"""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    denormalized = tensor * std_t + mean_t
    return torch.clamp(denormalized, 0, 1)


if __name__ == '__main__':
    test_visualizations()
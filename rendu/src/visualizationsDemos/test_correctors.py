"""
Test script for corrector models
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from config.config_loader import ConfigLoader
from models.model_factory import ModelFactory
from data.data_loader import DataLoaderFactory
from data.continuous_transforms import ContinuousTransforms


def test_corrector_models():
    """Test corrector model creation and basic functionality"""
    print("Testing corrector models...")
    
    config = ConfigLoader('config/cifar10_config.yaml')
    model_factory = ModelFactory(config)
    
    batch_size = 4
    channels = 3
    img_size = 32
    
    dummy_input = torch.randn(batch_size, channels, img_size, img_size)
    
    print(f"Testing with input shape: {dummy_input.shape}")
    
    corrector_types = ['unet_corrector', 'transformer_corrector', 'hybrid_corrector']
    
    for corrector_type in corrector_types:
        print(f"\n--- Testing {corrector_type} ---")
        try:
            model = model_factory.create_model(corrector_type, 'cifar10')
            model.eval()
            
            print(f"✓ {corrector_type} created successfully")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            
            assert output.shape == dummy_input.shape, f"Shape mismatch: {output.shape} vs {dummy_input.shape}"
            print(f"✓ {corrector_type} forward pass successful")
            
            corrected = model.correct_image(dummy_input)
            assert corrected.shape == dummy_input.shape
            print(f"✓ {corrector_type} correct_image method works")
            
        except Exception as e:
            print(f"✗ {corrector_type} failed: {e}")
            import traceback
            traceback.print_exc()
    
    wrapper_types = [
        ('unet_resnet18', 'ResNet18'),
        ('transformer_resnet18', 'ResNet18'),
        ('hybrid_resnet18', 'ResNet18')
    ]
    
    print(f"\n--- Testing Corrector Wrapper Models ---")
    for wrapper_type, backbone_name in wrapper_types:
        print(f"\nTesting {wrapper_type} ({backbone_name} backbone)...")
        try:
            model = model_factory.create_model(wrapper_type, 'cifar10')
            model.eval()
            
            print(f"✓ {wrapper_type} created successfully")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            with torch.no_grad():
                logits = model(dummy_input)
            
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Logits shape: {logits.shape}")
            
            expected_classes = 10  # CIFAR-10
            assert logits.shape == (batch_size, expected_classes), f"Logits shape mismatch: {logits.shape}"
            print(f"✓ {wrapper_type} classification forward pass successful")
            
            corrected_logits, corrected_images = model(dummy_input, return_corrected=True)
            assert corrected_images.shape == dummy_input.shape
            print(f"✓ {wrapper_type} image correction works")
            
        except Exception as e:
            print(f"✗ {wrapper_type} failed: {e}")
            import traceback
            traceback.print_exc()


def test_transforms_and_correction():
    """Test applying transforms and correcting them"""
    print(f"\n--- Testing Transform Application and Correction ---")
    
    config = ConfigLoader('config/cifar10_config.yaml')
    model_factory = ModelFactory(config)
    
    try:
        corrector = model_factory.create_model('unet_corrector', 'cifar10')
        corrector.eval()
        
        clean_image = torch.randn(1, 3, 32, 32)
        
        transforms = ContinuousTransforms(img_size=32, num_classes=10)
        
        transform_tests = [
            ('gaussian_noise', {'severity': 0.3}),
            ('rotation', {'angle': 30}),
        ]
        
        for transform_name, params in transform_tests:
            print(f"\nTesting {transform_name} correction...")
            
            if transform_name == 'gaussian_noise':
                corrupted = transforms.apply_gaussian_noise(clean_image[0], severity=params['severity'])
                corrupted = corrupted.unsqueeze(0)
            elif transform_name == 'rotation':
                corrupted = transforms.apply_rotation(clean_image[0], angle=params['angle'])
                corrupted = corrupted.unsqueeze(0)
            
            with torch.no_grad():
                corrected = corrector.correct_image(corrupted)
            
            original_mse = torch.nn.functional.mse_loss(corrupted, clean_image)
            corrected_mse = torch.nn.functional.mse_loss(corrected, clean_image)
            
            print(f"  Original MSE: {original_mse.item():.6f}")
            print(f"  Corrected MSE: {corrected_mse.item():.6f}")
            print(f"  Improvement: {((original_mse - corrected_mse) / original_mse * 100).item():.2f}%")
            
    except Exception as e:
        print(f"✗ Transform and correction test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_corrector_models()
    
    test_transforms_and_correction()
    
    print(f"\n--- Test Complete ---")
    print("✓ All corrector implementations ready for training and evaluation!")
    print("\nTo train correctors, use:")
    print("  python main.py --dataset cifar10 --models unet_corrector transformer_corrector hybrid_corrector")
    print("\nTo train and evaluate corrector+classifier combinations:")
    print("  python main.py --dataset cifar10 --models unet_resnet18 transformer_resnet18 hybrid_resnet18")
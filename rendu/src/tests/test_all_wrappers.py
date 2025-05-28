#!/usr/bin/env python3
"""Test script for all wrapped models with ResNet18 backbone"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.config_loader import ConfigLoader
from models.model_factory import ModelFactory

def test_all_wrapped_models():
    """Test creating and using all wrapped models"""
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'cifar10_config.yaml'
    config_loader = ConfigLoader(config_path)
    model_factory = ModelFactory(config_loader)
    
    # Test all wrapper types
    wrapper_models = {
        'blended_resnet18': 'BlendedWrapper with ResNet18',
        'ttt_resnet18': 'TTTWrapper with ResNet18', 
        'healer_resnet18': 'HealerWrapper with ResNet18'
    }
    
    dummy_input = torch.randn(2, 3, 32, 32)
    
    for model_type, description in wrapper_models.items():
        print(f"\nTesting {description}...")
        
        # Create model
        model = model_factory.create_model(model_type, 'cifar10')
        print(f"Created {description}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if hasattr(model, 'backbone'):
            backbone_params = sum(p.numel() for p in model.backbone.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Backbone parameters: {backbone_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(dummy_input)
            
            if isinstance(output, tuple):
                print(f"Output shape: {output[0].shape}")
                if isinstance(output[1], dict):
                    print(f"Additional outputs: {list(output[1].keys())}")
                else:
                    print(f"Additional output shape: {output[1].shape}")
            else:
                print(f"Output shape: {output.shape}")
        
        # Test specific wrapper functionality
        if model_type == 'ttt_resnet18':
            print("Testing TTT adaptation...")
            dummy_transform_labels = torch.randint(0, 4, (2,))
            adaptation_results = model.adapt_parameters(dummy_input, dummy_transform_labels)
            print(f"Adaptation loss: {adaptation_results['adaptation_loss']:.4f}")
            
        elif model_type == 'healer_resnet18':
            print("Testing healer preprocessing...")
            if hasattr(model, 'heal_input'):
                healed_input = model.heal_input(dummy_input)
                print(f"Healed input shape: {healed_input.shape}")
                print(f"Input range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
                print(f"Healed range: [{healed_input.min():.3f}, {healed_input.max():.3f}]")
        
        print(f"✓ {description} test passed")
    
    print("\n" + "="*60)
    print("All wrapped model tests passed successfully!")
    print("The following models are now available in the pipeline:")
    for model_type, description in wrapper_models.items():
        print(f"  - {model_type}: {description}")

if __name__ == "__main__":
    test_all_wrapped_models()
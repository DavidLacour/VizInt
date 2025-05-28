#!/usr/bin/env python3
"""Test script for wrapped models with ResNet18 backbone"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config_loader import ConfigLoader
from src.models.model_factory import ModelFactory

def test_wrapped_models():
    """Test creating and using wrapped models"""
    
    # Load configuration
    config_path = Path(__file__).parent / 'config' / 'cifar10_config.yaml'
    config_loader = ConfigLoader(config_path)
    model_factory = ModelFactory(config_loader)
    
    # Test BlendedWrapper with ResNet18
    print("Testing BlendedWrapper with ResNet18...")
    blended_resnet = model_factory.create_model('blended_resnet18', 'cifar10')
    print(f"Created BlendedWrapper with ResNet18 backbone")
    print(f"Number of parameters: {sum(p.numel() for p in blended_resnet.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 32, 32)
    output = blended_resnet(dummy_input)
    print(f"Output shape: {output.shape}")
    if hasattr(blended_resnet, 'predicted_transforms') and blended_resnet.predicted_transforms is not None:
        print(f"Predicted transforms: {blended_resnet.predicted_transforms}")
    print()
    
    # Test TTTWrapper with ResNet18
    print("Testing TTTWrapper with ResNet18...")
    ttt_resnet = model_factory.create_model('ttt_resnet18', 'cifar10')
    print(f"Created TTTWrapper with ResNet18 backbone")
    print(f"Number of parameters: {sum(p.numel() for p in ttt_resnet.parameters()):,}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in ttt_resnet.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    output = ttt_resnet(dummy_input)
    if isinstance(output, tuple):
        print(f"Output shape: {output[0].shape}")
        print(f"Transform predictions: {list(output[1].keys())}")
    else:
        print(f"Output shape: {output.shape}")
    if hasattr(ttt_resnet, 'predicted_transforms') and ttt_resnet.predicted_transforms is not None:
        print(f"Predicted transforms: {ttt_resnet.predicted_transforms}")
    
    # Test adaptation
    print("\nTesting TTT adaptation...")
    dummy_transform_labels = torch.randint(0, 4, (2,))  # Random transform labels
    adaptation_results = ttt_resnet.adapt_parameters(dummy_input, dummy_transform_labels)
    print(f"Adaptation loss: {adaptation_results['adaptation_loss']:.4f}")
    output_adapted = ttt_resnet(dummy_input)
    print("Adaptation completed successfully")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_wrapped_models()
#!/usr/bin/env python3
"""
Quick test to verify each architecture can be created and run a forward pass.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from experimental_vit import create_experimental_vit

def test_architecture(arch_name):
    """Test if an architecture can be created and do a forward pass"""
    print(f"\n{'='*50}")
    print(f"Testing: {arch_name}")
    print('='*50)
    
    try:
        # Create model
        model = create_experimental_vit(
            architecture_type=arch_name,
            img_size=224,
            num_classes=10
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        
        # Test with smaller image size for speed
        print(f"\nTesting with smaller image (32x32)...")
        model_small = create_experimental_vit(
            architecture_type=arch_name,
            img_size=32,
            num_classes=10
        )
        dummy_input_small = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output_small = model_small(dummy_input_small)
        print(f"✓ Small image forward pass successful")
        print(f"  Output shape: {output_small.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    architectures = ['fourier', 'elfatt', 'mamba', 'kan', 'hybrid', 'mixed']
    
    print("Quick Architecture Test")
    print("="*60)
    print(f"Testing {len(architectures)} architectures...")
    
    results = {}
    for arch in architectures:
        results[arch] = test_architecture(arch)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for arch, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{arch:10s}: {status}")
    
    # Count successes
    successes = sum(results.values())
    print(f"\nTotal: {successes}/{len(architectures)} architectures working")


if __name__ == "__main__":
    main()
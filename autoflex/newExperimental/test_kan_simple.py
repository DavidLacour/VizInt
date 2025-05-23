#!/usr/bin/env python3
"""
Simple test script for KAN with minimal parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Try to create a minimal working example
print("Testing KAN with minimal parameters...")

# Create tiny synthetic dataset
batch_size = 3
num_samples = 6
img_size = 32
num_classes = 10

# Generate random data
X_train = torch.randn(num_samples, 3, img_size, img_size)
y_train = torch.randint(0, num_classes, (num_samples,))

X_val = torch.randn(batch_size, 3, img_size, img_size)
y_val = torch.randint(0, num_classes, (batch_size,))

# Create datasets and loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"\nDataset created:")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples: {len(val_dataset)}")
print(f"  Batch size: {batch_size}")
print(f"  Image size: {img_size}x{img_size}")

# Try to import and test KAN
try:
    from kan_transformer import KANAttention, KANMLP
    print("\n✅ KAN modules imported successfully")
    
    # Test KAN attention with small dimensions
    dim = 64  # Small embedding dimension
    num_heads = 4
    seq_len = 9  # For 3x3 patches on 32x32 image
    
    kan_attn = KANAttention(dim=dim, num_heads=num_heads)
    test_input = torch.randn(batch_size, seq_len, dim)
    
    print(f"\nTesting KAN attention:")
    print(f"  Input shape: {test_input.shape}")
    
    try:
        output = kan_attn(test_input)
        print(f"  Output shape: {output.shape}")
        print("  ✅ KAN attention forward pass successful!")
    except Exception as e:
        print(f"  ❌ KAN attention error: {e}")
        
except ImportError as e:
    print(f"\n❌ Failed to import KAN modules: {e}")

# Try using experimental ViT with KAN
try:
    from experimental_vit import create_experimental_vit
    
    print("\n\nTesting full KAN model with small config:")
    model_config = {
        'img_size': img_size,
        'patch_size': 16,  # Larger patches for stability
        'in_chans': 3,
        'num_classes': num_classes,
        'embed_dim': 64,  # Small embedding
        'depth': 2,  # Only 2 layers
        'num_heads': 4,
        'mlp_ratio': 2.0,  # Smaller MLP
        'attention_type': 'kan',
        'mlp_type': 'standard'  # Use standard MLP to isolate KAN attention issues
    }
    
    print(f"Model config: {model_config}")
    
    model = create_experimental_vit(**model_config)
    model.eval()
    
    # Test forward pass
    with torch.no_grad():
        test_batch = torch.randn(batch_size, 3, img_size, img_size)
        print(f"\nTest input shape: {test_batch.shape}")
        
        try:
            output = model(test_batch)
            print(f"Model output shape: {output.shape}")
            print("✅ Model forward pass successful!")
        except Exception as e:
            print(f"❌ Model forward pass error: {e}")
            import traceback
            traceback.print_exc()
            
except Exception as e:
    print(f"\n❌ Failed to create model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Test completed!")
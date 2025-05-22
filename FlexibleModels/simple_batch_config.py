"""
Simple and reliable batch size configuration
"""

import torch

def get_batch_size(model_type: str, backbone_name: str, device: str = None) -> int:
    """
    Get a reliable batch size based on model type and backbone.
    Uses conservative, tested values that work on most systems.
    """
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Base batch sizes by model type
    base_batch_sizes = {
        'classification': 64,
        'healer': 32,
        'ttt': 32,
        'blended_ttt': 64
    }
    
    # Backbone adjustments (multiplicative factors)
    backbone_adjustments = {
        'vit_small': 1.0,
        'vit_base': 0.5,    # Larger model, reduce batch size
        'resnet18': 1.0,
        'resnet50': 1.0,
        'vgg16': 1.0,
        'deit_small': 1.0,
        'swin_small': 0.75   # Slightly reduce for swin
    }
    
    # Get base batch size
    base_size = base_batch_sizes.get(model_type, 64)
    
    # Apply backbone adjustment
    adjustment = backbone_adjustments.get(backbone_name, 1.0)
    adjusted_size = int(base_size * adjustment)
    
    # Ensure minimum batch size
    final_size = max(8, adjusted_size)
    
    # Reduce for CPU
    if device == 'cpu':
        final_size = min(final_size, 16)
    
    return final_size

def get_num_workers(device: str = None) -> int:
    """Get appropriate number of data loading workers"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        return 4  # Conservative number of workers for GPU
    else:
        return 2  # Fewer workers for CPU

# Tested batch size configurations
TESTED_BATCH_SIZES = {
    ('classification', 'vit_small'): 64,
    ('classification', 'vit_base'): 32,
    ('classification', 'resnet18'): 64,
    ('classification', 'resnet50'): 64,
    ('classification', 'vgg16'): 64,
    ('classification', 'deit_small'): 64,
    ('classification', 'swin_small'): 48,
    
    ('healer', 'vit_small'): 32,
    ('healer', 'vit_base'): 16,
    ('healer', 'resnet18'): 32,
    ('healer', 'resnet50'): 32,
    ('healer', 'vgg16'): 32,
    ('healer', 'deit_small'): 32,
    ('healer', 'swin_small'): 24,
    
    ('ttt', 'vit_small'): 32,
    ('ttt', 'vit_base'): 16,
    ('ttt', 'resnet18'): 32,
    ('ttt', 'resnet50'): 32,
    ('ttt', 'vgg16'): 32,
    ('ttt', 'deit_small'): 32,
    ('ttt', 'swin_small'): 24,
    
    ('blended_ttt', 'vit_small'): 64,
    ('blended_ttt', 'vit_base'): 32,
    ('blended_ttt', 'resnet18'): 64,
    ('blended_ttt', 'resnet50'): 64,
    ('blended_ttt', 'vgg16'): 64,
    ('blended_ttt', 'deit_small'): 64,
    ('blended_ttt', 'swin_small'): 48,
}

def get_tested_batch_size(model_type: str, backbone_name: str) -> int:
    """Get a tested and verified batch size for the combination"""
    key = (model_type, backbone_name)
    return TESTED_BATCH_SIZES.get(key, 32)  # Default to 32 if not found

if __name__ == "__main__":
    # Test the batch size function
    print("🔧 Batch Size Configuration Test")
    print("=" * 40)
    
    model_types = ['classification', 'healer', 'ttt', 'blended_ttt']
    backbones = ['vit_small', 'resnet50', 'vgg16']
    
    for model_type in model_types:
        print(f"\n{model_type.upper()}:")
        for backbone in backbones:
            batch_size = get_tested_batch_size(model_type, backbone)
            print(f"  {backbone:<12}: {batch_size:2d}")
    
    print(f"\nDevice: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Workers: {get_num_workers()}")

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
    ('classification', 'vit_small'): 200,
    ('classification', 'vit_base'): 200,
    ('classification', 'resnet18'): 200,
    ('classification', 'resnet50'): 200,
    ('classification', 'vgg16'): 200,
    ('classification', 'deit_small'): 200,
    ('classification', 'swin_small'): 200,
    
    ('healer', 'vit_small'): 200,
    ('healer', 'vit_base'): 200,
    ('healer', 'resnet18'): 200,
    ('healer', 'resnet50'): 200,
    ('healer', 'vgg16'): 200,
    ('healer', 'deit_small'): 200,
    ('healer', 'swin_small'): 200,
    
    ('ttt', 'vit_small'): 200,
    ('ttt', 'vit_base'): 200,
    ('ttt', 'resnet18'): 200,
    ('ttt', 'resnet50'): 200,
    ('ttt', 'vgg16'): 200,
    ('ttt', 'deit_small'): 200,
    ('ttt', 'swin_small'): 200,
    
    ('blended_ttt', 'vit_small'): 200,
    ('blended_ttt', 'vit_base'): 200,
    ('blended_ttt', 'resnet18'): 200,
    ('blended_ttt', 'resnet50'): 200,
    ('blended_ttt', 'vgg16'): 200,
    ('blended_ttt', 'deit_small'): 200,
    ('blended_ttt', 'swin_small'):200,
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

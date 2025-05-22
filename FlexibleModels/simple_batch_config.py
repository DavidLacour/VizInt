"""
Simple and reliable batch size configuration
Enhanced to support both pretrained and non-pretrained models
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
    # Include both pretrained and scratch versions
    backbone_adjustments = {
        'vit_small': 1.0,
        'vit_base': 0.5,    # Larger model, reduce batch size
        
        # ResNet models
        'resnet18': 1.0,
        'resnet18_pretrained': 1.0,
        'resnet18_scratch': 1.0,
        'resnet50': 1.0,
        'resnet50_pretrained': 1.0,
        'resnet50_scratch': 1.0,
        
        # VGG models
        'vgg16': 1.0,
        'vgg16_pretrained': 1.0,
        'vgg16_scratch': 1.0,
        
        # Timm models
        'deit_small': 1.0,
        'deit_small_pretrained': 1.0,
        'deit_small_scratch': 1.0,
        'swin_small': 0.75,   # Slightly reduce for swin
        'swin_small_pretrained': 0.75,
        'swin_small_scratch': 0.75,
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

# Enhanced tested batch size configurations including pretrained/scratch variants
TESTED_BATCH_SIZES = {
    # Custom ViT models
    ('classification', 'vit_small'): 200,
    ('classification', 'vit_base'): 200,
    ('healer', 'vit_small'): 200,
    ('healer', 'vit_base'): 200,
    ('ttt', 'vit_small'): 200,
    ('ttt', 'vit_base'): 200,
    ('blended_ttt', 'vit_small'): 200,
    ('blended_ttt', 'vit_base'): 200,
    
    # ResNet models - pretrained versions
    ('classification', 'resnet18_pretrained'): 200,
    ('classification', 'resnet50_pretrained'): 200,
    ('healer', 'resnet18_pretrained'): 200,
    ('healer', 'resnet50_pretrained'): 200,
    ('ttt', 'resnet18_pretrained'): 200,
    ('ttt', 'resnet50_pretrained'): 200,
    ('blended_ttt', 'resnet18_pretrained'): 200,
    ('blended_ttt', 'resnet50_pretrained'): 200,
    
    # ResNet models - scratch versions
    ('classification', 'resnet18_scratch'): 200,
    ('classification', 'resnet50_scratch'): 200,
    ('healer', 'resnet18_scratch'): 200,
    ('healer', 'resnet50_scratch'): 200,
    ('ttt', 'resnet18_scratch'): 200,
    ('ttt', 'resnet50_scratch'): 200,
    ('blended_ttt', 'resnet18_scratch'): 200,
    ('blended_ttt', 'resnet50_scratch'): 200,
    
    # VGG models - pretrained versions
    ('classification', 'vgg16_pretrained'): 200,
    ('healer', 'vgg16_pretrained'): 200,
    ('ttt', 'vgg16_pretrained'): 200,
    ('blended_ttt', 'vgg16_pretrained'): 200,
    
    # VGG models - scratch versions
    ('classification', 'vgg16_scratch'): 200,
    ('healer', 'vgg16_scratch'): 200,
    ('ttt', 'vgg16_scratch'): 200,
    ('blended_ttt', 'vgg16_scratch'): 200,
    
    # Timm models - pretrained versions
    ('classification', 'deit_small_pretrained'): 200,
    ('classification', 'swin_small_pretrained'): 200,
    ('healer', 'deit_small_pretrained'): 200,
    ('healer', 'swin_small_pretrained'): 200,
    ('ttt', 'deit_small_pretrained'): 200,
    ('ttt', 'swin_small_pretrained'): 200,
    ('blended_ttt', 'deit_small_pretrained'): 200,
    ('blended_ttt', 'swin_small_pretrained'): 200,
    
    # Timm models - scratch versions
    ('classification', 'deit_small_scratch'): 200,
    ('classification', 'swin_small_scratch'): 200,
    ('healer', 'deit_small_scratch'): 200,
    ('healer', 'swin_small_scratch'): 200,
    ('ttt', 'deit_small_scratch'): 200,
    ('ttt', 'swin_small_scratch'): 200,
    ('blended_ttt', 'deit_small_scratch'): 200,
    ('blended_ttt', 'swin_small_scratch'): 200,
    
    # Backward compatibility - original names (pretrained versions)
    ('classification', 'resnet18'): 200,
    ('classification', 'resnet50'): 200,
    ('classification', 'vgg16'): 200,
    ('classification', 'deit_small'): 200,
    ('classification', 'swin_small'): 200,
    ('healer', 'resnet18'): 200,
    ('healer', 'resnet50'): 200,
    ('healer', 'vgg16'): 200,
    ('healer', 'deit_small'): 200,
    ('healer', 'swin_small'): 200,
    ('ttt', 'resnet18'): 200,
    ('ttt', 'resnet50'): 200,
    ('ttt', 'vgg16'): 200,
    ('ttt', 'deit_small'): 200,
    ('ttt', 'swin_small'): 200,
    ('blended_ttt', 'resnet18'): 200,
    ('blended_ttt', 'resnet50'): 200,
    ('blended_ttt', 'vgg16'): 200,
    ('blended_ttt', 'deit_small'): 200,
    ('blended_ttt', 'swin_small'): 200,
}

def get_tested_batch_size(model_type: str, backbone_name: str) -> int:
    """Get a tested and verified batch size for the combination"""
    key = (model_type, backbone_name)
    return TESTED_BATCH_SIZES.get(key, 32)  # Default to 32 if not found

def get_pretrained_variants():
    """Get list of available pretrained vs scratch backbone variants"""
    pretrained_backbones = []
    scratch_backbones = []
    custom_backbones = []
    
    for backbone_name in TESTED_BATCH_SIZES.keys():
        if len(backbone_name) > 1:  # Skip model_type part
            backbone = backbone_name[1]
            if 'pretrained' in backbone:
                pretrained_backbones.append(backbone)
            elif 'scratch' in backbone:
                scratch_backbones.append(backbone)
            elif backbone in ['vit_small', 'vit_base']:
                custom_backbones.append(backbone)
    
    # Remove duplicates and sort
    pretrained_backbones = sorted(list(set(pretrained_backbones)))
    scratch_backbones = sorted(list(set(scratch_backbones)))
    custom_backbones = sorted(list(set(custom_backbones)))
    
    return {
        'pretrained': pretrained_backbones,
        'scratch': scratch_backbones,
        'custom': custom_backbones
    }

if __name__ == "__main__":
    # Test the batch size function
    print("🔧 Enhanced Batch Size Configuration Test")
    print("=" * 50)
    
    model_types = ['classification', 'healer', 'ttt', 'blended_ttt']
    
    # Test pretrained vs scratch comparison
    backbone_pairs = [
        ('resnet18_pretrained', 'resnet18_scratch'),
        ('resnet50_pretrained', 'resnet50_scratch'),
        ('vgg16_pretrained', 'vgg16_scratch'),
        ('deit_small_pretrained', 'deit_small_scratch'),
        ('swin_small_pretrained', 'swin_small_scratch'),
    ]
    
    print("\n📊 Pretrained vs Scratch Comparison:")
    for pretrained, scratch in backbone_pairs:
        print(f"\n{pretrained.replace('_pretrained', '').upper()}:")
        for model_type in model_types:
            pretrained_bs = get_tested_batch_size(model_type, pretrained)
            scratch_bs = get_tested_batch_size(model_type, scratch)
            print(f"  {model_type:<12}: Pretrained={pretrained_bs:3d}, Scratch={scratch_bs:3d}")
    
    print(f"\n💾 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"👥 Workers: {get_num_workers()}")
    
    # Show variant summary
    variants = get_pretrained_variants()
    print(f"\n📋 Available Variants:")
    print(f"  🏗️  Custom models: {len(variants['custom'])}")
    print(f"  ✅ Pretrained models: {len(variants['pretrained'])}")
    print(f"  🔨 Scratch models: {len(variants['scratch'])}")

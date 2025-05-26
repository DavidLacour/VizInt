# Baseline Models Training Summary

## Model Support by Script

| Script | VGG16 | VGG16 Pretrained | ResNet18 | ResNet18 Pretrained |
|--------|-------|------------------|----------|-------------------|
| **main_cifar10_all.py** | ❌ | ❌ | ✅ | ✅ |
| **main_baselines_3fc.py** | ✅* | ❌ | ✅* | ❌ |
| **main_baselines_3fc_integration.py** | ❌ | ❌ | ✅ | ✅ |

*Imported but not explicitly trained in the main function

## Details:

### main_cifar10_all.py (CIFAR-10)
- ✅ Trains ResNet18 baseline (custom implementation)
- ✅ Trains ResNet18 pretrained (torchvision model adapted for CIFAR-10)
- ❌ No VGG16 support

### main_baselines_3fc.py (Tiny ImageNet)
- ✅ Imports SimpleResNet18 and SimpleVGG16 from baseline_models
- ❌ But doesn't explicitly train them in the main pipeline
- Focus is on ViT, TTT3fc, and BlendedTTT3fc models

### main_baselines_3fc_integration.py (Tiny ImageNet)  
- ✅ Trains ResNet18 baseline (custom implementation)
- ✅ Trains ResNet18 pretrained (torchvision model)
- ❌ No VGG16 support

## Summary:
- **ResNet18** is well supported across all scripts
- **VGG16** is only imported in main_baselines_3fc.py but not actively used
- For comprehensive baseline comparisons, you would need to add VGG16 training to the scripts
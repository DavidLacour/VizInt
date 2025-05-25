"""
Check available ResNet models in the codebase and torchvision
"""

import torchvision.models as models

# Check available ResNet models in torchvision
print("=== Available ResNet models in torchvision ===")
resnet_models = [attr for attr in dir(models) if 'resnet' in attr.lower() and not attr.startswith('_')]
for model in sorted(resnet_models):
    print(f"- {model}")

print("\n=== ResNet models used in the codebase ===")
print("1. SimpleResNet18 (custom implementation)")
print("   - Location: baseline_models.py")
print("   - Based on: torchvision resnet18")
print("   - Modified for: Tiny ImageNet (64x64, 200 classes)")
print("   - Changes: Modified conv1 and removed maxpool")

print("\n2. Pretrained ResNet18")
print("   - Used in: main_baselines_3fc_integration.py, main_cifar10_all.py")
print("   - From: torchvision.models.resnet18(pretrained=True)")
print("   - Adapted for: Tiny ImageNet or CIFAR-10")

print("\n=== Summary ===")
print("Currently only ResNet18 is implemented in the codebase")
print("Other ResNet variants available in torchvision:")
other_resnets = ['resnet34', 'resnet50', 'resnet101', 'resnet152', 
                 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']
for model in other_resnets:
    if model in resnet_models:
        print(f"  - {model}")
# Pretrained vs Scratch Training Guide

This guide explains how to test both pretrained feature maps and non-pretrained (scratch) feature maps with your training system.

## Key Changes Made

### 1. Enhanced Backbone Configurations
- **Pretrained models**: `resnet50_pretrained`, `vgg16_pretrained`, `deit_small_pretrained`, etc.
- **Scratch models**: `resnet50_scratch`, `vgg16_scratch`, `deit_small_scratch`, etc.
- **Custom models**: `vit_small`, `vit_base` (always trained from scratch)

### 2. Automatic Training Adjustments
- **Scratch models** get higher learning rates, more epochs, and longer warmup
- **Pretrained models** use standard fine-tuning parameters
- Different weight decay strategies for each type

### 3. Easy Testing Interface
A new script `test_pretrained_vs_scratch.py` for convenient comparisons.

## Quick Start Examples

### 1. Compare ResNet50 Pretrained vs Scratch
```bash
python test_pretrained_vs_scratch.py --backbone resnet50 --model classification
```
This trains both `resnet50_pretrained` and `resnet50_scratch` with classification.

### 2. Test Multiple Backbones
```bash
python test_pretrained_vs_scratch.py --backbone resnet18 resnet50 deit_small --model classification
```

### 3. Only Train Pretrained Models
```bash
python test_pretrained_vs_scratch.py --backbone resnet50 --model classification --pretrained_only
```

### 4. Only Train From Scratch
```bash
python test_pretrained_vs_scratch.py --backbone resnet50 --model classification --scratch_only
```

### 5. Quick Test (Fewer Epochs)
```bash
python test_pretrained_vs_scratch.py --backbone resnet18 --model classification --epochs 10
```

### 6. See What Would Be Trained (Dry Run)
```bash
python test_pretrained_vs_scratch.py --backbone resnet50 --model classification --dry_run
```

## Using the Original Auto-Training System

You can also use the enhanced `auto_train_all.py` directly:

### Train Specific Backbones
```bash
python auto_train_all.py --backbones resnet50_pretrained resnet50_scratch vit_small
```

### Train Specific Models
```bash
python auto_train_all.py --models classification healer --backbones resnet18_pretrained resnet18_scratch
```

## Available Backbone Configurations

### Pretrained Models (with ImageNet weights)
- `resnet18_pretrained`
- `resnet50_pretrained`
- `vgg16_pretrained`
- `deit_small_pretrained`
- `swin_small_pretrained`

### Scratch Models (random initialization)
- `resnet18_scratch`
- `resnet50_scratch`
- `vgg16_scratch`
- `deit_small_scratch`
- `swin_small_scratch`

### Custom Models (always from scratch)
- `vit_small` - Custom ViT implementation
- `vit_base` - Custom ViT implementation

### Backward Compatibility
Original names still work and default to pretrained versions:
- `resnet18` → `resnet18_pretrained`
- `resnet50` → `resnet50_pretrained`
- `vgg16` → `vgg16_pretrained`

## Configuration Differences

The system automatically applies different training configurations:

### Scratch Models Get:
- **1.5x higher learning rate** (better for random initialization)
- **1.2x more epochs** (need more training from scratch)
- **1.5x longer warmup** (smoother start from random weights)
- **0.8x weight decay** (less regularization needed)

### Pretrained Models Get:
- **Standard learning rate** (fine-tuning rate)
- **Standard epochs** (transfer learning converges faster)
- **Standard warmup** (pretrained weights are already good)
- **Standard weight decay** (standard regularization)

## Monitoring and Analysis

### Check Training Progress
```bash
python training_monitor.py --progress
```

### Compare Model Performance
```bash
python training_monitor.py --compare
```

### View All Existing Models
```bash
python training_monitor.py --summary
```

### Generate Plots (requires matplotlib)
```bash
python training_monitor.py --plot
```

## Example Training Output

When you run the test script, you'll see output like:
```
🔧 Configuration Comparison:
🔍 Pretrained vs Scratch Comparison for classification
============================================================

📊 RESNET50:
  learning_rate  : Pretrained=0.0001   | Scratch=0.00015
  epochs         : Pretrained=50       | Scratch=60
  warmup_steps   : Pretrained=1000     | Scratch=1500
  weight_decay   : Pretrained=0.05     | Scratch=0.04

🚀 Starting training with 2 model configurations...
📋 Training Plan:
   1. classification   + resnet50_pretrained       (✅ PRETRAINED) - 50 epochs, LR=0.0001
   2. classification   + resnet50_scratch          (🔨 SCRATCH) - 60 epochs, LR=0.00015
```

## Model Directory Structure

After training, you'll get directories like:
```
bestmodel_resnet50_pretrained_classification/
bestmodel_resnet50_scratch_classification/
bestmodel_vit_small_classification/
...
```

## Tips for Experimentation

1. **Start small**: Use `--epochs 10` for quick tests
2. **Use dry run**: Always check with `--dry_run` first
3. **Monitor progress**: Use the training monitor to track results
4. **Compare systematically**: Train same architecture with both init strategies
5. **Check configs**: Use `python training_config.py` to see all configurations

## Research Questions You Can Answer

1. **Which initialization works better for your task?**
2. **How much does pretrained initialization help vs scratch?**
3. **Do different architectures benefit differently from pretraining?**
4. **What's the training time difference between pretrained and scratch?**
5. **How do the learned features differ between initialization strategies?**

The enhanced system makes it easy to systematically study these questions!

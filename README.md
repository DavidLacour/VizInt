# Vision Transformer Robustness Research Framework

This repository contains a comprehensive framework for training and evaluating robust vision models on CIFAR-10 and TinyImageNet datasets. The framework includes various model architectures, robustness techniques, and visualization tools.

The defauls commands shoud create folders in the directories above the VizInt git repo folders train everything and evaluate everything. 
It could take more than one day with a gaming laptop GPU. 

python main.py --dataset cifar10 

python main.py --dataset tinyimagenet 


## Models

### Base Models
- **VanillaViT**: Vision Transformer with configurable depth and embedding dimensions
- **ResNetBaseline**: ResNet18 trained from scratch
- **ResNetPretrained**: ResNet18 with ImageNet pretrained weights
- **ResNet18NotPretrainedRobust**: ResNet18 trained from scratch with continuous transforms

### Robustness-Enhanced Models
- **Healer**: Predicts and corrects transformations using Wiener deconvolution for noise and inverse transforms
- **TTT/TTT3fc**: Test-Time Training models that adapt during inference
- **BlendedTraining**: Predicts applied corruptions in the hope of getting a better feature map and more robust model.

### Corrector + Classifier Combinations
- **UNet + ResNet18/ViT**: UNet corrector preprocessing for classifiers
- **Transformer + ResNet18/ViT**: Transformer-based corrector
- **Hybrid + ResNet18/ViT**: Combined CNN and Transformer corrector

## Features

### Continuous Transforms
The framework applies various transformations with adjustable severity:
- **Gaussian Noise**: Additive noise with σ up to 0.5
- **Rotation**: Random rotations up to 360°
- **Affine**: Translation and shear transformations

### Robust Training
Based Models ending with `_robust` are trained with continuous transforms applied during training, making them more resilient to input perturbations, wrappers with robust means they contains a based model trained for robustness(with continuous transforms) experimental models are always trained for robustness as they need transformations augmentation to work.

### Healer Model Capabilities
- **Wiener Deconvolution**: Advanced frequency-domain denoising
- **Rotation Correction**: Inverse rotation based on predicted angles
- **Affine Correction**: Reverses translation and shear transformations
- **Transform Type Classification**: Identifies which transformation was applied

## Usage

### Command Line Arguments

Options:
  --dataset {cifar10,tinyimagenet}  Dataset to use (default: cifar10)
  --mode {train,evaluate,both}      Operation mode (default: both)
  --models MODEL [MODEL ...]        Models to train/evaluate (default: all)
  --skip_models MODEL [MODEL ...]   Models to skip
  --robust                          Train robust versions of TTT models
  --force_retrain                   Force retraining existing models
  --severities SEV [SEV ...]        Severity levels for evaluation
  --debug                           Enable debug mode with small dataset
  --device {cuda,cpu,auto}          Device to use
  --seed SEED                       Random seed
```

### Training Individual Models

```bash
# Train vanilla ViT
python main.py --dataset cifar10 --models vanilla_vit --mode train

# Train ResNet18 with robust training
python main.py --dataset cifar10 --models resnet18_not_pretrained_robust --mode train

# Train Healer model
python main.py --dataset cifar10 --models healer --mode train
```

### Evaluation

```bash
# Evaluate all models
python main.py --dataset cifar10 --mode evaluate

# Evaluate with custom severity levels
python main.py --dataset cifar10 --mode evaluate --severities 0.0 0.25 0.5 0.75 1.0

# Evaluate specific model
python main.py --dataset cifar10 --models resnet18_not_pretrained_robust --mode evaluate
```

## Visualization Demos

### Healer Visualization Demo
Demonstrates all Healer correction capabilities:

```bash
# CIFAR-10
python demo_healer_visualizations.py

# TinyImageNet
python demo_healer_visualizations.py --dataset tinyimagenet
```

This creates visualizations showing:
- Corrections for Gaussian noise, rotation, and affine transforms
- Comparison of different denoising methods
- Performance across severity levels
- Results with trained Healer models

### Wiener Denoising Demo
Focuses on the Wiener deconvolution method:

```bash
# CIFAR-10
python demo_healer_wiener.py

# TinyImageNet
python demo_healer_wiener.py --dataset tinyimagenet
```

Shows:
- Detailed denoising process with noise/residual maps
- PSNR metrics and improvements
- Performance across different noise levels

### Transform Visualization Demo
Visualizes the continuous transforms:

```bash
python demo_transforms.py
```

## Configuration

Configuration files are in YAML format:
- `config/cifar10_config.yaml`: CIFAR-10 specific settings
- `config/tinyimagenet_config.yaml`: TinyImageNet specific settings

Key configuration sections:
- **dataset**: Dataset paths and parameters
- **models**: Model-specific hyperparameters
- **training**: Training settings (epochs, learning rate, etc.)
- **evaluation**: Evaluation parameters and severity levels
- **model_combinations**: Defines which models to evaluate

### Adding New Models

1. Implement model in `models/` directory
2. Register in `model_factory.py`
3. Add to `all_models` list in `main.py`
4. Add to `all_model_types` in `model_evaluator.py`
5. Add configuration in YAML files

## Results

The evaluation produces:
- **Clean accuracy**: Performance on unmodified images
- **Robustness scores**: Performance at different transformation severities
- **Transform prediction accuracy**: For models that predict transformations
- **OOD performance**: Results on out-of-distribution transforms

Results are saved to:
- Checkpoints: `checkpoints/{dataset}/bestmodel_{model_name}/`
- Visualizations: `visualizations/{dataset}/`
- Logs: `experiment.log`

### Example Results Table
```
Model                     Clean    S0.3    S0.5    S0.7    S1.0
ResNet18_Pretrained       0.9064  0.5826  0.5533  0.5421  0.5398
ResNet18_Baseline         0.8636  0.5787  0.5392  0.5169  0.5071
ResNet18_NotPretrainedRobust  0.8521  0.7234  0.6891  0.6523  0.6102
VanillaViT_Robust         0.7395  0.5255  0.4774  0.4523  0.4540
```

## Advanced Features

### Test-Time Training (TTT)
TTT models adapt their parameters during inference by solving a self-supervised task (rotation prediction).

### Blended Training
Predict transforms augmentation corruputions in the hope of optaining a better feature map and a more robust model. 

### Corrector Models
Preprocessors that attempt to clean corrupted inputs before classification:
- **UNet**: CNN-based denoising
- **Transformer**: Attention-based correction
- **Hybrid**: Combines CNN and Transformer approaches

# 🏆 COMPREHENSIVE RESULTS - CIFAR-10

## Model Performance Results

| Model Combination | Description | Clean | S0.3 | S0.5 | S0.7 | S1.0 |
|---|---|---|---|---|---|---|
| ResNet18_Pretrained | ResNet18 (ImageNet pretrained) | 0.9064 | 0.5853 | 0.5555 | 0.5456 | 0.5415 |
| ResNet18_NotPretrainedRobust | ResNet18 (from scratch, robust training) | 0.8654 | 0.5616 | 0.5213 | 0.5132 | 0.5083 |
| ResNet18_Baseline | ResNet18 (from scratch) | 0.8636 | 0.5819 | 0.5434 | 0.5195 | 0.5050 |
| BlendedResNet18 | Blended wrapper with ResNet18 backbone | 0.8375 | 0.7654 | 0.7253 | 0.7026 | 0.6695 |
| Transformer+ResNet18 | Transformer corrector + ResNet18 classifier | 0.8263 | 0.5573 | 0.5079 | 0.4876 | 0.4845 |
| UNet+ResNet18 | UNet corrector + ResNet18 classifier | 0.8057 | 0.5602 | 0.5055 | 0.4879 | 0.4782 |
| Hybrid+ResNet18 | Hybrid corrector + ResNet18 classifier | 0.7935 | 0.5210 | 0.4809 | 0.4764 | 0.4542 |
| VanillaViT_Robust | Vanilla ViT (robust training) | 0.7395 | 0.5242 | 0.4690 | 0.4567 | 0.4581 |
| BlendedTraining | Blended Training (inherently robust) | 0.7328 | 0.4890 | 0.4753 | 0.4652 | 0.4567 |
| VanillaViT | Vanilla ViT (not robust) | 0.7319 | 0.5045 | 0.4823 | 0.4563 | 0.4456 |
| BlendedTraining3fc | Blended Training 3fc (inherently robust) | 0.7010 | 0.4773 | 0.4478 | 0.4354 | 0.4317 |
| UNet+ViT | UNet corrector + Vision Transformer | 0.6885 | 0.4516 | 0.4257 | 0.4161 | 0.4104 |
| Transformer+ViT | Transformer corrector + Vision Transformer | 0.6637 | 0.4881 | 0.4505 | 0.4276 | 0.4197 |
| HealerResNet18 | Healer wrapper with ResNet18 backbone | 0.6409 | 0.5151 | 0.4508 | 0.4132 | 0.3876 |
| Hybrid+ViT | Hybrid corrector + Vision Transformer | 0.6272 | 0.4058 | 0.3826 | 0.3814 | 0.3635 |
| Healer+VanillaViT_Robust | Healer + Vanilla ViT (robust) | 0.2886 | 0.2399 | 0.2440 | 0.2343 | 0.2286 |
| Healer+VanillaViT | Healer + Vanilla ViT (not robust) | 0.2304 | 0.1516 | 0.1461 | 0.1494 | 0.1477 |
| TTTResNet18 | TTT wrapper with ResNet18 backbone | 0.1023 | 0.1021 | 0.1013 | 0.0977 | 0.0979 |
| TTT | TTT (Test-Time Training) | 0.0946 | 0.1018 | 0.1021 | 0.0973 | 0.1003 |
| TTT3fc | TTT3fc (Test-Time Training with 3FC) | 0.0935 | 0.0994 | 0.1003 | 0.1045 | 0.0995 |

## 📊 ANALYSIS

- **🥇 Best Clean Data Performance:** ResNet18_Pretrained (0.9064)
- **🛡️ Most Transform Robust:** TTTResNet18 (0.4% drop)

## 📊 TRANSFORMATION ROBUSTNESS SUMMARY

| Model | Sev 0.0 | Sev 0.3 | Sev 0.5 | Sev 0.7 | Sev 1.0 | Avg Drop |
|---|---|---|---|---|---|---|
| ResNet18_Pretrained | 0.9064 | 0.5853 | 0.5555 | 0.5456 | 0.5415 | 0.3855 |
| ResNet18_NotPretrainedRobust | 0.8654 | 0.5616 | 0.5213 | 0.5132 | 0.5083 | 0.3921 |
| ResNet18_Baseline | 0.8636 | 0.5819 | 0.5434 | 0.5195 | 0.5050 | 0.3777 |
| BlendedResNet18 | 0.8375 | 0.7654 | 0.7253 | 0.7026 | 0.6695 | 0.1454 |
| Transformer+ResNet18 | 0.8263 | 0.5573 | 0.5079 | 0.4876 | 0.4845 | 0.3836 |
| UNet+ResNet18 | 0.8057 | 0.5602 | 0.5055 | 0.4879 | 0.4782 | 0.3696 |
| Hybrid+ResNet18 | 0.7935 | 0.5210 | 0.4809 | 0.4764 | 0.4542 | 0.3911 |
| VanillaViT_Robust | 0.7395 | 0.5242 | 0.4690 | 0.4567 | 0.4581 | 0.3550 |
| BlendedTraining | 0.7328 | 0.4890 | 0.4753 | 0.4652 | 0.4567 | 0.3565 |
| VanillaViT | 0.7319 | 0.5045 | 0.4823 | 0.4563 | 0.4456 | 0.3549 |
| BlendedTraining3fc | 0.7010 | 0.4773 | 0.4478 | 0.4354 | 0.4317 | 0.3608 |
| UNet+ViT | 0.6885 | 0.4516 | 0.4257 | 0.4161 | 0.4104 | 0.3813 |
| Transformer+ViT | 0.6637 | 0.4881 | 0.4505 | 0.4276 | 0.4197 | 0.3273 |
| HealerResNet18 | 0.6409 | 0.5151 | 0.4508 | 0.4132 | 0.3876 | 0.3109 |
| Hybrid+ViT | 0.6272 | 0.4058 | 0.3826 | 0.3814 | 0.3635 | 0.3888 |
| Healer+VanillaViT_Robust | 0.2886 | 0.2399 | 0.2440 | 0.2343 | 0.2286 | 0.1798 |
| Healer+VanillaViT | 0.2304 | 0.1516 | 0.1461 | 0.1494 | 0.1477 | 0.3546 |
| TTTResNet18 | 0.1023 | 0.1021 | 0.1013 | 0.0977 | 0.0979 | 0.0249 |
| TTT | 0.0946 | 0.1018 | 0.1021 | 0.0973 | 0.1003 | -0.0610 |
| TTT3fc | 0.0935 | 0.0994 | 0.1003 | 0.1045 | 0.0995 | -0.0794 |

## 🔍 HEALER GUIDANCE EVALUATION

### 🔍 Evaluating Healer+VanillaViT_Robust...
- **Severity 0.3:** Original: 0.5242, Healed: 0.2399, Improvement: -0.2843
- **Severity 0.5:** Original: 0.4690, Healed: 0.2440, Improvement: -0.2250
- **Severity 0.7:** Original: 0.4567, Healed: 0.2343, Improvement: -0.2224
- **Severity 1.0:** Original: 0.4581, Healed: 0.2286, Improvement: -0.2295

### 🔍 Evaluating Healer+VanillaViT...
- **Severity 0.3:** Original: 0.5242, Healed: 0.1516, Improvement: -0.3726
- **Severity 0.5:** Original: 0.4690, Healed: 0.1461, Improvement: -0.3229
- **Severity 0.7:** Original: 0.4567, Healed: 0.1494, Improvement: -0.3073
- **Severity 1.0:** Original: 0.4581, Healed: 0.1477, Improvement: -0.3104

## 🎯 TRANSFORMATION PREDICTION ACCURACY

| Model | Sev 0.3 | Sev 0.5 | Sev 0.7 | Sev 1.0 | Average |
|---|---|---|---|---|---|
| BlendedResNet18 | 0.9117 | 0.9393 | 0.9427 | 0.9398 | 0.9334 |
| BlendedTraining | 0.0983 | 0.0597 | 0.0540 | 0.0467 | 0.0647 |
| BlendedTraining3fc | 0.2330 | 0.2488 | 0.2503 | 0.2295 | 0.2404 |
| HealerResNet18 | 0.2135 | 0.1878 | 0.1745 | 0.1623 | 0.1845 |
| Healer+VanillaViT_Robust | 0.5775 | 0.5856 | 0.5669 | 0.5217 | 0.5629 |
| Healer+VanillaViT | 0.5848 | 0.5949 | 0.5655 | 0.5257 | 0.5677 |
| TTTResNet18 | 0.3633 | 0.3704 | 0.5288 | 0.5790 | 0.4604 |

## 📊 DETAILED TRANSFORM TYPE PREDICTION ACCURACY

### BlendedResNet18

| Transform Type | Sev 0.3 | Sev 0.5 | Sev 0.7 | Sev 1.0 |
|---|---|---|---|---|
| gaussian_noise | 0.9918 | 0.9988 | 0.9996 | 0.9988 |
| none | 0.8602 | 0.8707 | 0.8681 | 0.8658 |
| rotate | 0.9233 | 0.9414 | 0.9450 | 0.9419 |
| translate | 0.8746 | 0.9462 | 0.9589 | 0.9522 |

### BlendedTraining

| Transform Type | Sev 0.3 | Sev 0.5 | Sev 0.7 | Sev 1.0 |
|---|---|---|---|---|
| gaussian_noise | 0.2355 | 0.0956 | 0.0609 | 0.0559 |
| none | 0.1252 | 0.1193 | 0.1257 | 0.1085 |
| rotate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| translate | 0.0318 | 0.0227 | 0.0292 | 0.0211 |

### BlendedTraining3fc

| Transform Type | Sev 0.3 | Sev 0.5 | Sev 0.7 | Sev 1.0 |
|---|---|---|---|---|
| gaussian_noise | 0.2999 | 0.3652 | 0.3489 | 0.2459 |
| none | 0.4388 | 0.4360 | 0.4470 | 0.4452 |
| rotate | 0.0043 | 0.0008 | 0.0037 | 0.0037 |
| translate | 0.1952 | 0.1803 | 0.1996 | 0.2170 |

### HealerResNet18

| Transform Type | Sev 0.3 | Sev 0.5 | Sev 0.7 | Sev 1.0 |
|---|---|---|---|---|
| gaussian_noise | 0.7508 | 0.6442 | 0.5937 | 0.5144 |
| none | 0.0027 | 0.0024 | 0.0016 | 0.0041 |
| rotate | 0.0666 | 0.0613 | 0.0690 | 0.0674 |
| translate | 0.0319 | 0.0331 | 0.0429 | 0.0501 |

### Healer+VanillaViT_Robust

| Transform Type | Sev 0.3 | Sev 0.5 | Sev 0.7 | Sev 1.0 |
|---|---|---|---|---|
| gaussian_noise | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| none | 0.0671 | 0.0612 | 0.0578 | 0.0621 |
| rotate | 0.7408 | 0.7753 | 0.8061 | 0.7660 |
| translate | 0.5042 | 0.5084 | 0.4176 | 0.2555 |

### Healer+VanillaViT

| Transform Type | Sev 0.3 | Sev 0.5 | Sev 0.7 | Sev 1.0 |
|---|---|---|---|---|
| gaussian_noise | 0.9996 | 1.0000 | 1.0000 | 1.0000 |
| none | 0.0626 | 0.0645 | 0.0523 | 0.0567 |
| rotate | 0.7512 | 0.7923 | 0.8024 | 0.7831 |
| translate | 0.5235 | 0.5275 | 0.4218 | 0.2627 |

### TTTResNet18

| Transform Type | Sev 0.3 | Sev 0.5 | Sev 0.7 | Sev 1.0 |
|---|---|---|---|---|
| gaussian_noise | 0.0000 | 0.0052 | 0.6788 | 0.9948 |
| none | 0.0529 | 0.0515 | 0.0555 | 0.0500 |
| rotate | 0.5808 | 0.6580 | 0.6675 | 0.6616 |
| translate | 0.7908 | 0.7741 | 0.7241 | 0.6100 |

## 📏 PARAMETER PREDICTION ACCURACY (Mean Absolute Error)

### Healer+VanillaViT_Robust

| Parameter | Sev 0.3 | Sev 0.5 | Sev 0.7 | Sev 1.0 | Average |
|---|---|---|---|---|---|
| noise | 0.1371 | 0.0873 | 0.0162 | 0.1304 | 0.0928 |

### Healer+VanillaViT

| Parameter | Sev 0.3 | Sev 0.5 | Sev 0.7 | Sev 1.0 | Average |
|---|---|---|---|---|---|
| noise | 0.1367 | 0.0870 | 0.0164 | 0.1303 | 0.0926 |

## 🚀 OUT-OF-DISTRIBUTION (FUNKY TRANSFORMS) EVALUATION

This section evaluates model performance on extreme, funky transformations including color inversion, pixelation, extreme blur, masking, etc.

| Model Combination | Description | Funky OOD |
|---|---|---|
| BlendedResNet18 | Blended wrapper with ResNet18 backbone | 0.4761 |
| ResNet18_Pretrained | ResNet18 (ImageNet pretrained) | 0.4449 |
| ResNet18_NotPretrainedRobust | ResNet18 (from scratch, robust training) | 0.4303 |
| ResNet18_Baseline | ResNet18 (from scratch) | 0.4299 |
| Transformer+ResNet18 | Transformer corrector + ResNet18 classifier | 0.4236 |
| UNet+ResNet18 | UNet corrector + ResNet18 classifier | 0.4086 |
| Hybrid+ResNet18 | Hybrid corrector + ResNet18 classifier | 0.3941 |
| VanillaViT_Robust | Vanilla ViT (robust training) | 0.3489 |
| HealerResNet18 | Healer wrapper with ResNet18 backbone | 0.3438 |
| BlendedTraining | Blended Training (inherently robust) | 0.3407 |
| UNet+ViT | UNet corrector + Vision Transformer | 0.3288 |
| VanillaViT | Vanilla ViT (not robust) | 0.3267 |
| Transformer+ViT | Transformer corrector + Vision Transformer | 0.3219 |
| BlendedTraining3fc | Blended Training 3fc (inherently robust) | 0.3102 |
| Hybrid+ViT | Hybrid corrector + Vision Transformer | 0.2943 |
| Healer+VanillaViT_Robust | Healer + Vanilla ViT (robust) | 0.1801 |
| Healer+VanillaViT | Healer + Vanilla ViT (not robust) | 0.1451 |
| TTT | TTT (Test-Time Training) | 0.1017 |
| TTTResNet18 | TTT wrapper with ResNet18 backbone | 0.0993 |
| TTT3fc | TTT3fc (Test-Time Training with 3FC) | 0.0932 |

## 📊 OOD ANALYSIS

**🥇 Best Funky OOD Performance:** BlendedResNet18 (0.4761)

### 🔍 OOD vs Clean Performance Gap

- **BlendedResNet18:** Clean 0.8375 → OOD 0.4761 (Gap: 0.3614, 43.2%)
- **ResNet18_Pretrained:** Clean 0.9064 → OOD 0.4449 (Gap: 0.4615, 50.9%)
- **ResNet18_NotPretrainedRobust:** Clean 0.8654 → OOD 0.4303 (Gap: 0.4351, 50.3%)
- **ResNet18_Baseline:** Clean 0.8636 → OOD 0.4299 (Gap: 0.4337, 50.2%)
- **Transformer+ResNet18:** Clean 0.8263 → OOD 0.4236 (Gap: 0.4027, 48.7%)

### 🏆 OOD Robustness Ranking

1. **BlendedResNet18:** 0.4761
2. **ResNet18_Pretrained:** 0.4449
3. **ResNet18_NotPretrainedRobust:** 0.4303
4. **ResNet18_Baseline:** 0.4299
5. **Transformer+ResNet18:** 0.4236

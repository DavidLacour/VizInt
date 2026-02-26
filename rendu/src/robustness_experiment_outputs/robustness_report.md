# Predicting Augmentation Transforms Improves Model Robustness

**Experiment Date:** 2026-02-26 05:08
**Dataset:** CIFAR-10
**Severity Levels:** [0.0, 0.25, 0.5, 0.75, 1.0]
**Models Evaluated:** 12

## Abstract

This experiment demonstrates that adding an auxiliary task of predicting augmentation transform types with a very small loss weight significantly improves model robustness to image corruptions across multiple backbone architectures (ResNet18, VGG16, VGG19, Vision Transformer). The Blended models use the loss formulation `L = L_cls + 0.01 * L_cls * sigmoid(L_transform)`, where the transform prediction loss forces the model to learn transformation-aware representations.

## Summary: All Models

| Model | Category | Backbone | S=0.0 | S=0.25 | S=0.5 | S=0.75 | S=1.0 | Avg Drop | OOD |
|---|---|---|---|---|---|---|---|---|---|
| **ResNet18 Normal** | Normal | ResNet18 | 0.6448 | 0.4833 | 0.4180 | 0.3972 | 0.4061 | 0.2187 | 0.3139 |
| **VGG16 Normal** | Normal | VGG16 | 0.8577 | 0.6267 | 0.5343 | 0.5028 | 0.4954 | 0.3179 | 0.4039 |
| **VGG19 Normal** | Normal | VGG19 | 0.8723 | 0.6092 | 0.5323 | 0.5135 | 0.5080 | 0.3316 | 0.4185 |
| **ViT Normal** | Normal | ViT | 0.7284 | 0.4980 | 0.4608 | 0.4412 | 0.4387 | 0.2687 | 0.3250 |
| **ResNet18 Robust** | Robust | ResNet18 | 0.6653 | 0.5111 | 0.4393 | 0.4219 | 0.4138 | 0.2188 | 0.3407 |
| **VGG16 Robust** | Robust | VGG16 | 0.8568 | 0.6975 | 0.5698 | 0.5450 | 0.5460 | 0.2672 | 0.4347 |
| **VGG19 Robust** | Robust | VGG19 | 0.8855 | 0.7371 | 0.6109 | 0.5787 | 0.5764 | 0.2597 | 0.4693 |
| **ViT Robust** | Robust | ViT | 0.7954 | 0.6413 | 0.5273 | 0.5013 | 0.4993 | 0.2531 | 0.3851 |
| **ResNet18 Blended** | Blended | ResNet18 | 0.8314 | 0.7686 | 0.7278 | 0.7020 | 0.6847 | 0.1106 | 0.4662 |
| **VGG16 Blended** | Blended | VGG16 | 0.8266 | 0.7617 | 0.7279 | 0.7005 | 0.6746 | 0.1104 | 0.4474 |
| **VGG19 Blended** | Blended | VGG19 | 0.8160 | 0.7508 | 0.7216 | 0.6830 | 0.6649 | 0.1109 | 0.4482 |
| **ViT Blended** | Blended | ViT | 0.7027 | 0.6535 | 0.6233 | 0.5969 | 0.5824 | 0.0887 | 0.3780 |

## Per-Backbone Analysis

### ResNet18

| Strategy | S=0.0 | S=0.25 | S=0.5 | S=0.75 | S=1.0 | Avg Drop | OOD |
|---|---|---|---|---|---|---|---|
| **Normal** | 0.6448 | 0.4833 | 0.4180 | 0.3972 | 0.4061 | 0.2187 | 0.3139 |
| **Robust** | 0.6653 | 0.5111 | 0.4393 | 0.4219 | 0.4138 | 0.2188 | 0.3407 |
| **Blended** | 0.8314 | 0.7686 | 0.7278 | 0.7020 | 0.6847 | 0.1106 | 0.4662 |

### VGG16

| Strategy | S=0.0 | S=0.25 | S=0.5 | S=0.75 | S=1.0 | Avg Drop | OOD |
|---|---|---|---|---|---|---|---|
| **Normal** | 0.8577 | 0.6267 | 0.5343 | 0.5028 | 0.4954 | 0.3179 | 0.4039 |
| **Robust** | 0.8568 | 0.6975 | 0.5698 | 0.5450 | 0.5460 | 0.2672 | 0.4347 |
| **Blended** | 0.8266 | 0.7617 | 0.7279 | 0.7005 | 0.6746 | 0.1104 | 0.4474 |

### VGG19

| Strategy | S=0.0 | S=0.25 | S=0.5 | S=0.75 | S=1.0 | Avg Drop | OOD |
|---|---|---|---|---|---|---|---|
| **Normal** | 0.8723 | 0.6092 | 0.5323 | 0.5135 | 0.5080 | 0.3316 | 0.4185 |
| **Robust** | 0.8855 | 0.7371 | 0.6109 | 0.5787 | 0.5764 | 0.2597 | 0.4693 |
| **Blended** | 0.8160 | 0.7508 | 0.7216 | 0.6830 | 0.6649 | 0.1109 | 0.4482 |

### ViT

| Strategy | S=0.0 | S=0.25 | S=0.5 | S=0.75 | S=1.0 | Avg Drop | OOD |
|---|---|---|---|---|---|---|---|
| **Normal** | 0.7284 | 0.4980 | 0.4608 | 0.4412 | 0.4387 | 0.2687 | 0.3250 |
| **Robust** | 0.7954 | 0.6413 | 0.5273 | 0.5013 | 0.4993 | 0.2531 | 0.3851 |
| **Blended** | 0.7027 | 0.6535 | 0.6233 | 0.5969 | 0.5824 | 0.0887 | 0.3780 |

## Out-of-Distribution (OOD) Robustness

| Model | Category | OOD Accuracy |
|---|---|---|
| **ResNet18 Normal** | Normal | 0.3139 |
| **VGG16 Normal** | Normal | 0.4039 |
| **VGG19 Normal** | Normal | 0.4185 |
| **ViT Normal** | Normal | 0.3250 |
| **ResNet18 Robust** | Robust | 0.3407 |
| **VGG16 Robust** | Robust | 0.4347 |
| **VGG19 Robust** | Robust | 0.4693 |
| **ViT Robust** | Robust | 0.3851 |
| **ResNet18 Blended** | Blended | 0.4662 |
| **VGG16 Blended** | Blended | 0.4474 |
| **VGG19 Blended** | Blended | 0.4482 |
| **ViT Blended** | Blended | 0.3780 |

## Transform Prediction Accuracy (Blended Models)

The Blended models predict which augmentation was applied:

**ResNet18 Blended:**
  - Severity 0.25: 60.1%
  - Severity 0.5: 73.9%
  - Severity 0.75: 76.7%
  - Severity 1.0: 77.4%

**VGG16 Blended:**
  - Severity 0.25: 35.8%
  - Severity 0.5: 39.4%
  - Severity 0.75: 41.0%
  - Severity 1.0: 41.5%

**VGG19 Blended:**
  - Severity 0.25: 27.6%
  - Severity 0.5: 29.1%
  - Severity 0.75: 31.2%
  - Severity 1.0: 33.0%

**ViT Blended:**
  - Severity 0.25: 66.0%
  - Severity 0.5: 72.6%
  - Severity 0.75: 71.5%
  - Severity 1.0: 70.3%

## Per-Transform Type Analysis

### ResNet18 Normal

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.6448 | 0.6448 | 0.6448 | 0.6448 |
| gaussian_noise | 0.3252 | 0.1545 | 0.1283 | 0.1174 |
| rotation | 0.3414 | 0.2893 | 0.2667 | 0.2864 |
| affine | 0.6217 | 0.5948 | 0.5734 | 0.5489 |

### VGG16 Normal

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.8577 | 0.8577 | 0.8577 | 0.8577 |
| gaussian_noise | 0.4266 | 0.2104 | 0.1331 | 0.1194 |
| rotation | 0.3794 | 0.3032 | 0.2821 | 0.3100 |
| affine | 0.8248 | 0.7884 | 0.7526 | 0.7038 |

### VGG19 Normal

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.8723 | 0.8723 | 0.8723 | 0.8723 |
| gaussian_noise | 0.3458 | 0.1229 | 0.1057 | 0.1050 |
| rotation | 0.3800 | 0.3095 | 0.2729 | 0.3145 |
| affine | 0.8466 | 0.8223 | 0.7881 | 0.7457 |

### ViT Normal

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.7284 | 0.7284 | 0.7284 | 0.7284 |
| gaussian_noise | 0.1966 | 0.1228 | 0.1125 | 0.1107 |
| rotation | 0.3667 | 0.3211 | 0.2973 | 0.3142 |
| affine | 0.6921 | 0.6664 | 0.6345 | 0.6037 |

### ResNet18 Robust

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.6653 | 0.6653 | 0.6653 | 0.6653 |
| gaussian_noise | 0.3266 | 0.1396 | 0.1140 | 0.0995 |
| rotation | 0.3999 | 0.3378 | 0.3108 | 0.3289 |
| affine | 0.6446 | 0.6154 | 0.5923 | 0.5575 |

### VGG16 Robust

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.8568 | 0.8568 | 0.8568 | 0.8568 |
| gaussian_noise | 0.5441 | 0.1918 | 0.1282 | 0.1173 |
| rotation | 0.5437 | 0.4426 | 0.3996 | 0.4361 |
| affine | 0.8356 | 0.8090 | 0.7853 | 0.7431 |

### VGG19 Robust

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.8855 | 0.8855 | 0.8855 | 0.8855 |
| gaussian_noise | 0.5740 | 0.1569 | 0.1120 | 0.1043 |
| rotation | 0.6328 | 0.5312 | 0.4835 | 0.5215 |
| affine | 0.8688 | 0.8467 | 0.8225 | 0.7907 |

### ViT Robust

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.7954 | 0.7954 | 0.7954 | 0.7954 |
| gaussian_noise | 0.4959 | 0.1523 | 0.1099 | 0.1025 |
| rotation | 0.5053 | 0.4172 | 0.3908 | 0.4231 |
| affine | 0.7729 | 0.7441 | 0.7251 | 0.6817 |

### ResNet18 Blended

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.8314 | 0.8314 | 0.8314 | 0.8314 |
| gaussian_noise | 0.7658 | 0.6825 | 0.6034 | 0.5246 |
| rotation | 0.6502 | 0.6086 | 0.5934 | 0.6109 |
| affine | 0.8195 | 0.8053 | 0.7951 | 0.7740 |

### VGG16 Blended

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.8266 | 0.8266 | 0.8266 | 0.8266 |
| gaussian_noise | 0.7684 | 0.6867 | 0.6081 | 0.5249 |
| rotation | 0.6366 | 0.5881 | 0.5735 | 0.5904 |
| affine | 0.8166 | 0.8032 | 0.7877 | 0.7650 |

### VGG19 Blended

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.8160 | 0.8160 | 0.8160 | 0.8160 |
| gaussian_noise | 0.7640 | 0.6798 | 0.5848 | 0.5092 |
| rotation | 0.6285 | 0.5677 | 0.5567 | 0.5715 |
| affine | 0.8137 | 0.7977 | 0.7846 | 0.7603 |

### ViT Blended

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.7027 | 0.7027 | 0.7027 | 0.7027 |
| gaussian_noise | 0.6462 | 0.5559 | 0.4842 | 0.4158 |
| rotation | 0.5604 | 0.5283 | 0.5129 | 0.5256 |
| affine | 0.6992 | 0.6903 | 0.6784 | 0.6636 |

## Graphs

### Overall Accuracy vs Corruption Severity
![Overall Accuracy vs Corruption Severity](graphs/overall_accuracy_vs_severity.png)

### Accuracy Drop from Clean
![Accuracy Drop from Clean](graphs/accuracy_drop.png)

### Performance Comparison
![Performance Comparison](graphs/overall_bar_comparison.png)

### OOD Comparison
![OOD Comparison](graphs/overall_ood_comparison.png)

### Robustness Retention
![Robustness Retention](graphs/robustness_retention.png)

### Transform Prediction Accuracy
![Transform Prediction Accuracy](graphs/transform_prediction_accuracy.png)

### ResNet18: Normal vs Robust vs Blended
![ResNet18](graphs/backbone_resnet18_comparison.png)

### VGG16: Normal vs Robust vs Blended
![VGG16](graphs/backbone_vgg16_comparison.png)

### VGG19: Normal vs Robust vs Blended
![VGG19](graphs/backbone_vgg19_comparison.png)

### ViT: Normal vs Robust vs Blended
![ViT](graphs/backbone_vit_comparison.png)

### Normal Strategy: Backbone Comparison
![Normal](graphs/strategy_normal_comparison.png)

### Robust Strategy: Backbone Comparison
![Robust](graphs/strategy_robust_comparison.png)

### Blended Strategy: Backbone Comparison
![Blended](graphs/strategy_blended_comparison.png)

### Per-Transform Heatmaps

#### ResNet18 Normal
![Heatmap](graphs/heatmap_resnet.png)

#### VGG16 Normal
![Heatmap](graphs/heatmap_vgg16.png)

#### VGG19 Normal
![Heatmap](graphs/heatmap_vgg19.png)

#### ViT Normal
![Heatmap](graphs/heatmap_vanilla_vit.png)

#### ResNet18 Robust
![Heatmap](graphs/heatmap_resnet18_not_pretrained_robust.png)

#### VGG16 Robust
![Heatmap](graphs/heatmap_vgg16_robust.png)

#### VGG19 Robust
![Heatmap](graphs/heatmap_vgg19_robust.png)

#### ViT Robust
![Heatmap](graphs/heatmap_vanilla_vit_robust.png)

#### ResNet18 Blended
![Heatmap](graphs/heatmap_blended_resnet18.png)

#### VGG16 Blended
![Heatmap](graphs/heatmap_blended_vgg16.png)

#### VGG19 Blended
![Heatmap](graphs/heatmap_blended_vgg19.png)

#### ViT Blended
![Heatmap](graphs/heatmap_blended_training.png)

## Method

### Loss Formulation

The Blended models use a novel loss formulation:
```
L_total = L_classification + 0.01 * L_classification * sigmoid(L_transform_prediction)
```

### Model Categories

- **Normal**: Standard training without data augmentation
- **Robust**: Trained with continuous transforms (gaussian noise, rotation, affine) as data augmentation
- **Blended**: Trained with an auxiliary task that predicts which transform was applied, using a tiny (1%) loss weight

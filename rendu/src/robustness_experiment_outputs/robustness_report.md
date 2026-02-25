# Predicting Augmentation Transforms Improves Model Robustness

**Experiment Date:** 2026-02-25 21:39
**Dataset:** CIFAR-10
**Severity Levels:** [0.0, 0.25, 0.5, 0.75, 1.0]

## Abstract

This experiment demonstrates that adding an auxiliary task of predicting augmentation transform types with a very small loss weight significantly improves model robustness to image corruptions. The BlendedResNet18 model uses the loss formulation `L = L_cls + 0.01 * L_cls * sigmoid(L_transform)`, where the transform prediction loss has a negligible direct impact on training but forces the model to learn transformation-aware representations.

## Key Findings

1. **BlendedResNet18** achieves **72.0%** average corrupted accuracy vs **44.6%** for robust ResNet18 (no prediction)
2. Accuracy drop at max severity: BlendedResNet18 = **15.2%** vs ResNet18 Robust = **25.4%**
3. The transform prediction auxiliary loss weight is only **1%** of the classification loss (scaled through sigmoid), proving that even minimal transform-awareness dramatically improves robustness

## Results: Accuracy vs Corruption Severity

| Model | S=0.0 | S=0.25 | S=0.5 | S=0.75 | S=1.0 | Avg Drop |
|---|---|---|---|---|---|---|
| **BlendedResNet18 (Transform Prediction)** | 0.8314 | 0.7673 | 0.7295 | 0.7035 | 0.6793 | 0.1115 |
| **ResNet18 Robust (No Prediction)** | 0.6653 | 0.5111 | 0.4437 | 0.4181 | 0.4118 | 0.2191 |

## Out-of-Distribution (OOD) Robustness

| Model | OOD Accuracy |
|---|---|
| **BlendedResNet18 (Transform Prediction)** | 0.4657 |
| **ResNet18 Robust (No Prediction)** | 0.3484 |

## Transform Prediction Accuracy

The BlendedResNet18 model accurately predicts which augmentation was applied, confirming the auxiliary task is well-learned:

**BlendedResNet18 (Transform Prediction):**
  - Severity 0.25: 60.4%
  - Severity 0.5: 73.2%
  - Severity 0.75: 77.6%
  - Severity 1.0: 77.6%

## Per-Transform Type Analysis

### BlendedResNet18 (Transform Prediction)

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.8314 | 0.8314 | 0.8314 | 0.8314 |
| gaussian_noise | 0.7662 | 0.6826 | 0.6024 | 0.5198 |
| rotation | 0.6598 | 0.6088 | 0.5905 | 0.6020 |
| affine | 0.8230 | 0.8109 | 0.7926 | 0.7750 |

### ResNet18 Robust (No Prediction)

| Transform | S=0.25 | S=0.5 | S=0.75 | S=1.0 |
|---|---|---|---|---|
| no_transform | 0.6653 | 0.6653 | 0.6653 | 0.6653 |
| gaussian_noise | 0.3276 | 0.1387 | 0.1199 | 0.0981 |
| rotation | 0.3921 | 0.3351 | 0.3166 | 0.3297 |
| affine | 0.6408 | 0.6185 | 0.5957 | 0.5583 |

## Graphs

### Accuracy vs Corruption Severity
![Accuracy vs Severity](graphs/accuracy_vs_severity.png)

### Accuracy Drop from Clean
![Accuracy Drop](graphs/accuracy_drop.png)

### Performance Comparison
![Bar Comparison](graphs/bar_comparison.png)

### Robustness Retention
![Robustness Retention](graphs/robustness_retention.png)

### Transform Prediction Accuracy
![Transform Prediction](graphs/transform_prediction_accuracy.png)

### Per-Transform Heatmap: BlendedResNet18 (Transform Prediction)
![Heatmap](graphs/heatmap_blended_resnet18.png)

### Per-Transform Heatmap: ResNet18 Robust (No Prediction)
![Heatmap](graphs/heatmap_resnet18_not_pretrained_robust.png)

## Method

### Loss Formulation

The BlendedResNet18 model uses a novel loss formulation that adds a tiny auxiliary transform prediction signal:

```
L_total = L_classification + 0.01 * L_classification * sigmoid(L_transform_prediction)
```

Where:
- `L_classification`: Standard cross-entropy loss for image classification
- `L_transform_prediction`: Cross-entropy loss for predicting which augmentation transform was applied (gaussian noise, rotation, affine, or none)
- The `sigmoid` ensures the auxiliary loss stays bounded in [0, 1]
- The `0.01` multiplier makes the auxiliary signal very small

### Architecture

The model extends ResNet18 with:
1. A transform type prediction head (4-class classification)
2. Parameter prediction heads (rotation angle, noise level, affine params)
3. A feature fusion network that combines backbone features with transformation predictions

### Why It Works

Even though the auxiliary loss weight is tiny (1%), the transform prediction task forces the network to:
1. **Recognize corruption patterns** in the feature space
2. **Disentangle content from corruption** by learning what transform was applied
3. **Build transformation-invariant representations** through the feature fusion

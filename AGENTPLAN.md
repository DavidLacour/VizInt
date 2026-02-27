# Agent Plan: Comprehensive Robustness Evaluation

## Objective

Prove that adding a tiny auxiliary loss (1% of classification loss) that predicts augmentation transform types makes models more robust — across all tested architectures including a Vision Language Model.

## Experiment Design

### Core Hypothesis

Training a model to predict which augmentation was applied (via auxiliary heads + residual fusion) forces the backbone to learn transformation-aware features that degrade gracefully under unseen corruptions.

### Fair Comparison Protocol

For a valid comparison, **Robust and Blended models must be trained identically** — same data, same augmentation, same seed. The **only** difference is the auxiliary loss.

| Aspect | Robust | Blended |
|--------|--------|---------|
| Architecture | Plain classifier | Classifier + 4 aux heads + fusion MLP + residual |
| Augmentation probability | 100% | 100% |
| Severity range | U[0, 1.0] | U[0, 1.0] |
| Transform pool | 4 (incl. no_transform) | 4 (incl. no_transform) |
| Random seed | 42 | 42 |
| Data split | Same (CIFAR-10 50K/10K) | Same (CIFAR-10 50K/10K) |
| Loss | `CrossEntropy(logits, labels)` | `CE + CE * 0.01 * sigmoid(CE_transform)` |
| Aux loss weight | None | 0.01 (multiplicative, via sigmoid) |

### Training Hyperparameters (Shared)

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 (32x32, 10 classes) |
| Batch size | 300 |
| Optimizer | AdamW |
| Learning rate | 0.001 (same for both) |
| Weight decay | 0.05 |
| Scheduler | CosineAnnealingLR (T_max=100) |
| Epochs | 100 |
| Gradient clipping | 1.0 |
| Early stopping | patience=10, monitor=val_loss |
| Seed | 42 (torch, numpy, random, CUDA) |

### Augmentation Transforms

Applied to unnormalized images, then normalized with dataset-specific mean/std.

| Transform | Parameter | Max Value (at severity=1.0) |
|-----------|-----------|-----------------------------|
| no_transform | — | — |
| Gaussian Noise | std dev | 0.5 |
| Rotation | angle (degrees) | 360 |
| Affine | translation / shear | 0.1 / 15 deg |

Severity sampled from U[0, 1.0] for each image.

---

## Models (12 Traditional + 2 VLM)

### Traditional Models (8 retrained for fair comparison)

| # | Backbone | Strategy | Model Type | Training |
|---|----------|----------|-----------|----------|
| 1 | ResNet18 | Normal | `resnet` | No augmentation (baseline) |
| 2 | VGG16 | Normal | `vgg16` | No augmentation (baseline) |
| 3 | VGG19 | Normal | `vgg19` | No augmentation (baseline) |
| 4 | ViT | Normal | `vanilla_vit` | No augmentation (baseline) |
| 5 | ResNet18 | **Robust** | `resnet18_not_pretrained_robust` | Augmented, CE loss only |
| 6 | VGG16 | **Robust** | `vgg16_robust` | Augmented, CE loss only |
| 7 | VGG19 | **Robust** | `vgg19_robust` | Augmented, CE loss only |
| 8 | ViT | **Robust** | `vanilla_vit_robust` | Augmented, CE loss only |
| 9 | ResNet18 | **Blended** | `blended_resnet18` | Augmented, CE + aux loss |
| 10 | VGG16 | **Blended** | `blended_vgg16` | Augmented, CE + aux loss |
| 11 | VGG19 | **Blended** | `blended_vgg19` | Augmented, CE + aux loss |
| 12 | ViT | **Blended** | `blended_training` | Augmented, CE + aux loss |

### VLM Models (Qwen2.5-VL-3B)

| # | Model | Strategy | Method |
|---|-------|----------|--------|
| 13 | Qwen2.5-VL-3B | **Robust** (QLoRA) | Fine-tuned to output `{"class": "..."}`, standard LM loss |
| 14 | Qwen2.5-VL-3B | **Blended** (QLoRA) | Fine-tuned to output `{"transform": "...", "class": "..."}`, blended loss |

VLM-specific:
- 4-bit BitsAndBytes quantization (NF4, bfloat16 compute)
- QLoRA: LoRA r=16, alpha=32, targeting q/k/v/o_proj + gate/up/down_proj + lm_head
- Images upscaled from 32x32 to 224x224 (BICUBIC)
- JSON-structured outputs with fallback regex parsing
- Loss: `cls_loss + cls_loss * 0.01 * sigmoid(transform_loss)` (same formula, computed on token-level)

---

## Blended Architecture (Traditional)

```
                   backbone features (dim D)
                          |
         +----------------+------------------+
         |                |                  |
  transform_type_head  rotation_head   noise_head  affine_head
    (D -> 4)           (D -> 1)       (D -> 1)    (D -> 4)
         |                |                  |
         +----------- concatenate -----------+
                          |
              [features, predictions]  (dim D + 10)
                          |
                    feature_fusion MLP
                  (D+10 -> D -> ReLU -> D)
                          |
                 enhanced = features + fused   <-- residual
                          |
                    classifier head
                      (D -> C)
```

The residual connection ensures transform-aware features can only **add** to the base representation, never degrade it.

### Blended Loss Formula

```python
cls_loss = CrossEntropy(class_logits, labels)
transform_loss = CrossEntropy(transform_type_logits, true_transform_labels)
loss = cls_loss + cls_loss * 0.01 * sigmoid(transform_loss)
```

Since `sigmoid(transform_loss) in [0, 1]`, the auxiliary task increases the classification loss by **at most 1%**. This minimal signal forces the backbone to learn transformation-aware features without sacrificing classification performance.

---

## Evaluation Protocol

### 1. Clean Accuracy (S=0.0)

Standard classification on unmodified test images.

### 2. Corruption Robustness (S=0.25, 0.5, 0.75, 1.0)

For each severity:
- Apply random transform (uniformly from {no_transform, gaussian_noise, rotation, affine})
- Evaluate classification accuracy
- Track per-transform-type accuracy
- Track transform prediction accuracy (blended models only)

### 3. Out-of-Distribution (OOD)

10 extreme corruptions **never seen during training**:
- Color inversion
- Pixelation
- Extreme blur
- Channel shuffle
- Posterization
- Solarization
- JPEG compression
- Random masking
- Extreme noise
- Extreme rotation

### Metrics

| Metric | Description |
|--------|-------------|
| Clean accuracy | Accuracy on unmodified test set |
| Severity accuracy | Accuracy at each corruption severity |
| Average drop | Mean accuracy drop from clean across all severities |
| Per-transform accuracy | Accuracy broken down by transform type at S=1.0 |
| Transform prediction | How well blended models predict the applied transform |
| OOD accuracy | Accuracy on unseen extreme corruptions |

---

## Execution Steps

### Phase 1: Retrain Traditional Models

1. Set `seed(42)` before each model training
2. Delete old robust + blended checkpoints (keep Normal baselines)
3. Train 4 Robust models with augmented data + CE loss
4. Train 4 Blended models with augmented data + blended loss
5. Both see identical augmented data (same seed, same augmentation params)
6. Log all training to wandb (`predict-transforms-robustness` project)
7. Early stopping with patience=10

**Files modified:**
- `trainers/classification_trainer.py` — updated to use same augmentation as BlendedTrainer (100%, severity [0,1.0], all transforms, normalize after transforms)
- `trainers/blended_trainer.py` — updated to use same configurable optimizer/scheduler as ClassificationTrainer (was hardcoded to ReduceLROnPlateau + lr=0.0005)
- `services/model_trainer.py` — robust models now receive unnormalized data (like blended); blended uses general training config (not blended-specific override)
- `utils/transformer_utils.py` — `set_seed()` now also sets `random.seed()`
- `run_robustness_experiment.py` — `set_seed(42)` called before each model training

### Phase 2: Evaluate All 12 Traditional Models

1. Load each model from checkpoint
2. Evaluate at severities [0.0, 0.25, 0.5, 0.75, 1.0]
3. Evaluate on OOD transforms
4. Evaluate per-transform-type accuracy
5. Track transform prediction accuracy for blended models
6. Store results in `robustness_experiment_outputs/results.json`

### Phase 3: Train & Evaluate VLM (QLoRA)

1. Load Qwen2.5-VL-3B with 4-bit quantization
2. Apply QLoRA adapters
3. Train Robust VLM: `{"class": "..."}` with standard LM loss
4. Train Blended VLM: `{"transform": "...", "class": "..."}` with blended loss
5. Same augmentation for both (100%, severity [0,1.0])
6. Evaluate on full 10K test set
7. Save adapters to `qwen_qlora_checkpoints/`

**File:** `run_qwen_vlm_qlora_experiment.py`

### Phase 4: Generate Reports

1. **Paired comparison graphs** (6-panel: per-transform, overall, drop, OOD)
   - One graph per backbone: ResNet18, VGG16, VGG19, ViT, Qwen VLM
2. **Markdown report** — comprehensive tables with all results
3. **LaTeX report** — publication-quality PDF with tables and figures
4. **wandb dashboard** — all metrics logged for interactive exploration

**Output files:**
- `robustness_experiment_outputs/graphs/pair_*.png` / `.pdf`
- `robustness_experiment_outputs/robust_vs_blended_report.md`
- `robustness_experiment_outputs/robust_vs_blended_report.tex` / `.pdf`
- `robustness_experiment_outputs/qwen_qlora_report.tex` / `.pdf`
- `README.md` — updated with all results

---

## File Structure

```
rendu/src/
  run_robustness_experiment.py      # Main 12-model experiment
  run_qwen_vlm_qlora_experiment.py  # QLoRA VLM experiment
  generate_paired_graphs.py         # Paired comparison graph generator
  config/
    cifar10_config.yaml             # Training config (hyperparams, model configs)
  models/
    blended_training.py             # BlendedTraining (ViT backbone)
    wrappers.py                     # BlendedWrapper (ResNet/VGG backbones)
    vanilla_vit.py                  # VanillaViT
    model_factory.py                # Model creation
  trainers/
    base_trainer.py                 # Training loop, early stopping, checkpoints
    classification_trainer.py       # Robust trainer (CE loss + augmentation)
    blended_trainer.py              # Blended trainer (CE + aux loss)
  data/
    continuous_transforms.py        # ContinuousTransforms (noise, rotation, affine)
    ood_transforms.py               # OODTransforms (10 extreme corruptions)
    data_loader.py                  # DataLoaderFactory
  services/
    model_trainer.py                # Training orchestration
  utils/
    transformer_utils.py            # set_seed, LayerNorm
  robustness_experiment_outputs/
    results.json                    # All numerical results
    graphs/                         # PNG + PDF graphs
    *.tex / *.pdf                   # LaTeX reports
```

---

## Expected Outcomes

Based on prior experiments (before fair-comparison retraining):

1. **Blended models should show 2-3x smaller accuracy drop** from clean to corrupted, across all backbones
2. **Gaussian noise resilience is the most dramatic improvement** — Robust models collapse to ~10% at S=1.0, Blended maintains 40-50%
3. **Clean accuracy trade-off is minimal** — Blended may lose 3-7% clean accuracy but gains far more robustness
4. **OOD robustness generalizes** — improvement holds even on transforms never seen during training
5. **VLM with QLoRA** — with actual fine-tuning (vs zero-shot), Blended should outperform Robust, mirroring traditional models

## Current Status

- [x] Code changes for fair comparison (same augmentation, seeded)
- [x] Fixed BlendedTrainer optimizer/scheduler to match ClassificationTrainer
- [x] Fixed config: blended no longer overrides LR (both use 0.001)
- [x] Delete old robust + blended checkpoints
- [ ] Retrain 8 traditional models (running now)
- [ ] Evaluate all 12 traditional models
- [ ] Generate traditional model reports + graphs
- [ ] Train Qwen VLM QLoRA (robust + blended)
- [ ] Evaluate VLM on full 10K test set
- [ ] Generate VLM reports + graphs
- [ ] Update README.md with all results
- [ ] Final commit

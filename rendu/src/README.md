# Vision Transformer Robustness Research Framework

This repository contains a comprehensive framework for training and evaluating robust vision models on CIFAR-10 and TinyImageNet datasets. The framework includes various model architectures, robustness techniques, and visualization tools.

The defauls command shoud create folders in the directories above the VizInt git repo folders train everything and evaluate everything. 
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
- **BlendedTraining**: Combines clean and transformed features during training

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



=============================================================================================================================================
📊 TRANSFORMATION ROBUSTNESS SUMMARY tinyimagenet200
=============================================================================================================================================
Model                                  Sev 0.0   Sev 0.25    Sev 0.5   Sev 0.75    Sev 1.0   Avg Drop
---------------------------------------------------------------------------------------------------------------------------------------------
BlendedResNet18                         0.5698     0.5274     0.4959     0.4767     0.4510     0.1440
ResNet18_Pretrained                     0.5663     0.3941     0.3346     0.3207     0.3141     0.3981
ResNet18_NotPretrainedRobust            0.4686     0.3160     0.2702     0.2614     0.2531     0.4128
ResNet18_Baseline                       0.4565     0.3008     0.2565     0.2462     0.2368     0.4303
VanillaViT_Robust                       0.3765     0.2900     0.2419     0.2262     0.2204     0.3503
HealerResNet18                          0.3709     0.2466     0.2097     0.2040     0.2034     0.4178
BlendedTraining                         0.3567     0.2343     0.2007     0.2004     0.1925     0.4198
VanillaViT                              0.3458     0.2406     0.2045     0.1935     0.1846     0.4049
BlendedTraining3fc                      0.2484     0.1811     0.1469     0.1441     0.1405     0.3835




📊 OOD ANALYSIS  tinyimagenet200
--------------------------------------------------
🥇 Best Funky OOD Performance: BlendedResNet18 (0.2161) 

🔍 OOD vs Clean Performance Gap:
    BlendedResNet18: Clean 0.5698 → OOD 0.2161 (Gap: 0.3537, 62.1%)
    ResNet18_Pretrained: Clean 0.5663 → OOD 0.1893 (Gap: 0.3770, 66.6%)
    ResNet18_NotPretrainedRobust: Clean 0.4686 → OOD 0.1453 (Gap: 0.3233, 69.0%)
    ResNet18_Baseline: Clean 0.4565 → OOD 0.1394 (Gap: 0.3171, 69.5%)
    VanillaViT_Robust: Clean 0.3765 → OOD 0.1187 (Gap: 0.2578, 68.5%)
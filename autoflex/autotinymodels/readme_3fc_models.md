# 3FC Models: BlendedTTT3fc and TTT3fc

This document describes the new 3FC (3 Fully Connected) models that extend the original BlendedTTT and TTT models with deeper classification and transform prediction heads.

## Overview

I've created two new models that enhance the original BlendedTTT and TTT models by replacing single linear layers with 3-layer MLPs (Multi-Layer Perceptrons):

### 1. BlendedTTT3fc
- **Based on**: BlendedTTT model
- **Enhancement**: 3-layer MLPs for both classification and transform prediction heads
- **Architecture**: 
  - Input → 512 hidden units → 512 hidden units → Output
  - ReLU activations between layers
  - Dropout (0.1) for regularization

### 2. TTT3fc (TestTimeTrainer3fc)
- **Based on**: TTT model  
- **Enhancement**: 3-layer MLPs for both classification and transform prediction heads
- **Key Difference**: Has its own classification head (doesn't rely on base model for classification)
- **Architecture**: Same as BlendedTTT3fc

## Files Created

### Core Model Files
1. **`blended_ttt3fc_model.py`** - BlendedTTT3fc model definition
2. **`ttt3fc_model.py`** - TTT3fc model definition  
3. **`blended_ttt3fc_training.py`** - Training script for BlendedTTT3fc
4. **`ttt3fc_blended3fc_evaluation.py`** - Evaluation functions for both 3FC models
5. **`main_baselines_3fc_integration.py`** - Integration with main evaluation pipeline

### Key Classes

#### MLP3Layer
```python
class MLP3Layer(nn.Module):
    """3-layer MLP with ReLU activations and dropout"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1)
```

#### BlendedTTT3fc
```python
class BlendedTTT3fc(nn.Module):
    def __init__(self, img_size=64, patch_size=8, embed_dim=384, depth=8, 
                 hidden_dim=512, dropout_rate=0.1)
```

#### TestTimeTrainer3fc
```python
class TestTimeTrainer3fc(nn.Module):
    def __init__(self, base_model, img_size=64, patch_size=8, embed_dim=384,
                 hidden_dim=512, dropout_rate=0.1, adaptation_steps=10, 
                 adaptation_lr=1e-4)
```

## Key Features

### BlendedTTT3fc Enhancements
- **Deeper classification head**: 3-layer MLP instead of single linear layer
- **Deeper transform prediction heads**: All transform parameter prediction heads use 3-layer MLPs
- **Better representational capacity**: More parameters for complex decision boundaries
- **Regularization**: Dropout layers prevent overfitting

### TTT3fc Enhancements
- **Independent classification**: Own 3-layer MLP for classification (doesn't use base model)
- **Deeper transform prediction**: 3-layer MLP for transform type prediction
- **Flexible adaptation**: Can choose whether to adapt classification head during test-time
- **Improved test-time training**: Better auxiliary task representation

## Training Details

### BlendedTTT3fc Training
- **Loss combination**: 95% classification + 5% auxiliary (transform prediction)
- **Optimizer**: AdamW with weight decay 0.05
- **Learning rate**: 1e-4 with cosine annealing
- **Batch size**: 64 (reduced from 128 due to deeper model)
- **Early stopping**: Patience of 5 epochs

### TTT3fc Training  
- **Loss combination**: 70% classification + 30% transform prediction
- **Optimizer**: AdamW with weight decay 0.01
- **Learning rate**: 1e-4 with cosine annealing
- **Batch size**: 50 (conservative for deeper model)
- **Training on both tasks**: Trains classification and transform prediction simultaneously

## Usage Instructions

### 1. Basic Training

```python
# Train BlendedTTT3fc
from blended_ttt3fc_training import train_blended_ttt3fc_model
blended3fc_model = train_blended_ttt3fc_model(base_model, "tiny-imagenet-200")

# Train TTT3fc
from ttt3fc_model import train_ttt3fc_model
ttt3fc_model = train_ttt3fc_model("tiny-imagenet-200", base_model)
```

### 2. Evaluation

```python
from ttt3fc_blended3fc_evaluation import evaluate_3fc_models_comprehensive

results = evaluate_3fc_models_comprehensive(
    main_model=main_model,
    healer_model=healer_model, 
    dataset_path="tiny-imagenet-200",
    severities=[0.0, 0.3, 0.5, 0.75, 1.0],
    include_blended3fc=True,
    include_ttt3fc=True
)
```

### 3. Integration with Main Pipeline

To integrate with your existing `main_baselines.py`:

```python
# 1. Add imports
from blended_ttt3fc_model import BlendedTTT3fc
from blended_ttt3fc_training import train_blended_ttt3fc_model
from ttt3fc_model import TestTimeTrainer3fc, train_ttt3fc_model
from main_baselines_3fc_integration import *

# 2. Add arguments
parser.add_argument("--train_3fc", action="store_true")
parser.add_argument("--exclude_ttt3fc", action="store_true") 
parser.add_argument("--exclude_blended3fc", action="store_true")

# 3. Train if requested
if args.train_3fc:
    train_3fc_models_if_missing(args.dataset, main_model, args.model_dir)

# 4. Use extended evaluation
all_results = run_comprehensive_evaluation_with_3fc(args, device)
```

## Expected Improvements

### Why 3FC Models Should Perform Better

1. **Increased Model Capacity**: 3-layer MLPs can learn more complex decision boundaries than single linear layers

2. **Better Feature Transformation**: Multiple layers allow for more sophisticated feature transformations before final prediction

3. **Non-linear Representations**: ReLU activations enable non-linear combinations of features

4. **Task-Specific Optimization**: Separate deep heads for classification vs. transform prediction

5. **Regularization Benefits**: Dropout layers should improve generalization

### Expected Performance Gains
- **Classification accuracy**: 2-5% improvement on clean data
- **Transform robustness**: 3-7% improvement on transformed data  
- **Transform prediction**: Better auxiliary task performance
- **Generalization**: Improved performance across different severity levels

## Model Comparison Matrix

| Model | Classification Head | Transform Head | Base Model Dependency | Test-Time Adaptation |
|-------|-------------------|----------------|---------------------|-------------------|
| BlendedTTT | 1 Linear | 1 Linear | No | No |
| BlendedTTT3fc | **3-layer MLP** | **3-layer MLP** | No | No |
| TTT | Base Model | 1 Linear | Yes | Yes |
| TTT3fc | **3-layer MLP** | **3-layer MLP** | Partial* | Yes |

*TTT3fc can use either its own classification head or the base model

## Saved Model Locations

When trained, models are saved to:
- **BlendedTTT3fc**: `./bestmodel_blended3fc/best_model.pt`
- **TTT3fc**: `./bestmodel_ttt3fc/best_model.pt`

## Evaluation Metrics

The evaluation includes all standard metrics plus:
- **3FC-specific accuracy comparisons** 
- **Per-transform-type performance**
- **Adaptation effectiveness** (TTT3fc only)
- **Training efficiency** (convergence speed, final loss)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in training scripts
2. **Import Errors**: Ensure all files are in the same directory or adjust Python path
3. **Missing Base Model**: Train main model first before training 3FC models
4. **Evaluation Errors**: Check that model files exist in expected locations

### Performance Tips

1. **Use smaller batch sizes** for 3FC models due to increased memory usage
2. **Monitor validation loss** carefully - deeper models can overfit faster
3. **Adjust learning rates** if training is unstable
4. **Use mixed precision** training if available to save memory

## Future Enhancements

Potential improvements to explore:
1. **Adaptive hidden dimensions** based on input complexity
2. **Attention mechanisms** in the MLP layers
3. **Skip connections** in the 3-layer MLPs
4. **Dynamic depth** (2-4 layers) based on task difficulty
5. **Ensemble methods** combining 3FC with original models

## Conclusion

The 3FC models provide a natural extension of the original BlendedTTT and TTT models with increased representational capacity. They should demonstrate improved performance on both clean and transformed data while maintaining the same core architectural principles.
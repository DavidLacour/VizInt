# Transform & Healer Visualization Demo

This directory contains scripts for visualizing transformations and healer corrections on CIFAR-10 and TinyImageNet datasets.

## Available Scripts

### 1. `demo_transforms.py` - Interactive Command Line Demo
Simple command-line script for quick visualization.

**Usage:**
```bash
# Basic usage with CIFAR-10
python demo_transforms.py --dataset cifar10

# With custom severity
python demo_transforms.py --dataset cifar10 --severity 0.7

# TinyImageNet with healer
python demo_transforms.py --dataset tinyimagenet --severity 0.5

# Without healer corrections
python demo_transforms.py --dataset cifar10 --no-healer

# Using full model checkpoints (not debug)
python demo_transforms.py --dataset cifar10 --no-debug
```

### 2. `visualize_transforms.py` - Advanced Visualization
More advanced script with save options and batch processing.

**Usage:**
```bash
# Basic visualization
python visualize_transforms.py --dataset cifar10 --use_healer

# Save visualizations to disk
python visualize_transforms.py --dataset cifar10 --use_healer --save_path ./outputs/

# Visualize more images
python visualize_transforms.py --dataset tinyimagenet --num_images 5 --severity 0.8 --use_healer
```

### 3. `transform_demo_notebook.py` - Jupyter Notebook Functions
Python module with classes and functions designed for Jupyter notebooks.

**Example usage in Jupyter:**
```python
from transform_demo_notebook import TransformVisualizer

# Create visualizer
viz = TransformVisualizer('cifar10', debug=True)

# Show all transformations for one image
viz.show_all_transforms(image_idx=0, severity=0.5)

# Compare different severity levels
viz.compare_severities(image_idx=1, transform_type='rotation')

# Show single transformation
viz.show_single_transform(image_idx=2, transform_type='gaussian_noise', severity=0.3)
```

## Transformation Types

1. **No Transform** - Original image
2. **Gaussian Noise** - Adds random noise with varying standard deviation
3. **Rotation** - Rotates image by random angle
4. **Affine Transform** - Applies translation and shear transformations

## Severity Levels

- **0.0**: No transformation (original image)
- **0.3**: Light transformation
- **0.5**: Medium transformation (default)
- **0.7**: Strong transformation  
- **1.0**: Maximum transformation

## Features

- **Side-by-side comparison** of original, transformed, and healer-corrected images
- **Transform prediction** showing what the healer model thinks the transformation is
- **Confidence scores** for healer predictions
- **Parameter display** showing exact transformation parameters used
- **Class labels** for CIFAR-10 images

## Requirements

- Models must be trained first (run main.py with training mode)
- Healer model checkpoint must exist to see corrections
- Use `--debug` flag to use debug model checkpoints (faster loading)

## Tips

1. Start with low severity (0.3) to see subtle corrections
2. The healer works best on transformations it was trained on
3. Green prediction text = correct prediction, Red = incorrect
4. Use the notebook version for interactive exploration
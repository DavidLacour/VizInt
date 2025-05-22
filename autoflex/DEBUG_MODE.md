# Debug Mode for AutoFlex Training

Debug mode provides a lightweight way to test the training pipeline with minimal resources. Perfect for development, testing, and conda/pyenv environments.

## Features

- **Batch size of 1** - Minimal memory usage
- **Limited dataset** - Only 50 training samples, 20 validation samples
- **2 epochs only** - Quick iterations
- **No multiprocessing** - Simpler debugging
- **Disabled wandb** - No experiment tracking overhead

## Quick Start

### 1. Setup Environment

Check if your environment is ready:
```bash
python setup_debug.py
```

### 2. Install Dependencies

For conda:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy scipy scikit-learn pillow tqdm wandb -c conda-forge
```

For pip/pyenv:
```bash
pip install -r requirements-debug.txt
```

### 3. Run Debug Training

Train a single model:
```bash
python debug_train.py
```

Train with specific backbone:
```bash
python debug_train.py --backbone resnet18
```

Test all backbones:
```bash
python debug_train.py --test-all
```

List available backbones:
```bash
python debug_train.py --list-backbones
```

## Configuration

Debug mode automatically:
- Sets `DEBUG_MODE=1` environment variable
- Uses batch size of 1
- Limits dataset to 50 train + 20 val samples
- Runs for only 2 epochs
- Disables wandb tracking
- Uses single-threaded data loading

## Dataset Requirements

The script expects the tiny-imagenet-200 dataset in the parent directory:
```
../tiny-imagenet-200/
├── train/
│   ├── n01443537/
│   └── ...
└── val/
    ├── images/
    └── val_annotations.txt
```

Download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip

## Environment Variables

- `DEBUG_MODE=1` - Enables debug mode
- `WANDB_MODE=disabled` - Disables wandb tracking

## Expected Output

Debug training should complete in 1-2 minutes and show:
```
🐛 DEBUG MODE ACTIVATED
==================================================
Batch size: 1
Training samples: 50
Validation samples: 20
Epochs: 2
Learning rate: 0.0001
Workers: 0
Device: CUDA
==================================================

🎯 Training vit_small classification model in DEBUG MODE
📱 Device: cuda
✅ Model created: vit_small
📂 Loading dataset from: ../tiny-imagenet-200
🐛 Limited dataset to 50 samples
🐛 Limited dataset to 20 samples
📊 Train batches: 50, Val batches: 20

🚀 Starting debug training for 2 epochs...
...
✅ Debug training completed successfully!
🎉 Final validation accuracy: XX.XX%
```

## Troubleshooting

### Dataset Not Found
```bash
❌ Dataset not found at /path/to/tiny-imagenet-200
```
Solution: Download and extract the dataset to the correct location.

### CUDA Out of Memory
Even with batch size 1, some models might not fit on small GPUs. The script will automatically fall back to CPU.

### Import Errors
```bash
❌ torch (not installed)
```
Solution: Install missing dependencies using the commands above.

## Integration with Main Codebase

Debug mode is designed to be non-intrusive:
- Uses same model architectures as main training
- Compatible with all backbones
- Can be enabled/disabled via environment variables
- No changes needed to existing code

## Performance Expectations

Debug mode is designed for testing, not performance:
- **Speed**: 1-2 minutes per run
- **Memory**: <2GB GPU memory
- **Accuracy**: Not meaningful (too few samples)
- **Purpose**: Verify pipeline works correctly

For actual training, use the main training scripts with proper configurations.
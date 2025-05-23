# Experimental Vision Transformers Debug Summary

## Overview
Successfully set up and tested experimental vision transformer architectures using the autoflex environment with debug mode (batch_size=1, minimal samples).

## Architecture Status

### ✅ Working Architectures (5/6)
1. **Fourier** - Uses Fourier-based attention mechanism
   - Parameters: ~87M
   - Status: Fixed dimension mismatch issue, now working

2. **ELFATT** - Efficient Linear Function Attention
   - Parameters: ~86M
   - Status: Working out of the box

3. **Mamba** - State-space model based architecture
   - Parameters: ~215M (largest model)
   - Status: Working but slower due to model size

4. **Hybrid** - Combines multiple attention mechanisms
   - Parameters: ~91M
   - Status: Working out of the box

5. **Mixed** - Uses different attention types in different layers
   - Parameters: ~135M
   - Status: Working out of the box

### ❌ Not Working (1/6)
1. **KAN** - Kolmogorov-Arnold Network attention
   - Issue: Tensor dimension mismatch in forward pass
   - Status: Needs complete reimplementation of KANLinear layer

## Scripts Created

1. **debug_train_experimental.py** - Main debug training script with minimal resources
2. **quick_test_architectures.py** - Quick forward pass test for all architectures
3. **run_debug_training.sh** - Automated training script for all working architectures
4. **test_all_debug.sh** - Initial test script (slower version)

## Key Features Implemented

- Debug mode with batch_size=1 and 50 training/20 validation samples
- Automatic environment activation using autoflex conda environment
- Early stopping with patience=1 for quick testing
- Checkpoint saving and validation
- Comprehensive logging and error handling
- WANDB disabled in debug mode for faster execution

## Usage

To run debug training for all working architectures:
```bash
cd /home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex/newExperimental
./run_debug_training.sh
```

To test a single architecture:
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./autoflexenv
export DEBUG_MODE=1
python debug_train_experimental.py --architecture fourier --data-root ../cifar10
```

## Next Steps

1. Fix KAN architecture implementation or exclude it from production
2. Run full training without debug mode for performance evaluation
3. Integrate with the main auto_train_all.py workflow
4. Add proper hyperparameter tuning for each architecture

## Performance Notes

- Fourier, ELFATT, and Hybrid are fastest to train
- Mamba is significantly slower due to its 215M parameters
- All models successfully complete 2 epochs in debug mode
- Early stopping works correctly with patience=1
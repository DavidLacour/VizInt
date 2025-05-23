# Production Training for Experimental Vision Transformers

This directory contains production-ready training scripts for experimental vision transformer architectures.

## Available Architectures

1. **fourier** - Fourier-based attention mechanism
2. **elfatt** - Efficient Linear Function Attention  
3. **mamba** - State-space model based architecture
4. **hybrid** - Combines multiple attention mechanisms
5. **mixed** - Uses different attention types in different layers

Note: KAN (Kolmogorov-Arnold Network) architecture is currently excluded due to implementation issues.

## Production Settings

- **Epochs**: 300 (with early stopping, patience=20)
- **Batch Size**: 256 (128 for Mamba, 192 for Mixed)
- **Learning Rate**: 1e-3 with cosine schedule
- **Warmup Epochs**: 20
- **Weight Decay**: 0.05
- **Image Size**: 224x224

## Training Scripts

### 1. Cluster Array Job (Recommended)
Trains all architectures in parallel using SLURM array jobs:
```bash
sbatch submit_experimental_cluster.sh
```
This will launch 5 parallel jobs, one for each architecture.

### 2. Sequential Training
Trains all architectures one after another on a single GPU:
```bash
sbatch train_all_experimental.sh
```

### 3. Single Architecture Training
For training a specific architecture:
```bash
# With SLURM
sbatch train_single_arch.sh --arch fourier --dataset /path/to/dataset

# Without SLURM (local)
./train_single_arch.sh --arch fourier --dataset ../tiny-imagenet-200 --batch-size 256 --epochs 300
```

## Dataset Paths

Update the `DATASET_PATH` variable in the scripts to point to your dataset:
- Tiny-ImageNet: `../tiny-imagenet-200`
- CIFAR-100: `../cifar100`
- ImageNet: `/path/to/imagenet`

## Memory Requirements

- **Fourier, ELFATT, Hybrid**: ~40GB GPU memory with batch size 256
- **Mamba**: ~50GB GPU memory with batch size 128 (215M parameters)
- **Mixed**: ~45GB GPU memory with batch size 192

## Output Structure

```
results/
└── architecture_name_jobid/
    ├── checkpoints/
    │   ├── architecture_best.pt
    │   └── architecture_epoch_X.pt
    ├── training.log
    └── config.log
```

## Monitoring Training

Training progress is logged to:
1. WandB project: "experimental-vit-production"
2. SLURM output files: `experimental_vit_*.out`
3. Architecture-specific logs: `results/*/training.log`

## Quick Start

1. Ensure the autoflex conda environment is activated
2. Update dataset path in the chosen script
3. Submit to cluster:
   ```bash
   cd /home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex/newExperimental
   sbatch submit_experimental_cluster.sh
   ```

## Performance Notes

- Early stopping monitors validation accuracy with patience=20 epochs
- Checkpoints are saved for best model and every 10 epochs
- Automatic cleanup removes intermediate checkpoints to save space
- Use `--no-cleanup` flag to keep all checkpoints

## Troubleshooting

If training fails:
1. Check SLURM error files: `experimental_vit_*.err`
2. Verify GPU memory is sufficient for batch size
3. Ensure dataset path is correct
4. Check conda environment activation

For architecture-specific issues, see the training logs in the results directory.
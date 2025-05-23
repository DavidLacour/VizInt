# Directory Structure on SCITAS

## Project Layout

```
/scratch/izar/dlacour/
├── VizInt/
│   └── autoflex/
│       ├── newExperimental/
│       │   ├── train_experimental.py
│       │   ├── experimental_vit.py
│       │   ├── fourier_attention.py
│       │   ├── linear_attention.py
│       │   ├── vision_mamba.py
│       │   ├── kan_transformer.py
│       │   ├── hybrid_architectures.py
│       │   ├── submit_experimental_cluster.sh
│       │   ├── train_all_experimental.sh
│       │   └── train_single_arch.sh
│       ├── early_stopping_trainer.py
│       ├── tiny-imagenet-200/
│       ├── cifar10/
│       └── cifar100/
│
└── experimental_results/  # Results saved here (next to VizInt)
    ├── all_architectures_[JOB_ID]/
    │   ├── summary.log
    │   ├── fourier/
    │   │   ├── training.log
    │   │   └── checkpoints/
    │   │       ├── fourier_best_final.pt
    │   │       └── fourier_exp_best.pt
    │   ├── elfatt/
    │   │   ├── training.log
    │   │   └── checkpoints/
    │   │       ├── elfatt_best_final.pt
    │   │       └── elfatt_exp_best.pt
    │   ├── mamba/
    │   │   ├── training.log
    │   │   └── checkpoints/
    │   │       ├── mamba_best_final.pt
    │   │       └── mamba_exp_best.pt
    │   ├── hybrid/
    │   │   ├── training.log
    │   │   └── checkpoints/
    │   │       ├── hybrid_best_final.pt
    │   │       └── hybrid_exp_best.pt
    │   └── mixed/
    │       ├── training.log
    │       └── checkpoints/
    │           ├── mixed_best_final.pt
    │           └── mixed_exp_best.pt
    │
    └── [ARCH]_[JOB_ID]/  # For individual architecture runs
        ├── config.log
        ├── training.log
        └── checkpoints/
            ├── [arch]_best_final.pt
            └── [arch]_exp_best.pt
```

## Key Points

1. **Results Location**: All results are saved to `/scratch/izar/dlacour/experimental_results/` (next to VizInt, not inside it)

2. **Job Organization**:
   - Sequential training: `all_architectures_[JOB_ID]/`
   - Array job training: `[ARCHITECTURE]_[JOB_ID]/`
   - Manual training: `[ARCHITECTURE]_[TIMESTAMP]/`

3. **Checkpoint Naming**:
   - `*_best_final.pt`: Final best model after training completes
   - `*_exp_best.pt`: Best model during training (early stopping checkpoint)

4. **Log Files**:
   - `summary.log`: Overall training summary (sequential jobs only)
   - `training.log`: Detailed training output for each architecture
   - `config.log`: Configuration details for the run

## Accessing Results

After training completes, find your results:
```bash
# List all results
ls -la /scratch/izar/dlacour/experimental_results/

# Check specific job
ls -la /scratch/izar/dlacour/experimental_results/all_architectures_2646027/

# View summary
cat /scratch/izar/dlacour/experimental_results/all_architectures_2646027/summary.log

# Check training progress
tail -f /scratch/izar/dlacour/experimental_results/all_architectures_2646027/fourier/training.log
```

## Storage Estimates

Per architecture:
- Checkpoint files: ~350MB each (depending on architecture)
- Log files: ~10-50MB
- Total per architecture: ~400MB

Full training (5 architectures):
- Total storage: ~2GB
# SCITAS Cluster Setup Guide

## Quick Start

1. **Copy files to SCITAS** (from your local machine):
   ```bash
   ./copy_to_cluster.sh
   ```

2. **SSH to SCITAS**:
   ```bash
   ssh dlacour@izar.epfl.ch
   ```

3. **Navigate to project**:
   ```bash
   cd /scratch/izar/dlacour/VizInt/autoflex/newExperimental
   ```

4. **Submit training job**:
   ```bash
   # For sequential training of all architectures
   sbatch train_all_experimental.sh
   
   # OR for parallel training (array job)
   sbatch submit_experimental_cluster.sh
   ```

## Important Paths on SCITAS

- **Project root**: `/scratch/izar/dlacour/VizInt/autoflex`
- **Experimental directory**: `/scratch/izar/dlacour/VizInt/autoflex/newExperimental`
- **Conda installation**: `/home/dlacour/anaconda3`
- **Dataset path**: `/scratch/izar/dlacour/VizInt/autoflex/tiny-imagenet-200`

## Troubleshooting

### If conda activation fails:
```bash
# Initialize conda first
/home/dlacour/anaconda3/bin/conda init bash
source ~/.bashrc

# Then activate environment
cd /scratch/izar/dlacour/VizInt/autoflex
conda activate ./autoflexenv
```

### If dataset path is different:
Edit the `DATASET_PATH` variable in the scripts:
```bash
vim train_all_experimental.sh
# Change line: DATASET_PATH="/path/to/your/dataset"
```

### Check job status:
```bash
squeue -u dlacour
```

### View job output:
```bash
tail -f experimental_vit_all_*.out
```

### Cancel job:
```bash
scancel <job_id>
```

## Dataset Locations

Update the dataset path in scripts based on your setup:
- Tiny-ImageNet: `/scratch/izar/dlacour/VizInt/autoflex/tiny-imagenet-200`
- CIFAR-10: `/scratch/izar/dlacour/VizInt/autoflex/cifar10`
- CIFAR-100: `/scratch/izar/dlacour/VizInt/autoflex/cifar100`

## Resource Allocation

Current settings in scripts:
- Time: 48 hours (sequential) / 24 hours (parallel)
- Memory: 64GB
- GPUs: 1
- CPUs: 10

Adjust these in the SBATCH headers if needed.
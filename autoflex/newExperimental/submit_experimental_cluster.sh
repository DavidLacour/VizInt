#!/bin/bash
#SBATCH --job-name=exp_vit_train
#SBATCH --time=24:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=experimental_vit_%A_%a.out
#SBATCH --error=experimental_vit_%A_%a.err
#SBATCH --array=0-4  # 5 architectures (excluding KAN)

# Experimental Vision Transformer training on cluster
# Optimized for production with proper batch sizes and epochs

# Define architectures array (excluding KAN which has issues)
ARCHITECTURES=("fourier" "elfatt" "mamba" "hybrid" "mixed")

# Get architecture for this array job
ARCH=${ARCHITECTURES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Training Experimental ViT: $ARCH"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "=========================================="

# Set up environment variables
export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=1d6641b737cd13fe32a9371dd3780308fee23512
export CUDA_VISIBLE_DEVICES=0

# Set dataset path (adjust as needed)
# Update this path to your actual dataset location on SCITAS
DATASET_PATH="/scratch/izar/dlacour/tiny-imagenet-200"

# Architecture-specific configurations
case $ARCH in
    "mamba")
        # Mamba is memory intensive, reduce batch size
        BATCH_SIZE=128
        ACCUMULATION_STEPS=2
        ;;
    "mixed")
        # Mixed architecture needs moderate batch size
        BATCH_SIZE=192
        ACCUMULATION_STEPS=1
        ;;
    *)
        # Default batch size for other architectures
        BATCH_SIZE=256
        ACCUMULATION_STEPS=1
        ;;
esac

# Create output directory next to VizInt
OUTPUT_DIR="/scratch/izar/dlacour/experimental_results/${ARCH}_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Log configuration
echo "Configuration:" | tee "${OUTPUT_DIR}/config.log"
echo "- Architecture: $ARCH" | tee -a "${OUTPUT_DIR}/config.log"
echo "- Dataset: $DATASET_PATH" | tee -a "${OUTPUT_DIR}/config.log"
echo "- Batch size: $BATCH_SIZE" | tee -a "${OUTPUT_DIR}/config.log"
echo "- Accumulation steps: $ACCUMULATION_STEPS" | tee -a "${OUTPUT_DIR}/config.log"
echo "" | tee -a "${OUTPUT_DIR}/config.log"

# Run training
python train_experimental.py \
    --architecture $ARCH \
    --data-root $DATASET_PATH \
    --epochs 300 \
    --batch-size $BATCH_SIZE \
    --learning-rate 1e-3 \
    --weight-decay 0.05 \
    --img-size 224 \
    --warmup-epochs 20 \
    --checkpoint-dir "${OUTPUT_DIR}/checkpoints" \
    --wandb-project "experimental-vit-production" \
    --patience 20 \
    --min-delta 1e-4 \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully for $ARCH"
    
    # Save final model info
    echo "Training completed at: $(date)" >> "${OUTPUT_DIR}/config.log"
    
    # Find best model checkpoint
    BEST_MODEL=$(find "${OUTPUT_DIR}/checkpoints" -name "*best*.pt" | head -1)
    if [ ! -z "$BEST_MODEL" ]; then
        echo "Best model saved at: $BEST_MODEL" >> "${OUTPUT_DIR}/config.log"
        ls -lh "$BEST_MODEL" >> "${OUTPUT_DIR}/config.log"
    fi
else
    echo "❌ Training failed for $ARCH"
    echo "Training failed at: $(date)" >> "${OUTPUT_DIR}/config.log"
fi

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
#!/bin/bash

# Train a single experimental architecture
# For local testing or single GPU training

# Default values
ARCH="fourier"
DATASET="../tiny-imagenet-200"
BATCH_SIZE=256
EPOCHS=300

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--arch ARCH] [--dataset PATH] [--batch-size SIZE] [--epochs N]"
            echo "Available architectures: fourier, elfatt, mamba, hybrid, mixed"
            exit 1
            ;;
    esac
done

echo "======================================================"
echo "Training Experimental Vision Transformer"
echo "Architecture: $ARCH"
echo "Dataset: $DATASET"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Started at: $(date)"
echo "======================================================"

# Navigate to project directory
cd /home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./autoflexenv

# Set up environment variables
export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=1d6641b737cd13fe32a9371dd3780308fee23512

# Navigate to experimental directory
cd newExperimental

# Create output directory
OUTPUT_DIR="results/${ARCH}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Run training
python train_experimental.py \
    --architecture $ARCH \
    --data-root $DATASET \
    --epochs $EPOCHS \
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
    echo ""
    echo "✅ Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Find best model
    BEST_MODEL=$(find "${OUTPUT_DIR}/checkpoints" -name "*best*.pt" | head -1)
    if [ ! -z "$BEST_MODEL" ]; then
        echo "Best model: $BEST_MODEL"
        ls -lh "$BEST_MODEL"
    fi
else
    echo ""
    echo "❌ Training failed!"
fi

echo ""
echo "Completed at: $(date)"
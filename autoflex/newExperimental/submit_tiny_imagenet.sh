#!/bin/bash
#SBATCH --job-name=exp_vit_tiny200
#SBATCH --account=cs-503
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=exp_vit_tiny200_%j.out
#SBATCH --error=exp_vit_tiny200_%j.err

# Activate local conda environment
cd /home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./autoflexenv

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=/scratch/izar/$USER/torch_cache
export HF_HOME=/scratch/izar/$USER/hf_cache
export WANDB_DIR=/scratch/izar/$USER/wandb

# Create scratch directories if they don't exist
mkdir -p /scratch/izar/$USER/torch_cache
mkdir -p /scratch/izar/$USER/hf_cache
mkdir -p /scratch/izar/$USER/wandb
mkdir -p /scratch/izar/$USER/experimental_vit_checkpoints
mkdir -p /scratch/izar/$USER/experimental_vit_logs

# Change to project directory
cd /home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex/newExperimental

# Set the dataset path
DATASET_PATH="/home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex/tiny-imagenet-200"

# Create log files
LOG_DIR="/scratch/izar/$USER/experimental_vit_logs"
SUMMARY_FILE="$LOG_DIR/training_results_summary.txt"

# Initialize summary file
echo "Experimental ViT Training on Tiny ImageNet-200 - $(date)" > $SUMMARY_FILE
echo "===========================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Function to train a model
train_model() {
    local arch=$1
    local epochs=$2
    local batch_size=$3
    local lr=$4
    local weight_decay=$5
    
    echo "" >> $SUMMARY_FILE
    echo "Training $arch architecture..." | tee -a $SUMMARY_FILE
    echo "Parameters: epochs=$epochs, batch_size=$batch_size, lr=$lr, wd=$weight_decay" | tee -a $SUMMARY_FILE
    
    python train_experimental.py \
        --architecture $arch \
        --data-root $DATASET_PATH \
        --epochs $epochs \
        --batch-size $batch_size \
        --lr $lr \
        --weight-decay $weight_decay \
        --img-size 64 \
        --num-classes 200 \
        --checkpoint-dir /scratch/izar/$USER/experimental_vit_checkpoints/$arch \
        --project-name "experimental-vit-tiny200-$USER" \
        --run-name "${arch}_tiny200" \
        --seed 42
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "✓ $arch training completed successfully" | tee -a $SUMMARY_FILE
        
        # Find the best checkpoint
        BEST_CKPT=$(ls -t /scratch/izar/$USER/experimental_vit_checkpoints/$arch/*_best.pt 2>/dev/null | head -1)
        if [ -n "$BEST_CKPT" ]; then
            echo "  Best checkpoint: $BEST_CKPT" | tee -a $SUMMARY_FILE
        fi
    else
        echo "✗ $arch training failed" | tee -a $SUMMARY_FILE
    fi
    echo "-------------------------------------------" >> $SUMMARY_FILE
}

# Train different architectures with appropriate hyperparameters
# Note: Tiny ImageNet-200 has 64x64 images, 200 classes

# Only train working architectures
echo "Training architectures that work with Tiny ImageNet-200..." | tee -a $SUMMARY_FILE

# Efficient Linear Attention (ELFATT) - WORKING
train_model "elfatt" 100 128 0.001 0.05

# Vision Mamba - WORKING
train_model "mamba" 100 64 0.0005 0.01

# Hybrid Architecture - WORKING
train_model "hybrid" 100 96 0.0008 0.05

# Skip problematic architectures for now
echo "" >> $SUMMARY_FILE
echo "Note: Fourier, KAN, and Mixed architectures skipped due to compatibility issues" >> $SUMMARY_FILE

# Summary
echo "" >> $SUMMARY_FILE
echo "===========================================" >> $SUMMARY_FILE
echo "Training completed at $(date)" >> $SUMMARY_FILE
echo "===========================================" >> $SUMMARY_FILE

# Display summary
cat $SUMMARY_FILE

echo "All training jobs completed!"
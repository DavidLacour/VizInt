#!/bin/bash
#SBATCH --job-name=experimental_vit
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=experimental_vit_%j.out
#SBATCH --error=experimental_vit_%j.err

# Load required modules
module load gcc/9.3.0
module load cuda/11.8
module load python/3.9.7

# Set up Python environment
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=/scratch/$USER/torch_cache
export HF_HOME=/scratch/$USER/hf_cache

cd /home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./autoflexenv

# Create output directories
mkdir -p /scratch/$USER/experimental_vit_checkpoints
mkdir -p /scratch/$USER/experimental_vit_logs

# Set Python path to include parent directory
export PYTHONPATH=$PYTHONPATH:/home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex

# Training parameters
DATA_ROOT="../../tiny-imagenet-200"  # Using Tiny-ImageNet dataset
EPOCHS=100
BATCH_SIZE=128
LEARNING_RATE=1e-3
PATIENCE=10
MIN_DELTA=1e-4
IMG_SIZE=64  # Tiny-ImageNet image size

# All architectures to train
ARCHITECTURES=("fourier" "elfatt" "mamba" "kan" "hybrid" "mixed")

# Create results summary file
RESULTS_FILE="/scratch/$USER/experimental_vit_logs/training_results_summary.txt"
echo "Experimental ViT Training Results Summary" > $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "Started at: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Train all architectures
for ARCHITECTURE in "${ARCHITECTURES[@]}"; do
    echo "================================================"
    echo "Starting training for architecture: $ARCHITECTURE"
    echo "================================================"
    echo "Early stopping patience: $PATIENCE epochs"
    echo "Checkpoint cleanup: Enabled (only best model will be kept)"
    echo ""
    
    # Log to summary file
    echo "Architecture: $ARCHITECTURE" >> $RESULTS_FILE
    echo "Start time: $(date)" >> $RESULTS_FILE
    
    # Run training with early stopping and checkpoint management
    python train_experimental.py \
        --architecture $ARCHITECTURE \
        --data-root $DATA_ROOT \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE \
        --patience $PATIENCE \
        --min-delta $MIN_DELTA \
        --checkpoint-dir /scratch/$USER/experimental_vit_checkpoints \
        --wandb-project experimental-vit-$USER \
        --warmup-epochs 5 \
        --img-size $IMG_SIZE
    
    TRAIN_EXIT_CODE=$?
    
    # Check if training was successful
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully for $ARCHITECTURE!"
        echo "Training status: SUCCESS" >> $RESULTS_FILE
        
        # Run evaluation on the best model
        echo "Running evaluation for $ARCHITECTURE..."
        python evaluate_experimental.py \
            --architecture $ARCHITECTURE \
            --checkpoint-dir /scratch/$USER/experimental_vit_checkpoints \
            --data-root $DATA_ROOT \
            --img-size $IMG_SIZE 2>&1 | tee -a $RESULTS_FILE
        
        EVAL_EXIT_CODE=$?
        if [ $EVAL_EXIT_CODE -eq 0 ]; then
            echo "Evaluation completed successfully for $ARCHITECTURE!"
            echo "Evaluation status: SUCCESS" >> $RESULTS_FILE
        else
            echo "Evaluation failed for $ARCHITECTURE with exit code $EVAL_EXIT_CODE"
            echo "Evaluation status: FAILED (exit code $EVAL_EXIT_CODE)" >> $RESULTS_FILE
        fi
    else
        echo "Training failed for $ARCHITECTURE with exit code $TRAIN_EXIT_CODE"
        echo "Training status: FAILED (exit code $TRAIN_EXIT_CODE)" >> $RESULTS_FILE
    fi
    
    # List checkpoint files for this architecture
    echo "Checkpoint files for $ARCHITECTURE:"
    ls -la /scratch/$USER/experimental_vit_checkpoints/$ARCHITECTURE/
    
    echo "End time: $(date)" >> $RESULTS_FILE
    echo "----------------------------------------" >> $RESULTS_FILE
    echo "" >> $RESULTS_FILE
    
    # Add separator between architectures
    echo ""
    echo "================================================"
    echo ""
done

# Final summary
echo "All architectures training completed!"
echo "" >> $RESULTS_FILE
echo "Completed at: $(date)" >> $RESULTS_FILE

# Display summary
echo ""
echo "Training Summary:"
echo "================="
cat $RESULTS_FILE

# List all final checkpoints
echo ""
echo "All final checkpoint files:"
for ARCHITECTURE in "${ARCHITECTURES[@]}"; do
    echo "Architecture: $ARCHITECTURE"
    ls -la /scratch/$USER/experimental_vit_checkpoints/$ARCHITECTURE/ 2>/dev/null || echo "  No checkpoints found"
done 
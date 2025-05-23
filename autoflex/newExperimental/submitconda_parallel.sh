#!/bin/bash
#SBATCH --job-name=exp_vit_parallel
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:6
#SBATCH --output=exp_vit_parallel_%j.out
#SBATCH --error=exp_vit_parallel_%j.err

# Load required modules
module load gcc/9.3.0
module load cuda/11.8
module load python/3.9.7

# Set up Python environment
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
DATA_ROOT="/scratch/$USER/datasets"  # Update this to your dataset path
EPOCHS=100
BATCH_SIZE=128
LEARNING_RATE=1e-3
PATIENCE=10
MIN_DELTA=1e-4

# All architectures to train
ARCHITECTURES=("fourier" "elfatt" "mamba" "kan" "hybrid" "mixed")

# Function to train a single architecture
train_architecture() {
    local ARCHITECTURE=$1
    local GPU_ID=$2
    
    echo "[GPU $GPU_ID] Starting training for architecture: $ARCHITECTURE"
    
    # Set GPU for this process
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Run training
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
        --warmup-epochs 5 > /scratch/$USER/experimental_vit_logs/train_${ARCHITECTURE}.log 2>&1
    
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "[GPU $GPU_ID] Training completed for $ARCHITECTURE. Running evaluation..."
        
        # Run evaluation
        python evaluate_experimental.py \
            --architecture $ARCHITECTURE \
            --checkpoint-dir /scratch/$USER/experimental_vit_checkpoints \
            --data-root $DATA_ROOT > /scratch/$USER/experimental_vit_logs/eval_${ARCHITECTURE}.log 2>&1
        
        echo "[GPU $GPU_ID] Completed $ARCHITECTURE"
    else
        echo "[GPU $GPU_ID] Training failed for $ARCHITECTURE with exit code $TRAIN_EXIT_CODE"
    fi
}

# Export the function so it's available to parallel
export -f train_architecture
export DATA_ROOT EPOCHS BATCH_SIZE LEARNING_RATE PATIENCE MIN_DELTA PYTHONPATH

# Create a temporary file with architecture-GPU pairs
TEMP_FILE=$(mktemp)
for i in "${!ARCHITECTURES[@]}"; do
    echo "${ARCHITECTURES[$i]} $i" >> $TEMP_FILE
done

echo "Starting parallel training on 6 GPUs..."
echo "Architectures: ${ARCHITECTURES[@]}"
echo ""

# Run training in parallel using GNU parallel
cat $TEMP_FILE | parallel --colsep ' ' -j 6 train_architecture {1} {2}

# Clean up
rm $TEMP_FILE

# Wait for all jobs to complete
wait

echo ""
echo "All parallel training jobs completed!"
echo ""

# Create summary report
RESULTS_FILE="/scratch/$USER/experimental_vit_logs/parallel_training_summary.txt"
echo "Experimental ViT Parallel Training Results Summary" > $RESULTS_FILE
echo "================================================" >> $RESULTS_FILE
echo "Completed at: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Collect results
for ARCHITECTURE in "${ARCHITECTURES[@]}"; do
    echo "Architecture: $ARCHITECTURE" >> $RESULTS_FILE
    echo "------------------------" >> $RESULTS_FILE
    
    # Check if checkpoint exists
    if [ -f "/scratch/$USER/experimental_vit_checkpoints/$ARCHITECTURE/best_model_final.pt" ] || \
       [ -f "/scratch/$USER/experimental_vit_checkpoints/$ARCHITECTURE/${ARCHITECTURE}_exp_best.pt" ]; then
        echo "Status: SUCCESS" >> $RESULTS_FILE
        
        # Extract evaluation results if available
        if [ -f "/scratch/$USER/experimental_vit_logs/eval_${ARCHITECTURE}.log" ]; then
            grep -E "Top-1 Accuracy:|Top-5 Accuracy:" /scratch/$USER/experimental_vit_logs/eval_${ARCHITECTURE}.log >> $RESULTS_FILE 2>/dev/null || echo "Evaluation results not found" >> $RESULTS_FILE
        fi
    else
        echo "Status: FAILED or INCOMPLETE" >> $RESULTS_FILE
    fi
    
    echo "" >> $RESULTS_FILE
done

# Display summary
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

# Display individual logs location
echo ""
echo "Individual training logs available at:"
echo "/scratch/$USER/experimental_vit_logs/train_*.log"
echo ""
echo "Individual evaluation logs available at:"
echo "/scratch/$USER/experimental_vit_logs/eval_*.log"
#!/bin/bash
#SBATCH --job-name=exp_vit_all
#SBATCH --time=48:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=experimental_vit_all_%j.out
#SBATCH --error=experimental_vit_all_%j.err

# Train all experimental architectures sequentially
# For use when array jobs are not preferred

echo "======================================================"
echo "Training All Experimental Vision Transformers"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "======================================================"

# Navigate to project directory on SCITAS
cd /scratch/izar/dlacour/VizInt/autoflex

# Activate conda environment
source /home/dlacour/anaconda3/etc/profile.d/conda.sh
conda activate ./autoflexenv

# Set up environment variables
export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=1d6641b737cd13fe32a9371dd3780308fee23512
export CUDA_VISIBLE_DEVICES=0

# Navigate to experimental directory
cd newExperimental

# Set dataset path (adjust as needed)
# Update this path to your actual dataset location on SCITAS
DATASET_PATH="/scratch/izar/dlacour/VizInt/autoflex/tiny-imagenet-200"

# Define architectures (excluding KAN)
ARCHITECTURES=("fourier" "elfatt" "mamba" "hybrid" "mixed")

# Create main results directory next to VizInt
RESULTS_DIR="/scratch/izar/dlacour/experimental_results/all_architectures_${SLURM_JOB_ID}"
mkdir -p $RESULTS_DIR

# Summary log
SUMMARY_LOG="${RESULTS_DIR}/summary.log"
echo "Training Summary - Started at $(date)" > $SUMMARY_LOG
echo "=======================================" >> $SUMMARY_LOG

# Train each architecture
for ARCH in "${ARCHITECTURES[@]}"; do
    echo ""
    echo "======================================================"
    echo "Training: $ARCH"
    echo "Started at: $(date)"
    echo "======================================================"
    
    # Architecture-specific batch size
    case $ARCH in
        "mamba")
            BATCH_SIZE=128
            ;;
        "mixed")
            BATCH_SIZE=192
            ;;
        *)
            BATCH_SIZE=256
            ;;
    esac
    
    # Create architecture-specific directory
    ARCH_DIR="${RESULTS_DIR}/${ARCH}"
    mkdir -p $ARCH_DIR
    
    # Log start
    echo "" >> $SUMMARY_LOG
    echo "$ARCH - Started at $(date)" >> $SUMMARY_LOG
    echo "Batch size: $BATCH_SIZE" >> $SUMMARY_LOG
    
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
        --checkpoint-dir "${ARCH_DIR}/checkpoints" \
        --wandb-project "experimental-vit-production" \
        --patience 20 \
        --min-delta 1e-4 \
        2>&1 | tee "${ARCH_DIR}/training.log"
    
    # Check status
    if [ $? -eq 0 ]; then
        echo "✅ $ARCH: Training completed successfully"
        echo "$ARCH - ✅ Success - Completed at $(date)" >> $SUMMARY_LOG
        
        # Find and log best model
        BEST_MODEL=$(find "${ARCH_DIR}/checkpoints" -name "*best*.pt" | head -1)
        if [ ! -z "$BEST_MODEL" ]; then
            echo "Best model: $BEST_MODEL" >> $SUMMARY_LOG
            
            # Extract best accuracy from log if available
            BEST_ACC=$(grep "Best validation accuracy" "${ARCH_DIR}/training.log" | tail -1 | awk '{print $4}')
            if [ ! -z "$BEST_ACC" ]; then
                echo "Best validation accuracy: $BEST_ACC" >> $SUMMARY_LOG
            fi
        fi
    else
        echo "❌ $ARCH: Training failed"
        echo "$ARCH - ❌ Failed - at $(date)" >> $SUMMARY_LOG
    fi
done

echo ""
echo "======================================================"
echo "All Training Completed at: $(date)"
echo "======================================================"

# Final summary
echo "" >> $SUMMARY_LOG
echo "=======================================" >> $SUMMARY_LOG
echo "Training completed at $(date)" >> $SUMMARY_LOG

# Display summary
echo ""
echo "FINAL SUMMARY:"
cat $SUMMARY_LOG

echo ""
echo "Results saved to: $RESULTS_DIR"
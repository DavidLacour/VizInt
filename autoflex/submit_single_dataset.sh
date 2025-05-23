#!/bin/bash
#SBATCH --job-name=single_dataset_eval
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=single_dataset_%j.out
#SBATCH --error=single_dataset_%j.err

# Usage: sbatch submit_single_dataset.sh <dataset_path>
# Example: sbatch submit_single_dataset.sh oxford_pets

DATASET_PATH=${1:-"oxford_pets"}

# Activate local conda environment
cd /home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./autoflexenv

# Set up WandB
export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=1d6641b737cd13fe32a9371dd3780308fee23512

echo "=== SINGLE DATASET EVALUATION: $DATASET_PATH ===" | tee single_dataset_results.log
echo "Started at: $(date)" | tee -a single_dataset_results.log
echo "Job ID: $SLURM_JOB_ID" | tee -a single_dataset_results.log
echo "" | tee -a single_dataset_results.log

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset $DATASET_PATH not found!" | tee -a single_dataset_results.log
    exit 1
fi

# Key backbones for fast evaluation
BACKBONES=(
    "vit_small"
    "resnet18_pretrained" 
    "deit_small_pretrained"
)

echo "Dataset: $DATASET_PATH" | tee -a single_dataset_results.log
echo "Backbones to test: ${BACKBONES[@]}" | tee -a single_dataset_results.log
echo "" | tee -a single_dataset_results.log

# Phase 1: Train classification models
echo "=== Phase 1: Classification Models ===" | tee -a single_dataset_results.log
python3 auto_train_all.py \
    --dataset_path "$DATASET_PATH" \
    --models classification \
    --backbones "${BACKBONES[@]}" \
    --continue_on_error 2>&1 | tee -a single_dataset_results.log

# Phase 2: Train healer models
echo "=== Phase 2: Healer Models ===" | tee -a single_dataset_results.log
python3 auto_train_all.py \
    --dataset_path "$DATASET_PATH" \
    --models healer \
    --backbones "${BACKBONES[@]}" \
    --continue_on_error 2>&1 | tee -a single_dataset_results.log

# Phase 3: Train TTT models (requires classification base models)
echo "=== Phase 3: TTT Models ===" | tee -a single_dataset_results.log
python3 auto_train_all.py \
    --dataset_path "$DATASET_PATH" \
    --models ttt \
    --backbones "${BACKBONES[@]}" \
    --continue_on_error 2>&1 | tee -a single_dataset_results.log

# Final summary
echo "=== EVALUATION COMPLETED ===" | tee -a single_dataset_results.log
echo "Finished at: $(date)" | tee -a single_dataset_results.log

# List trained models
echo "Trained models:" | tee -a single_dataset_results.log
python3 auto_train_all.py --list | tee -a single_dataset_results.log

echo "Single dataset evaluation completed. Results saved to single_dataset_results.log"
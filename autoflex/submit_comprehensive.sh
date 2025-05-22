#!/bin/bash
#SBATCH --job-name=comprehensive_eval
#SBATCH --time=48:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=comprehensive_eval_%j.out
#SBATCH --error=comprehensive_eval_%j.err

# Activate environment
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda activate nanofm

# Set up WandB
export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=1d6641b737cd13fe32a9371dd3780308fee23512

# Create comprehensive evaluation log
echo "=== COMPREHENSIVE MODEL AND DATASET EVALUATION ===" > comprehensive_results.log
echo "Started at: $(date)" >> comprehensive_results.log
echo "Job ID: $SLURM_JOB_ID" >> comprehensive_results.log
echo "" >> comprehensive_results.log

# Define datasets to evaluate
DATASETS=(
    "oxford_pets"
    "hybrid_small_imagenetc" 
    "laionc_small"
    "../tiny-imagenet-200"
)

# Define model types to evaluate
MODEL_TYPES=(
    "classification"
    "healer"
)

# Define key backbones for comprehensive evaluation
BACKBONES=(
    "vit_small"
    "vit_base"
    "resnet18_pretrained"
    "resnet50_pretrained"
    "deit_small_pretrained"
    "swin_small_pretrained"
)

echo "Datasets to evaluate: ${DATASETS[@]}" >> comprehensive_results.log
echo "Model types: ${MODEL_TYPES[@]}" >> comprehensive_results.log
echo "Backbones: ${BACKBONES[@]}" >> comprehensive_results.log
echo "" >> comprehensive_results.log

# Function to train and evaluate on a single dataset
evaluate_dataset() {
    local dataset_path=$1
    local dataset_name=$(basename "$dataset_path")
    
    echo "========================================" >> comprehensive_results.log
    echo "EVALUATING DATASET: $dataset_name" >> comprehensive_results.log
    echo "Path: $dataset_path" >> comprehensive_results.log
    echo "Started at: $(date)" >> comprehensive_results.log
    echo "========================================" >> comprehensive_results.log
    
    # First, train all classification models (needed for TTT)
    echo "Phase 1: Training classification models..." >> comprehensive_results.log
    for backbone in "${BACKBONES[@]}"; do
        echo "Training classification with $backbone..." >> comprehensive_results.log
        python3 auto_train_all.py \
            --dataset_path "$dataset_path" \
            --models classification \
            --backbones "$backbone" \
            --continue_on_error 2>&1 | tee -a comprehensive_results.log
    done
    
    # Then train other model types
    echo "Phase 2: Training other model types..." >> comprehensive_results.log
    for model_type in "${MODEL_TYPES[@]}"; do
        if [ "$model_type" != "classification" ]; then
            for backbone in "${BACKBONES[@]}"; do
                echo "Training $model_type with $backbone..." >> comprehensive_results.log
                python3 auto_train_all.py \
                    --dataset_path "$dataset_path" \
                    --models "$model_type" \
                    --backbones "$backbone" \
                    --continue_on_error 2>&1 | tee -a comprehensive_results.log
            done
        fi
    done
    
    # Add TTT training if classification models exist
    echo "Phase 3: Training TTT models..." >> comprehensive_results.log
    for backbone in "${BACKBONES[@]}"; do
        echo "Training TTT with $backbone..." >> comprehensive_results.log
        python3 auto_train_all.py \
            --dataset_path "$dataset_path" \
            --models ttt \
            --backbones "$backbone" \
            --continue_on_error 2>&1 | tee -a comprehensive_results.log
    done
    
    echo "Dataset $dataset_name evaluation completed at: $(date)" >> comprehensive_results.log
    echo "" >> comprehensive_results.log
}

# Main evaluation loop
echo "Starting comprehensive evaluation..." >> comprehensive_results.log

for dataset in "${DATASETS[@]}"; do
    # Check if dataset exists
    if [ -d "$dataset" ]; then
        evaluate_dataset "$dataset"
    else
        echo "WARNING: Dataset $dataset not found, skipping..." >> comprehensive_results.log
    fi
done

# Final summary
echo "========================================" >> comprehensive_results.log
echo "COMPREHENSIVE EVALUATION COMPLETED" >> comprehensive_results.log
echo "Finished at: $(date)" >> comprehensive_results.log
echo "========================================" >> comprehensive_results.log

# List all trained models
echo "Listing all trained models:" >> comprehensive_results.log
python3 auto_train_all.py --list >> comprehensive_results.log

# Create summary of results
echo "" >> comprehensive_results.log
echo "SUMMARY OF TRAINED MODELS:" >> comprehensive_results.log
find . -name "best_model.pt" -type f | wc -l >> comprehensive_results.log
echo "model files found." >> comprehensive_results.log

echo "Comprehensive evaluation completed. Results saved to comprehensive_results.log"
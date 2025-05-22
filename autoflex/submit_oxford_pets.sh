#!/bin/bash
#SBATCH --job-name=oxford_pets_eval
#SBATCH --time=8:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=oxford_pets_eval_%j.out
#SBATCH --error=oxford_pets_eval_%j.err

# Dedicated Oxford-IIIT Pet Dataset evaluation

# Activate environment
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda activate nanofm

# Set up WandB
export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=1d6641b737cd13fe32a9371dd3780308fee23512

echo "=== OXFORD-IIIT PET DATASET EVALUATION ===" | tee oxford_pets_results.log
echo "Started at: $(date)" | tee -a oxford_pets_results.log
echo "Job ID: $SLURM_JOB_ID" | tee -a oxford_pets_results.log
echo "" | tee -a oxford_pets_results.log

# Check if Oxford pets dataset exists
if [ ! -d "oxford_pets" ]; then
    echo "ERROR: Oxford pets dataset not found!" | tee -a oxford_pets_results.log
    exit 1
fi

# Test the dataset loading first
echo "=== Testing Dataset Loading ===" | tee -a oxford_pets_results.log
python3 debug_multi_dataset.py --dataset oxford_pets --backbone vit_small 2>&1 | tee -a oxford_pets_results.log

# Full evaluation with multiple backbones
BACKBONES=(
    "vit_small"
    "vit_base"
    "resnet18_pretrained"
    "resnet50_pretrained"
    "deit_small_pretrained"
)

echo "=== Training Models on Oxford Pets ===" | tee -a oxford_pets_results.log
echo "Backbones: ${BACKBONES[@]}" | tee -a oxford_pets_results.log

# Train all model types with all backbones
python3 auto_train_all.py \
    --dataset_path oxford_pets \
    --models classification healer \
    --backbones "${BACKBONES[@]}" \
    --continue_on_error 2>&1 | tee -a oxford_pets_results.log

# Train TTT models (requires classification base models)
python3 auto_train_all.py \
    --dataset_path oxford_pets \
    --models ttt \
    --backbones "${BACKBONES[@]}" \
    --continue_on_error 2>&1 | tee -a oxford_pets_results.log

echo "=== Oxford Pets Evaluation Completed ===" | tee -a oxford_pets_results.log
echo "Finished at: $(date)" | tee -a oxford_pets_results.log

# Summary
python3 auto_train_all.py --list | tee -a oxford_pets_results.log

echo "Oxford pets evaluation completed. Results saved to oxford_pets_results.log"
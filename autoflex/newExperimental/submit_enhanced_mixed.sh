#!/bin/bash
#SBATCH --job-name=mixed_enhanced
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/mixed_enhanced_%j.out
#SBATCH --error=logs/mixed_enhanced_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
ARCHITECTURE="mixed"
DATA_ROOT="../../../tiny-imagenet-200"
SAVE_DIR="../../../experimentalmodels"
EPOCHS=100
BATCH_SIZE=96  # Mixed models: balanced batch size
PATIENCE=10

# Setup environment
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda activate nanofm

# Export wandb credentials
export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=1d6641b737cd13fe32a9371dd3780308fee23512

echo "Starting enhanced training for $ARCHITECTURE"
echo "Data root: $DATA_ROOT"
echo "Save directory: $SAVE_DIR"
echo "Time: $(date)"

# Run enhanced training with early stopping and proper saving
python train_experimental_enhanced.py \
    --architecture $ARCHITECTURE \
    --data-root $DATA_ROOT \
    --save-dir $SAVE_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --patience $PATIENCE \
    --wandb-project experimental-vit-enhanced

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Run evaluation with perturbations
    echo "Starting evaluation with perturbations..."
    python evaluate_experimental_perturbations.py \
        --model-path "$SAVE_DIR/$ARCHITECTURE/best_model.pt" \
        --architecture $ARCHITECTURE \
        --data-root $DATA_ROOT \
        --severities 0.0 0.1 0.3 0.5 0.75 1.0
        
    echo "Evaluation completed!"
else
    echo "Training failed!"
    exit 1
fi

echo "All tasks completed at $(date)"
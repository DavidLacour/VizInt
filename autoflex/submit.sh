#!/bin/bash
#SBATCH --job-name=flex         # Change as needed
#SBATCH --time=6:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1                    # Request 2 GPUs
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10               # Adjust CPU allocation if needed
#SBATCH --output=interactive_job_flex.out    # Output log file
#SBATCH --error=interactive_job_flex.err     # Error log file

CONFIG_FILE=$1
WANDB=$2
NUM_GPUS=$3

# Activate conda environment
cd /home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./autoflexenv
export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=1d6641b737cd13fe32a9371dd3780308fee23512 
python3 auto_train_all.py

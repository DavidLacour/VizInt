#!/bin/bash
#SBATCH --job-name=tiny_fc3          # Change as needed
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1                    # Request 2 GPUs
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10               # Adjust CPU allocation if needed
#SBATCH --output=interactive_fc3.out    # Output log file
#SBATCH --error=interactive_fc3.err     # Error log file

CONFIG_FILE=$1
WANDB=$2
NUM_GPUS=$3

~/miniconda3/bin/conda init bash
source ~/.bashrc
conda activate nanofm
export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=1d6641b737cd13fe32a9371dd3780308fee23512 
python3 main_baselines_3fc_integration.py --mode all

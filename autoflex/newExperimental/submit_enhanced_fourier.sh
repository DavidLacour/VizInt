#!/bin/bash
#SBATCH --job-name=exp_fourier_enhanced
#SBATCH --output=fourier_enhanced_%j.out
#SBATCH --error=fourier_enhanced_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

module load python/3.9
module load cuda/11.8

cd /home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex/newExperimental

python train_experimental_enhanced.py \
    --architecture fourier \
    --data-root ../tiny-imagenet-200 \
    --epochs 100 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --save-dir ../../../experimentalmodels
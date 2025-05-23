#!/bin/bash

# Script to copy experimental training files to SCITAS cluster
# Run this from your local machine

echo "Copying experimental training files to SCITAS cluster..."

# Define source and destination
LOCAL_DIR="/home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex/newExperimental"
REMOTE_HOST="dlacour@izar.epfl.ch"
REMOTE_DIR="/scratch/izar/dlacour/VizInt/autoflex/newExperimental"

# Files to copy
FILES=(
    "train_experimental.py"
    "experimental_vit.py"
    "fourier_attention.py"
    "linear_attention.py"
    "vision_mamba.py"
    "kan_transformer.py"
    "hybrid_architectures.py"
    "submit_experimental_cluster.sh"
    "train_all_experimental.sh"
    "train_single_arch.sh"
    "PRODUCTION_README.md"
)

# Copy parent directory files needed
PARENT_FILES=(
    "../early_stopping_trainer.py"
)

echo "Creating remote directory structure..."
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR"

echo "Copying experimental module files..."
for file in "${FILES[@]}"; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        echo "  - Copying $file"
        scp "$LOCAL_DIR/$file" "$REMOTE_HOST:$REMOTE_DIR/"
    else
        echo "  ! Warning: $file not found"
    fi
done

echo "Copying required parent directory files..."
for file in "${PARENT_FILES[@]}"; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        echo "  - Copying $(basename $file) to parent directory"
        scp "$LOCAL_DIR/$file" "$REMOTE_HOST:/scratch/izar/dlacour/VizInt/autoflex/"
    else
        echo "  ! Warning: $file not found"
    fi
done

echo ""
echo "Setting execute permissions on scripts..."
ssh $REMOTE_HOST "cd $REMOTE_DIR && chmod +x *.sh"

echo ""
echo "✅ Files copied to SCITAS cluster!"
echo ""
echo "To run on the cluster:"
echo "1. SSH to SCITAS: ssh dlacour@izar.epfl.ch"
echo "2. Navigate to: cd /scratch/izar/dlacour/VizInt/autoflex/newExperimental"
echo "3. Submit job: sbatch train_all_experimental.sh"
echo ""
echo "Or for array job (parallel training):"
echo "   sbatch submit_experimental_cluster.sh"
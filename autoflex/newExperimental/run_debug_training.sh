#!/bin/bash

# Run experimental architectures in debug mode using autoflex environment
# Optimized for quick testing with minimal resources

echo "🔧 Running Experimental Vision Transformers Debug Training"
echo "========================================================="
echo "Using autoflex environment with batch_size=1 and minimal samples"
echo ""

# Activate the autoflex environment
cd /home/david-lacour/Documents/transformerVision/githubs/VizInt/autoflex
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./autoflexenv

# Set debug mode
export DEBUG_MODE=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled  # Disable wandb for debug

# Change to experimental directory
cd newExperimental

# Test dataset - using CIFAR-10 for quick testing
DATA_ROOT="../cifar10"

# Create results directory
mkdir -p debug_results
mkdir -p debug_checkpoints

# Working architectures (excluding KAN which has issues)
ARCHITECTURES=("fourier" "elfatt" "mamba" "hybrid" "mixed")

echo "Testing architectures: ${ARCHITECTURES[@]}"
echo ""

# Function to run training for a single architecture
train_architecture() {
    local arch=$1
    echo "============================================"
    echo "Training: $arch"
    echo "Started at: $(date)"
    echo "============================================"
    
    # Run debug training
    python debug_train_experimental.py \
        --architecture "$arch" \
        --data-root "$DATA_ROOT" \
        --checkpoint-dir "debug_checkpoints" 2>&1 | tee "debug_results/${arch}_training.log"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✅ $arch: Training completed successfully"
        
        # Check if checkpoint was saved
        if [ -f "debug_checkpoints/${arch}/debug_final.pt" ]; then
            echo "✅ $arch: Checkpoint saved"
            # Get file size
            ls -lh "debug_checkpoints/${arch}/debug_final.pt" | awk '{print "   File size:", $5}'
        else
            echo "⚠️  $arch: No checkpoint found"
        fi
    else
        echo "❌ $arch: Training failed"
    fi
    
    echo ""
}

# Run training for each architecture
for arch in "${ARCHITECTURES[@]}"; do
    train_architecture "$arch"
done

echo "============================================"
echo "Debug training completed at: $(date)"
echo "============================================"

# Summary report
echo ""
echo "📊 TRAINING SUMMARY:"
echo "==================="

successful=0
failed=0

for arch in "${ARCHITECTURES[@]}"; do
    if grep -q "Debug training completed successfully" "debug_results/${arch}_training.log" 2>/dev/null; then
        echo "✅ $arch: Success"
        ((successful++))
        
        # Extract accuracy if available
        accuracy=$(grep "Best validation accuracy:" "debug_results/${arch}_training.log" | tail -1 | awk '{print $4}')
        if [ ! -z "$accuracy" ]; then
            echo "   Best val accuracy: $accuracy"
        fi
    else
        echo "❌ $arch: Failed"
        ((failed++))
    fi
done

echo ""
echo "Total: $successful/$((successful + failed)) architectures trained successfully"

# List all checkpoints
echo ""
echo "📁 SAVED CHECKPOINTS:"
echo "===================="
find debug_checkpoints -name "*.pt" -type f -exec ls -lh {} \; | awk '{print $9 ":", $5}'

echo ""
echo "✅ Debug training workflow completed!"
echo "   - Logs saved to: debug_results/"
echo "   - Checkpoints saved to: debug_checkpoints/"
echo ""
echo "To run full training, remove DEBUG_MODE and adjust parameters."
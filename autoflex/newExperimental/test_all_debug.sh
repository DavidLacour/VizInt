#!/bin/bash

# Test all experimental architectures in debug mode
# Using autoflex environment with minimal resources

echo "🔧 Testing Experimental Vision Transformers in Debug Mode"
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

# Change to experimental directory
cd newExperimental

# Test dataset - using CIFAR-10 for quick testing
DATA_ROOT="../cifar10"

# Create results directory
mkdir -p debug_results

# Test each architecture
ARCHITECTURES=("fourier" "elfatt" "mamba" "kan" "hybrid" "mixed")

echo "Testing architectures: ${ARCHITECTURES[@]}"
echo ""

# Test each architecture
for arch in "${ARCHITECTURES[@]}"; do
    echo "============================================"
    echo "Testing: $arch"
    echo "Started at: $(date)"
    echo "============================================"
    
    # Run debug training
    python debug_train_experimental.py \
        --architecture "$arch" \
        --data-root "$DATA_ROOT" \
        --checkpoint-dir "debug_checkpoints" 2>&1 | tee "debug_results/${arch}_debug.log"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✅ $arch: SUCCESS"
    else
        echo "❌ $arch: FAILED"
    fi
    
    echo ""
done

echo "============================================"
echo "Debug testing completed at: $(date)"
echo "Results saved to debug_results/"
echo "============================================"

# Summary
echo ""
echo "📊 SUMMARY:"
for arch in "${ARCHITECTURES[@]}"; do
    if grep -q "Debug training completed successfully" "debug_results/${arch}_debug.log" 2>/dev/null; then
        echo "  ✅ $arch: Passed"
    else
        echo "  ❌ $arch: Failed"
    fi
done
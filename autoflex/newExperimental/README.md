# Experimental Vision Transformers

This directory contains implementations of cutting-edge vision transformer architectures that go beyond standard self-attention mechanisms. All implementations are based on the latest research from 2024-2025.

## Architecture Implementations

### 1. Fourier Attention (`fourier_attention.py`)
- **FourierAttention**: Replaces dot-product kernels with generalized Fourier integral kernels
- **FNetBlock**: Google's FNet approach using Fourier Transform instead of self-attention
- Based on FourierFormer and FNet research

### 2. Linear Attention (`linear_attention.py`)
- **LinearAttention**: O(N) complexity attention mechanism
- **EfficientLinearAttention (ELFATT)**: Optimized for high-resolution vision tasks
- **GatedLinearAttention**: Hardware-efficient training with gating mechanisms

### 3. Vision Mamba (`vision_mamba.py`)
- **SelectiveScan**: Core Mamba mechanism with linear complexity
- **BidirectionalMamba**: Bidirectional processing for better context modeling
- **VisionMambaBackbone**: Complete vision backbone using state space models

### 4. KAN Transformer (`kan_transformer.py`)
- **KANLinear**: Kolmogorov-Arnold Network layers with learnable spline functions
- **KANAttention**: Attention mechanism using KAN layers
- **SineKAN**: Efficient KAN variant using sine activation functions

### 5. Hybrid Architectures (`hybrid_architectures.py`)
- **ConvStem**: Convolutional stem for local feature extraction
- **HybridAttention**: Combines global attention with local convolution
- **CNNTransformerHybrid**: Full hybrid CNN-Transformer architecture

### 6. Unified Framework (`experimental_vit.py`)
- **ExperimentalVisionTransformer**: Unified framework supporting all experimental approaches
- **create_experimental_vit**: Factory function for different architecture variants
- Supports mixing different attention mechanisms and components

## Quick Start

### Basic Usage

```python
from experimental_vit import create_experimental_vit

# Create a Fourier attention model
model = create_experimental_vit(
    architecture_type='fourier',
    img_size=224,
    num_classes=1000
)

# Create an ELFATT model
model = create_experimental_vit(
    architecture_type='elfatt',
    img_size=224,
    num_classes=1000
)

# Create a Vision Mamba model
model = create_experimental_vit(
    architecture_type='mamba',
    img_size=224,
    num_classes=1000
)
```

### Available Architecture Types

- `'fourier'`: Fourier-based attention mechanisms
- `'elfatt'`: Efficient Linear Fast Attention
- `'mamba'`: Vision Mamba with state space models
- `'kan'`: Kolmogorov-Arnold Network layers
- `'hybrid'`: CNN-Transformer hybrid architecture
- `'mixed'`: Combination of multiple approaches

## Testing

### Syntax Validation
```bash
python test_syntax.py
```

### Individual Component Testing
Each file can be run independently to test its components:
```bash
python fourier_attention.py
python linear_attention.py
python vision_mamba.py
python kan_transformer.py
python hybrid_architectures.py
python experimental_vit.py
```

## Running on EPFL Izar Cluster

### Prerequisites
1. Access to EPFL Izar SCITAS cluster
2. GPU partition access
3. Python 3.9+ environment

### Submission
```bash
# Submit the job
sbatch submit.sh

# Check job status
squeue -u $USER

# View job output
tail -f experimental_vit_JOBID.out
```

### Configuration
The submit script automatically:
- Sets up the required environment
- Creates a virtual environment with dependencies
- Runs experiments on multiple architectures
- Saves results to scratch and local directories

## Research Background

This implementation is based on comprehensive research of experimental vision transformer architectures from 2024-2025, including:

### Fourier-Based Attention
- **FourierFormer**: Generalizes attention with Fourier integral kernels
- **FNet**: Replaces self-attention with FFT for 80% speedup
- **Applications**: Scientific computing, efficient transformers

### State Space Models
- **Vision Mamba**: Linear complexity for visual sequences
- **Bidirectional processing**: Better context modeling
- **Advantages**: 2.8× faster than DeiT, 86.8% memory savings

### Linear Attention
- **ELFATT**: 4-7× speedups on high-resolution tasks
- **Hardware optimization**: FlashAttention-2 compatible
- **Scalability**: Linear complexity with maintained performance

### KAN Networks
- **Theoretical foundation**: Kolmogorov-Arnold representation theorem
- **Better interpretability**: Learnable activation functions on edges
- **Challenges**: Computational complexity, noise sensitivity

### Hybrid Architectures
- **Best of both worlds**: CNN local features + Transformer global modeling
- **Performance**: 7-11% improvement over pure architectures
- **Efficiency**: Optimized for edge deployment

## Performance Expectations

Based on research findings:

### Computational Efficiency (fastest to slowest)
1. **Mamba/SSM models**: Linear complexity
2. **Linear Attention (ELFATT)**: 4-7× speedup
3. **Fourier Attention**: 80% faster training
4. **Hybrid CNN-Transformer**: Balanced efficiency
5. **KAN models**: Slower due to spline computations

### Performance Quality (best to worst)
1. **Hybrid architectures**: Best overall performance
2. **Mamba models**: Superior for high-resolution tasks
3. **Fourier attention**: Excellent accuracy with efficiency
4. **KAN models**: High accuracy but noise sensitive
5. **Linear attention**: Good performance with speed gains

## Citation

If you use this code in your research, please cite the relevant papers from the research report (`experimental_vision_transformers_research_report.md`).

## Contributing

1. Add new experimental architectures in separate files
2. Update `experimental_vit.py` to include new architectures
3. Add configuration to `create_experimental_vit` factory function
4. Update tests and documentation

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Memory**: Reduce batch size or model dimensions
3. **Compilation Errors**: Check GCC and CUDA versions on cluster
4. **Performance Issues**: Monitor GPU utilization and memory usage

### Debug Mode
For local testing without GPU:
```python
# Use smaller models for debugging
model = create_experimental_vit(
    architecture_type='fourier',
    img_size=32,
    embed_dim=128,
    depth=2,
    num_classes=10
)
```
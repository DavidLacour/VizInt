#!/bin/bash
#SBATCH --job-name=experimental_vit
#SBATCH --account=izar-c3i
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=experimental_vit_%j.out
#SBATCH --error=experimental_vit_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$USER@epfl.ch

# Load required modules
module load gcc/9.3.0
module load cuda/11.8
module load python/3.9.7

# Set up Python environment
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=/scratch/$USER/torch_cache
export HF_HOME=/scratch/$USER/hf_cache

# Create scratch directories if they don't exist
mkdir -p /scratch/$USER/torch_cache
mkdir -p /scratch/$USER/hf_cache
mkdir -p /scratch/$USER/experimental_vit_results

# Navigate to working directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment (create if it doesn't exist)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-debug.txt
    # Install additional requirements for experimental features
    pip install mamba-ssm  # For Vision Mamba
    pip install flash-attn  # For efficient attention
else
    source venv/bin/activate
fi

echo "Starting experimental vision transformer training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"

# Set experiment configurations
export EXPERIMENT_NAME="experimental_vit_$(date +%Y%m%d_%H%M%S)"
export RESULTS_DIR="/scratch/$USER/experimental_vit_results/$EXPERIMENT_NAME"
mkdir -p $RESULTS_DIR

# Create experimental training script
cat > experimental_training.py << 'EOF'
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.getcwd())

# Import experimental architectures
from experimental_vit import create_experimental_vit

def get_dataset(dataset_name='cifar10', batch_size=32, img_size=224):
    """Load dataset for training."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        num_classes = 10
    elif dataset_name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, num_classes

def train_model(model, train_loader, test_loader, num_epochs=10, device='cuda'):
    """Train the experimental model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    results = {
        'train_losses': [],
        'train_accuracies': [],
        'test_accuracies': [],
        'epochs': [],
        'training_time': 0
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Testing
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        # Record results
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total
        avg_loss = train_loss / len(train_loader)
        
        results['train_losses'].append(avg_loss)
        results['train_accuracies'].append(train_acc)
        results['test_accuracies'].append(test_acc)
        results['epochs'].append(epoch)
        
        print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Loss: {avg_loss:.4f}')
        
        scheduler.step()
    
    results['training_time'] = time.time() - start_time
    return results

def main():
    """Main experimental training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Experimental configurations
    experiments = [
        {'name': 'fourier', 'config': {'architecture_type': 'fourier', 'img_size': 224, 'embed_dim': 384, 'depth': 6}},
        {'name': 'elfatt', 'config': {'architecture_type': 'elfatt', 'img_size': 224, 'embed_dim': 384, 'depth': 6}},
        {'name': 'hybrid', 'config': {'architecture_type': 'hybrid', 'img_size': 224, 'embed_dim': 384, 'depth': 6}},
        {'name': 'mamba', 'config': {'architecture_type': 'mamba', 'img_size': 224, 'embed_dim': 384, 'depth': 6}},
    ]
    
    # Dataset configuration
    dataset_name = 'cifar10'
    batch_size = 32
    num_epochs = 10
    
    # Load dataset
    train_loader, test_loader, num_classes = get_dataset(dataset_name, batch_size, img_size=224)
    
    all_results = {}
    
    for experiment in experiments:
        exp_name = experiment['name']
        exp_config = experiment['config']
        
        print(f"\n{'='*50}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*50}")
        
        try:
            # Create model
            model = create_experimental_vit(
                num_classes=num_classes,
                **exp_config
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {total_params:,}")
            
            # Train model
            results = train_model(model, train_loader, test_loader, num_epochs, device)
            
            # Save results
            all_results[exp_name] = {
                'config': exp_config,
                'total_params': total_params,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"Experiment {exp_name} completed successfully!")
            print(f"Best test accuracy: {max(results['test_accuracies']):.2f}%")
            
        except Exception as e:
            print(f"Experiment {exp_name} failed: {e}")
            all_results[exp_name] = {
                'config': exp_config,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Save all results
    results_file = os.path.join(os.environ.get('RESULTS_DIR', '.'), 'experimental_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll experiments completed! Results saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    for exp_name, result in all_results.items():
        if 'error' in result:
            print(f"{exp_name:12} | FAILED: {result['error']}")
        else:
            best_acc = max(result['results']['test_accuracies'])
            params = result['total_params']
            print(f"{exp_name:12} | Best Acc: {best_acc:6.2f}% | Params: {params:8,}")

if __name__ == "__main__":
    main()
EOF

# Run the experimental training
echo "Starting experimental training at $(date)"
python experimental_training.py

# Copy results to permanent storage if successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    cp -r $RESULTS_DIR/* ./experimental_results_$(date +%Y%m%d_%H%M%S)/
    echo "Results copied to local directory"
else
    echo "Training failed with exit code $?"
fi

echo "Job completed at $(date)"
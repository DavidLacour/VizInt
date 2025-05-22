"""
Debug mode configuration for quick testing with minimal resources.
Sets batch size to 1 and limits dataset to a few samples for fast iteration.
"""

import os
import torch
from pathlib import Path

class DebugConfig:
    """Configuration for debug mode with minimal resources"""
    
    def __init__(self):
        self.batch_size = 1
        self.num_samples_train = 50  # Very small training set
        self.num_samples_val = 20    # Very small validation set
        self.epochs = 2              # Just 2 epochs for testing
        self.num_workers = 0         # No multiprocessing for debugging
        self.pin_memory = False      # Simpler memory management
        self.learning_rate = 1e-4    # Standard learning rate
        self.weight_decay = 0.01
        self.warmup_steps = 5        # Minimal warmup
        self.patience = 1            # Early stopping after 1 epoch of no improvement
        
    def get_model_config(self, model_type: str) -> dict:
        """Get debug configuration for any model type"""
        return {
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'patience': self.patience,
            'dropout': 0.1,
            'optimizer': 'adamw',
            'scheduler': 'cosine_warmup',
            # Model-specific parameters
            'severity': 1.0 if model_type == 'healer' else None,
            'adaptation_steps': 2 if 'ttt' in model_type else None,
            'adaptation_lr': 1e-4 if 'ttt' in model_type else None,
        }
    
    def get_dataloader_config(self) -> dict:
        """Get data loading configuration for debug mode"""
        return {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'shuffle': True,
            'drop_last': False,
        }
    
    def limit_dataset_size(self, dataset):
        """Limit dataset to debug sample size"""
        if hasattr(dataset, 'samples') and len(dataset.samples) > self.num_samples_train:
            # For ImageFolder-style datasets
            dataset.samples = dataset.samples[:self.num_samples_train]
            dataset.targets = dataset.targets[:self.num_samples_train] if hasattr(dataset, 'targets') else None
        elif hasattr(dataset, 'imgs') and len(dataset.imgs) > self.num_samples_train:
            # Alternative attribute name
            dataset.imgs = dataset.imgs[:self.num_samples_train]
        return dataset
    
    def print_debug_info(self):
        """Print debug configuration information"""
        print("🐛 DEBUG MODE ACTIVATED")
        print("=" * 50)
        print(f"Batch size: {self.batch_size}")
        print(f"Training samples: {self.num_samples_train}")
        print(f"Validation samples: {self.num_samples_val}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Workers: {self.num_workers}")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print("=" * 50)

# Global debug config instance
debug_config = DebugConfig()

def is_debug_mode() -> bool:
    """Check if debug mode is enabled via environment variable"""
    return os.getenv('DEBUG_MODE', '').lower() in ['1', 'true', 'yes', 'on']

def get_debug_config() -> DebugConfig:
    """Get the debug configuration instance"""
    return debug_config

def setup_debug_environment():
    """Setup environment variables for debug mode"""
    os.environ['DEBUG_MODE'] = '1'
    os.environ['WANDB_MODE'] = 'disabled'  # Disable wandb in debug mode
    print("🐛 Debug environment activated")

if __name__ == "__main__":
    # Test debug configuration
    debug_config.print_debug_info()
    
    # Show example model configs
    model_types = ['classification', 'healer', 'ttt', 'blended_ttt']
    print("\n📋 Debug Model Configurations:")
    for model_type in model_types:
        config = debug_config.get_model_config(model_type)
        print(f"\n{model_type}:")
        for key, value in config.items():
            if value is not None:
                print(f"  {key}: {value}")
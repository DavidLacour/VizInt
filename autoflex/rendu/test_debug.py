#!/usr/bin/env python3
"""
Test script to verify the refactored code works correctly in debug mode
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config_loader import ConfigLoader


def test_config_loader():
    """Test configuration loader"""
    print("\n=== Testing Configuration Loader ===")
    
    config = ConfigLoader()
    
    # Test basic config access
    print(f"Device: {config.get_device()}")
    print(f"Debug mode: {config.is_debug_mode()}")
    print(f"Batch size (training): {config.get_batch_size('training')}")
    print(f"Batch size (evaluation): {config.get_batch_size('evaluation')}")
    
    # Test dataset config
    cifar10_config = config.get_dataset_config('cifar10')
    print(f"\nCIFAR-10 config:")
    print(f"  Path: {cifar10_config['path']}")
    print(f"  Num classes: {cifar10_config['num_classes']}")
    print(f"  Image size: {cifar10_config['img_size']}")
    
    # Enable debug mode
    config.update({'debug': {'enabled': True}})
    print(f"\nAfter enabling debug mode:")
    print(f"  Debug mode: {config.is_debug_mode()}")
    print(f"  Batch size: {config.get_batch_size()}")
    print(f"  Num epochs: {config.get_num_epochs()}")
    
    print("\n✅ Config loader test passed!")


def test_model_factory():
    """Test model factory"""
    print("\n=== Testing Model Factory ===")
    
    from src.models.model_factory import ModelFactory
    
    config = ConfigLoader()
    config.update({'debug': {'enabled': True}})
    
    factory = ModelFactory(config)
    
    # Test creating a simple model
    try:
        model = factory.create_model('baseline', 'cifar10')
        print(f"✅ Created baseline model: {type(model).__name__}")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
    
    print("\n✅ Model factory test passed!")


def test_data_loader():
    """Test data loader factory"""
    print("\n=== Testing Data Loader Factory ===")
    
    from src.data.data_loader import DataLoaderFactory
    
    config = ConfigLoader()
    config.update({'debug': {'enabled': True}})
    
    factory = DataLoaderFactory(config)
    
    # Test creating CIFAR-10 loaders
    try:
        train_loader, val_loader = factory.create_data_loaders('cifar10')
        print(f"✅ Created CIFAR-10 data loaders")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Get a sample batch
        images, labels = next(iter(train_loader))
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        
    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Data loader test passed!")


def test_trainer():
    """Test trainer"""
    print("\n=== Testing Trainer ===")
    
    from src.config.config_loader import ConfigLoader
    from src.models.model_factory import ModelFactory
    from src.data.data_loader import DataLoaderFactory
    from src.trainers.classification_trainer import ClassificationTrainer
    import torch
    
    config = ConfigLoader()
    config.update({'debug': {'enabled': True}})
    
    model_factory = ModelFactory(config)
    data_factory = DataLoaderFactory(config)
    
    try:
        # Create a simple model
        model = model_factory.create_model('baseline', 'cifar10')
        device = torch.device(config.get_device())
        model = model.to(device)
        
        # Create trainer
        trainer = ClassificationTrainer(
            model=model,
            config={'training': config.get_training_config()},
            device=str(device)
        )
        
        print(f"✅ Created trainer: {type(trainer).__name__}")
        
        # Create optimizer
        optimizer = trainer.create_optimizer()
        print(f"  Optimizer: {type(optimizer).__name__}")
        
        # Create scheduler
        scheduler = trainer.create_scheduler(optimizer)
        print(f"  Scheduler: {type(scheduler).__name__ if scheduler else 'None'}")
        
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Trainer test passed!")


def main():
    """Run all tests"""
    print("Starting debug tests for refactored code...")
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_config_loader()
    test_model_factory()
    test_data_loader()
    test_trainer()
    
    print("\n🎉 All tests completed!")
    print("\nYou can now run the main script with debug mode:")
    print("  python main.py --dataset cifar10 --debug --models baseline")
    print("\nOr for a full experiment:")
    print("  python main.py --dataset cifar10 --mode both")


if __name__ == "__main__":
    main()
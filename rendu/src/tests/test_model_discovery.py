#!/usr/bin/env python3
"""Test script to verify model discovery in evaluation pipeline"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.config_loader import ConfigLoader
from models.model_factory import ModelFactory
from data.data_loader import DataLoaderFactory
from services.model_evaluator import ModelEvaluator

def test_model_discovery():
    """Test that all model types are properly discovered"""
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'cifar10_config.yaml'
    config_loader = ConfigLoader(config_path)
    model_factory = ModelFactory(config_loader)
    data_factory = DataLoaderFactory(config_loader)
    
    # Create evaluator
    evaluator = ModelEvaluator(config_loader, model_factory, data_factory)
    
    # Check if wrapped models are in the all_model_types list
    print("Testing model discovery...")
    
    # Get the model combinations from config
    combinations = config_loader.get('model_combinations', [])
    
    print(f"\nFound {len(combinations)} model combinations in config:")
    for combo in combinations:
        print(f"  - {combo['name']}: {combo['main_model']}")
    
    # Check if wrapped models are included
    wrapped_models = ['blended_resnet18', 'ttt_resnet18', 'healer_resnet18']
    found_wrapped = []
    
    for combo in combinations:
        if combo['main_model'] in wrapped_models:
            found_wrapped.append(combo['main_model'])
    
    print(f"\nWrapped models found in combinations: {found_wrapped}")
    print(f"Missing wrapped models: {set(wrapped_models) - set(found_wrapped)}")
    
    # Test that evaluator recognizes these models
    print(f"\nEvaluator's all_model_types includes:")
    
    # Access the method directly to check the model types
    try:
        # Simulate the model loading logic
        available_models = evaluator._load_available_models('cifar10', wrapped_models)
        print(f"Models that could be loaded: {list(available_models.keys())}")
    except Exception as e:
        print(f"Model loading test failed: {e}")
    
    print("\n" + "="*60)
    print("Model discovery test completed!")

if __name__ == "__main__":
    test_model_discovery()
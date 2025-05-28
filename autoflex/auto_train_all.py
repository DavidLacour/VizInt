#!/usr/bin/env python3
"""
Automatic training script that trains all possible model and backbone combinations.
Supports force retraining and selective training based on existing models.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import json
from datetime import datetime

# Import the training functions and configurations
from flexible_training import train_classification_model, train_healer_model
from flexible_models import create_model
from ttt_model import train_ttt_model
from blended_ttt_model import BlendedTTT
from new_new import set_seed

# Import BACKBONE_CONFIGS properly
try:
    from flexible_models import BACKBONE_CONFIGS
except ImportError:
    # Fallback: import from backbone_factory directly
    from backbone_factory import BACKBONE_CONFIGS

DATASET_PATH="../tiny-imagenet-200"

class ModelTrainer:
    """Comprehensive model trainer that handles all combinations"""
    
    def __init__(self, dataset_path: str = "DATASET_PATH", force: bool = False):
        self.dataset_path = dataset_path
        self.force = force
        self.trained_models = {}
        self.failed_models = {}
        self.training_log = []
        
        # Store original backbone configs
        self.original_backbone_configs = BACKBONE_CONFIGS.copy()
        
        # Model types and their corresponding training functions
        self.model_types = {
            'classification': {
                'train_func': train_classification_model,
                'epochs': 50,
                'learning_rate': 1e-4,
                'requires_base_model': False
            },
            'healer': {
                'train_func': train_healer_model,
                'epochs': 15,
                'learning_rate': 5e-5,
                'requires_base_model': False,
                'severity': 1.0
            },
            'ttt': {
                'train_func': self.train_ttt_wrapper,
                'epochs': 10,
                'learning_rate': 1e-4,
                'requires_base_model': True
            },
            'blended_ttt': {
                'train_func': self.train_blended_ttt_wrapper,
                'epochs': 20,
                'learning_rate': 1e-4,
                'requires_base_model': False
            }
        }
        
        # Create results directory
        self.results_dir = Path("training_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Log file
        self.log_file = self.results_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def set_backbone_filter(self, backbone_names: List[str]):
        """Filter backbones to only train specified ones"""
        if backbone_names:
            # Create a filtered version without modifying the global BACKBONE_CONFIGS
            global BACKBONE_CONFIGS
            filtered_configs = {}
            for backbone in backbone_names:
                if backbone in self.original_backbone_configs:
                    filtered_configs[backbone] = self.original_backbone_configs[backbone]
                else:
                    print(f"⚠️  Warning: Unknown backbone '{backbone}', skipping.")
            
            # Update the global BACKBONE_CONFIGS for this session
            BACKBONE_CONFIGS.clear()
            BACKBONE_CONFIGS.update(filtered_configs)
            print(f"🎯 Training only backbones: {list(BACKBONE_CONFIGS.keys())}")
    
    def get_model_path(self, model_type: str, backbone_name: str) -> Path:
        """Get the path for a model's best checkpoint"""
        return Path(f"bestmodel_{backbone_name}_{model_type}") / "best_model.pt"
    
    def get_checkpoint_dir(self, model_type: str, backbone_name: str) -> Path:
        """Get the checkpoint directory for a model"""
        return Path(f"checkpoints_{backbone_name}_{model_type}")
    
    def model_exists(self, model_type: str, backbone_name: str) -> bool:
        """Check if a trained model already exists"""
        model_path = self.get_model_path(model_type, backbone_name)
        return model_path.exists()
    
    def delete_model_files(self, model_type: str, backbone_name: str):
        """Delete existing model files and checkpoints"""
        # Delete best model
        best_model_dir = self.get_model_path(model_type, backbone_name).parent
        if best_model_dir.exists():
            shutil.rmtree(best_model_dir)
            print(f"Deleted best model directory: {best_model_dir}")
        
        # Delete checkpoints
        checkpoint_dir = self.get_checkpoint_dir(model_type, backbone_name)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            print(f"Deleted checkpoint directory: {checkpoint_dir}")
    
    def find_compatible_backbones(self, model_type: str) -> List[str]:
        """Find backbones compatible with a given model type"""
        compatible_backbones = []
        
        for backbone_name, config in BACKBONE_CONFIGS.items():
            # Some models might have specific requirements
            if model_type == 'ttt' and config.get('type') == 'vgg':
                # Skip VGG for TTT as it might be too slow
                continue
            
            compatible_backbones.append(backbone_name)
        
        return compatible_backbones
    
    def get_base_model_path(self, backbone_name: str) -> Path:
        """Get path to base classification model for TTT training"""
        return self.get_model_path('classification', backbone_name)
    
    def train_ttt_wrapper(self, **kwargs):
        """Wrapper for TTT training that loads the base model"""
        backbone_name = kwargs.get('backbone_name')
        base_model_path = self.get_base_model_path(backbone_name)
        
        if not base_model_path.exists():
            raise FileNotFoundError(f"Base classification model not found at {base_model_path}")
        
        # Load base model
        base_model = create_model('classification', backbone_name, num_classes=200)
        
        checkpoint = torch.load(base_model_path, map_location='cpu')
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Remove backbone_name from kwargs and add base_model
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('backbone_name', None)
        kwargs_copy['base_model'] = base_model
        
        return train_ttt_model(**kwargs_copy)
    
    def train_blended_ttt_wrapper(self, **kwargs):
        """Wrapper for Blended TTT training"""
        # This would need to be implemented based on your blended TTT training logic
        # For now, we'll skip it
        print(f"Blended TTT training not yet implemented, skipping...")
        return None
    
    def train_single_model(self, model_type: str, backbone_name: str) -> bool:
        """Train a single model with specified type and backbone"""
        model_info = self.model_types[model_type]
        
        # Check if base model is required and exists
        if model_info.get('requires_base_model', False):
            base_model_path = self.get_base_model_path(backbone_name)
            if not base_model_path.exists():
                print(f"⚠️  Base model required for {model_type} with {backbone_name} but not found. Skipping.")
                return False
        
        print(f"🚀 Training {model_type} with {backbone_name} backbone...")
        
        try:
            # Get configuration from config manager
            from training_config import get_training_config
            from simple_batch_config import get_tested_batch_size, get_num_workers
            
            config = get_training_config(model_type, backbone_name)
            
            # Use tested batch size instead of config batch size for reliability
            batch_size = get_tested_batch_size(model_type, backbone_name)
            
            # Prepare training arguments with configuration
            train_kwargs = {
                'dataset_path': self.dataset_path,
                'backbone_name': backbone_name,
                'epochs': config.get('epochs', model_info['epochs']),
                'learning_rate': config.get('learning_rate', model_info['learning_rate']),
                'batch_size': batch_size,  # Use tested batch size
                'experiment_name': f"{model_type}_{backbone_name}_auto"
            }
            
            # Add model-specific arguments
            if model_type == 'healer':
                train_kwargs['severity'] = config.get('severity', model_info.get('severity', 1.0))
            
            print(f"📋 Using config: epochs={train_kwargs['epochs']}, lr={train_kwargs['learning_rate']}, batch_size={train_kwargs['batch_size']}")
            
            # Train the model
            model = model_info['train_func'](**train_kwargs)
            
            if model is not None:
                print(f"✅ Successfully trained {model_type} with {backbone_name}")
                self.trained_models[f"{model_type}_{backbone_name}"] = {
                    'model_type': model_type,
                    'backbone': backbone_name,
                    'status': 'success',
                    'config_used': config,
                    'timestamp': datetime.now().isoformat()
                }
                return True
            else:
                raise Exception("Training returned None")
                
        except Exception as e:
            print(f"❌ Failed to train {model_type} with {backbone_name}: {str(e)}")
            self.failed_models[f"{model_type}_{backbone_name}"] = {
                'model_type': model_type,
                'backbone': backbone_name,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def get_training_plan(self) -> List[Tuple[str, str]]:
        """Generate a training plan based on existing models and requirements"""
        training_plan = []
        
        # First, identify all possible combinations
        all_combinations = []
        for model_type in self.model_types.keys():
            compatible_backbones = self.find_compatible_backbones(model_type)
            for backbone_name in compatible_backbones:
                all_combinations.append((model_type, backbone_name))
        
        # Filter based on existing models and force flag
        for model_type, backbone_name in all_combinations:
            should_train = False
            
            if self.force:
                # Force retrain everything
                should_train = True
                if self.model_exists(model_type, backbone_name):
                    print(f"🗑️  Force mode: will delete and retrain {model_type} with {backbone_name}")
                    self.delete_model_files(model_type, backbone_name)
            else:
                # Only train if model doesn't exist
                if not self.model_exists(model_type, backbone_name):
                    should_train = True
                else:
                    print(f"⏭️  Skipping {model_type} with {backbone_name} (already exists)")
            
            if should_train:
                training_plan.append((model_type, backbone_name))
        
        return training_plan
    
    def train_all_models(self):
        """Train all possible model combinations"""
        print("🔍 Analyzing existing models and creating training plan...")
        
        training_plan = self.get_training_plan()
        
        if not training_plan:
            print("✨ All models already exist! Use --force to retrain everything.")
            return
        
        print(f"\n📋 Training Plan ({len(training_plan)} models):")
        for i, (model_type, backbone_name) in enumerate(training_plan, 1):
            print(f"  {i:2d}. {model_type:<15} + {backbone_name}")
        
        print(f"\n🎯 Starting training of {len(training_plan)} models...\n")
        
        # Training order: classification first (needed for TTT), then others
        ordered_plan = []
        
        # First pass: classification models
        for model_type, backbone_name in training_plan:
            if model_type == 'classification':
                ordered_plan.append((model_type, backbone_name))
        
        # Second pass: other models
        for model_type, backbone_name in training_plan:
            if model_type != 'classification':
                ordered_plan.append((model_type, backbone_name))
        
        # Train models in order
        for i, (model_type, backbone_name) in enumerate(ordered_plan, 1):
            print(f"\n{'='*60}")
            print(f"Training {i}/{len(ordered_plan)}: {model_type} + {backbone_name}")
            print(f"{'='*60}")
            
            success = self.train_single_model(model_type, backbone_name)
            
            # Log progress
            self.training_log.append({
                'step': i,
                'total_steps': len(ordered_plan),
                'model_type': model_type,
                'backbone': backbone_name,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })
            
            # Save progress
            self.save_progress()
        
        self.print_summary()
    
    def save_progress(self):
        """Save training progress to file"""
        progress_data = {
            'trained_models': self.trained_models,
            'failed_models': self.failed_models,
            'training_log': self.training_log,
            'dataset_path': self.dataset_path,
            'force_mode': self.force,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def print_summary(self):
        """Print training summary"""
        total_attempted = len(self.trained_models) + len(self.failed_models)
        successful = len(self.trained_models)
        failed = len(self.failed_models)
        
        print(f"\n{'='*60}")
        print(f"🎉 TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total attempted: {total_attempted}")
        print(f"✅ Successful:   {successful}")
        print(f"❌ Failed:       {failed}")
        
        if successful > 0:
            print(f"\n✅ Successfully trained models:")
            for model_name, info in self.trained_models.items():
                print(f"  • {info['model_type']:<15} + {info['backbone']}")
        
        if failed > 0:
            print(f"\n❌ Failed models:")
            for model_name, info in self.failed_models.items():
                print(f"  • {info['model_type']:<15} + {info['backbone']:<15} - {info['error'][:50]}...")
        
        print(f"\n📊 Results saved to: {self.log_file}")
        print(f"{'='*60}")
    
    def list_existing_models(self):
        """List all existing trained models"""
        print("📋 Existing trained models:")
        
        found_any = False
        for model_type in self.model_types.keys():
            compatible_backbones = self.find_compatible_backbones(model_type)
            for backbone_name in compatible_backbones:
                if self.model_exists(model_type, backbone_name):
                    model_path = self.get_model_path(model_type, backbone_name)
                    file_size = model_path.stat().st_size / (1024 * 1024)  # MB
                    print(f"  ✅ {model_type:<15} + {backbone_name:<15} ({file_size:.1f} MB)")
                    found_any = True
        
        if not found_any:
            print("  No trained models found.")
    
    def clean_all_models(self):
        """Delete all existing models and checkpoints"""
        print("🗑️  Cleaning all existing models and checkpoints...")
        
        deleted_count = 0
        for model_type in self.model_types.keys():
            compatible_backbones = self.find_compatible_backbones(model_type)
            for backbone_name in compatible_backbones:
                if self.model_exists(model_type, backbone_name):
                    self.delete_model_files(model_type, backbone_name)
                    deleted_count += 1
        
        print(f"🗑️  Deleted {deleted_count} model directories.")

def main():
    """Main function with comprehensive argument parsing"""
    # Get available backbones at the start
    available_backbones = list(BACKBONE_CONFIGS.keys())
    
    parser = argparse.ArgumentParser(
        description='Automatic training system for all model and backbone combinations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all missing models
  python auto_train_all.py
  
  # Force retrain everything
  python auto_train_all.py --force
  
  # List existing models
  python auto_train_all.py --list
  
  # Clean all models then train
  python auto_train_all.py --clean --force
  
  # Train specific model types only
  python auto_train_all.py --models classification healer
  
  # Train specific backbones only
  python auto_train_all.py --backbones vit_small resnet50
        """
    )
    
    parser.add_argument('--dataset_path', type=str, default='../tiny-imagenet-200',
                        help='Path to the dataset directory')
    
    parser.add_argument('--force', action='store_true',
                        help='Force retrain all models (deletes existing models)')
    
    parser.add_argument('--list', action='store_true',
                        help='List all existing trained models and exit')
    
    parser.add_argument('--clean', action='store_true',
                        help='Delete all existing models and checkpoints')
    
    parser.add_argument('--models', nargs='+', 
                        choices=['classification', 'healer', 'ttt', 'blended_ttt'],
                        help='Train only specified model types')
    
    parser.add_argument('--backbones', nargs='+',
                        choices=available_backbones,
                        help='Train only specified backbones')
    
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be trained without actually training')
    
    parser.add_argument('--continue_on_error', action='store_true', default=True,
                        help='Continue training other models if one fails')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ModelTrainer(dataset_path=args.dataset_path, force=args.force)
    
    # Filter model types if specified
    if args.models:
        trainer.model_types = {k: v for k, v in trainer.model_types.items() if k in args.models}
        print(f"🎯 Training only model types: {args.models}")
    
    # Filter backbones if specified
    if args.backbones:
        trainer.set_backbone_filter(args.backbones)
    
    # Handle different modes
    if args.list:
        trainer.list_existing_models()
        return
    
    if args.clean:
        trainer.clean_all_models()
        if not args.force:
            print("Models cleaned. Use --force to retrain everything.")
            return
    
    if args.dry_run:
        print("🔍 DRY RUN - showing what would be trained:")
        training_plan = trainer.get_training_plan()
        if training_plan:
            for i, (model_type, backbone_name) in enumerate(training_plan, 1):
                print(f"  {i:2d}. {model_type:<15} + {backbone_name}")
        else:
            print("  Nothing to train (all models exist)")
        return
    
    # Print available configurations
    print(f"🔧 Available model types: {list(trainer.model_types.keys())}")
    print(f"🔧 Available backbones: {list(BACKBONE_CONFIGS.keys())}")
    
    # Start training
    try:
        trainer.train_all_models()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        trainer.print_summary()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        trainer.print_summary()
        sys.exit(1)

if __name__ == "__main__":
    main()
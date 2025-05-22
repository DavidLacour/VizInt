#!/usr/bin/env python3
"""
Test script to compare pretrained vs scratch training for different backbone architectures.
This script allows you to easily test both pretrained feature maps and non-pretrained feature maps.
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Test pretrained vs scratch backbone training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ResNet50 with both pretrained and scratch initialization
  python test_pretrained_vs_scratch.py --backbone resnet50 --model classification
  
  # Train all ResNet variants
  python test_pretrained_vs_scratch.py --backbone resnet18 resnet50 --model classification
  
  # Train all models with DeiT small
  python test_pretrained_vs_scratch.py --backbone deit_small --model classification healer
  
  # Quick test with fewer epochs
  python test_pretrained_vs_scratch.py --backbone resnet18 --model classification --epochs 10
  
  # Train everything (be careful, this takes a long time!)
  python test_pretrained_vs_scratch.py --all
        """
    )
    
    # Backbone selection
    parser.add_argument('--backbone', nargs='+', 
                        choices=['resnet18', 'resnet50', 'vgg16', 'deit_small', 'swin_small'],
                        help='Backbone architectures to test')
    
    # Model type selection
    parser.add_argument('--model', nargs='+',
                        choices=['classification', 'healer', 'ttt', 'blended_ttt'],
                        default=['classification'],
                        help='Model types to train')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs (uses config defaults if not specified)')
    
    parser.add_argument('--dataset_path', type=str, default='../tiny-imagenet-200',
                        help='Path to the dataset')
    
    # Control options
    parser.add_argument('--force', action='store_true',
                        help='Force retrain existing models')
    
    parser.add_argument('--pretrained_only', action='store_true',
                        help='Only train pretrained models')
    
    parser.add_argument('--scratch_only', action='store_true',
                        help='Only train scratch models')
    
    parser.add_argument('--all', action='store_true',
                        help='Train all backbone and model combinations')
    
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be trained without actually training')
    
    # Analysis options
    parser.add_argument('--compare_only', action='store_true',
                        help='Only compare existing models without training')
    
    parser.add_argument('--analyze', action='store_true',
                        help='Run analysis after training')
    
    args = parser.parse_args()
    
    # Import here to avoid import errors if modules aren't available
    try:
        from auto_train_all import ModelTrainer
        from backbone_factory import BACKBONE_CONFIGS
        from training_config import config_manager
        from training_monitor import TrainingMonitor
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available.")
        sys.exit(1)
    
    # Determine which backbones to test
    if args.all:
        backbones_to_test = ['resnet18', 'resnet50', 'vgg16', 'deit_small', 'swin_small']
        models_to_test = ['classification', 'healer']
    elif args.backbone:
        backbones_to_test = args.backbone
        models_to_test = args.model
    else:
        print("❌ Please specify --backbone or --all")
        parser.print_help()
        sys.exit(1)
    
    # Generate the training plan
    training_plan = []
    
    for backbone_base in backbones_to_test:
        for model_type in models_to_test:
            # Add pretrained version
            if not args.scratch_only:
                if backbone_base in ['vit_small', 'vit_base']:
                    # Custom ViT models don't have pretrained versions
                    training_plan.append((model_type, backbone_base))
                else:
                    pretrained_name = f"{backbone_base}_pretrained"
                    training_plan.append((model_type, pretrained_name))
            
            # Add scratch version
            if not args.pretrained_only:
                if backbone_base in ['vit_small', 'vit_base']:
                    # Custom ViT models are already "scratch"
                    pass  # Already added above
                else:
                    scratch_name = f"{backbone_base}_scratch"
                    training_plan.append((model_type, scratch_name))
    
    if args.compare_only:
        print("📊 Comparing existing models...")
        monitor = TrainingMonitor()
        monitor.print_model_summary()
        monitor.generate_comparison_report()
        return
    
    if args.dry_run:
        print("🔍 DRY RUN - Training plan:")
        print("=" * 50)
        for i, (model_type, backbone_name) in enumerate(training_plan, 1):
            is_pretrained = "PRETRAINED" if not backbone_name.endswith('_scratch') and backbone_name not in ['vit_small', 'vit_base'] else "SCRATCH"
            if backbone_name in ['vit_small', 'vit_base']:
                is_pretrained = "CUSTOM"
            print(f"  {i:2d}. {model_type:<15} + {backbone_name:<25} ({is_pretrained})")
        
        print(f"\nTotal: {len(training_plan)} model(s) to train")
        return
    
    # Show configuration comparison
    print("🔧 Configuration Comparison:")
    config_manager.compare_pretrained_vs_scratch("classification")
    
    # Create trainer with filtered backbones
    trainer = ModelTrainer(dataset_path=args.dataset_path, force=args.force)
    
    # Filter model types if needed
    if args.model != ['classification']:
        trainer.model_types = {k: v for k, v in trainer.model_types.items() if k in args.model}
    
    # Override epochs if specified
    if args.epochs:
        for model_config in trainer.model_types.values():
            model_config['epochs'] = args.epochs
        print(f"🔧 Overriding epochs to {args.epochs}")
    
    # Get unique backbone names from training plan
    backbone_names = list(set([backbone for _, backbone in training_plan]))
    trainer.set_backbone_filter(backbone_names)
    
    print(f"\n🚀 Starting training with {len(training_plan)} model configurations...")
    print("📋 Training Plan:")
    for i, (model_type, backbone_name) in enumerate(training_plan, 1):
        config = config_manager.get_training_config(model_type, backbone_name)
        is_pretrained = "✅ PRETRAINED" if config.get("is_pretrained", True) else "🔨 SCRATCH"
        if backbone_name in ['vit_small', 'vit_base']:
            is_pretrained = "🔧 CUSTOM"
        epochs = config.get('epochs', 'default')
        lr = config.get('learning_rate', 'default')
        print(f"  {i:2d}. {model_type:<15} + {backbone_name:<25} ({is_pretrained}) - {epochs} epochs, LR={lr}")
    
    # Start training
    try:
        trainer.train_all_models()
        
        if args.analyze:
            print("\n📊 Running post-training analysis...")
            monitor = TrainingMonitor()
            monitor.print_model_summary()
            monitor.generate_comparison_report()
            
            # Try to create plots
            try:
                monitor.plot_performance_comparison()
            except ImportError:
                print("📊 Plotting requires matplotlib and seaborn. Install with: pip install matplotlib seaborn")
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Check the bestmodel_* directories for trained models")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

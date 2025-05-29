#!/usr/bin/env python3
"""
Unified main script for training and evaluating models on CIFAR-10 and TinyImageNet
Supports all model types: ViT, ResNet, TTT, BlendedTTT, and their 3FC variants
"""
import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config.config_loader import ConfigLoader
from models.model_factory import ModelFactory
from data.data_loader import DataLoaderFactory
from services.model_trainer import ModelTrainer
from services.model_evaluator import ModelEvaluator
from utils.visualization import create_evaluation_plots
from utils.transformer_utils import set_seed


def setup_logging(config: ConfigLoader):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '[%(asctime)s] %(levelname)s: %(message)s')
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('experiment.log')
        ]
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unified training and evaluation script")
    
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'tinyimagenet'], 
                      default='cifar10', help='Dataset to use')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to configuration file (auto-selects based on dataset if not provided)')
  
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'],
                      default='both', help='Mode of operation')
   
    parser.add_argument('--models', type=str, nargs='+',
                      default=['all'], help='Models to train/evaluate')
    parser.add_argument('--skip_models', type=str, nargs='+',
                      help='Models to skip')
  
    parser.add_argument('--robust', action='store_true',
                      help='Train robust models')
    parser.add_argument('--force_retrain', action='store_true',
                      help='Force retraining even if checkpoints exist')
    
    parser.add_argument('--severities', type=float, nargs='+',
                      help='Override severities for evaluation')
    
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with small dataset')
    
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                      help='Override device setting')
    parser.add_argument('--seed', type=int, help='Override random seed')
    
    return parser.parse_args()


def get_models_to_process(args, config):
    """Get list of models to process based on arguments"""
    all_models = ['vanilla_vit', 'healer', 'ttt', 'ttt3fc', 'blended_training', 'blended_training_3fc', 
                  'resnet', 'resnet_pretrained', 'blended_resnet18', 'ttt_resnet18', 'healer_resnet18']
    
    name_mapping = {
        'main': 'vanilla_vit',
        'blended': 'blended_training',
        'blended3fc': 'blended_training_3fc',
        'baseline': 'resnet',
        'pretrained': 'resnet_pretrained'
    }
    
    if 'all' in args.models:
        models = all_models
    else:
        models = [name_mapping.get(m, m) for m in args.models]
    
    if args.skip_models:
        skip_models = [name_mapping.get(m, m) for m in args.skip_models]
        models = [m for m in models if m not in skip_models]
    
    enabled_models = []
    for model in models:
        enabled_models.append(model)
    
    return enabled_models


def train_models(args, config, models_to_train):
    """Train all specified models"""
    logger = logging.getLogger('train_models')
    
    model_factory = ModelFactory(config)
    data_factory = DataLoaderFactory(config)
    
    checkpoint_dir = config.get_checkpoint_dir(args.dataset)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    trainer_service = ModelTrainer(config, model_factory, data_factory)
    
    trained_models = {}
    
    for model_name in models_to_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} model on {args.dataset}")
        logger.info(f"{'='*60}")
        
        # Check if model already exists and load it and don't if it already exists
        model_dir = checkpoint_dir / f"bestmodel_{model_name}"
        checkpoint_path = model_dir / "best_model.pt"
        
        if checkpoint_path.exists() and not args.force_retrain:
            logger.info(f"Model already exists at {checkpoint_path}, skipping training")
            continue
        
        model, history = trainer_service.train_model(
            model_type=model_name,
            dataset_name=args.dataset,
            robust_training=False
        )
        trained_models[model_name] = model
        
        # Always train robust version for vanilla_vit (and optionally for others if --robust flag is set)
        # Not for blended models as they're already robust
        should_train_robust = (model_name == 'vanilla_vit') or (args.robust and model_name in ['ttt', 'ttt3fc'])
        
        if should_train_robust and model_name in ['vanilla_vit', 'ttt', 'ttt3fc']:
            robust_model_name = f"{model_name}_robust"
            logger.info(f"\nTraining robust version: {robust_model_name}")
            
            # Check if robust model already exists
            robust_model_dir = checkpoint_dir / f"bestmodel_{robust_model_name}"
            robust_checkpoint_path = robust_model_dir / "best_model.pt"
            
            if robust_checkpoint_path.exists() and not args.force_retrain:
                logger.info(f"Robust model already exists at {robust_checkpoint_path}, skipping training")
            else:
                robust_model, robust_history = trainer_service.train_model(
                    model_type=robust_model_name,
                    dataset_name=args.dataset,
                    robust_training=True
                )
                trained_models[robust_model_name] = robust_model
    
    return trained_models


def evaluate_models(args, config, models_to_evaluate):
    """Evaluate all specified models"""
    logger = logging.getLogger('evaluate_models')
    
    # Create factories
    model_factory = ModelFactory(config)
    data_factory = DataLoaderFactory(config)
    
    # Get evaluation severities
    if args.severities:
        severities = args.severities
    else:
        severities = config.get('evaluation.severities', [0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Create evaluator service
    evaluator = ModelEvaluator(config, model_factory, data_factory)
    
    # Evaluate all model combinations
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating models on {args.dataset} with severities: {severities}")
    logger.info(f"{'='*60}")
    
    results = evaluator.evaluate_all_combinations(
        dataset_name=args.dataset,
        severities=severities,
        model_types=models_to_evaluate,
        include_ood=True
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Create visualizations
    vis_dir = Path(config.get('paths.visualization_dir', './visualizations'))
    vis_dir = vis_dir / args.dataset
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    create_evaluation_plots(results, severities, vis_dir)
    logger.info(f"Visualizations saved to {vis_dir}")
    
    return results


def main():
    """Main entry point"""
    args = parse_arguments()
    
    if args.config is None:
        if args.dataset == 'cifar10':
            args.config = 'config/cifar10_config.yaml'
        else:
            args.config = 'config/tinyimagenet_config.yaml'
    
    config = ConfigLoader(args.config)
    
    # Override config with command line arguments
    if args.debug:
        config.update({'debug': {'enabled': True}})
    if args.device:
        config.update({'general': {'device': args.device}})
    if args.seed:
        config.update({'general': {'seed': args.seed}})
    
    setup_logging(config)
    logger = logging.getLogger('main')
   
    seed = config.get('general.seed', 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
  
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {config.get_device()}")
    logger.info(f"Debug mode: {config.is_debug_mode()}")
    
    models_to_process = get_models_to_process(args, config)
    logger.info(f"Models to process: {models_to_process}")
    
    # Training phase
    if args.mode in ['train', 'both']:
        train_models(args, config, models_to_process)
    
    # Evaluation phase
    if args.mode in ['evaluate', 'both']:
        evaluate_models(args, config, models_to_process)
    
    logger.info("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
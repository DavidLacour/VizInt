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
    all_models = ['resnet18_not_pretrained_robust', 'blended_resnet18', 'blended_vgg', 'blended_vgg16', 'blended_vgg19', 
                  'vgg_robust', 'vgg16_robust', 'vgg19_robust']
    """all_models = ['vanilla_vit', 'healer', 'ttt', 'ttt3fc', 'blended_training', 'blended_training_3fc', 
                  'resnet', 'resnet_pretrained', 'resnet18_not_pretrained_robust', 'blended_resnet18', 'blended_vgg', 'blended_vgg16', 'blended_vgg19', 'ttt_resnet18', 'healer_resnet18',
                  'unet_corrector', 'transformer_corrector', 'hybrid_corrector',
                  'unet_resnet18', 'unet_vit',
                  'transformer_resnet18', 'transformer_vit',
                  'hybrid_resnet18', 'hybrid_vit']"""
    
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


def train_corrector(corrector_name, dataset_name, config, model_factory, data_factory, checkpoint_dir):
    """Train a corrector model"""
    logger = logging.getLogger('train_corrector')
    
    if 'unet' in corrector_name:
        corrector_type = 'unet'
    elif 'transformer' in corrector_name:
        corrector_type = 'transformer'
    elif 'hybrid' in corrector_name:
        corrector_type = 'hybrid'
    else:
        raise ValueError(f"Unknown corrector type: {corrector_name}")
    
    corrector_config = config.get_dataset_config(dataset_name).copy()
    corrector_config.update({
        'model_type': corrector_type,
        'loss_type': 'combined',  # Use combined L1 + L2 + perceptual loss
        'transform_types': ['gaussian_noise', 'rotation', 'affine'],
        'severity_range': [0.1, 0.8],
        'l1_weight': 1.0,
        'l2_weight': 0.5,
        'perceptual_weight': 0.1,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'weight_decay': 1e-5
    })
    
    if corrector_type == 'transformer':
        corrector_config.update({
            'corrector_patch_size': 8,
            'corrector_embed_dim': 768,
            'corrector_depth': 12,
            'corrector_head_dim': 64
        })
    elif corrector_type == 'hybrid':
        corrector_config.update({
            'corrector_embed_dim': 384,
            'corrector_depth': 6,
            'use_transformer': True,
            'use_cnn': True
        })
    
    model = model_factory.create_model(corrector_name, dataset_name)
    
    from trainers.corrector_trainer import CorrectorTrainer
    trainer = CorrectorTrainer(corrector_config)
    
    train_loader, val_loader = data_factory.create_data_loaders(dataset_name)
    
    if config.is_debug_mode():
        debug_checkpoint_dir = Path(config.get('debug.checkpoint_dir', '../../../debugmodelrendu/cifar10'))
        if not debug_checkpoint_dir.is_absolute():
            debug_checkpoint_dir = Path(__file__).parent.parent / debug_checkpoint_dir
        debug_checkpoint_dir = debug_checkpoint_dir.resolve()
        save_dir = debug_checkpoint_dir / f"bestmodel_{corrector_name}"
        logger.info(f"Debug mode: Using debug checkpoint directory {debug_checkpoint_dir}")
    else:
        save_dir = checkpoint_dir / f"bestmodel_{corrector_name}"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    logger.info(f"Starting {corrector_type} corrector training...")
    history = trainer.train(model, train_loader, val_loader, save_dir)
    
    logger.info(f"Corrector training completed. Best model saved to {save_dir}")
    return history


def train_models(args, config, models_to_train):
    """Train all specified models"""
    logger = logging.getLogger('train_models')
    
    model_factory = ModelFactory(config)
    data_factory = DataLoaderFactory(config)
    
    checkpoint_dir = config.get_checkpoint_dir(args.dataset)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    corrector_models = ['unet_corrector', 'transformer_corrector', 'hybrid_corrector']
    corrector_wrapper_models = [m for m in models_to_train if any(c in m for c in ['unet_', 'transformer_', 'hybrid_']) and m not in corrector_models]
    pure_correctors = [m for m in models_to_train if m in corrector_models]
    
    # Train pure correctors first
    if pure_correctors:
        from trainers.corrector_trainer import CorrectorTrainer
        logger.info(f"Training corrector models: {pure_correctors}")
        
        for corrector_name in pure_correctors:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {corrector_name} corrector on {args.dataset}")
            logger.info(f"{'='*60}")
            
            # Use debug checkpoint dir if in debug mode
            if config.is_debug_mode():
                debug_checkpoint_dir = Path(config.get('debug.checkpoint_dir', '../../../debugmodelrendu/cifar10'))
                if not debug_checkpoint_dir.is_absolute():
                    debug_checkpoint_dir = Path(__file__).parent.parent / debug_checkpoint_dir
                debug_checkpoint_dir = debug_checkpoint_dir.resolve()
                corrector_dir = debug_checkpoint_dir / f"bestmodel_{corrector_name}"
            else:
                corrector_dir = checkpoint_dir / f"bestmodel_{corrector_name}"
                
            corrector_checkpoint = corrector_dir / "best_model.pt"
            
            if corrector_checkpoint.exists() and not args.force_retrain:
                logger.info(f"Corrector already exists at {corrector_checkpoint}, skipping training")
                continue
            
            train_corrector(corrector_name, args.dataset, config, model_factory, data_factory, checkpoint_dir)
    
    trainer_service = ModelTrainer(config, model_factory, data_factory)
    
    trained_models = {}
    
    # List of combined corrector models that don't need training
    combined_corrector_models = ['unet_resnet18', 'unet_vit', 'transformer_resnet18', 
                                'transformer_vit', 'hybrid_resnet18', 'hybrid_vit']
    
    for model_name in models_to_train:
        # Skip pure correctors as they're trained separately
        if model_name in corrector_models:
            continue
            
        # Skip combined corrector models - they use pre-trained components
        if model_name in combined_corrector_models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Skipping {model_name} - uses pre-trained corrector and classifier")
            logger.info(f"{'='*60}")
            continue
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} model on {args.dataset}")
        logger.info(f"{'='*60}")
        
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
        
        # Handle models that are inherently robust separately
        if model_name in ['resnet18_not_pretrained_robust', 'vgg_robust', 'vgg16_robust', 'vgg19_robust']:
            continue
            
        # Always train robust version for vanilla_vit (and optionally for others if --robust flag is set)
        # Not for blended models as they're already robust
        should_train_robust = (model_name == 'vanilla_vit') or (args.robust and model_name in ['ttt', 'ttt3fc'])
        
        if should_train_robust and model_name in ['vanilla_vit', 'ttt', 'ttt3fc']:
            robust_model_name = f"{model_name}_robust"
            logger.info(f"\nTraining robust version: {robust_model_name}")
            
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
    
    # Train resnet18_not_pretrained_robust if it's in the models list
    if 'resnet18_not_pretrained_robust' in models_to_train:
        logger.info(f"\nTraining resnet18_not_pretrained_robust with continuous transforms")
        
        model_dir = checkpoint_dir / "bestmodel_resnet18_not_pretrained_robust"
        checkpoint_path = model_dir / "best_model.pt"
        
        if checkpoint_path.exists() and not args.force_retrain:
            logger.info(f"Model already exists at {checkpoint_path}, skipping training")
        else:
            model, history = trainer_service.train_model(
                model_type='resnet18_not_pretrained_robust',
                dataset_name=args.dataset,
                robust_training=True
            )
            trained_models['resnet18_not_pretrained_robust'] = model
    
    # Train VGG robust models if they're in the models list
    vgg_robust_models = ['vgg_robust', 'vgg16_robust', 'vgg19_robust']
    for vgg_model in vgg_robust_models:
        if vgg_model in models_to_train:
            logger.info(f"\nTraining {vgg_model} with continuous transforms")
            
            model_dir = checkpoint_dir / f"bestmodel_{vgg_model}"
            checkpoint_path = model_dir / "best_model.pt"
            
            if checkpoint_path.exists() and not args.force_retrain:
                logger.info(f"Model already exists at {checkpoint_path}, skipping training")
            else:
                model, history = trainer_service.train_model(
                    model_type=vgg_model,
                    dataset_name=args.dataset,
                    robust_training=True
                )
                trained_models[vgg_model] = model
    
    return trained_models


def evaluate_models(args, config, models_to_evaluate):
    """Evaluate all specified models"""
    logger = logging.getLogger('evaluate_models')
    
    model_factory = ModelFactory(config)
    data_factory = DataLoaderFactory(config)
    
    if args.severities:
        severities = args.severities
    else:
        severities = config.get('evaluation.severities', [0.0, 0.25, 0.5, 0.75, 1.0])
    
    evaluator = ModelEvaluator(config, model_factory, data_factory)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating models on {args.dataset} with severities: {severities}")
    logger.info(f"{'='*60}")
    
    results = evaluator.evaluate_all_combinations(
        dataset_name=args.dataset,
        severities=severities,
        model_types=models_to_evaluate,
        include_ood=True
    )
    
    evaluator.print_results(results)
    
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
    
    if args.mode in ['train', 'both']:
        train_models(args, config, models_to_process)
    
    if args.mode in ['evaluate', 'both']:
        evaluate_models(args, config, models_to_process)
    
    logger.info("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script to train all experimental architectures with customized settings for each.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Architecture-specific configurations
ARCHITECTURE_CONFIGS = {
    'fourier': {
        'batch_size': 128,
        'learning_rate': 1e-3,
        'epochs': 100,
        'patience': 10,
        'description': 'Fourier Transform-based attention'
    },
    'elfatt': {
        'batch_size': 128,
        'learning_rate': 1e-3,
        'epochs': 100,
        'patience': 10,
        'description': 'Efficient Linear Attention'
    },
    'mamba': {
        'batch_size': 64,  # Smaller batch size for Mamba
        'learning_rate': 5e-4,
        'epochs': 100,
        'patience': 10,
        'description': 'Vision Mamba (State Space Model)'
    },
    'kan': {
        'batch_size': 64,  # Smaller batch size for KAN
        'learning_rate': 5e-4,
        'epochs': 80,
        'patience': 8,
        'description': 'Kolmogorov-Arnold Network attention'
    },
    'hybrid': {
        'batch_size': 96,
        'learning_rate': 8e-4,
        'epochs': 100,
        'patience': 10,
        'description': 'Hybrid CNN-Transformer'
    },
    'mixed': {
        'batch_size': 96,
        'learning_rate': 8e-4,
        'epochs': 100,
        'patience': 10,
        'description': 'Mixed architecture (Fourier + Mamba)'
    }
}


def run_training(architecture, config, args):
    """Run training for a single architecture"""
    print(f"\n{'='*60}")
    print(f"Training {architecture}: {config['description']}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 'train_experimental.py',
        '--architecture', architecture,
        '--data-root', args.data_root,
        '--epochs', str(config['epochs']),
        '--batch-size', str(config['batch_size']),
        '--learning-rate', str(config['learning_rate']),
        '--patience', str(config['patience']),
        '--min-delta', str(args.min_delta),
        '--checkpoint-dir', args.checkpoint_dir,
        '--wandb-project', args.wandb_project,
        '--warmup-epochs', str(args.warmup_epochs)
    ]
    
    if args.img_size != 224:
        cmd.extend(['--img-size', str(args.img_size)])
    
    if args.no_cleanup:
        cmd.append('--no-cleanup')
    
    # Run training
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    success = result.returncode == 0
    
    return {
        'architecture': architecture,
        'success': success,
        'exit_code': result.returncode,
        'config': config,
        'stdout': result.stdout,
        'stderr': result.stderr
    }


def run_evaluation(architecture, args):
    """Run evaluation for a single architecture"""
    print(f"\nEvaluating {architecture}...")
    
    cmd = [
        sys.executable, 'evaluate_experimental.py',
        '--architecture', architecture,
        '--checkpoint-dir', args.checkpoint_dir,
        '--data-root', args.data_root,
        '--batch-size', str(args.eval_batch_size),
    ]
    
    if args.img_size != 224:
        cmd.extend(['--img-size', str(args.img_size)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract metrics from output
    metrics = {}
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Top-1 Accuracy:' in line:
                try:
                    acc = float(line.split(':')[1].split('(')[0].strip())
                    metrics['top1_accuracy'] = acc
                except:
                    pass
            elif 'Top-5 Accuracy:' in line:
                try:
                    acc = float(line.split(':')[1].split('(')[0].strip())
                    metrics['top5_accuracy'] = acc
                except:
                    pass
    
    return {
        'success': result.returncode == 0,
        'metrics': metrics,
        'stdout': result.stdout,
        'stderr': result.stderr
    }


def main():
    parser = argparse.ArgumentParser(description='Train all experimental vision transformer architectures')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--wandb-project', type=str, default='experimental-vit',
                       help='Weights & Biases project name')
    parser.add_argument('--architectures', nargs='+', 
                       choices=list(ARCHITECTURE_CONFIGS.keys()) + ['all'],
                       default=['all'],
                       help='Architectures to train')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                       help='Minimum improvement delta for early stopping')
    parser.add_argument('--eval-batch-size', type=int, default=256,
                       help='Batch size for evaluation')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Disable automatic checkpoint cleanup')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation after training')
    parser.add_argument('--results-file', type=str, default='training_results.json',
                       help='File to save results summary')
    
    args = parser.parse_args()
    
    # Determine which architectures to train
    if 'all' in args.architectures:
        architectures = list(ARCHITECTURE_CONFIGS.keys())
    else:
        architectures = args.architectures
    
    print(f"Will train the following architectures: {architectures}")
    print(f"Dataset: {args.data_root}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # Results tracking
    results = {
        'start_time': datetime.now().isoformat(),
        'args': vars(args),
        'architectures': {}
    }
    
    # Train each architecture
    for arch in architectures:
        config = ARCHITECTURE_CONFIGS[arch]
        
        # Training
        train_result = run_training(arch, config, args)
        results['architectures'][arch] = {
            'config': config,
            'training': {
                'success': train_result['success'],
                'exit_code': train_result['exit_code']
            }
        }
        
        # Save intermediate results
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Evaluation
        if train_result['success'] and not args.skip_evaluation:
            eval_result = run_evaluation(arch, args)
            results['architectures'][arch]['evaluation'] = {
                'success': eval_result['success'],
                'metrics': eval_result['metrics']
            }
            
            # Save updated results
            with open(args.results_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Final summary
    results['end_time'] = datetime.now().isoformat()
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for arch, result in results['architectures'].items():
        print(f"\n{arch}:")
        print(f"  Training: {'SUCCESS' if result['training']['success'] else 'FAILED'}")
        if 'evaluation' in result and result['evaluation']['success']:
            metrics = result['evaluation']['metrics']
            if 'top1_accuracy' in metrics:
                print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
            if 'top5_accuracy' in metrics:
                print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
    
    # Save final results
    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.results_file}")
    
    # Create markdown summary
    summary_file = args.results_file.replace('.json', '_summary.md')
    with open(summary_file, 'w') as f:
        f.write("# Experimental Vision Transformer Training Results\n\n")
        f.write(f"**Date**: {results['start_time']} to {results['end_time']}\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Architecture | Description | Training | Top-1 Acc | Top-5 Acc |\n")
        f.write("|--------------|-------------|----------|-----------|------------|\n")
        
        for arch, result in results['architectures'].items():
            desc = ARCHITECTURE_CONFIGS[arch]['description']
            train_status = '✓' if result['training']['success'] else '✗'
            
            top1 = '-'
            top5 = '-'
            if 'evaluation' in result and result['evaluation']['success']:
                metrics = result['evaluation']['metrics']
                if 'top1_accuracy' in metrics:
                    top1 = f"{metrics['top1_accuracy']:.4f}"
                if 'top5_accuracy' in metrics:
                    top5 = f"{metrics['top5_accuracy']:.4f}"
            
            f.write(f"| {arch} | {desc} | {train_status} | {top1} | {top5} |\n")
    
    print(f"Summary report saved to: {summary_file}")


if __name__ == "__main__":
    main()
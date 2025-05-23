#!/usr/bin/env python3
"""
Evaluation script for experimental models with perturbations
Similar to autotinymodels evaluation with continuous transforms
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import experimental models
from experimental_vit import create_experimental_vit

# Import transformation utilities from autotinymodels
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'autotinymodels'))
from new_new import ContinuousTransforms, TinyImageNetDataset


class PerturbedDataset(Dataset):
    """Dataset wrapper that applies perturbations"""
    
    def __init__(self, base_dataset, severity=0.0, transform_type=None):
        self.base_dataset = base_dataset
        self.severity = severity
        self.transform_type = transform_type
        self.continuous_transforms = ContinuousTransforms(severity) if severity > 0 else None
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        if self.continuous_transforms is not None:
            # Apply continuous transforms
            img = self.continuous_transforms(img)
            
        return img, label


def load_experimental_model(model_path: str, architecture: str, device: torch.device):
    """Load a trained experimental model"""
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    else:
        num_classes = 200  # Default for Tiny-ImageNet
        
    if 'img_size' in checkpoint:
        img_size = checkpoint['img_size']
    else:
        img_size = 224
        
    # Create model
    model = create_experimental_vit(
        img_size=img_size,
        patch_size=checkpoint.get('patch_size', 16),
        in_chans=3,
        num_classes=num_classes,
        embed_dim=checkpoint.get('embed_dim', 768),
        depth=checkpoint.get('depth', 12),
        num_heads=checkpoint.get('num_heads', 12),
        mlp_ratio=4.0,
        attention_type=architecture
    ).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model, img_size


def evaluate_model(model, data_loader, device, desc="Evaluating"):
    """Evaluate model on a dataset"""
    
    model.eval()
    correct = 0
    total = 0
    losses = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=desc):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = np.mean(losses)
    
    return accuracy, avg_loss


def evaluate_with_perturbations(
    model_path: str,
    architecture: str,
    data_root: str,
    severities: List[float] = [0.0, 0.1, 0.3, 0.5, 0.75, 1.0],
    batch_size: int = 128,
    num_workers: int = 4,
    save_results: bool = True,
    plot_results: bool = True,
):
    """Evaluate model with different perturbation severities"""
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {architecture} model from {model_path}")
    model, img_size = load_experimental_model(model_path, architecture, device)
    
    # Data transforms (without augmentation for evaluation)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    base_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    
    # Load validation dataset
    val_path = Path(data_root) / 'val'
    base_dataset = datasets.ImageFolder(val_path, transform=base_transform)
    
    # Results storage
    results = {
        'architecture': architecture,
        'model_path': str(model_path),
        'severities': severities,
        'accuracies': [],
        'losses': [],
        'clean_accuracy': None,
        'clean_loss': None,
    }
    
    # Evaluate with different severities
    for severity in severities:
        print(f"\nEvaluating with severity: {severity}")
        
        # Create dataset with perturbations
        if severity == 0.0:
            # Clean data - apply normalization
            transform_with_norm = transforms.Compose([
                base_transform,
                normalize
            ])
            eval_dataset = datasets.ImageFolder(val_path, transform=transform_with_norm)
        else:
            # Perturbed data
            perturbed_dataset = PerturbedDataset(base_dataset, severity=severity)
            # We need to apply normalization after perturbation
            class NormalizedDataset(Dataset):
                def __init__(self, dataset):
                    self.dataset = dataset
                    self.normalize = normalize
                    
                def __len__(self):
                    return len(self.dataset)
                    
                def __getitem__(self, idx):
                    img, label = self.dataset[idx]
                    img = self.normalize(img)
                    return img, label
                    
            eval_dataset = NormalizedDataset(perturbed_dataset)
        
        # Create data loader
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Evaluate
        accuracy, loss = evaluate_model(
            model, eval_loader, device, 
            desc=f"Severity {severity}"
        )
        
        print(f"Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
        
        results['accuracies'].append(accuracy)
        results['losses'].append(loss)
        
        if severity == 0.0:
            results['clean_accuracy'] = accuracy
            results['clean_loss'] = loss
    
    # Calculate robustness metrics
    if results['clean_accuracy'] is not None:
        results['average_accuracy'] = np.mean(results['accuracies'])
        results['accuracy_drop'] = results['clean_accuracy'] - np.mean(results['accuracies'][1:])
        results['relative_robustness'] = np.mean(results['accuracies'][1:]) / results['clean_accuracy']
    
    # Save results
    if save_results:
        save_path = Path(model_path).parent / f"evaluation_results_perturbations.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {save_path}")
    
    # Plot results
    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.plot(severities, results['accuracies'], 'o-', linewidth=2, markersize=8)
        plt.xlabel('Perturbation Severity', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'{architecture.upper()} Model Robustness to Perturbations', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Add text with key metrics
        plt.text(0.02, 0.98, f"Clean Accuracy: {results['clean_accuracy']:.1f}%",
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.text(0.02, 0.88, f"Avg Accuracy: {results['average_accuracy']:.1f}%",
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plot_path = Path(model_path).parent / f"robustness_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {plot_path}")
    
    return results


def evaluate_all_architectures(
    models_dir: str,
    data_root: str,
    architectures: List[str] = ['fourier', 'elfatt', 'mamba', 'kan', 'hybrid', 'mixed'],
    severities: List[float] = [0.0, 0.1, 0.3, 0.5, 0.75, 1.0],
    batch_size: int = 128,
):
    """Evaluate all experimental architectures and compare"""
    
    all_results = {}
    
    for arch in architectures:
        model_path = Path(models_dir) / arch / "best_model.pt"
        
        if model_path.exists():
            print(f"\n{'='*60}")
            print(f"Evaluating {arch.upper()} architecture")
            print('='*60)
            
            results = evaluate_with_perturbations(
                model_path=str(model_path),
                architecture=arch,
                data_root=data_root,
                severities=severities,
                batch_size=batch_size,
                save_results=True,
                plot_results=True,
            )
            
            all_results[arch] = results
        else:
            print(f"Model not found for {arch}: {model_path}")
    
    # Create comparison plot
    if len(all_results) > 1:
        plt.figure(figsize=(12, 8))
        
        for arch, results in all_results.items():
            plt.plot(results['severities'], results['accuracies'], 'o-', 
                    label=f"{arch.upper()} (clean: {results['clean_accuracy']:.1f}%)",
                    linewidth=2, markersize=6)
        
        plt.xlabel('Perturbation Severity', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Experimental Architectures Robustness Comparison', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        comparison_path = Path(models_dir) / "architecture_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nComparison plot saved to: {comparison_path}")
        
        # Save comparison results
        comparison_results = {
            'architectures': list(all_results.keys()),
            'severities': severities,
            'results': {
                arch: {
                    'accuracies': results['accuracies'],
                    'clean_accuracy': results['clean_accuracy'],
                    'average_accuracy': results.get('average_accuracy', 0),
                    'accuracy_drop': results.get('accuracy_drop', 0),
                }
                for arch, results in all_results.items()
            }
        }
        
        comparison_json_path = Path(models_dir) / "comparison_results.json"
        with open(comparison_json_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"Comparison results saved to: {comparison_json_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate experimental models with perturbations')
    parser.add_argument('--model-path', type=str,
                       help='Path to specific model checkpoint')
    parser.add_argument('--architecture', type=str,
                       choices=['fourier', 'elfatt', 'mamba', 'kan', 'hybrid', 'mixed'],
                       help='Architecture type (required if model-path is specified)')
    parser.add_argument('--models-dir', type=str, default='../../../experimentalmodels',
                       help='Directory containing all model folders')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to Tiny-ImageNet dataset root')
    parser.add_argument('--severities', type=float, nargs='+',
                       default=[0.0, 0.1, 0.3, 0.5, 0.75, 1.0],
                       help='Perturbation severity levels to evaluate')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--evaluate-all', action='store_true',
                       help='Evaluate all architectures in models-dir')
    
    args = parser.parse_args()
    
    if args.evaluate_all:
        # Evaluate all architectures
        evaluate_all_architectures(
            models_dir=args.models_dir,
            data_root=args.data_root,
            severities=args.severities,
            batch_size=args.batch_size,
        )
    elif args.model_path and args.architecture:
        # Evaluate single model
        evaluate_with_perturbations(
            model_path=args.model_path,
            architecture=args.architecture,
            data_root=args.data_root,
            severities=args.severities,
            batch_size=args.batch_size,
        )
    else:
        print("Please specify either --evaluate-all or both --model-path and --architecture")


if __name__ == "__main__":
    main()
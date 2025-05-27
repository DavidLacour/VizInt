#!/usr/bin/env python3
"""
Evaluation script for baseline models (ResNet18, VGG16, etc.)
Tests models on clean and corrupted data
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from baseline_models import SimpleResNet18, SimpleVGG16
from new_new import TinyImageNetDataset, ContinuousTransforms

def load_baseline_model(model_path, model_class, device):
    """Load a baseline model from checkpoint"""
    print(f"Loading model from {model_path}")
    model = model_class(num_classes=200)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_loader, device):
    """Evaluate model accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if len(batch) == 4:  # Corrupted data
                orig_images, trans_images, labels, params = batch
                images = trans_images.to(device)
            else:  # Clean data
                images, labels = batch
                images = images.to(device)
            
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy, correct, total

def evaluate_all_baselines(dataset_path, severities=[0.0, 0.3, 0.5, 0.75, 1.0]):
    """Evaluate all baseline models on clean and corrupted data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define models to evaluate
    models_config = {
        'resnet18': {
            'class': SimpleResNet18,
            'path': './bestmodel_resnet18_baseline/best_model.pt',
            'name': 'ResNet18'
        },
        'vgg16': {
            'class': SimpleVGG16,
            'path': './bestmodel_vgg16_baseline/best_model.pt',
            'name': 'VGG16'
        }
    }
    
    # Standard transforms
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Results storage
    all_results = {}
    
    # Load and evaluate each model
    for model_key, config in models_config.items():
        if not os.path.exists(config['path']):
            print(f"⚠️  {config['name']} model not found at {config['path']}, skipping...")
            continue
            
        print(f"\n{'='*80}")
        print(f"Evaluating {config['name']}")
        print(f"{'='*80}")
        
        # Load model
        model = load_baseline_model(config['path'], config['class'], device)
        model_results = {}
        
        # Evaluate on each severity
        for severity in severities:
            if severity == 0.0:
                # Clean data
                val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
                val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
                
                print(f"\nEvaluating on clean data (severity 0.0)...")
                accuracy, correct, total = evaluate_model(model, val_loader, device)
                
            else:
                # Corrupted data
                ood_transform = ContinuousTransforms(severity=severity)
                ood_val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val, ood_transform=ood_transform)
                
                def collate_fn(batch):
                    orig_imgs, trans_imgs, labels, params = zip(*batch)
                    return torch.stack(orig_imgs), torch.stack(trans_imgs), torch.tensor(labels), params
                
                ood_val_loader = DataLoader(ood_val_dataset, batch_size=128, shuffle=False, 
                                          num_workers=4, collate_fn=collate_fn)
                
                print(f"\nEvaluating on corrupted data (severity {severity})...")
                accuracy, correct, total = evaluate_model(model, ood_val_loader, device)
            
            model_results[severity] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            
            print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        all_results[model_key] = model_results
    
    return all_results

def print_results_table(all_results, severities):
    """Print a formatted results table"""
    print("\n" + "="*100)
    print("BASELINE MODELS EVALUATION RESULTS")
    print("="*100)
    
    # Header
    print(f"{'Model':<15}", end="")
    for severity in severities:
        if severity == 0.0:
            print(f"{'Clean':<12}", end="")
        else:
            print(f"{'Sev ' + str(severity):<12}", end="")
    print(f"{'Avg Drop':<12}")
    print("-"*100)
    
    # Results for each model
    for model_key, results in all_results.items():
        model_name = model_key.upper()
        print(f"{model_name:<15}", end="")
        
        clean_acc = results.get(0.0, {}).get('accuracy', 0)
        
        drops = []
        for severity in severities:
            if severity in results:
                acc = results[severity]['accuracy']
                print(f"{acc:.4f}      ", end="")
                
                if severity > 0 and clean_acc > 0:
                    drop = ((clean_acc - acc) / clean_acc) * 100
                    drops.append(drop)
            else:
                print(f"{'N/A':<12}", end="")
        
        # Average drop
        if drops:
            avg_drop = np.mean(drops)
            print(f"{avg_drop:.1f}%")
        else:
            print("N/A")
    
    print("="*100)

def plot_results(all_results, severities, save_path="baseline_evaluation.png"):
    """Create and save a plot comparing baseline models"""
    plt.figure(figsize=(10, 6))
    
    for model_key, results in all_results.items():
        model_name = model_key.upper()
        
        # Get accuracies for each severity
        accuracies = []
        valid_severities = []
        
        for severity in severities:
            if severity in results:
                accuracies.append(results[severity]['accuracy'])
                valid_severities.append(severity)
        
        if accuracies:
            plt.plot(valid_severities, accuracies, 'o-', label=model_name, linewidth=2, markersize=8)
    
    plt.xlabel('Corruption Severity', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Baseline Models Performance vs Corruption Severity', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models")
    parser.add_argument("--dataset", type=str, default="../tiny-imagenet-200",
                        help="Path to the Tiny ImageNet dataset")
    parser.add_argument("--severities", type=str, default="0.0,0.3,0.5,0.75,1.0",
                        help="Comma-separated list of corruption severities to evaluate")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save a comparison plot")
    parser.add_argument("--plot_path", type=str, default="baseline_evaluation.png",
                        help="Path to save the plot")
    args = parser.parse_args()
    
    # Parse severities
    severities = [float(s) for s in args.severities.split(',')]
    
    # Evaluate all models
    results = evaluate_all_baselines(args.dataset, severities)
    
    # Print results table
    print_results_table(results, severities)
    
    # Generate plot if requested
    if args.plot:
        plot_results(results, severities, args.plot_path)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for model_key, results in results.items():
        model_name = model_key.upper()
        clean_acc = results.get(0.0, {}).get('accuracy', 0)
        
        if clean_acc > 0:
            print(f"\n{model_name}:")
            print(f"  Clean accuracy: {clean_acc:.4f}")
            
            # Find worst drop
            worst_drop = 0
            worst_severity = 0
            for severity in severities:
                if severity > 0 and severity in results:
                    acc = results[severity]['accuracy']
                    drop = ((clean_acc - acc) / clean_acc) * 100
                    if drop > worst_drop:
                        worst_drop = drop
                        worst_severity = severity
            
            if worst_drop > 0:
                print(f"  Worst drop: {worst_drop:.1f}% at severity {worst_severity}")

if __name__ == "__main__":
    main()
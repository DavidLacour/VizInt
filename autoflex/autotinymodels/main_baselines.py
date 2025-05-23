import os
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path
import shutil
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader

from new_new import * 
# Import BlendedTTT modules
from blended_ttt_model import BlendedTTT
from blended_ttt_training import train_blended_ttt_model
from blended_ttt_evaluation import evaluate_models_with_blended

# Import new TTT modules
from ttt_model import TestTimeTrainer, train_ttt_model
from ttt_evaluation import evaluate_with_ttt, evaluate_with_test_time_adaptation
from robust_training import *

# Import baseline models - use enhanced version with early stopping
try:
    from baseline_models_enhanced import SimpleResNet18, SimpleVGG16, train_baseline_model_with_early_stopping as train_baseline_model
except ImportError:
    # Fallback to original if enhanced version not available
    from baseline_models import SimpleResNet18, SimpleVGG16, train_baseline_model


def evaluate_full_pipeline(main_model, healer_model, dataset_path, severities=[0.1,0.2,0.3,0.4,0.6], model_dir="./", include_blended=True, include_ttt=True):
    """
    Evaluate the full transformation healing pipeline on clean and transformed data.
    
    Args:
        main_model: The classification model
        healer_model: The transformation healer model
        dataset_path: Path to the dataset
        severities: List of severity levels to evaluate
        model_dir: Directory containing the models
        include_blended: Whether to include BlendedTTT in evaluation
        include_ttt: Whether to include TTT in evaluation
        
    Returns:
        all_results: Dictionary of evaluation results
    """
    all_results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # First evaluate on clean data (severity 0.0) - This is the "WITHOUT TRANSFORMS" evaluation
    print("\n" + "="*80)
    print("EVALUATING WITHOUT TRANSFORMS (CLEAN DATA)")
    print("="*80)
    
    # Define standard transforms for validation
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get validation dataset (clean data)
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
    
    # DataLoader for validation
    batch_size = 128 if torch.cuda.is_available() else 64
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Evaluate main model on clean data
    main_model.eval()
    main_correct = 0
    total = 0
    
    # Find blended_model and ttt_model if needed
    blended_model = None
    ttt_model = None
    
    if include_blended:
        blended_model_path = f"{model_dir}/bestmodel_blended/best_model.pt"
        if os.path.exists(blended_model_path):
            blended_model = load_blended_model(blended_model_path, main_model, device)
            
    if include_ttt:
        ttt_model_path = f"{model_dir}/bestmodel_ttt/best_model.pt"
        if os.path.exists(ttt_model_path):
            ttt_model = load_ttt_model(ttt_model_path, main_model, device)
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating models (clean data)"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass with main model
            outputs = main_model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Update metrics
            total += labels.size(0)
            main_correct += (predicted == labels).sum().item()
    
    # Calculate main model accuracy
    main_accuracy = main_correct / total
    
    # Initialize clean results dictionary with main model
    clean_results = {
        'main': {
            'accuracy': main_accuracy,
            'correct': main_correct,
            'total': total
        }
    }
    
    # Now evaluate healer model on clean data
    # The healer should pass through clean data unchanged
    healer_model.eval()
    healer_correct = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating healer (clean data)"):
            images, labels = images.to(device), labels.to(device)
            
            # Pass through healer (should minimally alter clean images)
            # Get predictions from healer
            healer_predictions = healer_model(images)
            # Apply "corrections" (should be minimal for clean data)
            corrected_images = healer_model.apply_correction(images, healer_predictions)
            
            # Forward pass with main model on corrected images
            outputs = main_model(corrected_images)
            _, predicted = torch.max(outputs, 1)
            
            # Update metrics
            healer_correct += (predicted == labels).sum().item()
    
    # Calculate healer accuracy
    healer_accuracy = healer_correct / total
    
    # Add healer results to clean_results
    clean_results['healer'] = {
        'accuracy': healer_accuracy,
        'correct': healer_correct,
        'total': total
    }
    
    # Evaluate BlendedTTT model on clean data if available
    blended_accuracy = None
    blended_correct = 0
    if include_blended and blended_model is not None:
        blended_model.eval()
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating BlendedTTT (clean data)"):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass with blended model
                outputs, _ = blended_model(images)
                _, predicted = torch.max(outputs, 1)
                
                # Update metrics
                blended_correct += (predicted == labels).sum().item()
        
        # Calculate blended accuracy
        blended_accuracy = blended_correct / total
        
        # Add blended results to clean_results
        clean_results['blended'] = {
            'accuracy': blended_accuracy,
            'correct': blended_correct,
            'total': total
        }
    else:
        clean_results['blended'] = None
    
    # Evaluate TTT model on clean data if available
    ttt_accuracy = None
    ttt_correct = 0
    ttt_adapted_correct = 0
    if include_ttt and ttt_model is not None:
        ttt_model.eval()
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating TTT (clean data)"):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass with TTT model (no adaptation)
                outputs, _ = ttt_model(images)
                _, predicted = torch.max(outputs, 1)
                
                # Update metrics
                ttt_correct += (predicted == labels).sum().item()
                
                # TTT with adaptation (for each batch)
                with torch.enable_grad(): # cant do ttt without 
                    adapted_outputs = ttt_model.adapt(images, reset=True)
                    _, adapted_predicted = torch.max(adapted_outputs, 1)
                
                # Update adapted metrics
                ttt_adapted_correct += (adapted_predicted == labels).sum().item()
        
        # Calculate TTT accuracies
        ttt_accuracy = ttt_correct / total
        ttt_adapted_accuracy = ttt_adapted_correct / total
        
        # Add TTT results to clean_results
        clean_results['ttt'] = {
            'accuracy': ttt_accuracy,
            'correct': ttt_correct,
            'total': total
        }
        
        clean_results['ttt_adapted'] = {
            'accuracy': ttt_adapted_accuracy,
            'correct': ttt_adapted_correct,
            'total': total
        }
    else:
        clean_results['ttt'] = None
        clean_results['ttt_adapted'] = None
    
    all_results[0.0] = clean_results
    
    # Print clean data results
    print(f"\nWITHOUT TRANSFORMS - Accuracy Results:")
    print(f"  Main Model: {main_accuracy:.4f}")
    print(f"  Healer Model: {healer_accuracy:.4f}")
    
    if include_blended and blended_model is not None:
        print(f"  BlendedTTT Model: {blended_accuracy:.4f}")
    
    if include_ttt and ttt_model is not None:
        print(f"  TTT Model: {ttt_accuracy:.4f}")
        print(f"  TTT Model (adapted): {ttt_adapted_accuracy:.4f}")
    
    # Now evaluate on transformed data with different severity levels - This is the "WITH TRANSFORMS" evaluation
    for severity in severities:
        print("\n" + "="*80)
        print(f"EVALUATING WITH TRANSFORMS (SEVERITY {severity})")
        print("="*80)
        
        # Create continuous transforms for OOD
        ood_transform = ContinuousTransforms(severity=severity)
        
        # Get OOD validation set
        ood_val_dataset = TinyImageNetDataset(
            dataset_path, "val", transform_val, ood_transform=ood_transform
        )
        
        # Simplified collate function
        def collate_fn(batch):
            orig_imgs, trans_imgs, labels, params = zip(*batch)
            return torch.stack(orig_imgs), torch.stack(trans_imgs), torch.tensor(labels), params
        
        # DataLoader for OOD validation
        ood_val_loader = DataLoader(
            ood_val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        # Metrics
        main_correct = 0
        healer_correct = 0
        total = 0
        
        # Per-transformation metrics
        transform_types = ['no_transform', 'gaussian_noise', 'rotation', 'affine']
        main_per_transform_metrics = {t: {'correct': 0, 'total': 0} for t in transform_types}
        healer_per_transform_metrics = {t: {'correct': 0, 'total': 0} for t in transform_types}
        
        # Additional metrics for BlendedTTT and TTT if included
        blended_correct = 0
        blended_per_transform_metrics = {t: {'correct': 0, 'total': 0} for t in transform_types}
        
        ttt_correct = 0
        ttt_per_transform_metrics = {t: {'correct': 0, 'total': 0} for t in transform_types}
        
        ttt_adapted_correct = 0
        ttt_adapted_per_transform_metrics = {t: {'correct': 0, 'total': 0} for t in transform_types}
        
        # Helper function to determine transform type consistently
        def get_transform_type(params):
            if isinstance(params, dict) and 'transform_type' in params:
                return params['transform_type']
            return 'no_transform'
        
        # Evaluate on OOD data
        main_model.eval()
        healer_model.eval()
        
        if include_blended and blended_model is not None:
            blended_model.eval()
            
        if include_ttt and ttt_model is not None:
            ttt_model.eval()
        
        with torch.no_grad():
            for orig_images, trans_images, labels, params in tqdm(ood_val_loader, desc=f"Evaluating (severity {severity})"):
                orig_images = orig_images.to(device)
                trans_images = trans_images.to(device)
                labels = labels.to(device)
                
                # Forward pass with main model
                main_outputs = main_model(trans_images)
                _, main_predicted = torch.max(main_outputs, 1)
                
                # Update main metrics
                total += labels.size(0)
                main_correct += (main_predicted == labels).sum().item()
                
                # Forward pass with healer + main model
                healer_predictions = healer_model(trans_images)
                corrected_images = healer_model.apply_correction(trans_images, healer_predictions)
                
                healer_outputs = main_model(corrected_images)
                _, healer_predicted = torch.max(healer_outputs, 1)
                
                # Update healer metrics
                healer_correct += (healer_predicted == labels).sum().item()
                
                # BlendedTTT evaluation if included
                if include_blended and blended_model is not None:
                    blended_outputs, _ = blended_model(trans_images)
                    _, blended_predicted = torch.max(blended_outputs, 1)
                    
                    # Update BlendedTTT metrics
                    blended_correct += (blended_predicted == labels).sum().item()
                
                # TTT evaluation if included
                if include_ttt and ttt_model is not None:
                    # TTT without adaptation
                    ttt_outputs, _ = ttt_model(trans_images)
                    _, ttt_predicted = torch.max(ttt_outputs, 1)
                    
                    # Update TTT metrics
                    ttt_correct += (ttt_predicted == labels).sum().item()
                    
                    # TTT with adaptation (for each batch)
                    with torch.enable_grad(): # cant do ttt without 
                        ttt_adapted_outputs = ttt_model.adapt(trans_images, reset=True)
                        _, ttt_adapted_predicted = torch.max(ttt_adapted_outputs, 1)
                    
                    # Update TTT adapted metrics
                    ttt_adapted_correct += (ttt_adapted_predicted == labels).sum().item()
                
                # Update per-transformation metrics
                for i, p in enumerate(params):
                    t_type = get_transform_type(p)
                    
                    # Main model metrics
                    main_per_transform_metrics[t_type]['total'] += 1
                    if main_predicted[i] == labels[i]:
                        main_per_transform_metrics[t_type]['correct'] += 1
                    
                    # Healer model metrics
                    healer_per_transform_metrics[t_type]['total'] += 1
                    if healer_predicted[i] == labels[i]:
                        healer_per_transform_metrics[t_type]['correct'] += 1
                    
                    # BlendedTTT metrics if included
                    if include_blended and blended_model is not None:
                        blended_per_transform_metrics[t_type]['total'] += 1
                        if blended_predicted[i] == labels[i]:
                            blended_per_transform_metrics[t_type]['correct'] += 1
                    
                    # TTT metrics if included
                    if include_ttt and ttt_model is not None:
                        ttt_per_transform_metrics[t_type]['total'] += 1
                        if ttt_predicted[i] == labels[i]:
                            ttt_per_transform_metrics[t_type]['correct'] += 1
                        
                        ttt_adapted_per_transform_metrics[t_type]['total'] += 1
                        if ttt_adapted_predicted[i] == labels[i]:
                            ttt_adapted_per_transform_metrics[t_type]['correct'] += 1
        
        # Calculate accuracies
        main_accuracy = main_correct / total
        healer_accuracy = healer_correct / total
        
        # Calculate per-transform accuracies for main and healer
        main_per_transform_acc = {}
        healer_per_transform_acc = {}
        
        for t_type in transform_types:
            if main_per_transform_metrics[t_type]['total'] > 0:
                main_per_transform_acc[t_type] = (
                    main_per_transform_metrics[t_type]['correct'] / 
                    main_per_transform_metrics[t_type]['total']
                )
            else:
                main_per_transform_acc[t_type] = 0.0
                
            if healer_per_transform_metrics[t_type]['total'] > 0:
                healer_per_transform_acc[t_type] = (
                    healer_per_transform_metrics[t_type]['correct'] / 
                    healer_per_transform_metrics[t_type]['total']
                )
            else:
                healer_per_transform_acc[t_type] = 0.0
        
        # Calculate BlendedTTT metrics if included
        blended_accuracy = None
        blended_per_transform_acc = {}
        
        if include_blended and blended_model is not None:
            blended_accuracy = blended_correct / total
            
            for t_type in transform_types:
                if blended_per_transform_metrics[t_type]['total'] > 0:
                    blended_per_transform_acc[t_type] = (
                        blended_per_transform_metrics[t_type]['correct'] / 
                        blended_per_transform_metrics[t_type]['total']
                    )
                else:
                    blended_per_transform_acc[t_type] = 0.0
        
        # Calculate TTT metrics if included
        ttt_accuracy = None
        ttt_per_transform_acc = {}
        ttt_adapted_accuracy = None
        ttt_adapted_per_transform_acc = {}
        
        if include_ttt and ttt_model is not None:
            ttt_accuracy = ttt_correct / total
            ttt_adapted_accuracy = ttt_adapted_correct / total
            
            for t_type in transform_types:
                if ttt_per_transform_metrics[t_type]['total'] > 0:
                    ttt_per_transform_acc[t_type] = (
                        ttt_per_transform_metrics[t_type]['correct'] / 
                        ttt_per_transform_metrics[t_type]['total']
                    )
                else:
                    ttt_per_transform_acc[t_type] = 0.0
                    
                if ttt_adapted_per_transform_metrics[t_type]['total'] > 0:
                    ttt_adapted_per_transform_acc[t_type] = (
                        ttt_adapted_per_transform_metrics[t_type]['correct'] / 
                        ttt_adapted_per_transform_metrics[t_type]['total']
                    )
                else:
                    ttt_adapted_per_transform_acc[t_type] = 0.0
        
        # Create results
        ood_results = {
            'main': {
                'accuracy': main_accuracy,
                'correct': main_correct,
                'total': total,
                'per_transform_acc': main_per_transform_acc
            },
            'healer': {
                'accuracy': healer_accuracy,
                'correct': healer_correct,
                'total': total,
                'per_transform_acc': healer_per_transform_acc
            }
        }
        
        # Add BlendedTTT results if included
        if include_blended:
            if blended_model is not None:
                ood_results['blended'] = {
                    'accuracy': blended_accuracy,
                    'correct': blended_correct,
                    'total': total,
                    'per_transform_acc': blended_per_transform_acc
                }
            else:
                ood_results['blended'] = None
        else:
            ood_results['blended'] = None
        
        # Add TTT results if included
        if include_ttt:
            if ttt_model is not None:
                ood_results['ttt'] = {
                    'accuracy': ttt_accuracy,
                    'correct': ttt_correct,
                    'total': total,
                    'per_transform_acc': ttt_per_transform_acc
                }
                
                ood_results['ttt_adapted'] = {
                    'accuracy': ttt_adapted_accuracy,
                    'correct': ttt_adapted_correct,
                    'total': total,
                    'per_transform_acc': ttt_adapted_per_transform_acc
                }
            else:
                ood_results['ttt'] = None
                ood_results['ttt_adapted'] = None
        else:
            ood_results['ttt'] = None
            ood_results['ttt_adapted'] = None
        
        all_results[severity] = ood_results
        
        # Print results
        print(f"\nWITH TRANSFORMS - Accuracy Results (Severity {severity}):")
        print(f"  Main Model: {main_accuracy:.4f}")
        print(f"  Healer Model: {healer_accuracy:.4f}")
        
        if include_blended and blended_model is not None:
            print(f"  BlendedTTT Model: {blended_accuracy:.4f}")
        
        if include_ttt and ttt_model is not None:
            print(f"  TTT Model: {ttt_accuracy:.4f}")
            print(f"  TTT Model (adapted): {ttt_adapted_accuracy:.4f}")
        
        # Calculate and print performance drop due to transformations
        print(f"\nPerformance Impact of Transforms (Drop from Clean Data):")
        main_drop = clean_results['main']['accuracy'] - main_accuracy
        healer_drop = clean_results['main']['accuracy'] - healer_accuracy
        
        print(f"  Main Model: {main_drop:.4f} ({main_drop/clean_results['main']['accuracy']*100:.1f}% drop)")
        print(f"  Healer Model: {healer_drop:.4f} ({healer_drop/clean_results['main']['accuracy']*100:.1f}% drop)")
        
        if include_blended and blended_model is not None:
            blended_drop = clean_results['main']['accuracy'] - blended_accuracy
            print(f"  BlendedTTT Model: {blended_drop:.4f} ({blended_drop/clean_results['main']['accuracy']*100:.1f}% drop)")
        
        if include_ttt and ttt_model is not None:
            ttt_drop = clean_results['main']['accuracy'] - ttt_accuracy
            ttt_adapted_drop = clean_results['main']['accuracy'] - ttt_adapted_accuracy
            print(f"  TTT Model: {ttt_drop:.4f} ({ttt_drop/clean_results['main']['accuracy']*100:.1f}% drop)")
            print(f"  TTT Model (adapted): {ttt_adapted_drop:.4f} ({ttt_adapted_drop/clean_results['main']['accuracy']*100:.1f}% drop)")
        
        # Print per-transformation accuracies
        print("\nPer-Transformation Accuracy:")
        for t_type in transform_types:
            print(f"  {t_type.upper()}:")
            print(f"    Main: {main_per_transform_acc[t_type]:.4f}")
            print(f"    Healer: {healer_per_transform_acc[t_type]:.4f}")
            
            if include_blended and blended_model is not None:
                print(f"    BlendedTTT: {blended_per_transform_acc[t_type]:.4f}")
            
            if include_ttt and ttt_model is not None:
                print(f"    TTT: {ttt_per_transform_acc[t_type]:.4f}")
                print(f"    TTT (adapted): {ttt_adapted_per_transform_acc[t_type]:.4f}")
    
    # Log results to wandb if available
    try:
        import wandb
        log_wandb_results_with_all_models(all_results)
    except:
        print("Note: wandb not available or error in logging.")
    
    return all_results


def compare_with_without_transforms(results):
    """
    Explicitly compare results with and without transforms.
    
    Args:
        results: Dictionary of evaluation results from evaluate_full_pipeline
    
    Returns:
        comparison: Dictionary of comparison metrics
    """
    comparison = {}
    
    # Get clean results (without transforms)
    if 0.0 not in results:
        print("Error: No clean data (without transforms) results available for comparison")
        return comparison
        
    clean_results = results[0.0]
    
    # Compare for each severity level (with transforms)
    for severity in [s for s in results.keys() if s > 0.0]:
        ood_results = results[severity]
        
        severity_comparison = {}
        
        # Compare main model
        if 'main' in clean_results and 'main' in ood_results:
            clean_acc = clean_results['main']['accuracy']
            ood_acc = ood_results['main']['accuracy']
            drop = clean_acc - ood_acc
            drop_percent = (drop / clean_acc) * 100 if clean_acc > 0 else 0
            
            severity_comparison['main'] = {
                'clean_accuracy': clean_acc,
                'transform_accuracy': ood_acc,
                'drop': drop,
                'drop_percent': drop_percent
            }
        
        # Compare healer model
        if 'healer' in clean_results and 'healer' in ood_results:
            clean_acc = clean_results['healer']['accuracy']
            ood_acc = ood_results['healer']['accuracy']
            drop = clean_acc - ood_acc
            drop_percent = (drop / clean_acc) * 100 if clean_acc > 0 else 0
            
            severity_comparison['healer'] = {
                'clean_accuracy': clean_acc,
                'transform_accuracy': ood_acc,
                'drop': drop,
                'drop_percent': drop_percent
            }
        
        # Compare BlendedTTT model
        if ('blended' in clean_results and clean_results['blended'] is not None and
            'blended' in ood_results and ood_results['blended'] is not None):
            clean_acc = clean_results['blended']['accuracy']
            ood_acc = ood_results['blended']['accuracy']
            drop = clean_acc - ood_acc
            drop_percent = (drop / clean_acc) * 100 if clean_acc > 0 else 0
            
            severity_comparison['blended'] = {
                'clean_accuracy': clean_acc,
                'transform_accuracy': ood_acc,
                'drop': drop,
                'drop_percent': drop_percent
            }
        
        # Compare TTT model
        if ('ttt' in clean_results and clean_results['ttt'] is not None and
            'ttt' in ood_results and ood_results['ttt'] is not None):
            clean_acc = clean_results['ttt']['accuracy']
            ood_acc = ood_results['ttt']['accuracy']
            drop = clean_acc - ood_acc
            drop_percent = (drop / clean_acc) * 100 if clean_acc > 0 else 0
            
            severity_comparison['ttt'] = {
                'clean_accuracy': clean_acc,
                'transform_accuracy': ood_acc,
                'drop': drop,
                'drop_percent': drop_percent
            }
        
        # Compare TTT adapted model
        if ('ttt_adapted' in clean_results and clean_results['ttt_adapted'] is not None and
            'ttt_adapted' in ood_results and ood_results['ttt_adapted'] is not None):
            clean_acc = clean_results['ttt_adapted']['accuracy']
            ood_acc = ood_results['ttt_adapted']['accuracy']
            drop = clean_acc - ood_acc
            drop_percent = (drop / clean_acc) * 100 if clean_acc > 0 else 0
            
            severity_comparison['ttt_adapted'] = {
                'clean_accuracy': clean_acc,
                'transform_accuracy': ood_acc,
                'drop': drop,
                'drop_percent': drop_percent
            }
        
        comparison[severity] = severity_comparison
    
    # Print comparison summary
    print("\n" + "="*80)
    print("SUMMARY: PERFORMANCE COMPARISON WITH VS. WITHOUT TRANSFORMS")
    print("="*80)
    
    for severity, severity_comparison in comparison.items():
        print(f"\nSeverity {severity}:")
        
        # Find which model has the smallest drop
        model_drops = {
            model_name: data['drop_percent'] 
            for model_name, data in severity_comparison.items()
        }
        
        if model_drops:
            best_model = min(model_drops.items(), key=lambda x: x[1])
            
            print(f"  Model Performance (Clean → Transformed):")
            for model_name, data in severity_comparison.items():
                print(f"    {model_name.ljust(12)}: {data['clean_accuracy']:.4f} → {data['transform_accuracy']:.4f} "
                      f"(Drop: {data['drop']:.4f}, {data['drop_percent']:.1f}%)")
            
            print(f"\n  Best Transform Robustness: {best_model[0]} with {best_model[1]:.1f}% drop")
    
    return comparison


def plot_transform_comparison(results, save_path="transform_comparison.png"):
    """
    Create and save a plot comparing model performance with and without transforms.
    
    Args:
        results: Dictionary of evaluation results from evaluate_full_pipeline
        save_path: Path to save the generated plot
    """
    if 0.0 not in results:
        print("Error: Cannot generate plot, no clean data results available")
        return
    
    clean_results = results[0.0]
    severities = sorted([s for s in results.keys() if s > 0.0])
    
    if not severities:
        print("Error: Cannot generate plot, no transform results available")
        return
    
    # Models to include in the plot
    models = ['main', 'healer', 'blended', 'ttt', 'ttt_adapted']
    model_names = {
        'main': 'Main Model',
        'healer': 'Healer Model',
        'blended': 'BlendedTTT',
        'ttt': 'TTT',
        'ttt_adapted': 'TTT (adapted)'
    }
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Collect data for plotting
    for model in models:
        if model not in clean_results or clean_results[model] is None:
            continue
            
        # Get clean (without transforms) accuracy
        clean_acc = clean_results[model]['accuracy']
        
        # Collect accuracies for each severity (with transforms)
        severity_accs = []
        for severity in severities:
            if model in results[severity] and results[severity][model] is not None:
                severity_accs.append(results[severity][model]['accuracy'])
            else:
                severity_accs.append(None)  # Mark missing data
        
        # Plot clean accuracy as a horizontal line
        plt.axhline(y=clean_acc, linestyle='--', alpha=0.5, 
                   label=f"{model_names[model]} (Clean: {clean_acc:.4f})")
        
        # Plot accuracies with transforms
        valid_points = [(sev, acc) for sev, acc in zip(severities, severity_accs) if acc is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, 'o-', linewidth=2, markersize=8, 
                    label=f"{model_names[model]} (With Transforms)")
    
    # Add labels and title
    plt.xlabel('Transform Severity', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Performance: With vs. Without Transforms', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Set y-axis limits with some padding
    plt.ylim(0, 1.05)
    
    # Adjust x-axis ticks
    plt.xticks(severities)
    
    # Ensure the directory exists before saving
    save_dir = os.path.dirname(save_path)
    if save_dir:  # Only create directory if save_path includes a directory
        os.makedirs(save_dir, exist_ok=True)
    
    # Save and show plot
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Transform comparison plot saved to {save_path}")


# Baseline helper functions
def train_baseline_resnet18(dataset_path):
    """Train a ResNet18 baseline model"""
    print("Training ResNet18 baseline model...")
    model = SimpleResNet18(num_classes=200)
    trained_model = train_baseline_model(
        model, dataset_path, 
        model_name="resnet18_baseline", 
        epochs=50, 
        lr=0.001
    )
    return trained_model

def load_baseline_model(model_path, device):
    """Load baseline ResNet18 model from checkpoint"""
    print(f"Loading baseline model from {model_path}")
    model = SimpleResNet18(num_classes=200)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def train_baseline_vgg16(dataset_path):
    """Train a VGG16 baseline model"""
    print("Training VGG16 baseline model...")
    model = SimpleVGG16(num_classes=200)
    trained_model = train_baseline_model(
        model, dataset_path, 
        model_name="vgg16_baseline", 
        epochs=50, 
        lr=0.001
    )
    return trained_model

def load_vgg16_model(model_path, device):
    """Load baseline VGG16 model from checkpoint"""
    print(f"Loading VGG16 model from {model_path}")
    model = SimpleVGG16(num_classes=200)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

# Pretrained model class and functions
class PretrainedResNet18(nn.Module):
    """
    ResNet18 with ImageNet pretrained weights, fine-tuned for Tiny ImageNet
    This shows the benefit of pretraining vs training from scratch
    """
    def __init__(self, num_classes=200):
        super(PretrainedResNet18, self).__init__()
        # Use ImageNet pretrained weights
        from torchvision import models
        self.resnet = models.resnet18(pretrained=True)  # ✅ PRETRAINED = TRUE
        
        # Modify first conv layer for 64x64 input (instead of 224x224)
        # We'll keep the pretrained weights but adjust the layer
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Initialize new conv layer with interpolated weights from pretrained layer
        with torch.no_grad():
            # Get center crop of the 7x7 pretrained kernel to make it 3x3
            old_weight = old_conv.weight
            new_weight = old_weight[:, :, 2:5, 2:5].clone()  # Extract center 3x3
            self.resnet.conv1.weight.copy_(new_weight)
        
        # Remove maxpool since input is smaller
        self.resnet.maxpool = nn.Identity()
        
        # Modify final layer for 200 classes (this will be randomly initialized)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

def train_pretrained_resnet18(dataset_path):
    """Train pretrained ResNet18 model with fine-tuning"""
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import os
    from tqdm import tqdm
    
    print("Training pretrained ResNet18 model (ImageNet → Tiny ImageNet)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PretrainedResNet18(num_classes=200).to(device)
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    train_dataset = TinyImageNetDataset(dataset_path, "train", transform_train)
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Fine-tuning setup (lower learning rate for pretrained features)
    criterion = nn.CrossEntropyLoss()
    
    # Different learning rates: lower for pretrained features, higher for new classifier
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name:  # Final classifier layer
            classifier_params.append(param)
        else:  # Pretrained backbone
            backbone_params.append(param)
    
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': 0.0001},    # Lower LR for pretrained features
        {'params': classifier_params, 'lr': 0.001}    # Higher LR for new classifier
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Train for fewer epochs since we start from pretrained weights
    epochs = 30  # Reduced from 50 since we have pretrained weights
    best_val_acc = 0.0
    
    print(f"Fine-tuning for {epochs} epochs...")
    print(f"Backbone LR: 0.0001, Classifier LR: 0.001")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("./bestmodel_pretrained_resnet18", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, "./bestmodel_pretrained_resnet18/best_model.pt")
            print(f"  ✅ New best pretrained model saved with val_acc: {val_acc:.4f}")
        
        scheduler.step()
        print()
    
    print(f"Fine-tuning completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Expected improvement over from-scratch: +10-15% accuracy")
    return model

def load_pretrained_model(model_path, device):
    """Load pretrained ResNet18 model from checkpoint"""
    print(f"Loading pretrained model from {model_path}")
    model = PretrainedResNet18(num_classes=200)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

class IdentityHealer(nn.Module):
    """Dummy healer that does nothing - for baseline comparison"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.zeros(x.size(0), 1, device=x.device)
        
    def apply_correction(self, images, predictions):
        return images

def compare_with_baseline(your_results, baseline_results):
    """Compare your model results with baseline ResNet18"""
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINE RESNET18")
    print("="*60)
    
    # Compare clean performance
    if 0.0 in your_results and 0.0 in baseline_results:
        your_clean = your_results[0.0]['main']['accuracy']
        baseline_clean = baseline_results[0.0]['main']['accuracy']
        improvement = your_clean - baseline_clean
        
        print(f"\nClean Data Performance:")
        print(f"  Your ViT Model: {your_clean:.4f} ({your_clean*100:.1f}%)")
        print(f"  Baseline:       {baseline_clean:.4f} ({baseline_clean*100:.1f}%)")
        print(f"  Improvement:    {improvement:.4f} ({improvement*100:.1f} percentage points)")
        
        if improvement > 0:
            print(f"  ✓ Your ViT is {improvement*100:.1f}% better on clean data")
        else:
            print(f"  ⚠ Your ViT is {abs(improvement)*100:.1f}% worse on clean data")
    
    # Compare transform robustness
    print(f"\nTransform Robustness Comparison:")
    for severity in sorted([s for s in your_results.keys() if s > 0]):
        if severity in baseline_results:
            your_acc = your_results[severity]['main']['accuracy']
            baseline_acc = baseline_results[severity]['main']['accuracy']
            improvement = your_acc - baseline_acc
            
            print(f"\n  Severity {severity}:")
            print(f"    Your ViT:       {your_acc:.4f}")
            print(f"    Baseline:       {baseline_acc:.4f}")
            print(f"    Improvement:    {improvement:.4f}")
            
            # Compare with healer if available
            if ('healer' in your_results[severity] and 
                your_results[severity]['healer'] is not None):
                your_healer_acc = your_results[severity]['healer']['accuracy']
                healer_vs_baseline = your_healer_acc - baseline_acc
                healer_vs_your_main = your_healer_acc - your_acc
                print(f"    Your ViT+Healer: {your_healer_acc:.4f}")
                print(f"    Healer vs Baseline: {healer_vs_baseline:.4f}")
                print(f"    Healer Benefit: {healer_vs_your_main:.4f}")

def compare_with_pretrained(your_results, pretrained_results):
    """Compare your model results with pretrained ResNet18"""
    print("\n" + "="*60)
    print("COMPARISON WITH PRETRAINED RESNET18 (ImageNet → Tiny ImageNet)")
    print("="*60)
    
    # Compare clean performance
    if 0.0 in your_results and 0.0 in pretrained_results:
        your_clean = your_results[0.0]['main']['accuracy']
        pretrained_clean = pretrained_results[0.0]['main']['accuracy']
        gap = pretrained_clean - your_clean
        
        print(f"\nClean Data Performance:")
        print(f"  Your ViT Model:     {your_clean:.4f} ({your_clean*100:.1f}%)")
        print(f"  Pretrained ResNet:  {pretrained_clean:.4f} ({pretrained_clean*100:.1f}%)")
        print(f"  Pretraining Gap:    {gap:.4f} ({gap*100:.1f} percentage points)")
        
        if gap > 0:
            print(f"  📈 Pretraining provides {gap*100:.1f}% advantage")
        else:
            print(f"  🎯 Your ViT overcomes pretraining by {abs(gap)*100:.1f}%!")
    
    # Compare transform robustness
    print(f"\nTransform Robustness Comparison:")
    for severity in sorted([s for s in your_results.keys() if s > 0]):
        if severity in pretrained_results:
            your_acc = your_results[severity]['main']['accuracy']
            pretrained_acc = pretrained_results[severity]['main']['accuracy']
            gap = pretrained_acc - your_acc
            
            print(f"\n  Severity {severity}:")
            print(f"    Your ViT:           {your_acc:.4f}")
            print(f"    Pretrained ResNet:  {pretrained_acc:.4f}")
            print(f"    Pretraining Gap:    {gap:.4f}")
            
            # Compare with healer if available
            if ('healer' in your_results[severity] and 
                your_results[severity]['healer'] is not None):
                your_healer_acc = your_results[severity]['healer']['accuracy']
                healer_vs_pretrained = your_healer_acc - pretrained_acc
                healer_vs_your_main = your_healer_acc - your_acc
                
                print(f"    Your ViT+Healer:    {your_healer_acc:.4f}")
                print(f"    Healer vs Pretrained: {healer_vs_pretrained:.4f}")
                print(f"    Healer Benefit:     {healer_vs_your_main:.4f}")
                
                if healer_vs_pretrained > 0:
                    print(f"    🎯 Healer beats pretraining by {healer_vs_pretrained*100:.1f}%!")

def compare_three_models(your_results, baseline_results, pretrained_results):
    """Compare all three: Your models vs Baseline vs Pretrained"""
    print("\n" + "="*80)
    print("🏆 THREE-WAY CHAMPIONSHIP: YOUR MODELS vs BASELINE vs PRETRAINED")
    print("="*80)
    
    # Clean data comparison
    if (0.0 in your_results and 0.0 in baseline_results and 0.0 in pretrained_results):
        your_clean = your_results[0.0]['main']['accuracy']
        baseline_clean = baseline_results[0.0]['main']['accuracy']
        pretrained_clean = pretrained_results[0.0]['main']['accuracy']
        
        # Check if healer beats everyone
        your_healer_clean = your_results[0.0]['healer']['accuracy'] if 'healer' in your_results[0.0] else your_clean
        
        print(f"\n🥇 CLEAN DATA LEADERBOARD:")
        models = [
            ("Your ViT", your_clean),
            ("Your ViT+Healer", your_healer_clean),
            ("Baseline ResNet18", baseline_clean),
            ("Pretrained ResNet18", pretrained_clean)
        ]
        
        # Sort by accuracy
        models.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, acc) in enumerate(models):
            medal = ["🥇", "🥈", "🥉", "4️⃣"][i] if i < 4 else f"{i+1}️⃣"
            print(f"  {medal} {name}: {acc:.4f} ({acc*100:.1f}%)")
        
        # Analysis
        winner = models[0]
        print(f"\n🏆 CLEAN DATA WINNER: {winner[0]} with {winner[1]:.4f}")
        
        if "Your" in winner[0]:
            print("🎉 Your approach wins on clean data!")
        elif "Pretrained" in winner[0]:
            print("📈 Pretraining dominates on clean data")
        else:
            print("📊 Simple baseline leads")
    
    # Transform robustness comparison
    print(f"\n🛡️ TRANSFORM ROBUSTNESS COMPARISON:")
    
    for severity in sorted([s for s in your_results.keys() if s > 0]):
        if severity in baseline_results and severity in pretrained_results:
            your_acc = your_results[severity]['main']['accuracy']
            baseline_acc = baseline_results[severity]['main']['accuracy']
            pretrained_acc = pretrained_results[severity]['main']['accuracy']
            
            your_healer_acc = (your_results[severity]['healer']['accuracy'] 
                             if 'healer' in your_results[severity] and your_results[severity]['healer'] 
                             else your_acc)
            
            print(f"\n  Severity {severity} Leaderboard:")
            models = [
                ("Your ViT", your_acc),
                ("Your ViT+Healer", your_healer_acc),
                ("Baseline ResNet18", baseline_acc),
                ("Pretrained ResNet18", pretrained_acc)
            ]
            
            models.sort(key=lambda x: x[1], reverse=True)
            
            for i, (name, acc) in enumerate(models):
                medal = ["🥇", "🥈", "🥉", "4️⃣"][i] if i < 4 else f"{i+1}️⃣"
                print(f"    {medal} {name}: {acc:.4f}")
            
            # Highlight if healer is particularly good at this severity
            if models[0][0] == "Your ViT+Healer":
                print(f"    🎯 Healer dominates at severity {severity}!")
    
    print(f"\n" + "="*80)
    print("💡 KEY INSIGHTS:")
    print("• Pretrained models usually win on clean data (ImageNet knowledge)")
    print("• Your Healer should excel at transform robustness")
    print("• Baseline shows the minimum performance bar")
    print("• Compare drops from clean→transformed to see robustness")
    print("="*80)


# 🚀 NEW COMPREHENSIVE EVALUATION FUNCTIONS

def evaluate_all_model_combinations(dataset_path, severities, model_dir, args, device):
    """
    Evaluate ALL model combinations:
    1. Main (not robust)
    2. Main robust  
    3. Healer + Main (not robust)
    4. Healer + Main robust
    5. TTT + Main (not robust)
    6. TTT + Main robust
    7. BlendedTTT + Main (not robust) 
    8. BlendedTTT + Main robust
    9. Baseline (ResNet18 from scratch)
    10. Pretrained (ResNet18 with ImageNet pretraining)
    """
    
    print("\n" + "="*100)
    print("🏆 COMPREHENSIVE MODEL EVALUATION - ALL COMBINATIONS")
    print("="*100)
    
    all_model_results = {}
    
    # Load all models
    models = {}
    
    # 1. Load main models
    main_model_path = f"{model_dir}/bestmodel_main/best_model.pt"
    robust_model_path = f"{model_dir}/bestmodel_robust/best_model.pt"
    healer_model_path = f"{model_dir}/bestmodel_healer/best_model.pt"
    
    if os.path.exists(main_model_path):
        models['main'] = load_main_model(main_model_path, device)
        print("✅ Loaded: Main Model (not robust)")
    else:
        print("❌ Missing: Main Model")
        return None
        
    if os.path.exists(robust_model_path):
        models['main_robust'] = load_main_model(robust_model_path, device)
        print("✅ Loaded: Main Model (robust)")
    else:
        print("⚠️  Missing: Main Model (robust) - will skip robust combinations")
        
    if os.path.exists(healer_model_path):
        models['healer'] = load_healer_model(healer_model_path, device)
        print("✅ Loaded: Healer Model")
    else:
        print("❌ Missing: Healer Model")
        return None
    
    # 2. Load TTT models
    if not args.exclude_ttt:
        ttt_model_path = f"{model_dir}/bestmodel_ttt/best_model.pt"
        if os.path.exists(ttt_model_path):
            models['ttt'] = load_ttt_model(ttt_model_path, models['main'], device)
            print("✅ Loaded: TTT Model (based on main)")
            
            # Create TTT model based on robust main if available
            if 'main_robust' in models:
                models['ttt_robust'] = load_ttt_model(ttt_model_path, models['main_robust'], device)
                print("✅ Loaded: TTT Model (based on robust)")
        else:
            print("⚠️  Missing: TTT Model - will skip TTT combinations")
    
    # 3. Load BlendedTTT models  
    if not args.exclude_blended:
        blended_model_path = f"{model_dir}/bestmodel_blended/best_model.pt"
        if os.path.exists(blended_model_path):
            models['blended'] = load_blended_model(blended_model_path, models['main'], device)
            print("✅ Loaded: BlendedTTT Model (based on main)")
            
            # Create BlendedTTT based on robust main if available
            if 'main_robust' in models:
                models['blended_robust'] = load_blended_model(blended_model_path, models['main_robust'], device)
                print("✅ Loaded: BlendedTTT Model (based on robust)")
        else:
            print("⚠️  Missing: BlendedTTT Model - will skip BlendedTTT combinations")
    
    # 4. Load baseline models
    if args.compare_baseline:
        baseline_model_path = f"{model_dir}/bestmodel_resnet18_baseline/best_model.pt"
        if os.path.exists(baseline_model_path):
            models['baseline'] = load_baseline_model(baseline_model_path, device)
            print("✅ Loaded: Baseline ResNet18 (from scratch)")
        else:
            print("⚠️  Missing: Baseline ResNet18 - use --train_baseline to train it")
    
    if args.compare_pretrained:
        pretrained_model_path = f"{model_dir}/bestmodel_pretrained_resnet18/best_model.pt"
        if os.path.exists(pretrained_model_path):
            models['pretrained'] = load_pretrained_model(pretrained_model_path, device)
            print("✅ Loaded: Pretrained ResNet18 (ImageNet)")
        else:
            print("⚠️  Missing: Pretrained ResNet18 - use --train_pretrained to train it")
    
    # Load VGG16 baseline model
    if args.compare_vgg16:
        vgg16_model_path = f"{model_dir}/bestmodel_vgg16_baseline/best_model.pt"
        if os.path.exists(vgg16_model_path):
            models['vgg16'] = load_vgg16_model(vgg16_model_path, device)
            print("✅ Loaded: Baseline VGG16 (from scratch)")
        else:
            print("⚠️  Missing: Baseline VGG16 - use --train_vgg16 to train it")
    
    print(f"\n📊 Evaluating {len(models)} model combinations on {len(severities)} severity levels...")
    
    # Define evaluation combinations
    combinations = [
        # Format: (name, main_model_key, healer_model_key, description)
        ("Main", "main", None, "Main ViT (not robust)"),
        ("Main_Robust", "main_robust", None, "Main ViT (robust training)"),
        ("Healer+Main", "main", "healer", "Healer + Main ViT (not robust)"),
        ("Healer+Main_Robust", "main_robust", "healer", "Healer + Main ViT (robust)"),
        ("TTT+Main", "ttt", None, "TTT + Main ViT (not robust)"),
        ("TTT+Main_Robust", "ttt_robust", None, "TTT + Main ViT (robust)"),
        ("BlendedTTT+Main", "blended", None, "BlendedTTT + Main ViT (not robust)"),
        ("BlendedTTT+Main_Robust", "blended_robust", None, "BlendedTTT + Main ViT (robust)"),
        ("Baseline", "baseline", None, "ResNet18 (from scratch)"),
        ("Pretrained", "pretrained", None, "ResNet18 (ImageNet pretrained)"),
        ("VGG16", "vgg16", None, "VGG16 (from scratch)"),
    ]
    
    # Evaluate each combination
    for combo_name, main_key, healer_key, description in combinations:
        if main_key not in models:
            print(f"⏭️  Skipping {combo_name}: {main_key} model not available")
            continue
            
        if healer_key and healer_key not in models:
            print(f"⏭️  Skipping {combo_name}: {healer_key} model not available")
            continue
            
        print(f"\n🔍 Evaluating: {description}")
        
        main_model = models[main_key]
        healer_model = models[healer_key] if healer_key else IdentityHealer().to(device)
        
        # Evaluate this combination
        results = evaluate_model_combination(main_model, healer_model, dataset_path, severities, device, main_key)
        
        all_model_results[combo_name] = {
            'results': results,
            'description': description
        }
    
    return all_model_results

def evaluate_model_combination(main_model, healer_model, dataset_path, severities, device, model_type):
    """Evaluate a specific model + healer combination"""
    results = {}
    
    # Standard validation transforms
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Evaluate on each severity level
    for severity in severities:
        print(f"    Severity {severity}...", end=" ")
        
        if severity == 0.0:
            # Clean data evaluation
            val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
            
            correct = 0
            total = 0
            
            main_model.eval()
            healer_model.eval()
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    # Apply healer if not identity
                    if not isinstance(healer_model, IdentityHealer):
                        healer_predictions = healer_model(images)
                        images = healer_model.apply_correction(images, healer_predictions)
                    
                    # Forward pass - handle different model types
                    if 'ttt' in model_type.lower():
                        outputs, _ = main_model(images)  # TTT models return tuple
                    elif 'blended' in model_type.lower():
                        outputs, _ = main_model(images)  # BlendedTTT models return tuple
                    else:
                        outputs = main_model(images)  # Regular models
                    
                    _, predicted = torch.max(outputs, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            results[severity] = accuracy
            print(f"{accuracy:.4f}")
            
        else:
            # Transformed data evaluation
            ood_transform = ContinuousTransforms(severity=severity)
            ood_val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val, ood_transform=ood_transform)
            
            def collate_fn(batch):
                orig_imgs, trans_imgs, labels, params = zip(*batch)
                return torch.stack(orig_imgs), torch.stack(trans_imgs), torch.tensor(labels), params
            
            ood_val_loader = DataLoader(ood_val_dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate_fn)
            
            correct = 0
            total = 0
            
            main_model.eval()
            healer_model.eval()
            
            with torch.no_grad():
                for orig_images, trans_images, labels, params in ood_val_loader:
                    trans_images, labels = trans_images.to(device), labels.to(device)
                    
                    # Apply healer if not identity
                    if not isinstance(healer_model, IdentityHealer):
                        healer_predictions = healer_model(trans_images)
                        trans_images = healer_model.apply_correction(trans_images, healer_predictions)
                    
                    # Forward pass - handle different model types
                    if 'ttt' in model_type.lower():
                        outputs, _ = main_model(trans_images)  # TTT models return tuple
                    elif 'blended' in model_type.lower():
                        outputs, _ = main_model(trans_images)  # BlendedTTT models return tuple
                    else:
                        outputs = main_model(trans_images)  # Regular models
                    
                    _, predicted = torch.max(outputs, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            results[severity] = accuracy
            print(f"{accuracy:.4f}")
    
    return results

def print_comprehensive_results(all_model_results, severities):
    """Print comprehensive comparison of all model combinations"""
    
    print("\n" + "="*120)
    print("🏆 COMPREHENSIVE RESULTS - ALL MODEL COMBINATIONS")
    print("="*120)
    
    # Create results table
    print(f"\n{'Model Combination':<30} {'Description':<35} ", end="")
    for severity in severities:
        if severity == 0.0:
            print(f"{'Clean':<8}", end="")
        else:
            print(f"{'S'+str(severity):<8}", end="")
    print()
    print("-" * 120)
    
    # Sort models by clean data performance
    model_items = [(name, data) for name, data in all_model_results.items()]
    model_items.sort(key=lambda x: x[1]['results'].get(0.0, 0), reverse=True)
    
    for name, data in model_items:
        results = data['results']
        description = data['description']
        
        print(f"{name:<30} {description:<35} ", end="")
        for severity in severities:
            if severity in results:
                acc = results[severity]
                print(f"{acc:.4f}  ", end="")
            else:
                print(f"{'--':<8}", end="")
        print()
    
    # Analysis section
    print("\n" + "="*120)
    print("📊 ANALYSIS")
    print("="*120)
    
    # Find best performers
    clean_best = max(model_items, key=lambda x: x[1]['results'].get(0.0, 0))
    print(f"🥇 Best Clean Data Performance: {clean_best[0]} ({clean_best[1]['results'][0.0]:.4f})")
    
    # Find best transform robustness (smallest drop from clean)
    robustness_scores = []
    for name, data in model_items:
        results = data['results']
        if 0.0 in results and len([s for s in severities if s > 0 and s in results]) > 0:
            clean_acc = results[0.0]
            avg_transform_acc = np.mean([results[s] for s in severities if s > 0 and s in results])
            drop = clean_acc - avg_transform_acc
            drop_percent = (drop / clean_acc) * 100 if clean_acc > 0 else 100
            robustness_scores.append((name, drop_percent, avg_transform_acc))
    
    if robustness_scores:
        most_robust = min(robustness_scores, key=lambda x: x[1])
        print(f"🛡️  Most Transform Robust: {most_robust[0]} ({most_robust[1]:.1f}% average drop)")
    
    # Compare key combinations
    print(f"\n🔍 KEY COMPARISONS:")
    
    # Your best vs pretrained
    your_best_clean = 0
    your_best_name = ""
    for name, data in model_items:
        if any(keyword in name.lower() for keyword in ['main', 'healer', 'ttt', 'blended']):
            if data['results'].get(0.0, 0) > your_best_clean:
                your_best_clean = data['results'].get(0.0, 0)
                your_best_name = name
    
    pretrained_acc = all_model_results.get('Pretrained', {}).get('results', {}).get(0.0, 0)
    if your_best_clean > 0 and pretrained_acc > 0:
        gap = pretrained_acc - your_best_clean
        if gap > 0:
            print(f"📈 Pretraining Advantage: {gap:.4f} ({gap*100:.1f} points)")
            print(f"   Your Best: {your_best_name} ({your_best_clean:.4f})")
            print(f"   Pretrained: ({pretrained_acc:.4f})")
        else:
            print(f"🎯 Your Model Wins! {your_best_name} beats pretraining by {abs(gap):.4f}")
    
    # Healer benefit analysis
    main_acc = all_model_results.get('Main', {}).get('results', {}).get(0.0, 0)
    healer_main_acc = all_model_results.get('Healer+Main', {}).get('results', {}).get(0.0, 0)
    
    if main_acc > 0 and healer_main_acc > 0:
        healer_benefit = healer_main_acc - main_acc
        print(f"🔧 Healer Benefit (Clean): {healer_benefit:.4f} ({healer_benefit*100:.1f} points)")
        
        # Check healer benefit on transforms
        avg_severities = [s for s in severities if s > 0]
        if avg_severities:
            main_transform_avg = np.mean([all_model_results['Main']['results'].get(s, 0) for s in avg_severities])
            healer_transform_avg = np.mean([all_model_results['Healer+Main']['results'].get(s, 0) for s in avg_severities])
            transform_benefit = healer_transform_avg - main_transform_avg
            print(f"🔧 Healer Benefit (Transforms): {transform_benefit:.4f} ({transform_benefit*100:.1f} points)")
    
    # Robust training benefit
    main_acc = all_model_results.get('Main', {}).get('results', {}).get(0.0, 0)
    main_robust_acc = all_model_results.get('Main_Robust', {}).get('results', {}).get(0.0, 0)
    
    if main_acc > 0 and main_robust_acc > 0:
        robust_cost = main_acc - main_robust_acc  # Usually robust training hurts clean performance
        print(f"💪 Robust Training Cost (Clean): {robust_cost:.4f} ({robust_cost*100:.1f} points)")

def run_comprehensive_evaluation(args, device):
    """Run the comprehensive evaluation of all model combinations"""
    
    # Evaluate all combinations
    all_results = evaluate_all_model_combinations(
        args.dataset, args.severities, args.model_dir, args, device
    )
    
    if all_results is None:
        print("❌ Cannot run comprehensive evaluation - missing required models")
        return None
    
    # Print comprehensive results
    print_comprehensive_results(all_results, args.severities)
    
    # Generate plots for key comparisons
    create_comprehensive_plots(all_results, args.severities, args.visualize_dir)
    
    return all_results

def create_comprehensive_plots(all_results, severities, save_dir):
    """Create comprehensive comparison plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Plot all model combinations
    for name, data in all_results.items():
        results = data['results']
        sev_list = sorted([s for s in severities if s in results])
        acc_list = [results[s] for s in sev_list]
        
        if len(sev_list) > 1:
            plt.plot(sev_list, acc_list, 'o-', linewidth=2, markersize=6, label=name)
    
    plt.xlabel('Transform Severity', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Comprehensive Model Comparison: All Combinations', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = f"{save_dir}/comprehensive_comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"📊 Comprehensive comparison plot saved to {save_path}")
    plt.close()
    
    # Create leaderboard plot (clean data)
    clean_results = [(name, data['results'].get(0.0, 0)) for name, data in all_results.items() if 0.0 in data['results']]
    clean_results.sort(key=lambda x: x[1], reverse=True)
    
    if clean_results:
        plt.figure(figsize=(12, 8))
        names, accs = zip(*clean_results)
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        
        bars = plt.barh(range(len(names)), accs, color=colors)
        plt.yticks(range(len(names)), names)
        plt.xlabel('Clean Data Accuracy', fontsize=12)
        plt.title('Clean Data Performance Leaderboard', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (name, acc) in enumerate(clean_results):
            plt.text(acc + 0.005, i, f'{acc:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        save_path = f"{save_dir}/clean_data_leaderboard.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"🏆 Clean data leaderboard saved to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Transform Healing with Vision Transformers")
    parser.add_argument("--mode", type=str, default="evaluate", choices=["train", "evaluate", "visualize", "all"],
                      help="Mode of operation: train, evaluate, visualize, or all")
    parser.add_argument("--dataset", type=str, default="../tiny-imagenet-200",
                      help="Path to the dataset")
    parser.add_argument("--model_dir", type=str, default="../../../currentmodels",
                      help="Directory to save/load models")
    parser.add_argument("--visualize_dir", type=str, default="visualizations",
                      help="Directory to save visualizations")
    parser.add_argument("--severity", type=float, default=0.5,
                      help="Severity of transformations for evaluation/visualization")
    parser.add_argument("--num_samples", type=int, default=5,
                      help="Number of samples to visualize")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--exclude_blended", action="store_true",
                      help="Whether to exclude BlendedTTT model from the pipeline")
    parser.add_argument("--exclude_ttt", action="store_true",
                      help="Whether to exclude TTT model from the pipeline")
    parser.add_argument("--skip_ttt", action="store_true",
                      help="Skip training TTT model (but still use it for evaluation if available)")
    parser.add_argument("--severities", type=str, default="0.0,0.3,0.5,0.75,1.0",
                      help="Comma-separated list of transformation severities to evaluate")
    
    # BASELINE ARGUMENTS
    parser.add_argument("--train_baseline", action="store_true",
                      help="Train baseline ResNet18 model")
    parser.add_argument("--compare_baseline", action="store_true",
                      help="Include baseline comparison in evaluation")
    parser.add_argument("--train_pretrained", action="store_true",
                      help="Train pretrained ResNet18 model (ImageNet → Tiny ImageNet)")
    parser.add_argument("--compare_pretrained", action="store_true",
                      help="Include pretrained model comparison in evaluation")
    parser.add_argument("--train_vgg16", action="store_true",
                      help="Train baseline VGG16 model")
    parser.add_argument("--compare_vgg16", action="store_true",
                      help="Include VGG16 comparison in evaluation")
    
    args = parser.parse_args()
    
    # Parse severities from string to list of floats
    args.severities = [float(s) for s in args.severities.split(',')]
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    main_model = None
    main_model_robust = None
    healer_model = None
    ttt_model = None
    blended_model = None
    baseline_model = None
    
    # Check if models exist before training
    main_model_path = f"{args.model_dir}/bestmodel_main/best_model.pt"
    robust_model_path = f"{args.model_dir}/bestmodel_robust/best_model.pt"
    healer_model_path = f"{args.model_dir}/bestmodel_healer/best_model.pt"
    ttt_model_path = f"{args.model_dir}/bestmodel_ttt/best_model.pt" if not args.exclude_ttt else None
    blended_model_path = f"{args.model_dir}/bestmodel_blended/best_model.pt" if not args.exclude_blended else None
    baseline_model_path = f"{args.model_dir}/bestmodel_resnet18_baseline/best_model.pt"
    
    # Training mode - FIXED: Now properly checks if models exist and trains missing ones
    if args.mode in ["train", "evaluate", "all"]:
        print(f"\n=== Checking for existing models (mode: {args.mode}) ===")
        
        # Check if main model exists
        if not os.path.exists(main_model_path):
            print("\n=== Training Main Classification Model ===")
            main_model = train_main_model(args.dataset)
        else:
            print(f"✓ Main model found at {main_model_path}")
            main_model = load_main_model(main_model_path, device)
        
        # Check if robust main model exists
        if not os.path.exists(robust_model_path):
            print("\n=== Training Robust Main Classification Model ===")
            main_model_robust = train_main_model_robust(args.dataset, severity=args.severity)
        else:
            print(f"✓ Robust model found at {robust_model_path}")
            main_model_robust = load_main_model(robust_model_path, device)
        
        # Check if healer model exists
        if not os.path.exists(healer_model_path):
            print("\n=== Training Transformation Healer Model ===")
            healer_model = train_healer_model(args.dataset, severity=args.severity)
        else:
            print(f"✓ Healer model found at {healer_model_path}")
            healer_model = load_healer_model(healer_model_path, device)
        
        # Train TTT model if not skipped, not excluded, and not already existing
        if not args.exclude_ttt and not args.skip_ttt:
            if ttt_model_path and not os.path.exists(ttt_model_path):
                print("\n=== Training Test-Time Training Model ===")
                if main_model is None:
                    main_model = load_main_model(main_model_path, device)
                ttt_model = train_ttt_model(args.dataset, base_model=main_model, severity=args.severity)
            elif ttt_model_path:
                print(f"✓ TTT model found at {ttt_model_path}")
                if main_model is None:
                    main_model = load_main_model(main_model_path, device)
                ttt_model = load_ttt_model(ttt_model_path, main_model, device)
        
        # Train BlendedTTT model if not excluded and not already existing
        if not args.exclude_blended:
            if blended_model_path and not os.path.exists(blended_model_path):
                print("\n=== Training BlendedTTT Model ===")
                if main_model is None:
                    main_model = load_main_model(main_model_path, device)
                blended_model = train_blended_ttt_model(main_model, args.dataset)
            elif blended_model_path:
                print(f"✓ BlendedTTT model found at {blended_model_path}")
                if main_model is None:
                    main_model = load_main_model(main_model_path, device)
                blended_model = load_blended_model(blended_model_path, main_model, device)
        
        # BASELINE: Check if baseline should be trained
        if args.train_baseline or (args.compare_baseline and not os.path.exists(baseline_model_path)):
            print("\n=== Training Baseline ResNet18 Model ===")
            baseline_model = train_baseline_resnet18(args.dataset)
        elif os.path.exists(baseline_model_path):
            print(f"✓ Baseline model found at {baseline_model_path}")
            if args.compare_baseline:
                baseline_model = load_baseline_model(baseline_model_path, device)
        
        # PRETRAINED: Check if pretrained model should be trained
        if args.train_pretrained or (args.compare_pretrained and not os.path.exists(pretrained_model_path)):
            print("\n=== Training Pretrained ResNet18 Model ===")
            pretrained_model = train_pretrained_resnet18(args.dataset)
        elif os.path.exists(pretrained_model_path):
            print(f"✓ Pretrained model found at {pretrained_model_path}")
            if args.compare_pretrained:
                pretrained_model = load_pretrained_model(pretrained_model_path, device)
        
        # VGG16: Check if VGG16 should be trained
        vgg16_model_path = f"{args.model_dir}/bestmodel_vgg16_baseline/best_model.pt"
        if args.train_vgg16 or (args.compare_vgg16 and not os.path.exists(vgg16_model_path)):
            print("\n=== Training Baseline VGG16 Model ===")
            vgg16_model = train_baseline_vgg16(args.dataset)
        elif os.path.exists(vgg16_model_path):
            print(f"✓ VGG16 model found at {vgg16_model_path}")
            if args.compare_vgg16:
                vgg16_model = load_vgg16_model(vgg16_model_path, device)
    
    # Evaluation mode
    if args.mode in ["evaluate", "visualize", "all"]:
        print("\n=== Comprehensive Evaluation With and Without Transforms ===")
        
        # Load any models that weren't already loaded during training phase
        if main_model is None and os.path.exists(main_model_path):
            main_model = load_main_model(main_model_path, device)
        
        if main_model_robust is None and os.path.exists(robust_model_path):
            main_model_robust = load_main_model(robust_model_path, device)
        
        if healer_model is None and os.path.exists(healer_model_path):
            healer_model = load_healer_model(healer_model_path, device)
        
        if not args.exclude_ttt and ttt_model is None and ttt_model_path and os.path.exists(ttt_model_path):
            ttt_model = load_ttt_model(ttt_model_path, main_model, device)
        
        if not args.exclude_blended and blended_model is None and blended_model_path and os.path.exists(blended_model_path):
            blended_model = load_blended_model(blended_model_path, main_model, device)
        
        if args.compare_baseline and baseline_model is None and os.path.exists(baseline_model_path):
            baseline_model = load_baseline_model(baseline_model_path, device)
        
        if args.compare_pretrained and pretrained_model is None and os.path.exists(pretrained_model_path):
            pretrained_model = load_pretrained_model(pretrained_model_path, device)
        
        # Check if we have the minimum required models
        if main_model is None or healer_model is None:
            print("❌ ERROR: Missing required models!")
            print("Required models status:")
            print(f"  Main model: {'✓ Found' if os.path.exists(main_model_path) else '❌ Missing'} at {main_model_path}")
            print(f"  Healer model: {'✓ Found' if os.path.exists(healer_model_path) else '❌ Missing'} at {healer_model_path}")
            print(f"\nDataset path: {args.dataset}")
            print(f"Dataset exists: {'✓' if os.path.exists(args.dataset) else '❌'}")
            print("\n💡 Tip: Models should be trained automatically in 'evaluate' mode.")
            print("If this error persists, try running:")
            print(f"   python main_modified.py --mode train --dataset {args.dataset}")
            return
        
        # 🚀 NEW: Run comprehensive evaluation of ALL model combinations
        print("\n" + "="*100)
        print("🔥 RUNNING COMPREHENSIVE EVALUATION OF ALL MODEL COMBINATIONS")
        print("="*100)
        
        all_results = run_comprehensive_evaluation(args, device)
        
        if all_results is None:
            print("❌ Comprehensive evaluation failed")
            return
    
    # Visualization mode
    if args.mode in ["visualize", "all"]:
        print("\n=== Generating Visualizations ===")
        
        # Make sure models are loaded before visualization
        if main_model is None:
            main_model = load_main_model(main_model_path, device)
        
        if healer_model is None:
            healer_model = load_healer_model(healer_model_path, device)
        
        if not args.exclude_ttt and ttt_model is None and os.path.exists(ttt_model_path):
            ttt_model = load_ttt_model(ttt_model_path, main_model, device)
        
        if not args.exclude_blended and blended_model is None and os.path.exists(blended_model_path):
            blended_model = load_blended_model(blended_model_path, main_model, device)
        
        # Make sure visualization directory exists
        os.makedirs(args.visualize_dir, exist_ok=True)
            
        # Generate visualizations for standard model
        visualize_transformations(
            model_dir=args.model_dir,
            dataset_path=args.dataset,
            num_samples=args.num_samples,
            severity=args.severity,
            save_dir=f"{args.visualize_dir}/standard",
            include_blended=not args.exclude_blended,
            include_ttt=not args.exclude_ttt
        )
        
        # Generate visualizations for robust model if available
        if main_model_robust is not None:
            # We need to temporarily save the robust model to the standard location
            # Save original main model
            orig_state_dict = None
            if main_model is not None:
                orig_state_dict = deepcopy(main_model.state_dict())
                
            # Replace main model with robust model
            tmp_model_dir = Path(f"{args.model_dir}/bestmodel_main_tmp")
            tmp_model_dir.mkdir(exist_ok=True)
            shutil.copy(
                f"{args.model_dir}/bestmodel_main/best_model.pt",
                f"{args.model_dir}/bestmodel_main_tmp/best_model.pt"
            )
            
            # Save robust model to main model path
            torch.save({
                'epoch': 0,
                'model_state_dict': main_model_robust.state_dict(),
                'val_acc': 0.0,
            }, f"{args.model_dir}/bestmodel_main/best_model.pt")
            
            # Generate visualizations with robust model
            visualize_transformations(
                model_dir=args.model_dir,
                dataset_path=args.dataset,
                num_samples=args.num_samples,
                severity=args.severity,
                save_dir=f"{args.visualize_dir}/robust",
                include_blended=not args.exclude_blended,
                include_ttt=not args.exclude_ttt
            )
            
            # Restore original main model
            if orig_state_dict is not None:
                shutil.copy(
                    f"{args.model_dir}/bestmodel_main_tmp/best_model.pt",
                    f"{args.model_dir}/bestmodel_main/best_model.pt"
                )
                shutil.rmtree(tmp_model_dir)
    
    print("\nExperiment completed!")


# Helper functions for loading models
def load_main_model(model_path, device):
    """Load the main classification model from a checkpoint"""
    print(f"Loading main model from {model_path}")
    main_model = create_vit_model(
        img_size=64, patch_size=8, in_chans=3, num_classes=200,
        embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
    )
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create a new state dict with the correct keys
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith("vit_model."):
            new_key = key[len("vit_model."):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    main_model.load_state_dict(new_state_dict)
    main_model = main_model.to(device)
    main_model.eval()
    return main_model

def load_healer_model(model_path, device):
    """Load the healer model from a checkpoint"""
    print(f"Loading healer model from {model_path}")
    healer_model = TransformationHealer(
        img_size=64, patch_size=8, in_chans=3,
        embed_dim=384, depth=6, head_dim=64
    )
    checkpoint = torch.load(model_path, map_location=device)
    healer_model.load_state_dict(checkpoint['model_state_dict'])
    healer_model = healer_model.to(device)
    healer_model.eval()
    return healer_model

def load_ttt_model(model_path, main_model, device):
    """Load the TTT model from a checkpoint"""
    print(f"Loading TTT model from {model_path}")
    ttt_model = TestTimeTrainer(
        base_model=main_model,
        img_size=64,
        patch_size=8,
        embed_dim=384
    )
    checkpoint = torch.load(model_path, map_location=device)
    ttt_model.load_state_dict(checkpoint['model_state_dict'])
    ttt_model = ttt_model.to(device)
    ttt_model.eval()
    return ttt_model

def load_blended_model(model_path, main_model, device):
    """Load the BlendedTTT model from a checkpoint"""
    print(f"Loading BlendedTTT model from {model_path}")
    blended_model = BlendedTTT(
        img_size=64,
        patch_size=8,
        embed_dim=384,
        depth=8
    )
    checkpoint = torch.load(model_path, map_location=device)
    blended_model.load_state_dict(checkpoint['model_state_dict'])
    blended_model = blended_model.to(device)
    blended_model.eval()
    return blended_model

def compare_models_performance(main_results, robust_results, main_label="Main", robust_label="Robust"):
    """Compare performance between two model pipelines"""
    print(f"\n{'='*20} Performance Comparison {'='*20}")
    print(f"Comparing {main_label} model vs {robust_label} model:")
    
    # Compare clean data performance
    if 0.0 in main_results and 0.0 in robust_results:
        print("\nClean Data Performance (WITHOUT TRANSFORMS):")
        main_clean_acc = main_results[0.0]['main']['accuracy']
        robust_clean_acc = robust_results[0.0]['main']['accuracy']
        diff = robust_clean_acc - main_clean_acc
        print(f"  {main_label} Model: {main_clean_acc:.4f}")
        print(f"  {robust_label} Model: {robust_clean_acc:.4f}")
        print(f"  Difference: {diff:.4f} ({diff/main_clean_acc*100:.1f}%)")
    
    # Compare OOD performance
    for severity in [s for s in main_results.keys() if s > 0]:
        if severity in robust_results:
            print(f"\nTransformed Data Performance (WITH TRANSFORMS, Severity {severity}):")
            
            # Base model comparison
            main_ood_acc = main_results[severity]['main']['accuracy']
            robust_ood_acc = robust_results[severity]['main']['accuracy']
            diff = robust_ood_acc - main_ood_acc
            print(f"  {main_label} Model: {main_ood_acc:.4f}")
            print(f"  {robust_label} Model: {robust_ood_acc:.4f}")
            print(f"  Difference: {diff:.4f} ({diff/main_ood_acc*100:.1f}%)")
            
            # Healer performance comparison
            if (main_results[severity]['healer'] is not None and 
                robust_results[severity]['healer'] is not None):
                
                main_healer_acc = main_results[severity]['healer']['accuracy']
                robust_healer_acc = robust_results[severity]['healer']['accuracy']
                diff = robust_healer_acc - main_healer_acc
                print(f"\n  {main_label} + Healer: {main_healer_acc:.4f}")
                print(f"  {robust_label} + Healer: {robust_healer_acc:.4f}")
                print(f"  Difference: {diff:.4f} ({diff/main_healer_acc*100:.1f}%)")
            
            # Per-transformation comparison
            if ('per_transform_acc' in main_results[severity]['main'] and
                'per_transform_acc' in robust_results[severity]['main']):
                
                print("\n  Per-Transformation Accuracy:")
                for t_type in ['no_transform', 'gaussian_noise', 'rotation', 'affine']:
                    main_t_acc = main_results[severity]['main']['per_transform_acc'][t_type]
                    robust_t_acc = robust_results[severity]['main']['per_transform_acc'][t_type]
                    diff = robust_t_acc - main_t_acc
                    
                    print(f"    {t_type.upper()}:")
                    print(f"      {main_label}: {main_t_acc:.4f}")
                    print(f"      {robust_label}: {robust_t_acc:.4f}")
                    print(f"      Difference: {diff:.4f} ({diff/main_t_acc*100:.1f}% {'better' if diff > 0 else 'worse'})")
    
    print(f"{'='*60}")


def log_wandb_results_with_all_models(all_results):
    """
    Log comprehensive evaluation results to Weights & Biases
    including all model types (main, healer, blended, ttt)
    """
    import wandb
    
    # Log clean data results
    if 0.0 in all_results:
        clean_results = all_results[0.0]
        clean_acc = {
            "eval/clean_accuracy": clean_results['main']['accuracy']
        }
        if 'healer' in clean_results and clean_results['healer'] is not None:
            clean_acc["eval/clean_healer_accuracy"] = clean_results['healer']['accuracy']
        wandb.log(clean_acc)
    
    # Log OOD results
    for severity, results in all_results.items():
        if severity == 0.0:
            continue  # Skip clean results, already logged
        
        # Main metrics
        ood_metrics = {
            f"eval/ood_s{severity}_accuracy": results['main']['accuracy'],
        }
        
        if 'healer' in results and results['healer'] is not None:
            ood_metrics[f"eval/ood_s{severity}_healer_accuracy"] = results['healer']['accuracy']
        
        if 'blended' in results and results['blended'] is not None:
            ood_metrics[f"eval/ood_s{severity}_blended_accuracy"] = results['blended']['accuracy']
        
        if 'ttt' in results and results['ttt'] is not None:
            ood_metrics[f"eval/ood_s{severity}_ttt_accuracy"] = results['ttt']['accuracy']
        
        if 'ttt_adapted' in results and results['ttt_adapted'] is not None:
            ood_metrics[f"eval/ood_s{severity}_ttt_adapted_accuracy"] = results['ttt_adapted']['accuracy']
        
        # Per-transformation metrics
        if 'per_transform_acc' in results['main']:
            for t_type, acc in results['main']['per_transform_acc'].items():
                ood_metrics[f"eval/ood_s{severity}_{t_type}_accuracy"] = acc
                
                if 'healer' in results and results['healer'] is not None and 'per_transform_acc' in results['healer']:
                    ood_metrics[f"eval/ood_s{severity}_{t_type}_healer_accuracy"] = (
                        results['healer']['per_transform_acc'][t_type]
                    )
                
                if 'blended' in results and results['blended'] is not None and 'per_transform_acc' in results['blended']:
                    ood_metrics[f"eval/ood_s{severity}_{t_type}_blended_accuracy"] = (
                        results['blended']['per_transform_acc'][t_type]
                    )
                
                if 'ttt' in results and results['ttt'] is not None and 'per_transform_acc' in results['ttt']:
                    ood_metrics[f"eval/ood_s{severity}_{t_type}_ttt_accuracy"] = (
                        results['ttt']['per_transform_acc'][t_type]
                    )
                
                if 'ttt_adapted' in results and results['ttt_adapted'] is not None and 'per_transform_acc' in results['ttt_adapted']:
                    ood_metrics[f"eval/ood_s{severity}_{t_type}_ttt_adapted_accuracy"] = (
                        results['ttt_adapted']['per_transform_acc'][t_type]
                    )
        
        # Log all metrics together
        wandb.log(ood_metrics)


if __name__ == "__main__":
    main()

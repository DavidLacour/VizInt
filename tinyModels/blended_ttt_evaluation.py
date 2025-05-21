import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from new_new import ContinuousTransforms, TinyImageNetDataset

import os
import torch
import wandb
import numpy as np
import shutil
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from copy import deepcopy

# Import our custom ViT model
from transformer_utils import set_seed, LayerNorm, Mlp, TransformerTrunk
from vit_implementation import create_vit_model, PatchEmbed, VisionTransformer



def evaluate_blended_model(model, loader, device):
    """
    Evaluate the BlendedTTT model on a dataset, using only classification output
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            # In debug mode, only evaluate one batch
            if DEBUG and batch_idx > 0:
                break
                
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass - only use classification outputs
            logits, _ = model(images)
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Avoid division by zero
    if total == 0:
        return 0.0
        
    return correct / total


def evaluate_models_with_blended(main_model, healer_model, ttt_model, blended_model, dataset_path="tiny-imagenet-200", severity=0.0):
    """Extended version of evaluate_models that includes the BlendedTTT model"""
    # Get the base results from the existing function
    results = evaluate_models(main_model, healer_model, ttt_model, dataset_path, severity)
    
    # Add BlendedTTT evaluation
    if blended_model is not None:
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define image transformations
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # For severity 0, we don't apply transformations
        if severity > 0:
            # Dataset with transformations
            ood_transform = ContinuousTransforms(severity=severity)
            val_dataset = TinyImageNetDataset(
                dataset_path, "val", transform_val, ood_transform=ood_transform
            )
        else:
            # Dataset without transformations (clean data)
            val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
        
        # In debug mode, use small subset of data
        if DEBUG:
            # Create a small subset of the data
            val_indices = list(range(10))  # Just 10 validation samples
            
            from torch.utils.data import Subset
            val_dataset = Subset(val_dataset, val_indices)
        
        # Set batch size based on debug mode
        batch_size = 1 if DEBUG else 64
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=0 if DEBUG else 4, pin_memory=True)
        
        # Ensure model is in eval mode
        blended_model.eval()
        
        # Initialize blended model metrics
        blended_results = {'correct': 0, 'total': 0}
        
        # For transformed data, track per-transformation metrics
        transform_types = ['gaussian_noise', 'rotation', 'affine']
        if severity > 0:
            blended_results['per_transform'] = {t: {'correct': 0, 'total': 0} for t in transform_types}
        
        # Helper function to determine transform type
        def determine_transform_type(params):
            """Determine transform type by examining parameter values"""
            try:
                # Check for each transform parameter
                if 'transform_type' in params:
                    t_type = params['transform_type']
                    if isinstance(t_type, str):
                        if 'noise' in t_type.lower() or 'gaussian' in t_type.lower():
                            return 'gaussian_noise'
                        elif 'rot' in t_type.lower():
                            return 'rotation'
                        elif 'affine' in t_type.lower():
                            return 'affine'
                
                # If not found by name, check by parameter values
                if params.get('noise_std', 0.0) > 0.01:
                    return 'gaussian_noise'
                elif abs(params.get('rotation_angle', 0.0)) > 0.01:
                    return 'rotation'
                elif (abs(params.get('translate_x', 0.0)) > 0.001 or 
                      abs(params.get('translate_y', 0.0)) > 0.001 or
                      abs(params.get('shear_x', 0.0)) > 0.001 or
                      abs(params.get('shear_y', 0.0)) > 0.001):
                    return 'affine'
                
                # Default case
                return 'gaussian_noise'
                
            except Exception as e:
                print(f"Error determining transform type: {e}")
                return 'gaussian_noise'  # Default
        
        with torch.no_grad():
            if severity > 0:
                # Evaluation with transformations
                for batch_idx, (orig_images, transformed_images, labels, transform_params) in enumerate(val_loader):
                    # In debug mode, limit to a small number of batches
                    if DEBUG and batch_idx > 3:
                        break
                        
                    transformed_images = transformed_images.to(device)
                    labels = labels.to(device)
                    
                    # Only use classification output
                    logits, _ = blended_model(transformed_images)
                    blended_preds = torch.argmax(logits, dim=1)
                    
                    blended_results['correct'] += (blended_preds == labels).sum().item()
                    blended_results['total'] += labels.size(0)
                    
                    # Track per-transformation accuracy
                    for i, params in enumerate(transform_params):
                        # Use the helper function to determine transform type
                        t_type = determine_transform_type(params)
                        
                        blended_results['per_transform'][t_type]['total'] += 1
                        if blended_preds[i] == labels[i]:
                            blended_results['per_transform'][t_type]['correct'] += 1
            else:
                # Clean data evaluation
                for batch_idx, (images, labels) in enumerate(val_loader):
                    # In debug mode, limit to a small number of batches
                    if DEBUG and batch_idx > 3:
                        break
                        
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Only use classification output
                    logits, _ = blended_model(images)
                    blended_preds = torch.argmax(logits, dim=1)
                    
                    blended_results['correct'] += (blended_preds == labels).sum().item()
                    blended_results['total'] += labels.size(0)
        
        # Calculate overall accuracy
        if blended_results['total'] > 0:
            blended_results['accuracy'] = blended_results['correct'] / blended_results['total']
            
            # Calculate per-transformation accuracies
            if severity > 0 and 'per_transform' in blended_results:
                blended_results['per_transform_acc'] = {}
                for t_type in transform_types:
                    t_total = blended_results['per_transform'][t_type]['total']
                    if t_total > 0:
                        blended_results['per_transform_acc'][t_type] = (
                            blended_results['per_transform'][t_type]['correct'] / t_total
                        )
                    else:
                        blended_results['per_transform_acc'][t_type] = 0.0
        
        # Add blended results to overall results
        results['blended'] = blended_results
    else:
        results['blended'] = None
    
    return results


def evaluate_full_pipeline_with_blended(main_model, healer_model, ttt_model, blended_model, dataset_path="tiny-imagenet-200", severities=[0.3]):
    """
    Comprehensive evaluation across multiple severities including the BlendedTTT model.
    """
    all_results = {}
    
    # In debug mode, use a minimal set of severities if not explicitly provided
    if DEBUG and len(severities) > 1 and not any(s == 0.0 for s in severities):
        print("DEBUG MODE: Testing with minimal severities (0.0 and first provided)")
        severities = [0.0, severities[0]]
    
    # First evaluate on clean data
    print("\nEvaluating on clean data (no transformations)...")
    clean_results = evaluate_models_with_blended(main_model, healer_model, ttt_model, blended_model, dataset_path, severity=0.0)
    all_results[0.0] = clean_results
    
    # Print clean data results
    print(f"Clean Data Accuracy:")
    print(f"  Main Model: {clean_results['main']['accuracy']:.4f}")
    if clean_results['ttt'] is not None:
        print(f"  TTT Model: {clean_results['ttt']['accuracy']:.4f}")
    if clean_results['blended'] is not None:
        print(f"  BlendedTTT Model: {clean_results['blended']['accuracy']:.4f}")
    
    # Then evaluate on transformed data at different severities
    for severity in severities:
        if severity == 0.0:
            continue  # Skip, already evaluated
            
        print(f"\nEvaluating with severity {severity}...")
        ood_results = evaluate_models_with_blended(main_model, healer_model, ttt_model, blended_model, dataset_path, severity=severity)
        all_results[severity] = ood_results
        
        # Print OOD results
        print(f"OOD Accuracy (Severity {severity}):")
        print(f"  Main Model: {ood_results['main']['accuracy']:.4f}")
        if ood_results['healer'] is not None:
            print(f"  Healer Model: {ood_results['healer']['accuracy']:.4f}")
        if ood_results['ttt'] is not None:
            print(f"  TTT Model: {ood_results['ttt']['accuracy']:.4f}")
        if ood_results['blended'] is not None:
            print(f"  BlendedTTT Model: {ood_results['blended']['accuracy']:.4f}")
            
        # Calculate and print robustness metrics (drop compared to clean data)
        if severity > 0.0:
            main_drop = clean_results['main']['accuracy'] - ood_results['main']['accuracy']
            print(f"\nAccuracy Drop from Clean Data:")
            print(f"  Main Model: {main_drop:.4f} ({main_drop/clean_results['main']['accuracy']*100:.1f}%)")
            
            if ood_results['healer'] is not None:
                healer_drop = clean_results['main']['accuracy'] - ood_results['healer']['accuracy']
                print(f"  Healer Model: {healer_drop:.4f} ({healer_drop/clean_results['main']['accuracy']*100:.1f}%)")
                
            if ood_results['ttt'] is not None and clean_results['ttt'] is not None:
                ttt_drop = clean_results['ttt']['accuracy'] - ood_results['ttt']['accuracy']
                print(f"  TTT Model: {ttt_drop:.4f} ({ttt_drop/clean_results['ttt']['accuracy']*100:.1f}%)")
                
            if ood_results['blended'] is not None and clean_results['blended'] is not None:
                blended_drop = clean_results['blended']['accuracy'] - ood_results['blended']['accuracy']
                print(f"  BlendedTTT Model: {blended_drop:.4f} ({blended_drop/clean_results['blended']['accuracy']*100:.1f}%)")
    
    # Log comprehensive results to wandb
    log_results_to_wandb_with_blended(all_results)
    
    return all_results


def log_results_to_wandb_with_blended(all_results):
    """Log comprehensive results to wandb with detailed tables and charts."""
    # Overall accuracy across severities
    severity_data = []
    transform_data = {t: [] for t in ['gaussian_noise', 'rotation', 'affine']}
    
    for severity, results in all_results.items():
        # Skip missing or malformed results
        if not isinstance(results, dict) or 'main' not in results:
            continue
            
        # Overall accuracy row
        row = [severity, results['main']['accuracy']]
        
        if severity > 0.0:
            if results['healer'] is not None:
                row.append(results['healer']['accuracy'])
            else:
                row.append(None)
        else:
            row.append(None)  # Healer not applicable to clean data
            
        if results['ttt'] is not None:
            row.append(results['ttt']['accuracy'])
        else:
            row.append(None)
            
        if results['blended'] is not None:
            row.append(results['blended']['accuracy'])
        else:
            row.append(None)
            
        # Add robustness metrics (only for OOD data)
        if severity > 0.0 and 0.0 in all_results:
            clean_acc = all_results[0.0]['main']['accuracy']
            main_drop = clean_acc - results['main']['accuracy'] 
            row.append(main_drop)
            
            if results['healer'] is not None:
                healer_drop = clean_acc - results['healer']['accuracy']
                row.append(healer_drop)
            else:
                row.append(None)
                
            if results['ttt'] is not None and all_results[0.0]['ttt'] is not None:
                ttt_drop = all_results[0.0]['ttt']['accuracy'] - results['ttt']['accuracy']
                row.append(ttt_drop)
            else:
                row.append(None)
                
            if results['blended'] is not None and all_results[0.0]['blended'] is not None:
                blended_drop = all_results[0.0]['blended']['accuracy'] - results['blended']['accuracy']
                row.append(blended_drop)
            else:
                row.append(None)
        else:
            # For clean data, no drop
            row.extend([0.0, None, 0.0, 0.0])
            
        severity_data.append(row)
        
        # Per-transformation data
        if severity > 0.0 and 'per_transform_acc' in results['main']:
            for t_type in transform_data.keys():
                if t_type in results['main']['per_transform_acc']:
                    t_row = [
                        severity,
                        t_type,
                        results['main']['per_transform_acc'][t_type]
                    ]
                    
                    if results['healer'] is not None and 'per_transform_acc' in results['healer']:
                        t_row.append(results['healer']['per_transform_acc'][t_type])
                    else:
                        t_row.append(None)
                        
                    if results['ttt'] is not None and 'per_transform_acc' in results['ttt']:
                        t_row.append(results['ttt']['per_transform_acc'][t_type])
                    else:
                        t_row.append(None)
                        
                    if results['blended'] is not None and 'per_transform_acc' in results['blended']:
                        t_row.append(results['blended']['per_transform_acc'][t_type])
                    else:
                        t_row.append(None)
                        
                    transform_data[t_type].append(t_row)
    
    # Log overall accuracy table
    columns = [
        "Severity", "Main Acc", "Healer Acc", "TTT Acc", "Blended Acc",
        "Main Drop", "Healer Drop", "TTT Drop", "Blended Drop"
    ]
    wandb.log({"overall_accuracy": wandb.Table(data=severity_data, columns=columns)})
    
    # Log per-transformation tables
    for t_type, rows in transform_data.items():
        if rows:  # Only log if we have data
            t_columns = ["Severity", "Transform", "Main Acc", "Healer Acc", "TTT Acc", "Blended Acc"]
            wandb.log({f"transform_{t_type}": wandb.Table(data=rows, columns=t_columns)})
    
    # Create line chart for accuracy vs severity
    accs_by_severity = {
        'severity': [],
        'main': [],
        'healer': [],
        'ttt': [],
        'blended': []
    }
    
    for severity, results in sorted(all_results.items()):
        accs_by_severity['severity'].append(severity)
        accs_by_severity['main'].append(results['main']['accuracy'])
        
        if results['healer'] is not None and 'accuracy' in results['healer']:
            accs_by_severity['healer'].append(results['healer']['accuracy'])
        elif severity > 0.0:
            accs_by_severity['healer'].append(0)  # Placeholder for OOD data
        else:
            accs_by_severity['healer'].append(None)  # Placeholder for clean data
            
        if results['ttt'] is not None:
            accs_by_severity['ttt'].append(results['ttt']['accuracy'])
        else:
            accs_by_severity['ttt'].append(0)  # Placeholder
            
        if results['blended'] is not None:
            accs_by_severity['blended'].append(results['blended']['accuracy'])
        else:
            accs_by_severity['blended'].append(0)  # Placeholder
    
    # Log chart data
    for i, severity in enumerate(accs_by_severity['severity']):
        wandb.log({
            "chart/severity": severity,
            "chart/main_acc": accs_by_severity['main'][i],
            "chart/healer_acc": accs_by_severity['healer'][i] if i < len(accs_by_severity['healer']) else None,
            "chart/ttt_acc": accs_by_severity['ttt'][i],
            "chart/blended_acc": accs_by_severity['blended'][i]
        })
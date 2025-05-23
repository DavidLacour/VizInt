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


MAX_ROTATION = 360.0 
MAX_STD_GAUSSIAN_NOISE = 0.5
MAX_TRANSLATION_AFFINE = 0.1
MAX_SHEAR_ANGLE = 15.0
DEBUG = False 


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

def evaluate_full_pipeline(main_model, healer_model, dataset_path, severities, include_blended=True, include_ttt=True):
    """
    Evaluate the full transformation healing pipeline on clean and transformed data.
    
    Args:
        main_model: The classification model
        healer_model: The transformation healer model
        dataset_path: Path to the dataset
        severities: List of severity levels to evaluate
        include_blended: Whether to include BlendedTTT in evaluation
        include_ttt: Whether to include TTT in evaluation
        
    Returns:
        all_results: Dictionary of evaluation results
    """
    all_results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # First evaluate on clean data (severity 0.0)
    print("\nEvaluating on clean data (no transformations)...")
    
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
        blended_model_path = f"{args.model_dir}/bestmodel_blended/best_model.pt"
        if os.path.exists(blended_model_path):
            blended_model = load_blended_model(blended_model_path, main_model, device)
            
    if include_ttt:
        ttt_model_path = f"{args.model_dir}/bestmodel_ttt/best_model.pt"
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
    print(f"Clean Data Accuracy:")
    print(f"  Main Model: {main_accuracy:.4f}")
    print(f"  Healer Model: {healer_accuracy:.4f}")
    
    if include_blended and blended_model is not None:
        print(f"  BlendedTTT Model: {blended_accuracy:.4f}")
    
    if include_ttt and ttt_model is not None:
        print(f"  TTT Model: {ttt_accuracy:.4f}")
        print(f"  TTT Model (adapted): {ttt_adapted_accuracy:.4f}")
    
    # Now evaluate on transformed data with different severity levels
    for severity in severities:
        print(f"\nEvaluating with severity {severity}...")
        
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
        print(f"OOD Accuracy (Severity {severity}):")
        print(f"  Main Model: {main_accuracy:.4f}")
        print(f"  Healer Model: {healer_accuracy:.4f}")
        
        if include_blended and blended_model is not None:
            print(f"  BlendedTTT Model: {blended_accuracy:.4f}")
        
        if include_ttt and ttt_model is not None:
            print(f"  TTT Model: {ttt_accuracy:.4f}")
            print(f"  TTT Model (adapted): {ttt_adapted_accuracy:.4f}")
        
        # Calculate robustness metrics
        main_drop = clean_results['main']['accuracy'] - main_accuracy
        healer_drop = clean_results['main']['accuracy'] - healer_accuracy
        
        print(f"\nAccuracy Drop from Clean Data:")
        print(f"  Main Model: {main_drop:.4f} ({main_drop/clean_results['main']['accuracy']*100:.1f}%)")
        print(f"  Healer Model: {healer_drop:.4f} ({healer_drop/clean_results['main']['accuracy']*100:.1f}%)")
        
        if include_blended and blended_model is not None:
            blended_drop = clean_results['main']['accuracy'] - blended_accuracy
            print(f"  BlendedTTT Model: {blended_drop:.4f} ({blended_drop/clean_results['main']['accuracy']*100:.1f}%)")
        
        if include_ttt and ttt_model is not None:
            ttt_drop = clean_results['main']['accuracy'] - ttt_accuracy
            ttt_adapted_drop = clean_results['main']['accuracy'] - ttt_adapted_accuracy
            print(f"  TTT Model: {ttt_drop:.4f} ({ttt_drop/clean_results['main']['accuracy']*100:.1f}%)")
            print(f"  TTT Model (adapted): {ttt_adapted_drop:.4f} ({ttt_adapted_drop/clean_results['main']['accuracy']*100:.1f}%)")
        
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
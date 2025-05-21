import os
import torch
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


def visualize_transformations(model_dir, dataset_path, num_samples=5, severity=0.5, save_dir="visualizations", include_blended=True, include_ttt=True):
    """
    Visualize original, transformed, and corrected images to illustrate the healing process.
    
    Args:
        model_dir: Directory containing the trained models
        dataset_path: Path to the dataset
        num_samples: Number of samples to visualize
        severity: Severity of transformations to apply
        save_dir: Directory to save visualizations
        include_blended: Whether to include BlendedTTT in visualizations
        include_ttt: Whether to include TTT in visualizations
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    # Main classification model
    main_model = create_vit_model(
        img_size=64, patch_size=8, in_chans=3, num_classes=200,
        embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
    )
    main_checkpoint = torch.load(f"{model_dir}/bestmodel_main/best_model.pt", map_location=device)
    
    # Create a new state dict with the correct keys for the main model
    new_state_dict = {}
    for key, value in main_checkpoint['model_state_dict'].items():
        if key.startswith("vit_model."):
            new_key = key[len("vit_model."):]
            new_state_dict[new_key] = value
    
    main_model.load_state_dict(new_state_dict)
    main_model = main_model.to(device)
    main_model.eval()
    
    # Healer model
    healer_model = TransformationHealer(
        img_size=64, patch_size=8, in_chans=3,
        embed_dim=384, depth=6, head_dim=64
    )
    healer_checkpoint = torch.load(f"{model_dir}/bestmodel_healer/best_model.pt", map_location=device)
    healer_model.load_state_dict(healer_checkpoint['model_state_dict'])
    healer_model = healer_model.to(device)
    healer_model.eval()
    
    # BlendedTTT model (if requested)
    blended_model = None
    if include_blended:
        blended_model = BlendedTTT(
            base_model=main_model,
            img_size=64, 
            patch_size=8,
            embed_dim=384, 
            depth=8
        )
        blended_checkpoint = torch.load(f"{model_dir}/bestmodel_blended/best_model.pt", map_location=device)
        blended_model.load_state_dict(blended_checkpoint['model_state_dict'])
        blended_model = blended_model.to(device)
        blended_model.eval()
    
    # TTT model (if requested)
    ttt_model = None
    if include_ttt:
        ttt_model = TestTimeTrainer(
            base_model=main_model,
            img_size=64,
            patch_size=8,
            embed_dim=384
        )
        ttt_checkpoint = torch.load(f"{model_dir}/bestmodel_ttt/best_model.pt", map_location=device)
        ttt_model.load_state_dict(ttt_checkpoint['model_state_dict'])
        ttt_model = ttt_model.to(device)
        ttt_model.eval()
    
    # Setup transformations
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Initialize continuous transforms
    ood_transform = ContinuousTransforms(severity=severity)
    
    # Load dataset - use validation set for visualization
    val_dataset = TinyImageNetDataset(
        dataset_path, "val", transform_val, ood_transform=ood_transform
    )
    
    # For inverse normalization to display images
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    # Generate visualizations
    transform_names = ['no_transform', 'gaussian_noise', 'rotation', 'affine']
    
    with torch.no_grad():
        # Visualize each transformation type
        for t_idx, t_type in enumerate(transform_names):
            if t_type == 'no_transform':
                continue  # Skip no_transform for visualization
                
            # Determine number of columns based on which models to include
            num_cols = 3  # Original, Transformed, Healer
            if include_blended:
                num_cols += 1
            if include_ttt:
                num_cols += 1
                
            plt.figure(figsize=(5 * num_cols, 5 * num_samples))
            plt.suptitle(f"{t_type.upper()} Transformation (Severity: {severity})", fontsize=16)
            
            for i in range(num_samples):
                # Get a random sample
                idx = np.random.randint(0, len(val_dataset))
                orig_img, trans_img, label, params = val_dataset[idx]
                
                # Convert to tensors and move to device
                orig_img = orig_img.unsqueeze(0).to(device)
                
                # Apply specific transformation
                trans_img, params = ood_transform.apply_transforms(
                    orig_img.squeeze(0).cpu(), 
                    transform_type=t_type, 
                    severity=severity,
                    return_params=True
                )
                trans_img = trans_img.unsqueeze(0).to(device)
                
                # Predict transformations with healer
                healer_predictions = healer_model(trans_img)
                
                # Get prediction details
                pred_type_idx = torch.argmax(healer_predictions['transform_type_logits'], dim=1).item()
                pred_type = transform_names[pred_type_idx]
                
                # Apply correction with healer model
                healer_corrected_img = healer_model.apply_correction(trans_img, healer_predictions)
                
                # Variables for optional models
                blended_pred_type = None
                ttt_pred_type = None
                
                # Apply correction with BlendedTTT if available
                if include_blended and blended_model is not None:
                    _, blended_predictions = blended_model(trans_img)
                    blended_pred_type_idx = torch.argmax(blended_predictions['transform_type_logits'], dim=1).item()
                    blended_pred_type = transform_names[blended_pred_type_idx]
                    
                    # Get model predictions for classification
                    blended_pred = torch.argmax(blended_model(trans_img)[0], dim=1).item()
                
                # Get TTT model predictions if available
                if include_ttt and ttt_model is not None:
                    # For TTT visualization, we'll just extract the predicted transform type
                    _, transform_logits = ttt_model(trans_img)
                    ttt_pred_type_idx = torch.argmax(transform_logits, dim=1).item()
                    ttt_pred_type = transform_names[ttt_pred_type_idx]
                    
                    # For classification prediction
                    ttt_pred = torch.argmax(ttt_model.adapt(trans_img, reset=True), dim=1).item()
                
                # Get model predictions
                orig_pred = torch.argmax(main_model(orig_img), dim=1).item()
                trans_pred = torch.argmax(main_model(trans_img), dim=1).item()
                healer_corrected_pred = torch.argmax(main_model(healer_corrected_img), dim=1).item()
                
                # Prepare images for display by denormalizing
                orig_disp = inv_normalize(orig_img.squeeze(0).cpu())
                trans_disp = inv_normalize(trans_img.squeeze(0).cpu())
                healer_corrected_disp = inv_normalize(healer_corrected_img.squeeze(0).cpu())
                
                # Convert to PIL for display
                to_pil = transforms.ToPILImage()
                orig_pil = to_pil(torch.clamp(orig_disp, 0, 1))
                trans_pil = to_pil(torch.clamp(trans_disp, 0, 1))
                healer_corrected_pil = to_pil(torch.clamp(healer_corrected_disp, 0, 1))
                
                # Base column index
                start_col = 1
                
                # Original (always first column)
                plt.subplot(num_samples, num_cols, i*num_cols + start_col)
                plt.imshow(orig_pil)
                plt.title(f"Original\nPrediction: {orig_pred}")
                plt.axis('off')
                start_col += 1
                
                # Transformed (always second column)
                plt.subplot(num_samples, num_cols, i*num_cols + start_col)
                plt.imshow(trans_pil)
                plt.title(f"Transformed ({t_type})\nPrediction: {trans_pred}")
                plt.axis('off')
                start_col += 1
                
                # Healer corrected (always third column)
                plt.subplot(num_samples, num_cols, i*num_cols + start_col)
                plt.imshow(healer_corrected_pil)
                plt.title(f"Healer Corrected (Detected: {pred_type})\nPrediction: {healer_corrected_pred}")
                plt.axis('off')
                start_col += 1
                
                # BlendedTTT (optional fourth column)
                if include_blended:
                    plt.subplot(num_samples, num_cols, i*num_cols + start_col)
                    plt.imshow(trans_pil)  # For BlendedTTT we just show the classification result
                    plt.title(f"BlendedTTT (Detected: {blended_pred_type})\nPrediction: {blended_pred}")
                    plt.axis('off')
                    start_col += 1
                
                # TTT model (optional fifth column)
                if include_ttt:
                    plt.subplot(num_samples, num_cols, i*num_cols + start_col)
                    plt.imshow(trans_pil)  # For TTT we just show the classification result
                    plt.title(f"TTT (Detected: {ttt_pred_type})\nPrediction: {ttt_pred}")
                    plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{t_type}_visualization.png")
            plt.close()
            
            print(f"Saved visualization for {t_type} transformation")


def main():
    parser = argparse.ArgumentParser(description="Transform Healing with Vision Transformers")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "visualize", "all"],
                      help="Mode of operation: train, evaluate, visualize, or all")
    parser.add_argument("--dataset", type=str, default="tiny-imagenet-200",
                      help="Path to the dataset")
    parser.add_argument("--model_dir", type=str, default="./",
                      help="Directory to save/load models")
    parser.add_argument("--visualize_dir", type=str, default="visualizations",
                      help="Directory to save visualizations")
    parser.add_argument("--severity", type=float, default=0.5,
                      help="Severity of transformations for evaluation/visualization")
    parser.add_argument("--num_samples", type=int, default=5,
                      help="Number of samples to visualize")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    # Replace include with exclude options
    parser.add_argument("--exclude_blended", action="store_true",
                      help="Whether to exclude BlendedTTT model from the pipeline")
    parser.add_argument("--exclude_ttt", action="store_true",
                      help="Whether to exclude TTT model from the pipeline")
    parser.add_argument("--skip_ttt", action="store_true",
                      help="Skip training TTT model (but still use it for evaluation if available)")
    args = parser.parse_args()
    
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
    
    # Check if models exist before training
    main_model_path = f"{args.model_dir}/bestmodel_main/best_model.pt"
    robust_model_path = f"{args.model_dir}/bestmodel_robust/best_model.pt"
    healer_model_path = f"{args.model_dir}/bestmodel_healer/best_model.pt"
    # Changed from if args.include_ttt to if not args.exclude_ttt
    ttt_model_path = f"{args.model_dir}/bestmodel_ttt/best_model.pt" if not args.exclude_ttt else None
    # Changed from if args.include_blended to if not args.exclude_blended
    blended_model_path = f"{args.model_dir}/bestmodel_blended/best_model.pt" if not args.exclude_blended else None
    
    # Training mode
    if args.mode in ["train", "all"]:
        # Check if main model exists
        if not os.path.exists(main_model_path):
            print("\n=== Training Main Classification Model ===")
            main_model = train_main_model(args.dataset)
        else:
            print(f"\n=== Main Classification Model found at {main_model_path}, skipping training ===")
            main_model = load_main_model(main_model_path, device)
        
        # Check if robust main model exists
        if not os.path.exists(robust_model_path):
            print("\n=== Training Robust Main Classification Model ===")
            main_model_robust = train_main_model_robust(args.dataset, severity=args.severity)
        else:
            print(f"\n=== Robust Main Classification Model found at {robust_model_path}, skipping training ===")
            main_model_robust = load_main_model(robust_model_path, device)
        
        # Check if healer model exists
        if not os.path.exists(healer_model_path):
            print("\n=== Training Transformation Healer Model ===")
            healer_model = train_healer_model(args.dataset, severity=args.severity)
        else:
            print(f"\n=== Healer Model found at {healer_model_path}, skipping training ===")
            healer_model = load_healer_model(healer_model_path, device)
        
        # Train TTT model if not skipped, not excluded, and not already existing
        if not args.exclude_ttt and not args.skip_ttt:
            if not os.path.exists(ttt_model_path):
                print("\n=== Training Test-Time Training Model ===")
                if main_model is None:
                    main_model = load_main_model(main_model_path, device)
                ttt_model = train_ttt_model(args.dataset, base_model=main_model, severity=args.severity)
            else:
                print(f"\n=== TTT Model found at {ttt_model_path}, skipping training ===")
                if main_model is None:
                    main_model = load_main_model(main_model_path, device)
                ttt_model = load_ttt_model(ttt_model_path, main_model, device)
        
        # Train BlendedTTT model if not excluded and not already existing
        if not args.exclude_blended:
            if not os.path.exists(blended_model_path):
                print("\n=== Training BlendedTTT Model ===")
                if main_model is None:
                    main_model = load_main_model(main_model_path, device)
                blended_model = train_blended_ttt_model(main_model, args.dataset)
            else:
                print(f"\n=== BlendedTTT Model found at {blended_model_path}, skipping training ===")
                if main_model is None:
                    main_model = load_main_model(main_model_path, device)
                blended_model = load_blended_model(blended_model_path, main_model, device)
    
    # Evaluation mode
    if args.mode in ["evaluate", "all"]:
        print("\n=== Comprehensive Evaluation ===")
        # Evaluate at multiple severity levels
        severities = [0.2, 0.5, 0.8]
        
        # Make sure models are loaded before evaluation
        if main_model is None:
            main_model = load_main_model(main_model_path, device)
        
        if main_model_robust is None and os.path.exists(robust_model_path):
            main_model_robust = load_main_model(robust_model_path, device)
        
        if healer_model is None:
            healer_model = load_healer_model(healer_model_path, device)
        
        if not args.exclude_ttt and ttt_model is None and os.path.exists(ttt_model_path):
            ttt_model = load_ttt_model(ttt_model_path, main_model, device)
        
        if not args.exclude_blended and blended_model is None and os.path.exists(blended_model_path):
            blended_model = load_blended_model(blended_model_path, main_model, device)
        
        # Evaluate standard main model
        print("\n--- Evaluating Standard Main Model Pipeline ---")
        main_results = evaluate_full_pipeline(
            main_model, healer_model, 
            args.dataset, severities
        )
        
        # Evaluate robust main model if available
        if main_model_robust is not None:
            print("\n--- Evaluating Robust Main Model Pipeline ---")
            robust_results = evaluate_full_pipeline(
                main_model_robust, healer_model, 
                args.dataset, severities
            )
            
            print("\n--- Comparing Standard vs Robust Model Performance ---")
            compare_models_performance(main_results, robust_results, "Standard", "Robust")
        
        # Include TTT evaluation if not excluded
        if not args.exclude_ttt and ttt_model is not None:
            print("\n--- Evaluating TTT Pipeline ---")
            # Replace the call to non-existent function with evaluate_full_pipeline_with_blended_and_ttt
            ttt_results = evaluate_full_pipeline_with_blended_and_ttt(
                main_model, healer_model, ttt_model, None,  # Pass None for blended_model
                args.dataset, severities
            )
        
        # Include BlendedTTT evaluation if not excluded
        if not args.exclude_blended and blended_model is not None:
            print("\n--- Evaluating BlendedTTT Pipeline ---")
            blended_results = evaluate_full_pipeline_with_blended_and_ttt(
                main_model, healer_model, ttt_model, blended_model, 
                args.dataset, severities
            )
    
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
        base_model=main_model,
        img_size=64,
        patch_size=8,
        embed_dim=384,
        depth=4
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
        print("\nClean Data Performance:")
        main_clean_acc = main_results[0.0]['main']['accuracy']
        robust_clean_acc = robust_results[0.0]['main']['accuracy']
        diff = robust_clean_acc - main_clean_acc
        print(f"  {main_label} Model: {main_clean_acc:.4f}")
        print(f"  {robust_label} Model: {robust_clean_acc:.4f}")
        print(f"  Difference: {diff:.4f} ({diff/main_clean_acc*100:.1f}%)")
    
    # Compare OOD performance
    for severity in [s for s in main_results.keys() if s > 0]:
        if severity in robust_results:
            print(f"\nOOD Performance (Severity {severity}):")
            
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

def evaluate_main_model_only(main_model, dataset_path, severities):
    """
    Evaluate only the main model on clean and transformed data.
    
    Args:
        main_model: The classification model
        dataset_path: Path to the dataset
        severities: List of severity levels to evaluate
    
    Returns:
        all_results: Dictionary of evaluation results
    """
    all_results = {}
    
    # Evaluate on clean data (no transformations)
    print("\nEvaluating main model on clean data...")
    
    # Use device with PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define image transformations for validation
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get clean data validation set
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
    
    # Evaluate on clean data
    main_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating on clean data"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = main_model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Update metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate accuracy
    clean_accuracy = correct / total
    clean_results = {
        'main': {
            'accuracy': clean_accuracy,
            'correct': correct,
            'total': total
        },
        'healer': None,  # No healer model
        'blended': None,  # No blended model
        'ttt': None       # No TTT model
    }
    
    all_results[0.0] = clean_results
    
    print(f"Clean Data Accuracy: {clean_accuracy:.4f}")
    
    # Evaluate on OOD data
    for severity in severities:
        print(f"\nEvaluating main model with severity {severity}...")
        
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
        correct = 0
        total = 0
        
        # Per-transformation metrics
        transform_types = ['no_transform', 'gaussian_noise', 'rotation', 'affine']
        per_transform_metrics = {t: {'correct': 0, 'total': 0} for t in transform_types}
        
        # Helper function to determine transform type consistently
        def get_transform_type(params):
            if isinstance(params, dict) and 'transform_type' in params:
                return params['transform_type']
            return 'no_transform'
        
        # Evaluate on OOD data
        with torch.no_grad():
            for orig_images, trans_images, labels, params in tqdm(ood_val_loader, desc=f"Evaluating (severity {severity})"):
                orig_images = orig_images.to(device)
                trans_images = trans_images.to(device)
                labels = labels.to(device)
                
                # Forward pass on transformed images
                outputs = main_model(trans_images)
                _, predicted = torch.max(outputs, 1)
                
                # Update metrics
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update per-transformation metrics
                for i, p in enumerate(params):
                    t_type = get_transform_type(p)
                    per_transform_metrics[t_type]['total'] += 1
                    if predicted[i] == labels[i]:
                        per_transform_metrics[t_type]['correct'] += 1
        
        # Calculate accuracies
        ood_accuracy = correct / total
        
        # Calculate per-transform accuracies
        per_transform_acc = {}
        for t_type in transform_types:
            if per_transform_metrics[t_type]['total'] > 0:
                per_transform_acc[t_type] = (
                    per_transform_metrics[t_type]['correct'] / per_transform_metrics[t_type]['total']
                )
            else:
                per_transform_acc[t_type] = 0.0
        
        # Create results
        ood_results = {
            'main': {
                'accuracy': ood_accuracy,
                'correct': correct,
                'total': total,
                'per_transform_acc': per_transform_acc
            },
            'healer': None,  # No healer model
            'blended': None,  # No blended model
            'ttt': None       # No TTT model
        }
        
        all_results[severity] = ood_results
        
        print(f"OOD Accuracy (Severity {severity}): {ood_accuracy:.4f}")
        
        # Calculate robustness metrics
        main_drop = clean_accuracy - ood_accuracy
        print(f"Accuracy Drop from Clean Data: {main_drop:.4f} ({main_drop/clean_accuracy*100:.1f}%)")
        
        # Print per-transformation accuracies
        print("\nPer-Transformation Accuracy:")
        for t_type in transform_types:
            print(f"  {t_type.upper()}: {per_transform_acc[t_type]:.4f}")
    
    # Log results to wandb if available
    try:
        import wandb
        log_wandb_results_main_only(all_results)
    except:
        print("Note: wandb not available or error in logging.")
    
    return all_results

def evaluate_full_pipeline_with_blended_only(main_model, blended_model, dataset_path, severities):
    """
    Evaluate the main model and BlendedTTT model.
    
    Args:
        main_model: The classification model
        blended_model: The BlendedTTT model
        dataset_path: Path to the dataset
        severities: List of severity levels to evaluate
    
    Returns:
        all_results: Dictionary of evaluation results
    """
    all_results = {}
    
    # First evaluate on clean data (severity 0.0)
    print("\nEvaluating on clean data (no transformations)...")
    
    # For clean data, we just use the main model
    clean_results = evaluate_main_model_only(main_model, dataset_path, [])
    all_results[0.0] = clean_results[0.0]
    
    # Print clean data results
    print(f"Clean Data Accuracy:")
    print(f"  Main Model: {clean_results[0.0]['main']['accuracy']:.4f}")
    
    # Then evaluate on transformed data with different models
    for severity in severities:
        print(f"\nEvaluating with severity {severity}...")
        
        # Basic main model evaluation
        main_only_results = evaluate_main_model_only(main_model, dataset_path, [severity])
        
        # BlendedTTT evaluation
        print(f"Evaluating BlendedTTT model...")
        blended_results = evaluate_models_with_blended(main_model, blended_model, dataset_path, severity=severity)
        
        # Create combined results
        ood_results = {
            'main': main_only_results[severity]['main'],
            'healer': None,  # No healer model
            'blended': {
                'accuracy': blended_results['blended_accuracy'],
                'per_transform_acc': blended_results['per_transform_acc']['blended']
            },
            'ttt': None,
            'ttt_adapted': None
        }
        
        all_results[severity] = ood_results
        
        # Print results
        print(f"OOD Accuracy (Severity {severity}):")
        print(f"  Main Model: {ood_results['main']['accuracy']:.4f}")
        print(f"  BlendedTTT Model: {ood_results['blended']['accuracy']:.4f}")
        
        # Calculate robustness metrics
        main_drop = clean_results[0.0]['main']['accuracy'] - ood_results['main']['accuracy']
        blended_drop = clean_results[0.0]['main']['accuracy'] - ood_results['blended']['accuracy']
        
        print(f"\nAccuracy Drop from Clean Data:")
        print(f"  Main Model: {main_drop:.4f} ({main_drop/clean_results[0.0]['main']['accuracy']*100:.1f}%)")
        print(f"  BlendedTTT Model: {blended_drop:.4f} ({blended_drop/clean_results[0.0]['main']['accuracy']*100:.1f}%)")
        
        # Print per-transformation accuracies
        print("\nPer-Transformation Accuracy:")
        for t_type in ['no_transform', 'gaussian_noise', 'rotation', 'affine']:
            print(f"  {t_type.upper()}:")
            print(f"    Main: {ood_results['main']['per_transform_acc'][t_type]:.4f}")
            print(f"    BlendedTTT: {ood_results['blended']['per_transform_acc'][t_type]:.4f}")
    
    # Log results to wandb if available
    try:
        import wandb
        log_wandb_results_with_blended_only(all_results)
    except:
        print("Note: wandb not available or error in logging.")
    
    return all_results

def log_wandb_results_main_only(all_results):
    """
    Log evaluation results to Weights & Biases for main model only
    """
    import wandb
    
    # Log clean data results
    if 0.0 in all_results:
        clean_results = all_results[0.0]
        clean_acc = {
            "eval/clean_accuracy": clean_results['main']['accuracy']
        }
        wandb.log(clean_acc)
    
    # Log OOD results
    for severity, results in all_results.items():
        if severity == 0.0:
            continue  # Skip clean results, already logged
        
        # Main metrics
        ood_metrics = {
            f"eval/ood_s{severity}_accuracy": results['main']['accuracy'],
        }
        
        # Per-transformation metrics
        if 'per_transform_acc' in results['main']:
            for t_type, acc in results['main']['per_transform_acc'].items():
                ood_metrics[f"eval/ood_s{severity}_{t_type}_accuracy"] = acc
        
        # Log all metrics together
        wandb.log(ood_metrics)

def log_wandb_results_with_blended_only(all_results):
    """
    Log evaluation results to Weights & Biases for main and BlendedTTT models
    """
    import wandb
    
    # Log clean data results
    if 0.0 in all_results:
        clean_results = all_results[0.0]
        clean_acc = {
            "eval/clean_accuracy": clean_results['main']['accuracy']
        }
        wandb.log(clean_acc)
    
    # Log OOD results
    for severity, results in all_results.items():
        if severity == 0.0:
            continue  # Skip clean results, already logged
        
        # Main metrics
        ood_metrics = {
            f"eval/ood_s{severity}_accuracy": results['main']['accuracy'],
        }
        
        if results['blended'] is not None:
            ood_metrics[f"eval/ood_s{severity}_blended_accuracy"] = results['blended']['accuracy']
        
        # Per-transformation metrics
        if 'per_transform_acc' in results['main']:
            for t_type, acc in results['main']['per_transform_acc'].items():
                ood_metrics[f"eval/ood_s{severity}_{t_type}_accuracy"] = acc
                
                if results['blended'] is not None and 'per_transform_acc' in results['blended']:
                    ood_metrics[f"eval/ood_s{severity}_{t_type}_blended_accuracy"] = (
                        results['blended']['per_transform_acc'][t_type]
                    )
        
        # Log all metrics together
        wandb.log(ood_metrics)

def log_wandb_results_with_ttt_only(all_results):
    """
    Log evaluation results to Weights & Biases for main and TTT models
    """
    import wandb
    
    # Log clean data results
    if 0.0 in all_results:
        clean_results = all_results[0.0]
        clean_acc = {
            "eval/clean_accuracy": clean_results['main']['accuracy']
        }
        wandb.log(clean_acc)
    
    # Log OOD results
    for severity, results in all_results.items():
        if severity == 0.0:
            continue  # Skip clean results, already logged
        
        # Main metrics
        ood_metrics = {
            f"eval/ood_s{severity}_accuracy": results['main']['accuracy'],
        }
        
        if 'ttt' in results and results['ttt'] is not None:
            ood_metrics[f"eval/ood_s{severity}_ttt_accuracy"] = results['ttt']['accuracy']
        
        if 'ttt_adapted' in results and results['ttt_adapted'] is not None:
            ood_metrics[f"eval/ood_s{severity}_ttt_adapted_accuracy"] = results['ttt_adapted']['accuracy']
        
        # Per-transformation metrics
        if 'per_transform_acc' in results['main']:
            for t_type, acc in results['main']['per_transform_acc'].items():
                ood_metrics[f"eval/ood_s{severity}_{t_type}_accuracy"] = acc
                
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

def log_wandb_results_with_blended_and_ttt(all_results):
    """
    Log evaluation results to Weights & Biases for main, BlendedTTT, and TTT models
    """
    import wandb
    
    # Log clean data results
    if 0.0 in all_results:
        clean_results = all_results[0.0]
        clean_acc = {
            "eval/clean_accuracy": clean_results['main']['accuracy']
        }
        wandb.log(clean_acc)
    
    # Log OOD results
    for severity, results in all_results.items():
        if severity == 0.0:
            continue  # Skip clean results, already logged
        
        # Main metrics
        ood_metrics = {
            f"eval/ood_s{severity}_accuracy": results['main']['accuracy'],
        }
        
        if results['blended'] is not None:
            ood_metrics[f"eval/ood_s{severity}_blended_accuracy"] = results['blended']['accuracy']
        
        if 'ttt' in results and results['ttt'] is not None:
            ood_metrics[f"eval/ood_s{severity}_ttt_accuracy"] = results['ttt']['accuracy']
        
        if 'ttt_adapted' in results and results['ttt_adapted'] is not None:
            ood_metrics[f"eval/ood_s{severity}_ttt_adapted_accuracy"] = results['ttt_adapted']['accuracy']
        
        # Per-transformation metrics
        if 'per_transform_acc' in results['main']:
            for t_type, acc in results['main']['per_transform_acc'].items():
                ood_metrics[f"eval/ood_s{severity}_{t_type}_accuracy"] = acc
                
                if results['blended'] is not None and 'per_transform_acc' in results['blended']:
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

def evaluate_full_pipeline_with_ttt_only(main_model, ttt_model, dataset_path, severities):
    """
    Evaluate the main model and TTT model.
    
    Args:
        main_model: The classification model
        ttt_model: The TTT model
        dataset_path: Path to the dataset
        severities: List of severity levels to evaluate
    
    Returns:
        all_results: Dictionary of evaluation results
    """
    all_results = {}
    
    # First evaluate on clean data (severity 0.0)
    print("\nEvaluating on clean data (no transformations)...")
    
    # For clean data, we don't use TTT, just main model
    # Use evaluate_main_model_only for clean data evaluation
    clean_results = evaluate_main_model_only(main_model, dataset_path, [])
    all_results[0.0] = clean_results[0.0]
    
    # Print clean data results
    print(f"Clean Data Accuracy:")
    print(f"  Main Model: {clean_results[0.0]['main']['accuracy']:.4f}")
    
    # Then evaluate on transformed data with both models
    for severity in severities:
        print(f"\nEvaluating with severity {severity}...")
        
        # Basic evaluation of main model only
        main_only_results = evaluate_main_model_only(main_model, dataset_path, [severity])
        
        # TTT evaluation
        print(f"Evaluating TTT model (pre-adaptation)...")
        ttt_standard_results = evaluate_with_ttt(main_model, ttt_model, dataset_path, severity=severity)
        
        print(f"Evaluating TTT model with test-time adaptation...")
        ttt_adaptation_results = evaluate_with_test_time_adaptation(
            main_model, ttt_model, dataset_path, severity=severity
        )
        
        # Create combined results
        ood_results = {
            'main': main_only_results[severity]['main'],
            'healer': None,  # No healer model
            'blended': None,  # No blended model
            'ttt': {
                'accuracy': ttt_standard_results['ttt_accuracy'],
                'per_transform_acc': ttt_standard_results['per_transform_acc']['ttt']
            },
            'ttt_adapted': {
                'accuracy': ttt_adaptation_results['ttt_accuracy'],
                'per_transform_acc': ttt_adaptation_results['per_transform_acc']['ttt']
            }
        }
        
        all_results[severity] = ood_results
        
        # Print results
        print(f"OOD Accuracy (Severity {severity}):")
        print(f"  Main Model: {ood_results['main']['accuracy']:.4f}")
        print(f"  TTT Model: {ood_results['ttt']['accuracy']:.4f}")
        print(f"  TTT Model (adapted): {ood_results['ttt_adapted']['accuracy']:.4f}")
        
        # Calculate robustness metrics
        main_drop = clean_results[0.0]['main']['accuracy'] - ood_results['main']['accuracy']
        ttt_drop = clean_results[0.0]['main']['accuracy'] - ood_results['ttt']['accuracy']
        ttt_adapted_drop = clean_results[0.0]['main']['accuracy'] - ood_results['ttt_adapted']['accuracy']
        
        print(f"\nAccuracy Drop from Clean Data:")
        print(f"  Main Model: {main_drop:.4f} ({main_drop/clean_results[0.0]['main']['accuracy']*100:.1f}%)")
        print(f"  TTT Model: {ttt_drop:.4f} ({ttt_drop/clean_results[0.0]['main']['accuracy']*100:.1f}%)")
        print(f"  TTT Model (adapted): {ttt_adapted_drop:.4f} ({ttt_adapted_drop/clean_results[0.0]['main']['accuracy']*100:.1f}%)")
        
        # Print per-transformation accuracies
        print("\nPer-Transformation Accuracy:")
        for t_type in ['no_transform', 'gaussian_noise', 'rotation', 'affine']:
            print(f"  {t_type.upper()}:")
            print(f"    Main: {ood_results['main']['per_transform_acc'][t_type]:.4f}")
            print(f"    TTT: {ood_results['ttt']['per_transform_acc'][t_type]:.4f}")
            print(f"    TTT (adapted): {ood_results['ttt_adapted']['per_transform_acc'][t_type]:.4f}")
    
    # Log results to wandb if available
    try:
        import wandb
        log_wandb_results_with_ttt_only(all_results)
    except:
        print("Note: wandb not available or error in logging.")
    
    return all_results

def evaluate_full_pipeline_with_blended_and_ttt(main_model, ttt_model, blended_model, dataset_path, severities):
    """
    Evaluate the full pipeline with main, TTT, and BlendedTTT models.
    
    Args:
        main_model: The classification model
        ttt_model: The test-time training model
        blended_model: The blended TTT model
        dataset_path: Path to the dataset
        severities: List of severity levels to evaluate
    
    Returns:
        all_results: Dictionary of evaluation results
    """
    all_results = {}
    
    # First evaluate on clean data (severity 0.0)
    print("\nEvaluating on clean data (no transformations)...")
    
    # For clean data, we just use the main model
    clean_results = evaluate_main_model_only(main_model, dataset_path, [])
    all_results[0.0] = clean_results[0.0]
    
    # Print clean data results
    print(f"Clean Data Accuracy:")
    print(f"  Main Model: {clean_results[0.0]['main']['accuracy']:.4f}")
    
    # Then evaluate on transformed data with both models
    for severity in severities:
        print(f"\nEvaluating with severity {severity}...")
        
        # Basic main model evaluation
        main_only_results = evaluate_main_model_only(main_model, dataset_path, [severity])
        
        # BlendedTTT evaluation
        print(f"Evaluating BlendedTTT model...")
        blended_results = evaluate_models_with_blended(main_model, blended_model, dataset_path, severity=severity)
        
        # TTT evaluation
        print(f"Evaluating TTT model (pre-adaptation)...")
        ttt_standard_results = evaluate_with_ttt(main_model, ttt_model, dataset_path, severity=severity)
        
        print(f"Evaluating TTT model with test-time adaptation...")
        ttt_adaptation_results = evaluate_with_test_time_adaptation(
            main_model, ttt_model, dataset_path, severity=severity
        )
        
        # Create combined results
        ood_results = {
            'main': main_only_results[severity]['main'],
            'healer': None,  # No healer model
            'blended': {
                'accuracy': blended_results['blended_accuracy'],
                'per_transform_acc': blended_results['per_transform_acc']['blended']
            },
            'ttt': {
                'accuracy': ttt_standard_results['ttt_accuracy'],
                'per_transform_acc': ttt_standard_results['per_transform_acc']['ttt']
            },
            'ttt_adapted': {
                'accuracy': ttt_adaptation_results['ttt_accuracy'],
                'per_transform_acc': ttt_adaptation_results['per_transform_acc']['ttt']
            }
        }
        
        all_results[severity] = ood_results
        
        # Print results
        print(f"OOD Accuracy (Severity {severity}):")
        print(f"  Main Model: {ood_results['main']['accuracy']:.4f}")
        print(f"  BlendedTTT Model: {ood_results['blended']['accuracy']:.4f}")
        print(f"  TTT Model: {ood_results['ttt']['accuracy']:.4f}")
        print(f"  TTT Model (adapted): {ood_results['ttt_adapted']['accuracy']:.4f}")
        
        # Calculate robustness metrics
        main_drop = clean_results[0.0]['main']['accuracy'] - ood_results['main']['accuracy']
        blended_drop = clean_results[0.0]['main']['accuracy'] - ood_results['blended']['accuracy']
        ttt_drop = clean_results[0.0]['main']['accuracy'] - ood_results['ttt']['accuracy']
        ttt_adapted_drop = clean_results[0.0]['main']['accuracy'] - ood_results['ttt_adapted']['accuracy']
        
        print(f"\nAccuracy Drop from Clean Data:")
        print(f"  Main Model: {main_drop:.4f} ({main_drop/clean_results[0.0]['main']['accuracy']*100:.1f}%)")
        print(f"  BlendedTTT Model: {blended_drop:.4f} ({blended_drop/clean_results[0.0]['main']['accuracy']*100:.1f}%)")
        print(f"  TTT Model: {ttt_drop:.4f} ({ttt_drop/clean_results[0.0]['main']['accuracy']*100:.1f}%)")
        print(f"  TTT Model (adapted): {ttt_adapted_drop:.4f} ({ttt_adapted_drop/clean_results[0.0]['main']['accuracy']*100:.1f}%)")
        
        # Print per-transformation accuracies
        print("\nPer-Transformation Accuracy:")
        for t_type in ['no_transform', 'gaussian_noise', 'rotation', 'affine']:
            print(f"  {t_type.upper()}:")
            print(f"    Main: {ood_results['main']['per_transform_acc'][t_type]:.4f}")
            print(f"    BlendedTTT: {ood_results['blended']['per_transform_acc'][t_type]:.4f}")
            print(f"    TTT: {ood_results['ttt']['per_transform_acc'][t_type]:.4f}")
            print(f"    TTT (adapted): {ood_results['ttt_adapted']['per_transform_acc'][t_type]:.4f}")
    
    # Log results to wandb if available
    try:
        import wandb
        log_wandb_results_with_blended_and_ttt(all_results)
    except:
        print("Note: wandb not available or error in logging.")
    
    return all_results

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

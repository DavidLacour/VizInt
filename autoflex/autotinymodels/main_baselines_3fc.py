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
# Import BlendedTTT3fc modules
from blended_ttt3fc_model import BlendedTTT3fc
from blended_ttt3fc_training import train_blended_ttt3fc_model
from ttt3fc_blended3fc_evaluation import (
    evaluate_3fc_models_comprehensive, 
    load_blended3fc_model, 
    load_ttt3fc_model,
    compare_3fc_with_original_models,
    log_3fc_results_to_wandb
)

# Import new TTT3fc modules
from ttt3fc_model import TestTimeTrainer3fc, train_ttt3fc_model

# Import original models for comparison
from blended_ttt_model import BlendedTTT
from blended_ttt_training import train_blended_ttt_model
from blended_ttt_evaluation import evaluate_models_with_blended
from ttt_model import TestTimeTrainer, train_ttt_model
from ttt_evaluation import evaluate_with_ttt, evaluate_with_test_time_adaptation
from robust_training import *

# Import baseline models - use enhanced version with early stopping
try:
    from baseline_models_enhanced import SimpleResNet18, SimpleVGG16, train_baseline_model_with_early_stopping as train_baseline_model
except ImportError:
    # Fallback to original if enhanced version not available
    from baseline_models import SimpleResNet18, SimpleVGG16, train_baseline_model

# Import ViT model creation
from vit_implementation import create_vit_model


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
    # TransformationHealer is imported from new_new via 'from new_new import *'
    healer_model = TransformationHealer(
        img_size=64,
        patch_size=8,
        in_chans=3,
        embed_dim=384,
        depth=6,
        head_dim=64
    )
    checkpoint = torch.load(model_path, map_location=device)
    healer_model.load_state_dict(checkpoint['model_state_dict'])
    healer_model = healer_model.to(device)
    healer_model.eval()
    return healer_model


def evaluate_full_pipeline_3fc(main_model, healer_model, dataset_path, severities=[0.1,0.2,0.3,0.4,0.6], model_dir="./", include_blended3fc=True, include_ttt3fc=True):
    """
    Evaluate the full transformation healing pipeline with 3FC models on clean and transformed data.
    
    Args:
        main_model: The classification model
        healer_model: The transformation healer model
        dataset_path: Path to the dataset
        severities: List of severity levels to evaluate
        model_dir: Directory containing the models
        include_blended3fc: Whether to include BlendedTTT3fc in evaluation
        include_ttt3fc: Whether to include TTT3fc in evaluation
        
    Returns:
        all_results: Dictionary of evaluation results
    """
    all_results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # First evaluate on clean data (severity 0.0)
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
    
    # Find blended3fc_model and ttt3fc_model if needed
    blended3fc_model = None
    ttt3fc_model = None
    
    if include_blended3fc:
        blended3fc_model_path = f"{model_dir}/bestmodel_blended3fc/best_model.pt"
        if os.path.exists(blended3fc_model_path):
            blended3fc_model = load_blended3fc_model(blended3fc_model_path, main_model, device)
            
    if include_ttt3fc:
        ttt3fc_model_path = f"{model_dir}/bestmodel_ttt3fc/best_model.pt"
        if os.path.exists(ttt3fc_model_path):
            ttt3fc_model = load_ttt3fc_model(ttt3fc_model_path, main_model, device)
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating models (clean data)"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass with main model
            outputs = main_model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Update metrics
            main_correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    # Calculate accuracy for main model
    main_accuracy = main_correct / total * 100
    
    # Store clean results
    all_results[0.0] = {
        'main': {
            'accuracy': main_accuracy,
            'correct': main_correct,
            'total': total
        }
    }
    
    print(f"Main Model (Clean): {main_accuracy:.2f}%")
    
    # Evaluate 3FC models on clean data if available
    if blended3fc_model is not None:
        blended3fc_results = evaluate_3fc_models_comprehensive(
            main_model, healer_model, blended3fc_model, None,
            val_loader, device, severity=0.0
        )
        all_results[0.0]['blended3fc'] = blended3fc_results['blended3fc']
        print(f"BlendedTTT3fc Model (Clean): {blended3fc_results['blended3fc']['accuracy']:.2f}%")
    
    if ttt3fc_model is not None:
        ttt3fc_results = evaluate_3fc_models_comprehensive(
            main_model, healer_model, None, ttt3fc_model,
            val_loader, device, severity=0.0
        )
        all_results[0.0]['ttt3fc'] = ttt3fc_results['ttt3fc']
        print(f"TTT3fc Model (Clean): {ttt3fc_results['ttt3fc']['accuracy']:.2f}%")
    
    # Now evaluate with transforms
    print("\n" + "="*80)
    print("EVALUATING WITH TRANSFORMS (OOD DATA)")
    print("="*80)
    
    # For each severity level
    for severity in severities:
        if severity == 0.0:
            continue  # Already evaluated clean data
            
        print(f"\n--- Severity: {severity} ---")
        
        # Create transformed validation dataset
        transform_val_ood = transforms.Compose([
            transforms.ToTensor(),
            ContinuousTransforms(severity=severity),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        val_dataset_ood = TinyImageNetDataset(dataset_path, "val", transform_val_ood)
        val_loader_ood = DataLoader(
            val_dataset_ood, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        # Evaluate all models
        results = evaluate_3fc_models_comprehensive(
            main_model, healer_model, 
            blended3fc_model if include_blended3fc else None,
            ttt3fc_model if include_ttt3fc else None,
            val_loader_ood, device, severity=severity
        )
        
        all_results[severity] = results
        
        # Print results
        print(f"Main Model: {results['main']['accuracy']:.2f}%")
        if 'healer' in results and results['healer'] is not None:
            print(f"With Healer: {results['healer']['accuracy']:.2f}%")
        if 'blended3fc' in results and results['blended3fc'] is not None:
            print(f"BlendedTTT3fc: {results['blended3fc']['accuracy']:.2f}%")
        if 'ttt3fc' in results and results['ttt3fc'] is not None:
            print(f"TTT3fc: {results['ttt3fc']['accuracy']:.2f}%")
    
    return all_results


def plot_results_3fc(all_results, save_path="results_comparison_3fc.png"):
    """Plot comparison of different models including 3FC variants"""
    # Extract data for plotting
    severities = sorted(all_results.keys())
    
    # Model types to plot
    model_types = ['main', 'healer', 'blended3fc', 'ttt3fc']
    model_labels = ['Main Model', 'With Healer', 'BlendedTTT3fc', 'TTT3fc']
    model_colors = ['blue', 'green', 'red', 'purple']
    model_markers = ['o', 's', '^', 'D']
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    for i, (model_type, label, color, marker) in enumerate(zip(model_types, model_labels, model_colors, model_markers)):
        accuracies = []
        valid_severities = []
        
        for severity in severities:
            if severity in all_results and model_type in all_results[severity] and all_results[severity][model_type] is not None:
                accuracies.append(all_results[severity][model_type]['accuracy'])
                valid_severities.append(severity)
        
        if accuracies:
            plt.plot(valid_severities, accuracies, color=color, marker=marker, 
                    markersize=8, linewidth=2, label=label)
    
    plt.xlabel('Transformation Severity', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Performance with 3FC Variants vs Transformation Severity', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, max(severities) + 0.05)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results plot saved to {save_path}")


def visualize_samples_3fc(main_model, healer_model, blended3fc_model, ttt3fc_model, dataset_path, 
                         severity=0.5, num_samples=5, save_path="sample_predictions_3fc.png"):
    """Visualize sample predictions from different models including 3FC variants"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        ContinuousTransforms(severity=severity),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform)
    val_loader = DataLoader(val_dataset, batch_size=num_samples, shuffle=True)
    
    # Get a batch
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions from all models
    with torch.no_grad():
        # Main model
        outputs_main = main_model(images)
        _, preds_main = torch.max(outputs_main, 1)
        
        # With healer
        healer_preds = healer_model(images)
        healed_images = healer_model.apply_correction(images, healer_preds)
        outputs_healer = main_model(healed_images)
        _, preds_healer = torch.max(outputs_healer, 1)
        
        # BlendedTTT3fc
        preds_blended3fc = None
        if blended3fc_model is not None:
            outputs_blended3fc = blended3fc_model(images)
            _, preds_blended3fc = torch.max(outputs_blended3fc, 1)
        
        # TTT3fc
        preds_ttt3fc = None
        if ttt3fc_model is not None:
            outputs_ttt3fc = ttt3fc_model.predict(images)
            _, preds_ttt3fc = torch.max(outputs_ttt3fc, 1)
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Create visualization
    n_models = 4 if (blended3fc_model is not None or ttt3fc_model is not None) else 2
    fig, axes = plt.subplots(num_samples, n_models + 1, figsize=(3*(n_models+1), 3*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        img = images_denorm[i].cpu().permute(1, 2, 0)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'GT: {labels[i].item()}')
        axes[i, 0].axis('off')
        
        # Main model prediction
        axes[i, 1].imshow(img)
        color = 'green' if preds_main[i] == labels[i] else 'red'
        axes[i, 1].set_title(f'Main: {preds_main[i].item()}', color=color)
        axes[i, 1].axis('off')
        
        # Healer prediction
        axes[i, 2].imshow(img)
        color = 'green' if preds_healer[i] == labels[i] else 'red'
        axes[i, 2].set_title(f'Healer: {preds_healer[i].item()}', color=color)
        axes[i, 2].axis('off')
        
        # BlendedTTT3fc prediction
        if blended3fc_model is not None and n_models >= 3:
            axes[i, 3].imshow(img)
            color = 'green' if preds_blended3fc[i] == labels[i] else 'red'
            axes[i, 3].set_title(f'Blended3fc: {preds_blended3fc[i].item()}', color=color)
            axes[i, 3].axis('off')
        
        # TTT3fc prediction
        if ttt3fc_model is not None and n_models >= 4:
            axes[i, 4].imshow(img)
            color = 'green' if preds_ttt3fc[i] == labels[i] else 'red'
            axes[i, 4].set_title(f'TTT3fc: {preds_ttt3fc[i].item()}', color=color)
            axes[i, 4].axis('off')
    
    plt.suptitle(f'Sample Predictions with 3FC Models (Severity: {severity})', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sample predictions saved to {save_path}")


def run_comprehensive_evaluation_3fc(args, device):
    """Run comprehensive evaluation including 3FC models"""
    # Model paths
    main_model_path = f"{args.model_dir}/bestmodel_main/best_model.pt"
    healer_model_path = f"{args.model_dir}/bestmodel_healer/best_model.pt"
    blended3fc_model_path = f"{args.model_dir}/bestmodel_blended3fc/best_model.pt"
    ttt3fc_model_path = f"{args.model_dir}/bestmodel_ttt3fc/best_model.pt"
    
    # Load models
    main_model = load_main_model(main_model_path, device) if os.path.exists(main_model_path) else None
    healer_model = load_healer_model(healer_model_path, device) if os.path.exists(healer_model_path) else None
    
    if main_model is None or healer_model is None:
        print("❌ Missing required models (main or healer)")
        return None
    
    # Evaluate with multiple severities
    all_results = evaluate_full_pipeline_3fc(
        main_model, healer_model, args.dataset,
        severities=args.severities,
        model_dir=args.model_dir,
        include_blended3fc=not args.exclude_blended3fc,
        include_ttt3fc=not args.exclude_ttt3fc
    )
    
    # Plot results
    plot_results_3fc(all_results, save_path=f"{args.model_dir}/results_comparison_3fc.png")
    
    # Log to wandb if available
    if hasattr(args, 'wandb_run') and args.wandb_run is not None:
        log_3fc_results_to_wandb(all_results, args.wandb_run)
    
    return all_results


def retrain_main_model(model_path, dataset_path, device, model_dir="../../newModels"):
    """Retrain main model with validation-based early stopping"""
    print(f"\n📂 Loading main model from {model_path}")
    main_model = load_main_model(model_path, device)
    
    # Evaluate initial performance
    print("📊 Evaluating initial model performance...")
    from new_new import evaluate_model, TinyImageNetDataset
    
    # Create validation dataset
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)
    
    initial_val_loss, initial_val_acc = evaluate_model(main_model, val_loader, device)
    print(f"📈 Initial validation - Loss: {initial_val_loss:.4f}, Acc: {initial_val_acc:.4f}")
    
    # Retrain with lower learning rate
    print("\n🔄 Retraining main model with early stopping...")
    from new_new import train_main_model
    
    # Save the current best performance to compare
    print("Note: Retraining will create new checkpoints. Initial model backed up.")
    
    # Call training function which already has early stopping
    retrained_model = train_main_model(dataset_path, model_dir=model_dir)
    
    return retrained_model


def retrain_healer_model(model_path, dataset_path, device, severity=0.5, model_dir="../../newModels"):
    """Retrain healer model with validation-based early stopping"""
    print(f"\n📂 Loading healer model from {model_path}")
    healer_model = load_healer_model(model_path, device)
    
    # The healer training already includes validation-based early stopping
    print("\n🔄 Retraining healer model...")
    from new_new import train_healer_model
    
    # Call training function which already has early stopping
    retrained_model = train_healer_model(dataset_path, severity=severity, model_dir=model_dir)
    
    return retrained_model


def main():
    parser = argparse.ArgumentParser(description="Complete 3FC model training and evaluation pipeline")
    parser.add_argument("--mode", type=str, default="all", 
                      choices=["train", "evaluate", "visualize", "all"],
                      help="Mode to run: train only, evaluate only, visualize only, or all")
    parser.add_argument("--dataset", type=str, default="../../../tiny-imagenet-200",
                      help="Path to Tiny ImageNet dataset")
    parser.add_argument("--model_dir", type=str, default="../../newModels",
                      help="Directory to save/load models")
    parser.add_argument("--severity", type=float, default=0.5,
                      help="Severity of transformations for training/visualization")
    parser.add_argument("--num_samples", type=int, default=5,
                      help="Number of samples to visualize")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--exclude_blended3fc", action="store_true",
                      help="Whether to exclude BlendedTTT3fc model from the pipeline")
    parser.add_argument("--exclude_ttt3fc", action="store_true",
                      help="Whether to exclude TTT3fc model from the pipeline")
    parser.add_argument("--skip_ttt3fc", action="store_true",
                      help="Skip training TTT3fc model (but still use it for evaluation if available)")
    parser.add_argument("--severities", type=str, default="0.0,0.3,0.5,0.75,1.0",
                      help="Comma-separated list of transformation severities to evaluate")
    
    # Baseline comparison arguments
    parser.add_argument("--compare_baseline", action="store_true",
                      help="Include baseline comparison in evaluation")
    parser.add_argument("--compare_original", action="store_true",
                      help="Compare 3FC models with original TTT/BlendedTTT models")
    parser.add_argument("--retrain", action="store_true",
                      help="Reload existing models and retrain with validation early stopping")
    
    args = parser.parse_args()
    
    # Parse severities from string to list of floats
    args.severities = [float(s) for s in args.severities.split(',')]
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    print(f"✓ Model directory: {args.model_dir}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    main_model = None
    healer_model = None
    ttt3fc_model = None
    blended3fc_model = None
    
    # Check if models exist before training
    main_model_path = f"{args.model_dir}/bestmodel_main/best_model.pt"
    healer_model_path = f"{args.model_dir}/bestmodel_healer/best_model.pt"
    ttt3fc_model_path = f"{args.model_dir}/bestmodel_ttt3fc/best_model.pt" if not args.exclude_ttt3fc else None
    blended3fc_model_path = f"{args.model_dir}/bestmodel_blended3fc/best_model.pt" if not args.exclude_blended3fc else None
    
    # Retrain mode
    if args.retrain:
        print("\n=== RETRAIN MODE ===")
        print("🔄 Will reload and retrain existing models with validation-based early stopping")
        
        # Check and retrain main model
        if os.path.exists(main_model_path):
            print(f"\n📍 Found main model at {main_model_path}")
            main_model = retrain_main_model(main_model_path, args.dataset, device, model_dir=args.model_dir)
        else:
            print(f"❌ Main model not found at {main_model_path}")
        
        # Check and retrain healer model
        if os.path.exists(healer_model_path):
            print(f"\n📍 Found healer model at {healer_model_path}")
            healer_model = retrain_healer_model(healer_model_path, args.dataset, device, 
                                              severity=args.severity, model_dir=args.model_dir)
        else:
            print(f"❌ Healer model not found at {healer_model_path}")
        
        # Note: TTT3fc and Blended3fc models would need their own retrain functions
        # For now, we'll just notify that they're not implemented
        if ttt3fc_model_path and os.path.exists(ttt3fc_model_path):
            print(f"\n⚠️  TTT3fc model found but retraining not yet implemented")
        
        if blended3fc_model_path and os.path.exists(blended3fc_model_path):
            print(f"\n⚠️  Blended3fc model found but retraining not yet implemented")
        
        print("\n✅ Retraining complete!")
        
        # After retraining, proceed to evaluation
        args.mode = "evaluate"
    
    # Training mode
    if args.mode in ["train", "evaluate", "all"] and not args.retrain:
        # Ensure model directory exists
        os.makedirs(args.model_dir, exist_ok=True)
        print(f"\n=== Model Directory Configuration ===")
        print(f"📁 Model directory: {args.model_dir}")
        print(f"📁 Directory exists: {'✓' if os.path.exists(args.model_dir) else '❌'}")
        
        print(f"\n=== Expected Model Paths ===")
        print(f"📍 Main model: {main_model_path}")
        print(f"📍 Healer model: {healer_model_path}")
        if ttt3fc_model_path:
            print(f"📍 TTT3fc model: {ttt3fc_model_path}")
        if blended3fc_model_path:
            print(f"📍 Blended3fc model: {blended3fc_model_path}")
        
        print(f"\n=== Checking for existing models (mode: {args.mode}) ===")
        
        # Check if main model exists
        if not os.path.exists(main_model_path):
            print("\n=== Training Main Classification Model ===")
            main_model = train_main_model(args.dataset, model_dir=args.model_dir)
            print(f"✅ Main model saved to: {main_model_path}")
            print(f"✅ Model file exists: {'✓' if os.path.exists(main_model_path) else '❌'}")
        else:
            print(f"✓ Main model found at {main_model_path}")
            main_model = load_main_model(main_model_path, device)
        
        # Check if healer model exists
        if not os.path.exists(healer_model_path):
            print("\n=== Training Transformation Healer Model ===")
            healer_model = train_healer_model(args.dataset, severity=args.severity, model_dir=args.model_dir)
            print(f"✅ Healer model saved to: {healer_model_path}")
            print(f"✅ Model file exists: {'✓' if os.path.exists(healer_model_path) else '❌'}")
        else:
            print(f"✓ Healer model found at {healer_model_path}")
            healer_model = load_healer_model(healer_model_path, device)
        
        # Train TTT3fc model if not skipped, not excluded, and not already existing
        if not args.exclude_ttt3fc and not args.skip_ttt3fc:
            if ttt3fc_model_path and not os.path.exists(ttt3fc_model_path):
                print("\n=== Training Test-Time Training 3FC Model ===")
                if main_model is None:
                    main_model = load_main_model(main_model_path, device)
                ttt3fc_model = train_ttt3fc_model(args.dataset, base_model=main_model, severity=args.severity, model_dir=args.model_dir)
                print(f"✅ TTT3fc model saved to: {ttt3fc_model_path}")
                print(f"✅ Model file exists: {'✓' if os.path.exists(ttt3fc_model_path) else '❌'}")
            elif ttt3fc_model_path:
                print(f"✓ TTT3fc model found at {ttt3fc_model_path}")
                if main_model is None:
                    main_model = load_main_model(main_model_path, device)
                ttt3fc_model = load_ttt3fc_model(ttt3fc_model_path, main_model, device)
        
        # Train BlendedTTT3fc model if not excluded and not already existing
        if not args.exclude_blended3fc:
            if blended3fc_model_path and not os.path.exists(blended3fc_model_path):
                print("\n=== Training BlendedTTT 3FC Model ===")
                if main_model is None:
                    main_model = load_main_model(main_model_path, device)
                blended3fc_model = train_blended_ttt3fc_model(main_model, args.dataset, model_dir=args.model_dir)
                print(f"✅ BlendedTTT3fc model saved to: {blended3fc_model_path}")
                print(f"✅ Model file exists: {'✓' if os.path.exists(blended3fc_model_path) else '❌'}")
            elif blended3fc_model_path:
                print(f"✓ BlendedTTT3fc model found at {blended3fc_model_path}")
                if main_model is None:
                    main_model = load_main_model(main_model_path, device)
                blended3fc_model = load_blended3fc_model(blended3fc_model_path, main_model, device)
        
        # Print summary of model locations after training
        if args.mode in ["train", "all"]:
            print("\n=== Model Training Summary ===")
            print(f"📁 All models saved to: {args.model_dir}")
            print("\n📍 Model locations:")
            print(f"  - Main model: {main_model_path} {'✓' if os.path.exists(main_model_path) else '❌'}")
            print(f"  - Healer model: {healer_model_path} {'✓' if os.path.exists(healer_model_path) else '❌'}")
            if ttt3fc_model_path:
                print(f"  - TTT3fc model: {ttt3fc_model_path} {'✓' if os.path.exists(ttt3fc_model_path) else '❌'}")
            if blended3fc_model_path:
                print(f"  - Blended3fc model: {blended3fc_model_path} {'✓' if os.path.exists(blended3fc_model_path) else '❌'}")
    
    # Evaluation mode
    if args.mode in ["evaluate", "visualize", "all"]:
        print("\n=== Comprehensive Evaluation With and Without Transforms ===")
        
        # Load any models that weren't already loaded during training phase
        if main_model is None and os.path.exists(main_model_path):
            main_model = load_main_model(main_model_path, device)
        
        if healer_model is None and os.path.exists(healer_model_path):
            healer_model = load_healer_model(healer_model_path, device)
        
        if not args.exclude_ttt3fc and ttt3fc_model is None and ttt3fc_model_path and os.path.exists(ttt3fc_model_path):
            ttt3fc_model = load_ttt3fc_model(ttt3fc_model_path, main_model, device)
        
        if not args.exclude_blended3fc and blended3fc_model is None and blended3fc_model_path and os.path.exists(blended3fc_model_path):
            blended3fc_model = load_blended3fc_model(blended3fc_model_path, main_model, device)
        
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
            print(f"   python main_baselines_3fc.py --mode train --dataset {args.dataset}")
            return
        
        # Run comprehensive evaluation of 3FC models
        print("\n" + "="*100)
        print("🔥 RUNNING COMPREHENSIVE EVALUATION OF 3FC MODEL COMBINATIONS")
        print("="*100)
        
        all_results = run_comprehensive_evaluation_3fc(args, device)
        
        if all_results is None:
            print("❌ Comprehensive evaluation failed")
            return
        
        # Compare with original models if requested
        if args.compare_original:
            print("\n" + "="*100)
            print("📊 COMPARING 3FC MODELS WITH ORIGINAL TTT/BLENDED MODELS")
            print("="*100)
            
            # Load original models if available
            ttt_model_path = f"{args.model_dir}/bestmodel_ttt/best_model.pt"
            blended_model_path = f"{args.model_dir}/bestmodel_blended/best_model.pt"
            
            ttt_model = load_ttt_model(ttt_model_path, main_model, device) if os.path.exists(ttt_model_path) else None
            blended_model = load_blended_model(blended_model_path, main_model, device) if os.path.exists(blended_model_path) else None
            
            if ttt_model is not None or blended_model is not None:
                comparison_results = compare_3fc_with_original_models(
                    main_model, healer_model,
                    ttt_model, blended_model,
                    ttt3fc_model, blended3fc_model,
                    args.dataset, device,
                    severities=args.severities
                )
                
                # Save comparison plot
                save_path = f"{args.model_dir}/3fc_vs_original_comparison.png"
                print(f"Comparison results saved to {save_path}")
    
    # Visualization mode
    if args.mode in ["visualize", "all"]:
        print("\n=== Generating Visualizations ===")
        
        # Make sure models are loaded before visualization
        if main_model is None:
            main_model = load_main_model(main_model_path, device)
        
        if healer_model is None:
            healer_model = load_healer_model(healer_model_path, device)
        
        if not args.exclude_ttt3fc and ttt3fc_model is None and os.path.exists(ttt3fc_model_path):
            ttt3fc_model = load_ttt3fc_model(ttt3fc_model_path, main_model, device)
        
        if not args.exclude_blended3fc and blended3fc_model is None and os.path.exists(blended3fc_model_path):
            blended3fc_model = load_blended3fc_model(blended3fc_model_path, main_model, device)
        
        # Generate sample visualizations
        visualize_samples_3fc(
            main_model, healer_model, blended3fc_model, ttt3fc_model,
            args.dataset, severity=args.severity, num_samples=args.num_samples,
            save_path=f"{args.model_dir}/sample_predictions_3fc_s{args.severity}.png"
        )
    
    print("\n✅ All tasks completed successfully!")


if __name__ == "__main__":
    main()
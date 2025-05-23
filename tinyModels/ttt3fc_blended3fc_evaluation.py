import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from new_new import ContinuousTransforms, TinyImageNetDataset
from torchvision import transforms
import os

import torch.nn.functional as F

# Import the new models
from blended_ttt3fc_model import BlendedTTT3fc
from ttt3fc_model import TestTimeTrainer3fc


def evaluate_blended3fc_model(model, loader, device):
    """
    Evaluate the BlendedTTT3fc model on a dataset, using only classification output
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
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


def evaluate_ttt3fc_model(model, loader, device, use_adaptation=False, adapt_classification=True):
    """
    Evaluate the TTT3fc model on a dataset
    
    Args:
        model: TTT3fc model
        loader: DataLoader
        device: Device to run on
        use_adaptation: Whether to use test-time adaptation
        adapt_classification: Whether to adapt classification head during adaptation
    
    Returns:
        accuracy: Classification accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    if use_adaptation:
        # Evaluation with adaptation
        for batch_idx, batch in enumerate(loader):
            # Handle different batch formats
            if len(batch) == 2:
                images, labels = batch
                transform_labels = None
            elif len(batch) == 4:
                orig_images, transformed_images, labels, params = batch
                images = transformed_images
                # Extract transform types if available
                transform_labels = [
                    {'no_transform': 0, 'gaussian_noise': 1, 'rotation': 2, 'affine': 3}
                    .get(p.get('transform_type', 'no_transform'), 0)
                    for p in params
                ]
                transform_labels = torch.tensor(transform_labels, device=device)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            images, labels = images.to(device), labels.to(device)
            
            # Adapt to this batch
            adapted_logits = model.adapt(
                images, 
                transform_labels, 
                reset=True, 
                adapt_classification=adapt_classification
            )
            
            # Calculate accuracy
            _, predicted = torch.max(adapted_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    else:
        # Standard evaluation without adaptation
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # Handle different batch formats
                if len(batch) == 2:
                    images, labels = batch
                elif len(batch) == 4:
                    orig_images, transformed_images, labels, params = batch
                    images = transformed_images
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass without adaptation
                logits, _ = model(images, use_base_model=False)
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
    # Avoid division by zero
    if total == 0:
        return 0.0
        
    return correct / total


def load_blended3fc_model(model_path, device):
    """Load BlendedTTT3fc model from checkpoint"""
    print(f"Loading BlendedTTT3fc model from {model_path}")
    model = BlendedTTT3fc(
        img_size=64,
        patch_size=8,
        embed_dim=384,
        depth=8,
        hidden_dim=512,
        dropout_rate=0.1
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_ttt3fc_model(model_path, main_model, device):
    """Load TTT3fc model from checkpoint"""
    print(f"Loading TTT3fc model from {model_path}")
    model = TestTimeTrainer3fc(
        base_model=main_model,
        img_size=64,
        patch_size=8,
        embed_dim=384,
        hidden_dim=512,
        dropout_rate=0.1,
        adaptation_steps=5,
        adaptation_lr=1e-4
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def evaluate_3fc_models_comprehensive(
    main_model, 
    healer_model, 
    dataset_path, 
    severities=[0.0, 0.3, 0.5, 0.75, 1.0], 
    model_dir="./",
    include_blended3fc=True,
    include_ttt3fc=True
):
    """
    Comprehensive evaluation of the 3FC models along with existing models
    
    Args:
        main_model: The main classification model
        healer_model: The healer model
        dataset_path: Path to dataset
        severities: List of severity levels to evaluate
        model_dir: Directory containing model checkpoints
        include_blended3fc: Whether to include BlendedTTT3fc
        include_ttt3fc: Whether to include TTT3fc
        
    Returns:
        Dictionary of results for all models and severities
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}
    
    # Load 3FC models if available
    blended3fc_model = None
    ttt3fc_model = None
    
    if include_blended3fc:
        blended3fc_model_path = f"{model_dir}/bestmodel_blended3fc/best_model.pt"
        if os.path.exists(blended3fc_model_path):
            blended3fc_model = load_blended3fc_model(blended3fc_model_path, device)
            print("✅ Loaded: BlendedTTT3fc Model")
        else:
            print("⚠️  Missing: BlendedTTT3fc Model")
    
    if include_ttt3fc:
        ttt3fc_model_path = f"{model_dir}/bestmodel_ttt3fc/best_model.pt"
        if os.path.exists(ttt3fc_model_path):
            ttt3fc_model = load_ttt3fc_model(ttt3fc_model_path, main_model, device)
            print("✅ Loaded: TTT3fc Model")
        else:
            print("⚠️  Missing: TTT3fc Model")
    
    # Define transforms
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Evaluate on each severity level
    for severity in severities:
        print(f"\n🔍 Evaluating 3FC models at severity {severity}")
        
        # Create appropriate dataset
        if severity == 0.0:
            # Clean data
            val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=128, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True
            )
        else:
            # Transformed data
            ood_transform = ContinuousTransforms(severity=severity)
            ood_val_dataset = TinyImageNetDataset(
                dataset_path, "val", transform_val, ood_transform=ood_transform
            )
            
            def collate_fn(batch):
                orig_imgs, trans_imgs, labels, params = zip(*batch)
                return torch.stack(orig_imgs), torch.stack(trans_imgs), torch.tensor(labels), params
            
            val_loader = DataLoader(
                ood_val_dataset, 
                batch_size=128, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True,
                collate_fn=collate_fn
            )
        
        # Initialize results for this severity
        severity_results = {}
        
        # Evaluate main model on clean/transformed data
        main_model.eval()
        main_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluating Main (severity {severity})"):
                if len(batch) == 2:
                    images, labels = batch
                else:
                    orig_images, images, labels, params = batch
                
                images, labels = images.to(device), labels.to(device)
                
                outputs = main_model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                main_correct += (predicted == labels).sum().item()
        
        main_accuracy = main_correct / total
        severity_results['main'] = {'accuracy': main_accuracy, 'correct': main_correct, 'total': total}
        
        # Evaluate healer + main model
        healer_model.eval()
        healer_correct = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluating Healer (severity {severity})"):
                if len(batch) == 2:
                    images, labels = batch
                else:
                    orig_images, images, labels, params = batch
                
                images, labels = images.to(device), labels.to(device)
                
                # Apply healer
                healer_predictions = healer_model(images)
                corrected_images = healer_model.apply_correction(images, healer_predictions)
                
                # Forward pass with main model
                outputs = main_model(corrected_images)
                _, predicted = torch.max(outputs, 1)
                
                healer_correct += (predicted == labels).sum().item()
        
        healer_accuracy = healer_correct / total
        severity_results['healer'] = {'accuracy': healer_accuracy, 'correct': healer_correct, 'total': total}
        
        # Evaluate BlendedTTT3fc model
        if include_blended3fc and blended3fc_model is not None:
            blended3fc_accuracy = evaluate_blended3fc_model(blended3fc_model, val_loader, device)
            blended3fc_correct = int(blended3fc_accuracy * total)
            severity_results['blended3fc'] = {
                'accuracy': blended3fc_accuracy, 
                'correct': blended3fc_correct, 
                'total': total
            }
            print(f"  BlendedTTT3fc: {blended3fc_accuracy:.4f}")
        else:
            severity_results['blended3fc'] = None
        
        # Evaluate TTT3fc model (both with and without adaptation)
        if include_ttt3fc and ttt3fc_model is not None:
            # Without adaptation
            ttt3fc_accuracy = evaluate_ttt3fc_model(ttt3fc_model, val_loader, device, use_adaptation=False)
            ttt3fc_correct = int(ttt3fc_accuracy * total)
            severity_results['ttt3fc'] = {
                'accuracy': ttt3fc_accuracy, 
                'correct': ttt3fc_correct, 
                'total': total
            }
            print(f"  TTT3fc (no adapt): {ttt3fc_accuracy:.4f}")
            
            # With adaptation
            if severity > 0.0:  # Only do adaptation on transformed data
                ttt3fc_adapted_accuracy = evaluate_ttt3fc_model(
                    ttt3fc_model, val_loader, device, 
                    use_adaptation=True, adapt_classification=True
                )
                ttt3fc_adapted_correct = int(ttt3fc_adapted_accuracy * total)
                severity_results['ttt3fc_adapted'] = {
                    'accuracy': ttt3fc_adapted_accuracy, 
                    'correct': ttt3fc_adapted_correct, 
                    'total': total
                }
                print(f"  TTT3fc (adapted): {ttt3fc_adapted_accuracy:.4f}")
            else:
                severity_results['ttt3fc_adapted'] = severity_results['ttt3fc']  # Same as non-adapted for clean data
        else:
            severity_results['ttt3fc'] = None
            severity_results['ttt3fc_adapted'] = None
        
        all_results[severity] = severity_results
        
        # Print summary for this severity
        print(f"\nSummary for severity {severity}:")
        print(f"  Main Model: {main_accuracy:.4f}")
        print(f"  Healer Model: {healer_accuracy:.4f}")
        
        if severity_results['blended3fc'] is not None:
            print(f"  BlendedTTT3fc: {severity_results['blended3fc']['accuracy']:.4f}")
        
        if severity_results['ttt3fc'] is not None:
            print(f"  TTT3fc: {severity_results['ttt3fc']['accuracy']:.4f}")
            if severity_results['ttt3fc_adapted'] != severity_results['ttt3fc']:
                print(f"  TTT3fc (adapted): {severity_results['ttt3fc_adapted']['accuracy']:.4f}")
    
    return all_results


def compare_3fc_with_original_models(results_3fc, results_original):
    """
    Compare the performance of 3FC models with original models
    
    Args:
        results_3fc: Results from 3FC models evaluation
        results_original: Results from original models evaluation
    """
    print("\n" + "="*80)
    print("🔬 COMPARISON: 3FC MODELS vs ORIGINAL MODELS")
    print("="*80)
    
    for severity in sorted(results_3fc.keys()):
        if severity not in results_original:
            continue
            
        print(f"\nSeverity {severity}:")
        
        # Compare BlendedTTT vs BlendedTTT3fc
        if ('blended' in results_original[severity] and results_original[severity]['blended'] is not None and
            'blended3fc' in results_3fc[severity] and results_3fc[severity]['blended3fc'] is not None):
            
            orig_acc = results_original[severity]['blended']['accuracy']
            new_acc = results_3fc[severity]['blended3fc']['accuracy']
            improvement = new_acc - orig_acc
            
            print(f"  BlendedTTT Comparison:")
            print(f"    Original: {orig_acc:.4f}")
            print(f"    3FC:      {new_acc:.4f}")
            print(f"    Improvement: {improvement:.4f} ({improvement*100:.1f} points)")
        
        # Compare TTT vs TTT3fc
        if ('ttt' in results_original[severity] and results_original[severity]['ttt'] is not None and
            'ttt3fc' in results_3fc[severity] and results_3fc[severity]['ttt3fc'] is not None):
            
            orig_acc = results_original[severity]['ttt']['accuracy']
            new_acc = results_3fc[severity]['ttt3fc']['accuracy']
            improvement = new_acc - orig_acc
            
            print(f"  TTT Comparison:")
            print(f"    Original: {orig_acc:.4f}")
            print(f"    3FC:      {new_acc:.4f}")
            print(f"    Improvement: {improvement:.4f} ({improvement*100:.1f} points)")
        
        # Compare TTT adapted vs TTT3fc adapted
        if ('ttt_adapted' in results_original[severity] and results_original[severity]['ttt_adapted'] is not None and
            'ttt3fc_adapted' in results_3fc[severity] and results_3fc[severity]['ttt3fc_adapted'] is not None):
            
            orig_acc = results_original[severity]['ttt_adapted']['accuracy']
            new_acc = results_3fc[severity]['ttt3fc_adapted']['accuracy']
            improvement = new_acc - orig_acc
            
            print(f"  TTT Adapted Comparison:")
            print(f"    Original: {orig_acc:.4f}")
            print(f"    3FC:      {new_acc:.4f}")
            print(f"    Improvement: {improvement:.4f} ({improvement*100:.1f} points)")


def log_3fc_results_to_wandb(results):
    """Log 3FC model results to wandb"""
    try:
        import wandb
        
        for severity, severity_results in results.items():
            log_data = {
                f"3fc_eval/severity": severity,
            }
            
            if severity_results['main'] is not None:
                log_data[f"3fc_eval/main_accuracy_s{severity}"] = severity_results['main']['accuracy']
            
            if severity_results['healer'] is not None:
                log_data[f"3fc_eval/healer_accuracy_s{severity}"] = severity_results['healer']['accuracy']
            
            if severity_results['blended3fc'] is not None:
                log_data[f"3fc_eval/blended3fc_accuracy_s{severity}"] = severity_results['blended3fc']['accuracy']
            
            if severity_results['ttt3fc'] is not None:
                log_data[f"3fc_eval/ttt3fc_accuracy_s{severity}"] = severity_results['ttt3fc']['accuracy']
            
            if severity_results['ttt3fc_adapted'] is not None:
                log_data[f"3fc_eval/ttt3fc_adapted_accuracy_s{severity}"] = severity_results['ttt3fc_adapted']['accuracy']
            
            wandb.log(log_data)
    except:
        print("Note: wandb not available or error in logging 3FC results.")

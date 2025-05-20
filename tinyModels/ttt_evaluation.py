import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from new_new import ContinuousTransforms, TinyImageNetDataset

def evaluate_with_ttt(main_model, ttt_model, dataset_path, severity=0.5, batch_size=64):
    """
    Evaluate the TTT model on test-time adaptation.
    
    Args:
        main_model: Base classification model
        ttt_model: Test-time training model
        dataset_path: Path to the dataset
        severity: Severity of transformations
        batch_size: Batch size for evaluation
        
    Returns:
        results: Dictionary of evaluation results
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure models are in eval mode
    main_model.eval()
    ttt_model.eval()
    
    # Define image transformations
    transform_val = torch.nn.Sequential(
        torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    
    # Create continuous transforms for OOD
    ood_transform = ContinuousTransforms(severity=severity)
    
    # Create validation dataset with OOD transforms
    val_dataset = TinyImageNetDataset(
        dataset_path, "val", transform_val, ood_transform=ood_transform
    )
    
    # Simplified collate function
    def collate_fn(batch):
        orig_imgs, trans_imgs, labels, params = zip(*batch)
        
        orig_tensor = torch.stack(orig_imgs)
        trans_tensor = torch.stack(trans_imgs)
        labels_tensor = torch.tensor(labels)
        
        # Extract transform types
        transform_types = []
        for p in params:
            if isinstance(p, dict) and 'transform_type' in p:
                t_type = p['transform_type']
                if t_type == 'no_transform':
                    transform_types.append(0)
                elif t_type == 'gaussian_noise':
                    transform_types.append(1)
                elif t_type == 'rotation':
                    transform_types.append(2)
                elif t_type == 'affine':
                    transform_types.append(3)
                else:
                    transform_types.append(0)
            else:
                transform_types.append(0)
        
        transform_types_tensor = torch.tensor(transform_types)
        
        # Keep params as a list of dictionaries
        return orig_tensor, trans_tensor, labels_tensor, transform_types_tensor, params
    
    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize metrics
    main_correct = 0
    ttt_correct = 0
    total = 0
    
    # Per-transformation metrics
    transform_types = ['no_transform', 'gaussian_noise', 'rotation', 'affine']
    per_transform_metrics = {
        'main': {t: {'correct': 0, 'total': 0} for t in transform_types},
        'ttt': {t: {'correct': 0, 'total': 0} for t in transform_types}
    }
    
    # Evaluation loop
    with torch.no_grad():
        for orig_images, transformed_images, labels, transform_types, params in tqdm(val_loader, desc="Evaluating TTT"):
            # Move to device
            orig_images = orig_images.to(device)
            transformed_images = transformed_images.to(device)
            labels = labels.to(device)
            transform_types = transform_types.to(device)
            
            batch_size = transformed_images.size(0)
            
            # Main model predictions on transformed images
            main_outputs = main_model(transformed_images)
            main_preds = torch.argmax(main_outputs, dim=1)
            
            # TTT model predictions with adaptation
            ttt_logits, _ = ttt_model(transformed_images)
            ttt_preds = torch.argmax(ttt_logits, dim=1)
            
            # Update metrics
            main_correct += (main_preds == labels).sum().item()
            ttt_correct += (ttt_preds == labels).sum().item()
            total += batch_size
            
            # Update per-transformation metrics
            for i, (param, t_type) in enumerate(zip(params, transform_types)):
                # Get transform type name
                t_name = transform_types[t_type.item()]
                
                # Update counts
                per_transform_metrics['main'][t_name]['total'] += 1
                per_transform_metrics['ttt'][t_name]['total'] += 1
                
                if main_preds[i] == labels[i]:
                    per_transform_metrics['main'][t_name]['correct'] += 1
                
                if ttt_preds[i] == labels[i]:
                    per_transform_metrics['ttt'][t_name]['correct'] += 1
    
    # Calculate overall accuracies
    main_accuracy = main_correct / total
    ttt_accuracy = ttt_correct / total
    
    # Calculate per-transformation accuracies
    per_transform_acc = {
        'main': {},
        'ttt': {}
    }
    
    for model_name in ['main', 'ttt']:
        for t_name in transform_types:
            total_t = per_transform_metrics[model_name][t_name]['total']
            if total_t > 0:
                per_transform_acc[model_name][t_name] = (
                    per_transform_metrics[model_name][t_name]['correct'] / total_t
                )
            else:
                per_transform_acc[model_name][t_name] = 0.0
    
    # Compile results
    results = {
        'main_accuracy': main_accuracy,
        'ttt_accuracy': ttt_accuracy,
        'improvement': ttt_accuracy - main_accuracy,
        'per_transform_acc': per_transform_acc
    }
    
    return results

def evaluate_with_test_time_adaptation(main_model, ttt_model, dataset_path, severity=0.5, batch_size=16):
    """
    Evaluate the TTT model with full test-time adaptation.
    
    This performs actual test-time training on each batch.
    
    Args:
        main_model: Base classification model
        ttt_model: Test-time training model
        dataset_path: Path to the dataset
        severity: Severity of transformations
        batch_size: Batch size for evaluation (smaller for adaptation)
        
    Returns:
        results: Dictionary of evaluation results
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure models are in eval mode
    main_model.eval()
    ttt_model.eval()
    
    # Define image transformations
    transform_val = torch.nn.Sequential(
        torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    
    # Create continuous transforms for OOD
    ood_transform = ContinuousTransforms(severity=severity)
    
    # Create validation dataset with OOD transforms
    val_dataset = TinyImageNetDataset(
        dataset_path, "val", transform_val, ood_transform=ood_transform
    )
    
    # Simplified collate function
    def collate_fn(batch):
        orig_imgs, trans_imgs, labels, params = zip(*batch)
        
        orig_tensor = torch.stack(orig_imgs)
        trans_tensor = torch.stack(trans_imgs)
        labels_tensor = torch.tensor(labels)
        
        # Keep params as a list of dictionaries
        return orig_tensor, trans_tensor, labels_tensor, params
    
    # Create validation dataloader with smaller batch size for adaptation
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize metrics
    main_correct = 0
    ttt_correct = 0
    total = 0
    
    # Per-transformation metrics
    transform_types = ['no_transform', 'gaussian_noise', 'rotation', 'affine']
    per_transform_metrics = {
        'main': {t: {'correct': 0, 'total': 0} for t in transform_types},
        'ttt': {t: {'correct': 0, 'total': 0} for t in transform_types}
    }
    
    # Helper function to determine transform type
    def get_transform_type(params):
        if isinstance(params, dict) and 'transform_type' in params:
            return params['transform_type']
        return 'no_transform'
    
    # Evaluation with adaptation loop
    for orig_images, transformed_images, labels, params in tqdm(val_loader, desc="Evaluating with Adaptation"):
        # Move to device
        orig_images = orig_images.to(device)
        transformed_images = transformed_images.to(device)
        labels = labels.to(device)
        
        batch_size = transformed_images.size(0)
        
        # Main model predictions on transformed images
        with torch.no_grad():
            main_outputs = main_model(transformed_images)
            main_preds = torch.argmax(main_outputs, dim=1)
        
        # TTT model predictions with full adaptation
        # Extract transform types for supervision if available
        transform_labels = []
        for p in params:
            t_type = get_transform_type(p)
            if t_type == 'no_transform':
                transform_labels.append(0)
            elif t_type == 'gaussian_noise':
                transform_labels.append(1)
            elif t_type == 'rotation':
                transform_labels.append(2)
            elif t_type == 'affine':
                transform_labels.append(3)
            else:
                transform_labels.append(0)
        
        transform_labels = torch.tensor(transform_labels, device=device)
        
        # Adapt the model on this batch
        adapted_logits = ttt_model.adapt(transformed_images, transform_labels, reset=True)
        ttt_preds = torch.argmax(adapted_logits, dim=1)
        
        # Update metrics
        main_correct += (main_preds == labels).sum().item()
        ttt_correct += (ttt_preds == labels).sum().item()
        total += batch_size
        
        # Update per-transformation metrics
        for i, p in enumerate(params):
            # Get transform type name
            t_name = get_transform_type(p)
            
            # Update counts
            per_transform_metrics['main'][t_name]['total'] += 1
            per_transform_metrics['ttt'][t_name]['total'] += 1
            
            if main_preds[i] == labels[i]:
                per_transform_metrics['main'][t_name]['correct'] += 1
            
            if ttt_preds[i] == labels[i]:
                per_transform_metrics['ttt'][t_name]['correct'] += 1
    
    # Calculate overall accuracies
    main_accuracy = main_correct / total
    ttt_accuracy = ttt_correct / total
    
    # Calculate per-transformation accuracies
    per_transform_acc = {
        'main': {},
        'ttt': {}
    }
    
    for model_name in ['main', 'ttt']:
        for t_name in transform_types:
            total_t = per_transform_metrics[model_name][t_name]['total']
            if total_t > 0:
                per_transform_acc[model_name][t_name] = (
                    per_transform_metrics[model_name][t_name]['correct'] / total_t
                )
            else:
                per_transform_acc[model_name][t_name] = 0.0
    
    # Compile results
    results = {
        'main_accuracy': main_accuracy,
        'ttt_accuracy': ttt_accuracy,
        'improvement': ttt_accuracy - main_accuracy,
        'per_transform_acc': per_transform_acc
    }
    
    return results

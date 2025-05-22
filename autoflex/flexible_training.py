import argparse
import torch
import wandb
from pathlib import Path
from flexible_models import create_model, BACKBONE_CONFIGS
from new_new import (
    TinyImageNetDataset, ContinuousTransforms, HealerLoss,
    find_optimal_batch_size, set_seed
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np

def train_classification_model(
    dataset_path="tiny-imagenet-200", 
    backbone_name="vit_small",
    epochs=50,
    batch_size=None,
    learning_rate=1e-4,
    experiment_name=None
):
    """
    Train a classification model with the specified backbone using enhanced early stopping.
    
    Args:
        dataset_path: Path to the dataset
        backbone_name: Name of the backbone to use
        epochs: Number of training epochs
        batch_size: Batch size (if None, will be automatically determined)
        learning_rate: Learning rate
        experiment_name: Name for wandb experiment
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Setup wandb
    if experiment_name is None:
        experiment_name = f"classification_{backbone_name}"
    
    wandb.init(project="flexible-vit-tiny-imagenet", name=experiment_name)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using backbone: {backbone_name}")
    
    # Create model
    model = create_model('classification', backbone_name, num_classes=200)
    model.to(device)
    
    # Find optimal batch size if not specified
    if batch_size is None:
        batch_size = find_optimal_batch_size(model, img_size=64, starting_batch_size=128, device=device)
    
    # Define image transformations
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset and DataLoader
    train_dataset = TinyImageNetDataset(dataset_path, "train", transform_train)
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # DataLoaders
    num_workers = 8 if torch.cuda.is_available() else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    # Training parameters
    warmup_steps = 1000
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    def get_lr(step, total_steps, warmup_steps, base_lr):
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)
        else:
            decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
            return base_lr * 0.5 * (1 + np.cos(np.pi * decay_ratio))
    
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: get_lr(step, total_steps, warmup_steps, 1.0)
    )
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Logging
    wandb.config.update({
        "model": "classification",
        "backbone": backbone_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps
    })
    
    # Enhanced early stopping trainer
    from early_stopping_trainer import EarlyStoppingTrainer, get_early_stopping_config
    
    # Get early stopping configuration
    early_stop_config = get_early_stopping_config('classification', backbone_name)
    
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints_{backbone_name}_classification"
    best_model_dir = Path(f"bestmodel_{backbone_name}_classification")
    best_model_dir.mkdir(exist_ok=True)
    
    # Initialize enhanced trainer
    trainer = EarlyStoppingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=early_stop_config.patience,
        min_delta=early_stop_config.min_delta,
        restore_best_weights=early_stop_config.restore_best_weights,
        save_dir=checkpoint_dir,
        model_name=f"{backbone_name}_classification",
        cleanup_checkpoints=early_stop_config.cleanup_checkpoints,
        verbose=True
    )
    
    # Train with early stopping
    results = trainer.train(
        epochs=epochs,
        criterion=criterion,
        log_wandb=True,
        log_prefix="classification/"
    )
    
    # Save the final best model in the expected location
    if results['best_model_path']:
        best_model_path = best_model_dir / "best_model.pt"
        
        # Load the best checkpoint and save in expected format
        checkpoint = torch.load(results['best_model_path'], map_location='cpu')
        final_checkpoint = {
            'epoch': checkpoint['epoch'],
            'model_state_dict': checkpoint['model_state_dict'],
            'backbone_name': backbone_name,
            'val_acc': checkpoint['val_acc'],
            'val_loss': checkpoint['val_loss'],
            'training_results': results
        }
        
        torch.save(final_checkpoint, best_model_path)
        print(f"💾 Final model saved to: {best_model_path}")
    
    print(f"✅ Training completed!")
    print(f"   🏆 Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"   📊 Total epochs trained: {results['total_epochs']}")
    print(f"   ⏹️  Early stopped: {'Yes' if results['early_stopped'] else 'No'}")
    
    wandb.finish()
    return model

def train_healer_model(
    dataset_path="tiny-imagenet-200",
    backbone_name="vit_small", 
    severity=1.0,
    epochs=15,
    batch_size=None,
    learning_rate=5e-5,
    experiment_name=None
):
    """Train a healer model with the specified backbone using enhanced early stopping."""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Setup wandb
    if experiment_name is None:
        experiment_name = f"healer_{backbone_name}"
    
    wandb.init(project="flexible-vit-tiny-imagenet", name=experiment_name)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using backbone: {backbone_name}")
    
    # Create model
    healer_model = create_model('healer', backbone_name)
    healer_model.to(device)
    
    # Custom loss wrapper for healer model
    class HealerModelWithLoss(nn.Module):
        def __init__(self, healer_model, healer_loss):
            super().__init__()
            self.healer_model = healer_model
            self.healer_loss = healer_loss
        
        def forward(self, x):
            return self.healer_model(x)
    
    # Loss function
    healer_loss = HealerLoss()
    model_with_loss = HealerModelWithLoss(healer_model, healer_loss)
    
    # Find optimal batch size if not specified
    if batch_size is None:
        batch_size = 50  # Conservative for healer model
    
    # Define image transformations
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create continuous transforms for OOD
    ood_transform = ContinuousTransforms(severity=severity)
    
    # Dataset and DataLoader with OOD transforms
    train_dataset = TinyImageNetDataset(
        dataset_path, "train", transform_train, ood_transform=ood_transform
    )
    
    # Create a validation set (20% of training data)
    dataset_size = len(train_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"Training set size for healer: {train_size}")
    print(f"Validation set size for healer: {val_size}")
    
    # Custom trainer for healer model
    class HealerEarlyStoppingTrainer(EarlyStoppingTrainer):
        def __init__(self, *args, **kwargs):
            self.healer_loss_fn = kwargs.pop('healer_loss_fn')
            super().__init__(*args, **kwargs)
        
        def extract_transform_params(self, transform_params):
            """Extract transformation parameters from batch"""
            transform_type_mapping = {
                'no_transform': 0, 'no_transfrom': 0,
                'gaussian_noise': 1, 'rotation': 2, 'affine': 3
            }
            
            transform_types, severity_values = [], []
            noise_std_values, rotation_angle_values = [], []
            translate_x_values, translate_y_values = [], []
            shear_x_values, shear_y_values = [], []
            
            for params in transform_params:
                if isinstance(params, dict):
                    transform_type = params.get('transform_type', 'no_transform')
                    transform_types.append(transform_type_mapping.get(transform_type, 0))
                    severity_values.append(float(params.get('severity', 1.0)))
                    noise_std_values.append(float(params.get('noise_std', 0.0)))
                    rotation_angle_values.append(float(params.get('rotation_angle', 0.0)))
                    translate_x_values.append(float(params.get('translate_x', 0.0)))
                    translate_y_values.append(float(params.get('translate_y', 0.0)))
                    shear_x_values.append(float(params.get('shear_x', 0.0)))
                    shear_y_values.append(float(params.get('shear_y', 0.0)))
                else:
                    transform_types.append(0)
                    severity_values.append(1.0)
                    noise_std_values.extend([0.0])
                    rotation_angle_values.extend([0.0])
                    translate_x_values.extend([0.0])
                    translate_y_values.extend([0.0])
                    shear_x_values.extend([0.0])
                    shear_y_values.extend([0.0])
            
            targets = {
                'transform_type_idx': torch.tensor(transform_types),
                'severity': torch.tensor(severity_values).unsqueeze(1),
                'noise_std': torch.tensor(noise_std_values).unsqueeze(1),
                'rotation_angle': torch.tensor(rotation_angle_values).unsqueeze(1),
                'translate_x': torch.tensor(translate_x_values).unsqueeze(1),
                'translate_y': torch.tensor(translate_y_values).unsqueeze(1),
                'shear_x': torch.tensor(shear_x_values).unsqueeze(1),
                'shear_y': torch.tensor(shear_y_values).unsqueeze(1)
            }
            return targets
        
        def train_epoch(self, epoch: int, criterion=None) -> Tuple[float, float]:
            """Custom training epoch for healer model"""
            self.model.train()
            train_loss = 0.0
            transform_type_correct = 0
            total_samples = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.verbose)
            
            for batch_idx, (orig_images, transformed_images, labels, transform_params) in enumerate(progress_bar):
                orig_images = orig_images.to(self.device)
                transformed_images = transformed_images.to(self.device)
                
                # Extract and prepare target tensors
                targets = self.extract_transform_params(transform_params)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(transformed_images)
                
                # Calculate loss
                loss, loss_dict = self.healer_loss_fn(predictions, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                
                # Calculate transform type accuracy
                pred_types = torch.argmax(predictions['transform_type_logits'], dim=1)
                transform_type_correct += (pred_types == targets['transform_type_idx']).sum().item()
                total_samples += len(pred_types)
                
                # Update progress bar
                current_acc = transform_type_correct / total_samples if total_samples > 0 else 0
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{current_acc:.4f}"
                })
            
            train_loss /= len(self.train_loader)
            train_acc = transform_type_correct / total_samples if total_samples > 0 else 0
            
            return train_loss, train_acc
        
        def validate_epoch(self, epoch: int, criterion=None) -> Tuple[float, float]:
            """Custom validation epoch for healer model"""
            self.model.eval()
            val_loss = 0.0
            transform_type_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch_idx, (orig_images, transformed_images, labels, transform_params) in enumerate(self.val_loader):
                    orig_images = orig_images.to(self.device)
                    transformed_images = transformed_images.to(self.device)
                    
                    # Extract and prepare target tensors
                    targets = self.extract_transform_params(transform_params)
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                    
                    # Forward pass
                    predictions = self.model(transformed_images)
                    
                    # Calculate loss
                    loss, loss_dict = self.healer_loss_fn(predictions, targets)
                    
                    # Update metrics
                    val_loss += loss.item()
                    
                    # Calculate transform type accuracy
                    pred_types = torch.argmax(predictions['transform_type_logits'], dim=1)
                    transform_type_correct += (pred_types == targets['transform_type_idx']).sum().item()
                    total_samples += len(pred_types)
            
            val_loss /= len(self.val_loader)
            val_acc = transform_type_correct / total_samples if total_samples > 0 else 0
            
            return val_loss, val_acc
    
    # Collate function
    def collate_fn(batch):
        orig_imgs, trans_imgs, labels, params = zip(*batch)
        return torch.stack(orig_imgs), torch.stack(trans_imgs), torch.tensor(labels), params
    
    # DataLoaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=False, collate_fn=collate_fn
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(healer_model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Logging
    wandb.config.update({
        "model": "transformation_healer",
        "backbone": backbone_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "ood_severity": severity
    })
    
    # Enhanced early stopping trainer
    from early_stopping_trainer import get_early_stopping_config
    
    # Get early stopping configuration
    early_stop_config = get_early_stopping_config('healer', backbone_name)
    
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints_{backbone_name}_healer"
    best_model_dir = Path(f"bestmodel_{backbone_name}_healer")
    best_model_dir.mkdir(exist_ok=True)
    
    # Initialize enhanced trainer
    trainer = HealerEarlyStoppingTrainer(
        model=healer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=early_stop_config.patience,
        min_delta=early_stop_config.min_delta,
        restore_best_weights=early_stop_config.restore_best_weights,
        save_dir=checkpoint_dir,
        model_name=f"{backbone_name}_healer",
        cleanup_checkpoints=early_stop_config.cleanup_checkpoints,
        verbose=True,
        healer_loss_fn=healer_loss
    )
    
    # Train with early stopping
    results = trainer.train(
        epochs=epochs,
        criterion=None,  # We handle loss internally
        log_wandb=True,
        log_prefix="healer/"
    )
    
    # Save the final best model in the expected location
    if results['best_model_path']:
        best_model_path = best_model_dir / "best_model.pt"
        
        # Load the best checkpoint and save in expected format
        checkpoint = torch.load(results['best_model_path'], map_location='cpu')
        final_checkpoint = {
            'epoch': checkpoint['epoch'],
            'model_state_dict': checkpoint['model_state_dict'],
            'backbone_name': backbone_name,
            'val_loss': checkpoint['val_loss'],
            'val_acc': checkpoint['val_acc'],
            'training_results': results
        }
        
        torch.save(final_checkpoint, best_model_path)
        print(f"💾 Final model saved to: {best_model_path}")
    
    print(f"✅ Healer training completed!")
    print(f"   🏆 Best validation loss: {results['best_val_loss']:.4f}")
    print(f"   🎯 Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"   📊 Total epochs trained: {results['total_epochs']}")
    print(f"   ⏹️  Early stopped: {'Yes' if results['early_stopped'] else 'No'}")
    
    wandb.finish()
    return healer_model

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Train models with different backbones')
    parser.add_argument('--model_type', type=str, choices=['classification', 'healer'], 
                        default='classification', help='Type of model to train')
    parser.add_argument('--backbone', type=str, choices=list(BACKBONE_CONFIGS.keys()), 
                        default='vit_small', help='Backbone to use')
    parser.add_argument('--dataset_path', type=str, default='tiny-imagenet-200', 
                        help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--severity', type=float, default=1.0, 
                        help='Severity for healer model transformations')
    parser.add_argument('--experiment_name', type=str, default=None, 
                        help='Name for wandb experiment')
    
    args = parser.parse_args()
    
    print(f"Training {args.model_type} model with {args.backbone} backbone")
    print(f"Available backbones: {list(BACKBONE_CONFIGS.keys())}")
    
    if args.model_type == 'classification':
        model = train_classification_model(
            dataset_path=args.dataset_path,
            backbone_name=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            experiment_name=args.experiment_name
        )
    elif args.model_type == 'healer':
        model = train_healer_model(
            dataset_path=args.dataset_path,
            backbone_name=args.backbone,
            severity=args.severity,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            experiment_name=args.experiment_name
        )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
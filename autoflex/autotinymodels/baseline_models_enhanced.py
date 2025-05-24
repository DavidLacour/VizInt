import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import early stopping trainer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from early_stopping_trainer import EarlyStoppingTrainer

class SimpleResNet18(nn.Module):
    """
    ResNet18 adapted for Tiny ImageNet (64x64, 200 classes)
    This is a proven architecture that should give good baseline results
    """
    def __init__(self, num_classes=200):
        super(SimpleResNet18, self).__init__()
        # Use torchvision ResNet18 but modify for our input size and classes
        self.resnet = models.resnet18(pretrained=False)
        
        # Modify first conv layer for 64x64 input (instead of 224x224)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool since input is smaller
        
        # Modify final layer for 200 classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

class SimpleCNN(nn.Module):
    """
    Simple CNN baseline - should train quickly and give decent results
    """
    def __init__(self, num_classes=200):
        super(SimpleCNN, self).__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EfficientNetB0_Custom(nn.Module):
    """
    EfficientNet-B0 adapted for Tiny ImageNet
    Good balance of performance and efficiency
    """
    def __init__(self, num_classes=200):
        super(EfficientNetB0_Custom, self).__init__()
        from torchvision.models import efficientnet_b0
        
        self.efficientnet = efficientnet_b0(pretrained=False)
        
        # Modify for 200 classes
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, 
            num_classes
        )
        
    def forward(self, x):
        return self.efficientnet(x)

class SimpleVGG16(nn.Module):
    """
    VGG16 adapted for Tiny ImageNet (64x64, 200 classes)
    Classic architecture that should give good baseline results
    """
    def __init__(self, num_classes=200):
        super(SimpleVGG16, self).__init__()
        # Use torchvision VGG16 but modify for our input size and classes
        self.vgg = models.vgg16(pretrained=False)
        
        # VGG16 already works well with 64x64 input, but we need to adjust the classifier
        # The feature extractor will output 512 * 2 * 2 = 2048 features for 64x64 input
        self.vgg.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        return self.vgg(x)

def train_baseline_model_with_early_stopping(
    model, 
    dataset_path, 
    model_name="baseline", 
    epochs=50, 
    lr=0.001,
    patience=5,
    min_delta=1e-4,
    batch_size=128,
    num_workers=4,
    save_dir="./bestmodel",
    use_wandb=False,
    wandb_project=None,
    cleanup_checkpoints=True
):
    """
    Enhanced training function for baseline models with early stopping
    
    Args:
        model: The model to train
        dataset_path: Path to the dataset
        model_name: Name for saving the model
        epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
        min_delta: Minimum improvement delta
        batch_size: Batch size for training
        num_workers: Number of data loader workers
        save_dir: Directory to save models
        use_wandb: Whether to use wandb logging
        wandb_project: Wandb project name
        cleanup_checkpoints: Whether to cleanup intermediate checkpoints
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
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
    
    # Import dataset from new_new.py
    from new_new import TinyImageNetDataset
    
    train_dataset = TinyImageNetDataset(dataset_path, "train", transform_train)
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Create save directory
    save_path = Path(save_dir) / f"{model_name}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize early stopping trainer
    trainer = EarlyStoppingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        save_dir=str(save_path),
        model_name=model_name,
        cleanup_checkpoints=cleanup_checkpoints,
        verbose=True
    )
    
    print(f"\nTraining {model_name} with early stopping")
    print(f"Max epochs: {epochs}")
    print(f"Patience: {patience}")
    print(f"Min delta: {min_delta}")
    print(f"Save directory: {save_path}")
    
    # Train with early stopping
    results = trainer.train(
        epochs=epochs,
        criterion=criterion,
        log_wandb=use_wandb,
        log_prefix=f"{model_name}/"
    )
    
    # Save final best model in the expected format
    if results['best_model_path']:
        # Load best checkpoint
        checkpoint = torch.load(results['best_model_path'], map_location='cpu')
        
        # Save in the format expected by main_baselines.py
        final_save_path = Path(f"./bestmodel_{model_name}")
        final_save_path.mkdir(exist_ok=True)
        
        torch.save({
            'epoch': checkpoint['epoch'],
            'model_state_dict': checkpoint['model_state_dict'],
            'optimizer_state_dict': checkpoint['optimizer_state_dict'],
            'val_acc': checkpoint['val_acc'],
            'val_loss': checkpoint['val_loss'],
        }, final_save_path / "best_model.pt")
        
        print(f"\nBest model saved to: {final_save_path / 'best_model.pt'}")
        print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    
    return model, results

# Backward compatibility - keep the original function signature
def train_baseline_model(model, dataset_path, model_name="baseline", epochs=50, lr=0.001):
    """
    Original training function - now uses early stopping by default
    """
    return train_baseline_model_with_early_stopping(
        model=model,
        dataset_path=dataset_path,
        model_name=model_name,
        epochs=epochs,
        lr=lr,
        patience=5,  # Default patience
        min_delta=1e-4,  # Default min delta
        cleanup_checkpoints=True
    )[0]  # Return just the model for backward compatibility

# Example usage and expected performance
def get_expected_performance():
    """
    Expected performance on Tiny ImageNet for these baseline models
    (trained from scratch, no pretraining)
    """
    expected_results = {
        "SimpleResNet18": {
            "clean_accuracy": "~45-55%",
            "training_time": "~2-3 hours on GPU",
            "description": "Reliable baseline, well-tested architecture"
        },
        "SimpleCNN": {
            "clean_accuracy": "~35-45%", 
            "training_time": "~1-2 hours on GPU",
            "description": "Fast to train, good for quick experiments"
        },
        "EfficientNetB0": {
            "clean_accuracy": "~50-60%",
            "training_time": "~3-4 hours on GPU", 
            "description": "Best performance, but slower to train"
        },
        "SimpleVGG16": {
            "clean_accuracy": "~40-50%",
            "training_time": "~2-3 hours on GPU",
            "description": "Classic architecture, good baseline"
        }
    }
    return expected_results

# Training script example
if __name__ == "__main__":
    # Choose your baseline model
    print("Available baseline models:")
    for name, info in get_expected_performance().items():
        print(f"  {name}: {info['description']} - Expected: {info['clean_accuracy']}")
    
    # Example: Train ResNet18 baseline with early stopping
    print("\nTraining ResNet18 baseline with early stopping...")
    model = SimpleResNet18(num_classes=200)
    
    # Train the model with early stopping
    # trained_model, results = train_baseline_model_with_early_stopping(
    #     model, 
    #     "path/to/tiny-imagenet-200", 
    #     "resnet18_baseline",
    #     patience=5,
    #     min_delta=1e-4
    # )
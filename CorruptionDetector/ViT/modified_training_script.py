import os
import re
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

# Import our custom ViT model
from transformer_utils import set_seed
from vit_implementation import create_vit_model

# Initialize wandb
wandb.init(project="vit-hybrid-small-imagenetc", name="custom-vit-resnet")

class HybridImagenetCDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.root_dir) if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Extract class from filename using regex
        # Format: ILSVRC2012_val_00000057_class118_pixelate_intensity3.JPEG
        match = re.search(r'class(\d+)', img_name)
        if match:
            # Class indices are 0-based for the model
            class_id = int(match.group(1)) - 1  # Subtract 1 if your classes start from 1
        else:
            class_id = 0  # Default class if pattern not found
        
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, class_id


# Custom loss wrapper for compatibility with the training script
class CustomViTWithLoss(torch.nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, pixel_values, labels=None):
        # Forward pass through ViT model
        logits = self.vit_model(pixel_values)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return type('OutputsWithLoss', (), {
            'loss': loss,
            'logits': logits
        })


def train():
    # Set seed for reproducibility
    set_seed(42)
    
    # Create checkpoint directories
    checkpoints_dir = Path("checkpoints")
    best_model_dir = Path("bestmodel")
    
    # Create directories if they don't exist
    checkpoints_dir.mkdir(exist_ok=True)
    best_model_dir.mkdir(exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize our custom model
    vit_model = create_vit_model(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        head_dim=64,
        mlp_ratio=4.0,
        use_resnet_stem=True
    )
    
    # Wrap it with loss calculation for compatibility
    model = CustomViTWithLoss(vit_model)
    model.to(device)
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset and DataLoader
    dataset_path = "hybrid_small_imagenetc"  # Path to your dataset folder
    train_dataset = HybridImagenetCDataset(dataset_path, "train", transform)
    val_dataset = HybridImagenetCDataset(dataset_path, "val", transform)
    test_dataset = HybridImagenetCDataset(dataset_path, "test", transform)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Training parameters
    num_epochs = 30
    learning_rate = 2e-5
    warmup_steps = 500
    patience = 5  # Early stopping patience
    
    # Optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (custom implementation of linear warmup)
    def get_lr(step, total_steps, warmup_steps):
        if step < warmup_steps:
            return learning_rate * (step / warmup_steps)
        return learning_rate * (1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: get_lr(step, total_steps, warmup_steps) / learning_rate
    )
    
    # Logging with wandb
    wandb.config.update({
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "model": "custom-vit-resnet",
        "dataset": "hybrid_small_imagenetc"
    })
    
    # Initialize early stopping variables
    best_val_acc = 0
    early_stop_counter = 0
    best_epoch = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        # Training step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Validation step
        val_loss, val_acc = validate(model, val_loader, device)
        
        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            
            # Save checkpoint
            checkpoint_path = checkpoints_dir / f"model_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
            # Save best model
            best_model_path = best_model_dir / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            
            # Log to wandb
            wandb.save(str(checkpoint_path))
            wandb.save(str(best_model_path))
            
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
            
            # Track the best epoch for later reference
            best_epoch = epoch + 1
        else:
            early_stop_counter += 1
            
            # Still save a checkpoint but don't mark as best
            checkpoint_path = checkpoints_dir / f"model_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Clean up old checkpoints except the best one
    print("Cleaning up checkpoints to save disk space...")
    for checkpoint_file in checkpoints_dir.glob("*.pt"):
        if f"model_epoch{best_epoch}.pt" != checkpoint_file.name:
            checkpoint_file.unlink()
            print(f"Deleted {checkpoint_file}")
    
    # Test the best model
    model.load_state_dict(torch.load(best_model_dir / "best_model.pt"))
    test_loss, test_acc = validate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Log final test metrics and artifacts
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_model": wandb.Artifact(
            "best_model", type="model",
            description=f"Best model with validation accuracy {best_val_acc:.4f}"
        )
    })
    
    print(f"Training completed. Best model saved at: {best_model_dir / 'best_model.pt'}")
    print(f"Final test accuracy: {test_acc:.4f}")
    
    wandb.finish()

def validate(model, loader, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            
            # Update metrics
            val_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    val_loss /= len(loader)
    val_acc = accuracy_score(all_labels, all_preds)
    
    return val_loss, val_acc

if __name__ == "__main__":
    train()

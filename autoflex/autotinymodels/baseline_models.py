import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

def train_baseline_model(model, dataset_path, model_name="baseline", epochs=50, lr=0.001):
    """
    Training function for baseline models
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import os
    from tqdm import tqdm
    
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
    
    # Assuming you have TinyImageNetDataset from your existing code
    # You'll need to import this from your new_new.py file
    from new_new import TinyImageNetDataset
    
    train_dataset = TinyImageNetDataset(dataset_path, "train", transform_train)
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    best_val_acc = 0.0
    
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
            os.makedirs(f"./bestmodel_{model_name}", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f"./bestmodel_{model_name}/best_model.pt")
            print(f"  New best model saved with val_acc: {val_acc:.4f}")
        
        scheduler.step()
        print()
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return model

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
        }
    }
    return expected_results

# Training script example
if __name__ == "__main__":
    # Choose your baseline model
    print("Available baseline models:")
    for name, info in get_expected_performance().items():
        print(f"  {name}: {info['description']} - Expected: {info['clean_accuracy']}")
    
    # Train ResNet18 baseline (recommended)
    print("\nTraining ResNet18 baseline...")
    model = SimpleResNet18(num_classes=200)
    
    # Train the model (you'll need to provide your dataset path)
    # trained_model = train_baseline_model(model, "path/to/tiny-imagenet-200", "resnet18_baseline")
    
    # You can then use this trained model in your evaluation pipeline
    # by loading it the same way you load your other models
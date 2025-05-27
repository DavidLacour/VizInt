import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import necessary modules
from vit_implementation import create_vit_model
from transformer_utils import set_seed
from ttt3fc_model import TestTimeTrainer3fc
from blended_ttt3fc_model import BlendedTTT3fc
from blended_ttt_model import BlendedTTT
from ttt_model import TestTimeTrainer

def get_cifar10_loaders(batch_size=32, num_workers=4):
    """Get CIFAR-10 data loaders"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download CIFAR-10 if needed
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def create_cifar10_vit(device):
    """Create a smaller ViT for CIFAR-10"""
    model = create_vit_model(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        head_dim=64,
        mlp_ratio=4.0,
        use_resnet_stem=False
    )
    return model.to(device)

def train_main_model_cifar10(device, epochs=20):
    """Train main model on CIFAR-10"""
    print("\n=== Training Main Model on CIFAR-10 ===")
    
    model = create_cifar10_vit(device)
    train_loader, test_loader = get_cifar10_loaders()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100. * train_correct / train_total
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': test_acc
            }, 'cifar10_main_model.pt')
        
        scheduler.step()
    
    return model, best_acc

def train_robust_model_cifar10(device, epochs=20):
    """Train robust model on CIFAR-10 with augmentations"""
    print("\n=== Training Robust Model on CIFAR-10 ===")
    
    model = create_cifar10_vit(device)
    
    # Enhanced augmentations for robust training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = CIFAR10(root='./data', train=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    _, test_loader = get_cifar10_loaders()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train Robust]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f"Epoch {epoch+1}: Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': test_acc
            }, 'cifar10_robust_model.pt')
        
        scheduler.step()
    
    return model, best_acc

def train_ttt3fc_cifar10(main_model, device, epochs=10):
    """Train TTT3fc model on CIFAR-10"""
    print("\n=== Training TTT3fc Model on CIFAR-10 ===")
    
    # Create TTT3fc model
    ttt3fc_model = TestTimeTrainer3fc(
        base_model=main_model,
        img_size=32,
        patch_size=4,
        embed_dim=256,
        num_classes=10,
        transform_types=4  # no_transform, rotation, color_jitter, gaussian_noise
    ).to(device)
    
    # Create dataset with transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = CIFAR10(root='./data', train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Train only the transform predictor
    optimizer = optim.AdamW(ttt3fc_model.transform_predictor.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(epochs):
        ttt3fc_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TTT3fc]")
        for images, labels in pbar:
            images = images.to(device)
            batch_size = images.size(0)
            
            # Apply different transformations
            transformed_images = []
            transform_labels = []
            
            for i in range(batch_size):
                img = images[i]
                transform_type = np.random.randint(0, 4)
                
                if transform_type == 0:
                    # No transform
                    transformed_img = img
                elif transform_type == 1:
                    # Rotation
                    angle = np.random.uniform(-30, 30)
                    transformed_img = transforms.functional.rotate(img, angle)
                elif transform_type == 2:
                    # Color jitter
                    jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
                    transformed_img = jitter(img)
                else:
                    # Gaussian noise
                    noise = torch.randn_like(img) * 0.1
                    transformed_img = img + noise
                
                transformed_images.append(transformed_img)
                transform_labels.append(transform_type)
            
            transformed_images = torch.stack(transformed_images)
            transform_labels = torch.tensor(transform_labels, device=device)
            
            optimizer.zero_grad()
            
            # Forward through transform predictor
            features = ttt3fc_model.base_model.forward_features(transformed_images)
            transform_pred = ttt3fc_model.transform_predictor(features)
            
            loss = criterion(transform_pred, transform_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = transform_pred.max(1)
            correct += predicted.eq(transform_labels).sum().item()
            total += transform_labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Transform Prediction Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state_dict': ttt3fc_model.state_dict(),
                'accuracy': acc
            }, 'cifar10_ttt3fc_model.pt')
    
    return ttt3fc_model

def evaluate_all_models_cifar10(device):
    """Evaluate all model combinations on CIFAR-10"""
    print("\n=== Evaluating All Model Combinations on CIFAR-10 ===")
    
    _, test_loader = get_cifar10_loaders()
    results = {}
    
    # Load models
    print("Loading models...")
    
    # Main models
    main_model = create_cifar10_vit(device)
    if os.path.exists('cifar10_main_model.pt'):
        checkpoint = torch.load('cifar10_main_model.pt', map_location=device)
        main_model.load_state_dict(checkpoint['model_state_dict'])
    main_model.eval()
    
    robust_model = create_cifar10_vit(device)
    if os.path.exists('cifar10_robust_model.pt'):
        checkpoint = torch.load('cifar10_robust_model.pt', map_location=device)
        robust_model.load_state_dict(checkpoint['model_state_dict'])
    robust_model.eval()
    
    # TTT3fc models
    ttt3fc_main = None
    ttt3fc_robust = None
    
    if os.path.exists('cifar10_ttt3fc_model.pt'):
        ttt3fc_main = TestTimeTrainer3fc(main_model, img_size=32, patch_size=4, embed_dim=256, num_classes=10)
        checkpoint = torch.load('cifar10_ttt3fc_model.pt', map_location=device)
        ttt3fc_main.load_state_dict(checkpoint['model_state_dict'])
        ttt3fc_main = ttt3fc_main.to(device)
        ttt3fc_main.eval()
        
        # Create TTT3fc with robust model
        ttt3fc_robust = TestTimeTrainer3fc(robust_model, img_size=32, patch_size=4, embed_dim=256, num_classes=10)
        # Load only transform predictor weights
        ttt3fc_robust.transform_predictor.load_state_dict(ttt3fc_main.transform_predictor.state_dict())
        ttt3fc_robust = ttt3fc_robust.to(device)
        ttt3fc_robust.eval()
    
    # Evaluate each model
    def evaluate_model(model, name):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Evaluating {name}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = 100. * correct / total
        results[name] = accuracy
        print(f"{name}: {accuracy:.2f}%")
        return accuracy
    
    # 1. Evaluate main models
    evaluate_model(main_model, "Main Model")
    evaluate_model(robust_model, "Robust Main Model")
    
    # 2. Evaluate TTT3fc combinations
    if ttt3fc_main:
        # TTT3fc + Main
        def evaluate_ttt3fc(ttt_model, name):
            correct = 0
            total = 0
            for images, labels in tqdm(test_loader, desc=f"Evaluating {name}"):
                images, labels = images.to(device), labels.to(device)
                # Adapt and classify
                adapted_logits = ttt_model.adapt(images, None, reset=True, adapt_classification=True)
                _, predicted = adapted_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            accuracy = 100. * correct / total
            results[name] = accuracy
            print(f"{name}: {accuracy:.2f}%")
            return accuracy
        
        evaluate_ttt3fc(ttt3fc_main, "TTT3fc + Main")
        evaluate_ttt3fc(ttt3fc_robust, "TTT3fc + Robust")
    
    # Note: For CIFAR-10, we're focusing on main models and TTT3fc
    # Additional models (healer, BlendedTTT, etc.) would need to be trained separately
    
    # Print summary
    print("\n=== EVALUATION SUMMARY (CIFAR-10) ===")
    print("-" * 40)
    for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:<25}: {accuracy:>6.2f}%")
    print("-" * 40)
    
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(42)
    
    # Train models if they don't exist
    if not os.path.exists('cifar10_main_model.pt'):
        main_model, main_acc = train_main_model_cifar10(device, epochs=20)
        print(f"Main model best accuracy: {main_acc:.2f}%")
    else:
        print("Main model already exists, loading...")
        main_model = create_cifar10_vit(device)
        checkpoint = torch.load('cifar10_main_model.pt', map_location=device)
        main_model.load_state_dict(checkpoint['model_state_dict'])
    
    if not os.path.exists('cifar10_robust_model.pt'):
        robust_model, robust_acc = train_robust_model_cifar10(device, epochs=20)
        print(f"Robust model best accuracy: {robust_acc:.2f}%")
    
    # Train TTT3fc
    if not os.path.exists('cifar10_ttt3fc_model.pt'):
        ttt3fc_model = train_ttt3fc_cifar10(main_model, device, epochs=10)
        print("TTT3fc model training completed")
    
    # Evaluate all models
    results = evaluate_all_models_cifar10(device)
    
    # Save results
    import json
    with open('cifar10_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to cifar10_evaluation_results.json")

if __name__ == "__main__":
    main()
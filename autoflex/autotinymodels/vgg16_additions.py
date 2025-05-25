"""
VGG16 additions for main_cifar10_all.py
"""

# Add this function to main_cifar10_all.py

def train_vgg16_baseline(train_loader, val_loader, pretrained=False):
    """Train VGG16 baseline on CIFAR-10"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if pretrained:
        # Use pretrained VGG16 and adapt for CIFAR-10
        import torchvision.models as models
        model = models.vgg16(pretrained=True)
        
        # Modify classifier for CIFAR-10 (10 classes)
        # VGG16 expects 224x224, but CIFAR-10 is 32x32
        # So we need to adjust the classifier
        num_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(512, 256),  # Reduced from original due to smaller input
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, NUM_CLASSES),
        )
        model_name = "vgg16_pretrained"
    else:
        # Use custom SimpleVGG16 for CIFAR-10
        from baseline_models import SimpleVGG16
        model = SimpleVGG16(num_classes=NUM_CLASSES)
        model_name = "vgg16_baseline"
    
    model = model.to(device)
    
    # Training configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training loop with early stopping
    epochs = 100
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    print(f"🏋️ Training {model_name} on CIFAR-10...")
    
    for epoch in range(epochs):
        # Training phase
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
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100 * train_correct / train_total
        
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
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            save_path = os.path.join(CHECKPOINT_PATH, f"bestmodel_{model_name}")
            os.makedirs(save_path, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, os.path.join(save_path, "best_model.pt"))
            
            print(f"💾 Saved best model with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
                break
        
        scheduler.step()
    
    print(f"✅ Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return model, best_val_acc


# Add to the evaluation function (in evaluate_all_models):
"""
# Add these to model_configs list:
("VGG16 Baseline", "bestmodel_vgg16_baseline", "vgg"),
("VGG16 Pretrained", "bestmodel_vgg16_pretrained", "vgg"),

# Add this to the model loading section:
elif model_type == "vgg":
    if "pretrained" in model_dir:
        import torchvision.models as models
        model = models.vgg16(pretrained=False)
        num_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, NUM_CLASSES),
        )
    else:
        from baseline_models import SimpleVGG16
        model = SimpleVGG16(num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
"""

# Add to the training section in main():
"""
if not args.retrain and not args.skip_training and (args.train_all or args.train_baselines):
    # VGG16 models
    vgg16_path = os.path.join(CHECKPOINT_PATH, "bestmodel_vgg16_baseline", "best_model.pt")
    vgg16_pretrained_path = os.path.join(CHECKPOINT_PATH, "bestmodel_vgg16_pretrained", "best_model.pt")
    
    if os.path.exists(vgg16_path):
        print(f"\n✓ VGG16 baseline already exists at {vgg16_path}")
    else:
        print("\n=== TRAINING VGG16 BASELINE ===")
        train_vgg16_baseline(train_loader, val_loader, pretrained=False)
        
    if os.path.exists(vgg16_pretrained_path):
        print(f"\n✓ Pretrained VGG16 baseline already exists at {vgg16_pretrained_path}")
    else:
        print("\n=== TRAINING VGG16 BASELINE (pretrained) ===")
        train_vgg16_baseline(train_loader, val_loader, pretrained=True)
"""
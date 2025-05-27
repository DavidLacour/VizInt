"""
VGG16 additions for main_baselines_3fc_integration.py (Tiny ImageNet)
"""

# Add this function to main_baselines_3fc_integration.py

def train_vgg16_baseline(dataset_path, pretrained=False):
    """Train VGG16 baseline on Tiny ImageNet"""
    from baseline_models import SimpleVGG16, train_baseline_model
    
    if pretrained:
        # For pretrained VGG16 on Tiny ImageNet
        import torchvision.models as models
        model = models.vgg16(pretrained=True)
        
        # Adapt for Tiny ImageNet (200 classes, 64x64 images)
        # Modify the classifier
        model.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),  # Adjusted for 64x64 input
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 200),  # 200 classes for Tiny ImageNet
        )
        model_name = "vgg16_pretrained"
    else:
        # Use custom SimpleVGG16
        model_name = "vgg16_baseline"
        trained_model = train_baseline_model(
            dataset_path,
            model_name=model_name,
            model_class=SimpleVGG16,
            num_epochs=50,
            batch_size=128,
            learning_rate=0.01
        )
        return trained_model
    
    # For pretrained model, use train_baseline_model_with_early_stopping if available
    if pretrained:
        try:
            from baseline_models_enhanced import train_baseline_model_with_early_stopping
            trained_model = train_baseline_model_with_early_stopping(
                dataset_path,
                model_name=model_name,
                model=model,
                num_epochs=50,
                batch_size=128,
                learning_rate=0.001  # Lower LR for fine-tuning
            )
        except:
            # Fallback to regular training
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # Manual training loop here if needed
            print("Warning: Using basic training for pretrained VGG16")
            
    return trained_model


# Add to train_all_models_if_missing function:
"""
# 5. Train VGG16 Baselines
vgg16_baseline_path = f"{model_dir}/bestmodel_vgg16_baseline/best_model.pt"
if not os.path.exists(vgg16_baseline_path):
    print("\n🔧 Training VGG16 Baseline Model...")
    vgg16_model = train_vgg16_baseline(dataset_path, pretrained=False)
    models['vgg16_baseline'] = vgg16_model
else:
    print("✅ VGG16 baseline model already exists")
    from baseline_models import SimpleVGG16
    vgg16_model = SimpleVGG16(num_classes=200)
    checkpoint = torch.load(vgg16_baseline_path, map_location=device)
    vgg16_model.load_state_dict(checkpoint['model_state_dict'])
    vgg16_model = vgg16_model.to(device)
    models['vgg16_baseline'] = vgg16_model

vgg16_pretrained_path = f"{model_dir}/bestmodel_vgg16_pretrained/best_model.pt"
if not os.path.exists(vgg16_pretrained_path):
    print("\n🔧 Training VGG16 Pretrained Model...")
    vgg16_pretrained = train_vgg16_baseline(dataset_path, pretrained=True)
    models['vgg16_pretrained'] = vgg16_pretrained
else:
    print("✅ VGG16 pretrained model already exists")
    import torchvision.models as models
    vgg16_pretrained = models.vgg16(pretrained=False)
    vgg16_pretrained.classifier = nn.Sequential(
        nn.Linear(512 * 2 * 2, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 200),
    )
    checkpoint = torch.load(vgg16_pretrained_path, map_location=device)
    vgg16_pretrained.load_state_dict(checkpoint['model_state_dict'])
    vgg16_pretrained = vgg16_pretrained.to(device)
    models['vgg16_pretrained'] = vgg16_pretrained
"""

# Add to evaluation section:
"""
# Add VGG16 baselines to evaluation
if 'vgg16_baseline' in models:
    print("\n📊 Evaluating VGG16 Baseline...")
    vgg16_acc = evaluate_model(
        models['vgg16_baseline'],
        val_loader,
        device,
        criterion
    )
    results['VGG16 Baseline'] = vgg16_acc
    print(f"VGG16 Baseline Accuracy: {vgg16_acc:.2f}%")

if 'vgg16_pretrained' in models:
    print("\n📊 Evaluating VGG16 Pretrained...")
    vgg16_pre_acc = evaluate_model(
        models['vgg16_pretrained'],
        val_loader,
        device,
        criterion
    )
    results['VGG16 Pretrained'] = vgg16_pre_acc
    print(f"VGG16 Pretrained Accuracy: {vgg16_pre_acc:.2f}%")
"""
# Enhanced evaluation function to add to main_cifar10_all.py

def evaluate_all_models_comprehensive(val_loader):
    """Evaluate all trained models and combinations on CIFAR-10"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE MODEL EVALUATION ON CIFAR-10")
    print("="*80)
    
    # Load base models first
    models = {}
    
    # 1. Load Main ViT
    main_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_main", "best_model.pt")
    if os.path.exists(main_model_path):
        main_model = create_vit_model(
            img_size=IMG_SIZE, patch_size=4, in_chans=3, num_classes=NUM_CLASSES,
            embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
        )
        checkpoint = torch.load(main_model_path, map_location=device)
        main_model.load_state_dict(checkpoint['model_state_dict'])
        main_model = main_model.to(device)
        main_model.eval()
        models['main'] = main_model
        print("✅ Loaded Main ViT model")
    
    # 2. Load Robust ViT
    robust_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_robust", "best_model.pt")
    if os.path.exists(robust_model_path):
        robust_model = create_vit_model(
            img_size=IMG_SIZE, patch_size=4, in_chans=3, num_classes=NUM_CLASSES,
            embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
        )
        checkpoint = torch.load(robust_model_path, map_location=device)
        robust_model.load_state_dict(checkpoint['model_state_dict'])
        robust_model = robust_model.to(device)
        robust_model.eval()
        models['robust'] = robust_model
        print("✅ Loaded Robust ViT model")
    
    # 3. Load Healer model (need to train this for CIFAR-10 if not exists)
    healer_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_healer", "best_model.pt")
    if os.path.exists(healer_model_path):
        from new_new import TransformationHealer
        healer_model = TransformationHealer(
            img_size=IMG_SIZE, patch_size=4, in_chans=3,
            embed_dim=384, depth=6, head_dim=64
        )
        checkpoint = torch.load(healer_model_path, map_location=device)
        healer_model.load_state_dict(checkpoint['model_state_dict'])
        healer_model = healer_model.to(device)
        healer_model.eval()
        models['healer'] = healer_model
        print("✅ Loaded Healer model")
    
    # 4. Load TTT models
    ttt_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_ttt", "best_model.pt")
    if os.path.exists(ttt_model_path) and 'main' in models:
        ttt_model = TestTimeTrainer(base_model=models['main'], img_size=IMG_SIZE, patch_size=4, embed_dim=384)
        checkpoint = torch.load(ttt_model_path, map_location=device)
        ttt_model.load_state_dict(checkpoint['model_state_dict'])
        ttt_model = ttt_model.to(device)
        ttt_model.eval()
        models['ttt'] = ttt_model
        print("✅ Loaded TTT model")
    
    # 5. Load TTT3fc model
    ttt3fc_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_ttt3fc", "best_model.pt")
    if os.path.exists(ttt3fc_model_path) and 'main' in models:
        ttt3fc_model = TestTimeTrainer3fc(base_model=models['main'], img_size=IMG_SIZE, patch_size=4, embed_dim=384)
        checkpoint = torch.load(ttt3fc_model_path, map_location=device)
        ttt3fc_model.load_state_dict(checkpoint['model_state_dict'])
        ttt3fc_model = ttt3fc_model.to(device)
        ttt3fc_model.eval()
        models['ttt3fc'] = ttt3fc_model
        print("✅ Loaded TTT3fc model")
    
    # 6. Load Blended models
    blended_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_blended", "best_model.pt")
    if os.path.exists(blended_model_path):
        blended_model = BlendedTTT(img_size=IMG_SIZE, patch_size=4, embed_dim=384, depth=8, num_classes=NUM_CLASSES)
        checkpoint = torch.load(blended_model_path, map_location=device)
        blended_model.load_state_dict(checkpoint['model_state_dict'])
        blended_model = blended_model.to(device)
        blended_model.eval()
        models['blended'] = blended_model
        print("✅ Loaded BlendedTTT model")
    
    blended3fc_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_blended3fc", "best_model.pt")
    if os.path.exists(blended3fc_model_path):
        blended3fc_model = BlendedTTT3fc(img_size=IMG_SIZE, patch_size=4, embed_dim=384, depth=8, num_classes=NUM_CLASSES)
        checkpoint = torch.load(blended3fc_model_path, map_location=device)
        blended3fc_model.load_state_dict(checkpoint['model_state_dict'])
        blended3fc_model = blended3fc_model.to(device)
        blended3fc_model.eval()
        models['blended3fc'] = blended3fc_model
        print("✅ Loaded BlendedTTT3fc model")
    
    print("\n" + "-"*80)
    print("EVALUATING MODEL COMBINATIONS")
    print("-"*80)
    
    # Evaluation function for standard models
    def evaluate_standard_model(model, name):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Evaluating {name}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        results[name] = accuracy
        print(f"✅ {name}: {accuracy:.4f}")
        return accuracy
    
    # Evaluation function for healer combinations
    def evaluate_healer_combo(main_model, healer_model, name):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Evaluating {name}"):
                images, labels = images.to(device), labels.to(device)
                # Apply healer
                healer_output = healer_model(images)
                healed_images = healer_model.apply_correction(images, healer_output)
                # Classify with main model
                outputs = main_model(healed_images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        results[name] = accuracy
        print(f"✅ {name}: {accuracy:.4f}")
        return accuracy
    
    # Evaluation function for TTT combinations
    def evaluate_ttt_combo(ttt_model, name):
        correct = 0
        total = 0
        for images, labels in tqdm(val_loader, desc=f"Evaluating {name}"):
            images, labels = images.to(device), labels.to(device)
            # Adapt and classify
            adapted_logits = ttt_model.adapt(images, None, reset=True, adapt_classification=True)
            _, predicted = torch.max(adapted_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        results[name] = accuracy
        print(f"✅ {name}: {accuracy:.4f}")
        return accuracy
    
    # 1. Evaluate standalone models
    if 'main' in models:
        evaluate_standard_model(models['main'], "Main Model")
    
    if 'robust' in models:
        evaluate_standard_model(models['robust'], "Robust Main Model")
    
    if 'blended' in models:
        evaluate_standard_model(models['blended'], "BlendedTTT")
    
    if 'blended3fc' in models:
        evaluate_standard_model(models['blended3fc'], "BlendedTTT3fc")
    
    # 2. Evaluate Healer combinations
    if 'healer' in models:
        if 'main' in models:
            evaluate_healer_combo(models['main'], models['healer'], "Healer + Main")
        
        if 'robust' in models:
            evaluate_healer_combo(models['robust'], models['healer'], "Healer + Robust")
    
    # 3. Evaluate TTT combinations
    if 'ttt' in models:
        evaluate_ttt_combo(models['ttt'], "TTT + Main")
        
        # Create TTT with robust base
        if 'robust' in models:
            ttt_robust = TestTimeTrainer(base_model=models['robust'], img_size=IMG_SIZE, patch_size=4, embed_dim=384)
            # Copy transform predictor weights from trained TTT
            ttt_robust.transform_predictor.load_state_dict(models['ttt'].transform_predictor.state_dict())
            ttt_robust = ttt_robust.to(device)
            ttt_robust.eval()
            evaluate_ttt_combo(ttt_robust, "TTT + Robust")
    
    # 4. Evaluate TTT3fc combinations
    if 'ttt3fc' in models:
        evaluate_ttt_combo(models['ttt3fc'], "TTT3fc + Main")
        
        # Create TTT3fc with robust base
        if 'robust' in models:
            ttt3fc_robust = TestTimeTrainer3fc(base_model=models['robust'], img_size=IMG_SIZE, patch_size=4, embed_dim=384)
            # Copy transform predictor weights from trained TTT3fc
            ttt3fc_robust.transform_predictor.load_state_dict(models['ttt3fc'].transform_predictor.state_dict())
            ttt3fc_robust = ttt3fc_robust.to(device)
            ttt3fc_robust.eval()
            evaluate_ttt_combo(ttt3fc_robust, "TTT3fc + Robust")
    
    # Also evaluate ResNet baselines if they exist
    resnet_configs = [
        ("ResNet18 Baseline", "bestmodel_resnet18_baseline"),
        ("ResNet18 Pretrained", "bestmodel_resnet18_pretrained"),
    ]
    
    for name, model_dir in resnet_configs:
        model_path = os.path.join(CHECKPOINT_PATH, model_dir, "best_model.pt")
        if os.path.exists(model_path):
            if "pretrained" in model_dir:
                import torchvision.models as models
                model = models.resnet18(pretrained=False)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
                model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            else:
                model = SimpleResNet18(num_classes=NUM_CLASSES)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            evaluate_standard_model(model, name)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model':<30} {'Accuracy':>10}")
    print("-" * 45)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, acc) in enumerate(sorted_results):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        print(f"{medal} {name:<27} {acc:>10.4f}")
    
    return results
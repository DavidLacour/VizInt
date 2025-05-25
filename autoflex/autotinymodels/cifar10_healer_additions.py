"""
Additional functions to add to main_cifar10_all.py for healer support and comprehensive evaluation
"""

def train_healer_model_cifar10(train_loader, val_loader):
    """Train healer model for CIFAR-10"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*80)
    print("🏥 TRAINING HEALER MODEL FOR CIFAR-10")
    print("="*80)
    
    # Import healer model
    from new_new import TransformationHealer
    
    # Create healer model
    healer_model = TransformationHealer(
        img_size=IMG_SIZE,
        patch_size=4,
        in_chans=3,
        embed_dim=384,
        depth=6,
        head_dim=64
    ).to(device)
    
    # Training parameters
    optimizer = optim.AdamW(healer_model.parameters(), lr=0.001, weight_decay=0.05)
    criterion = nn.MSELoss()  # For reconstruction
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training loop
    epochs = 30
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        healer_model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, _ in pbar:
            images = images.to(device)
            
            # Apply random transformations
            batch_size = images.size(0)
            transformed_images = []
            
            for i in range(batch_size):
                img = images[i]
                # Random transformation
                if np.random.rand() > 0.5:
                    # Add noise
                    noise = torch.randn_like(img) * 0.1
                    transformed_img = img + noise
                else:
                    # Random rotation
                    angle = np.random.uniform(-15, 15)
                    transformed_img = transforms.functional.rotate(img, angle)
                transformed_images.append(transformed_img)
            
            transformed_images = torch.stack(transformed_images)
            
            optimizer.zero_grad()
            
            # Forward pass
            healer_output = healer_model(transformed_images)
            healed_images = healer_model.apply_correction(transformed_images, healer_output)
            
            # Loss: reconstruction loss
            loss = criterion(healed_images, images)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation phase
        healer_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                
                # Apply transformations
                batch_size = images.size(0)
                transformed_images = []
                
                for i in range(batch_size):
                    img = images[i]
                    if np.random.rand() > 0.5:
                        noise = torch.randn_like(img) * 0.1
                        transformed_img = img + noise
                    else:
                        angle = np.random.uniform(-15, 15)
                        transformed_img = transforms.functional.rotate(img, angle)
                    transformed_images.append(transformed_img)
                
                transformed_images = torch.stack(transformed_images)
                
                healer_output = healer_model(transformed_images)
                healed_images = healer_model.apply_correction(transformed_images, healer_output)
                loss = criterion(healed_images, images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(CHECKPOINT_PATH, "bestmodel_healer")
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'model_state_dict': healer_model.state_dict(),
                'val_loss': best_val_loss,
                'epoch': epoch
            }, os.path.join(save_path, "best_model.pt"))
            print(f"💾 Saved best model with val loss: {best_val_loss:.4f}")
        
        scheduler.step()
    
    print(f"✅ Healer training completed. Best val loss: {best_val_loss:.4f}")
    return healer_model


# Add to the main() function's training section:
"""
if not args.retrain and not args.skip_training and (args.train_all or args.train_healer):
    healer_model_path = os.path.join(CHECKPOINT_PATH, "bestmodel_healer", "best_model.pt")
    
    if os.path.exists(healer_model_path):
        print(f"\n✓ Healer model already exists at {healer_model_path}")
    else:
        print("\n=== TRAINING HEALER MODEL ===")
        train_healer_model_cifar10(train_loader, val_loader)
"""

# Add --train_healer argument to argparse:
"""
parser.add_argument("--train_healer", action="store_true", help="Train Healer model")
"""

# Replace the evaluate_all_models function with the comprehensive version from evaluate_all_models_enhanced.py
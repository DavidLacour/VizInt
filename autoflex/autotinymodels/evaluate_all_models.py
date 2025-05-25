import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import necessary modules
from new_new import TinyImageNetDataset, ContinuousTransforms
from vit_implementation import create_vit_model
from transformer_utils import set_seed
# from robust_training import load_robust_model  # Not needed, using same loader
from blended_ttt_model import BlendedTTT
from blended_ttt3fc_model import BlendedTTT3fc
from ttt_model import TestTimeTrainer
from ttt3fc_model import TestTimeTrainer3fc
from ttt3fc_blended3fc_evaluation import load_blended3fc_model, load_ttt3fc_model

def load_main_model(model_path, device):
    """Load the main classification model"""
    print(f"Loading main model from {model_path}")
    main_model = create_vit_model(
        img_size=64, patch_size=8, in_chans=3, num_classes=200,
        embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
    )
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle state dict keys
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith("vit_model."):
            new_key = key[len("vit_model."):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    main_model.load_state_dict(new_state_dict)
    main_model = main_model.to(device)
    main_model.eval()
    return main_model

def load_healer_model(model_path, device):
    """Load the healer model"""
    print(f"Loading healer model from {model_path}")
    from new_new import TransformationHealer
    
    healer_model = TransformationHealer(
        img_size=64, patch_size=8, in_chans=3,
        embed_dim=384, depth=6, head_dim=64
    )
    checkpoint = torch.load(model_path, map_location=device)
    healer_model.load_state_dict(checkpoint['model_state_dict'])
    healer_model = healer_model.to(device)
    healer_model.eval()
    return healer_model

def evaluate_model(model, loader, device, model_name="Model"):
    """Evaluate a single model on clean data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating {model_name}"):
            images, labels = images.to(device), labels.to(device)
            
            # Handle different model types
            if hasattr(model, 'forward_features'):
                # Standard model
                outputs = model(images)
            else:
                # Model with special forward method
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Get logits
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def evaluate_healer_combo(main_model, healer_model, loader, device, model_name="Healer+Main"):
    """Evaluate main model with healer assistance"""
    main_model.eval()
    healer_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating {model_name}"):
            images, labels = images.to(device), labels.to(device)
            
            # Apply healer
            healer_output = healer_model(images)
            healed_images = healer_model.apply_correction(images, healer_output)
            
            # Classify with main model
            outputs = main_model(healed_images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def evaluate_ttt_combo(ttt_model, loader, device, model_name="TTT"):
    """Evaluate TTT model with adaptation"""
    ttt_model.eval()
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc=f"Evaluating {model_name}"):
        images, labels = images.to(device), labels.to(device)
        
        # Adapt and classify
        adapted_logits = ttt_model.adapt(images, None, reset=True, adapt_classification=True)
        
        _, predicted = torch.max(adapted_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def evaluate_all_combinations(model_dir, dataset_path, device):
    """Evaluate all model combinations"""
    results = {}
    
    # Set up data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = TinyImageNetDataset(dataset_path, split='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load models
    print("\n=== Loading Models ===")
    
    # Main models
    main_model_path = f"{model_dir}/bestmodel_main/best_model.pt"
    robust_model_path = f"{model_dir}/bestmodel_robust/best_model.pt"
    main_model = load_main_model(main_model_path, device) if os.path.exists(main_model_path) else None
    robust_model = load_main_model(robust_model_path, device) if os.path.exists(robust_model_path) else None
    
    # Healer model
    healer_model_path = f"{model_dir}/bestmodel_healer/best_model.pt"
    healer_model = load_healer_model(healer_model_path, device) if os.path.exists(healer_model_path) else None
    
    # TTT models
    ttt_model_path = f"{model_dir}/bestmodel_ttt/best_model.pt"
    ttt3fc_model_path = f"{model_dir}/bestmodel_ttt3fc/best_model.pt"
    
    # Blended models
    blended_model_path = f"{model_dir}/bestmodel_blended/best_model.pt"
    blended3fc_model_path = f"{model_dir}/bestmodel_blended3fc/best_model.pt"
    
    print("\n=== Evaluating Models ===")
    
    # 1. Main model alone
    if main_model:
        results['Main Model'] = evaluate_model(main_model, val_loader, device, "Main Model")
        print(f"Main Model: {results['Main Model']:.2f}%")
    
    # 2. Robust main model alone
    if robust_model:
        results['Robust Main Model'] = evaluate_model(robust_model, val_loader, device, "Robust Main Model")
        print(f"Robust Main Model: {results['Robust Main Model']:.2f}%")
    
    # 3. BlendedTTT
    if os.path.exists(blended_model_path) and main_model and healer_model:
        blended_model = BlendedTTT(main_model, healer_model)
        checkpoint = torch.load(blended_model_path, map_location=device)
        blended_model.load_state_dict(checkpoint['model_state_dict'])
        blended_model = blended_model.to(device)
        results['BlendedTTT'] = evaluate_model(blended_model, val_loader, device, "BlendedTTT")
        print(f"BlendedTTT: {results['BlendedTTT']:.2f}%")
    
    # 4. BlendedTTT3fc
    if os.path.exists(blended3fc_model_path):
        blended3fc_model = load_blended3fc_model(blended3fc_model_path, device)
        results['BlendedTTT3fc'] = evaluate_model(blended3fc_model, val_loader, device, "BlendedTTT3fc")
        print(f"BlendedTTT3fc: {results['BlendedTTT3fc']:.2f}%")
    
    # 5. Healer + Main
    if healer_model and main_model:
        results['Healer + Main'] = evaluate_healer_combo(main_model, healer_model, val_loader, device, "Healer + Main")
        print(f"Healer + Main: {results['Healer + Main']:.2f}%")
    
    # 6. Healer + Robust
    if healer_model and robust_model:
        results['Healer + Robust'] = evaluate_healer_combo(robust_model, healer_model, val_loader, device, "Healer + Robust")
        print(f"Healer + Robust: {results['Healer + Robust']:.2f}%")
    
    # 7. TTT + Main
    if os.path.exists(ttt_model_path) and main_model:
        ttt_model = TestTimeTrainer(main_model, img_size=64, patch_size=8, embed_dim=384)
        checkpoint = torch.load(ttt_model_path, map_location=device)
        ttt_model.load_state_dict(checkpoint['model_state_dict'])
        ttt_model = ttt_model.to(device)
        results['TTT + Main'] = evaluate_ttt_combo(ttt_model, val_loader, device, "TTT + Main")
        print(f"TTT + Main: {results['TTT + Main']:.2f}%")
    
    # 8. TTT + Robust
    if os.path.exists(ttt_model_path) and robust_model:
        ttt_model_robust = TestTimeTrainer(robust_model, img_size=64, patch_size=8, embed_dim=384)
        checkpoint = torch.load(ttt_model_path, map_location=device)
        # Load transform predictor weights only
        ttt_model_robust.transform_predictor.load_state_dict(
            {k.replace('transform_predictor.', ''): v 
             for k, v in checkpoint['model_state_dict'].items() 
             if 'transform_predictor' in k}
        )
        ttt_model_robust = ttt_model_robust.to(device)
        results['TTT + Robust'] = evaluate_ttt_combo(ttt_model_robust, val_loader, device, "TTT + Robust")
        print(f"TTT + Robust: {results['TTT + Robust']:.2f}%")
    
    # 9. TTT3fc + Main
    if os.path.exists(ttt3fc_model_path) and main_model:
        ttt3fc_model = load_ttt3fc_model(ttt3fc_model_path, main_model, device)
        results['TTT3fc + Main'] = evaluate_ttt_combo(ttt3fc_model, val_loader, device, "TTT3fc + Main")
        print(f"TTT3fc + Main: {results['TTT3fc + Main']:.2f}%")
    
    # 10. TTT3fc + Robust
    if os.path.exists(ttt3fc_model_path) and robust_model:
        ttt3fc_model_robust = TestTimeTrainer3fc(robust_model, img_size=64, patch_size=8, embed_dim=384)
        checkpoint = torch.load(ttt3fc_model_path, map_location=device)
        # Load transform predictor weights only
        ttt3fc_model_robust.transform_predictor.load_state_dict(
            {k.replace('transform_predictor.', ''): v 
             for k, v in checkpoint['model_state_dict'].items() 
             if 'transform_predictor' in k}
        )
        ttt3fc_model_robust = ttt3fc_model_robust.to(device)
        results['TTT3fc + Robust'] = evaluate_ttt_combo(ttt3fc_model_robust, val_loader, device, "TTT3fc + Robust")
        print(f"TTT3fc + Robust: {results['TTT3fc + Robust']:.2f}%")
    
    return results

def main():
    # Configuration
    model_dir = "../../newModels"
    dataset_path = "../../../tinyimagenet200"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Evaluate all combinations
    results = evaluate_all_combinations(model_dir, dataset_path, device)
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print("-" * 40)
    for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:<25}: {accuracy:>6.2f}%")
    print("-" * 40)
    
    # Save results
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to evaluation_results.json")

if __name__ == "__main__":
    main()
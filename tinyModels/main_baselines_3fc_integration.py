# Complete integration file for 3FC models with main evaluation pipeline
# This file contains all necessary functions to add BlendedTTT3fc and TTT3fc models
# to your existing main_baselines.py evaluation pipeline

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from copy import deepcopy

# Import dataset and transform utilities
from new_new import TinyImageNetDataset, ContinuousTransforms

# Import the new 3FC models and their training/evaluation functions
from blended_ttt3fc_model import BlendedTTT3fc
from blended_ttt3fc_training import train_blended_ttt3fc_model
from ttt3fc_model import TestTimeTrainer3fc, train_ttt3fc_model
from ttt3fc_blended3fc_evaluation import (
    evaluate_3fc_models_comprehensive, 
    load_blended3fc_model, 
    load_ttt3fc_model,
    compare_3fc_with_original_models,
    log_3fc_results_to_wandb
)

# Import ViT model creation utilities
from vit_implementation import create_vit_model
from transformer_utils import set_seed


class IdentityHealer(nn.Module):
    """Dummy healer that does nothing - for baseline comparison"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.zeros(x.size(0), 1, device=x.device)
        
    def apply_correction(self, images, predictions):
        return images


def load_main_model(model_path, device):
    """Load the main classification model from a checkpoint"""
    print(f"Loading main model from {model_path}")
    main_model = create_vit_model(
        img_size=64, patch_size=8, in_chans=3, num_classes=200,
        embed_dim=384, depth=8, head_dim=64, mlp_ratio=4.0, use_resnet_stem=True
    )
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create a new state dict with the correct keys
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
    """Load the healer model from a checkpoint"""
    print(f"Loading healer model from {model_path}")
    from new_new import TransformationHealer  # Assuming this exists in new_new.py
    
    healer_model = TransformationHealer(
        img_size=64, patch_size=8, in_chans=3,
        embed_dim=384, depth=6, head_dim=64
    )
    checkpoint = torch.load(model_path, map_location=device)
    healer_model.load_state_dict(checkpoint['model_state_dict'])
    healer_model = healer_model.to(device)
    healer_model.eval()
    return healer_model


def load_ttt_model(model_path, main_model, device):
    """Load the TTT model from a checkpoint"""
    print(f"Loading TTT model from {model_path}")
    from ttt_model import TestTimeTrainer  # Original TTT model
    
    ttt_model = TestTimeTrainer(
        base_model=main_model,
        img_size=64,
        patch_size=8,
        embed_dim=384
    )
    checkpoint = torch.load(model_path, map_location=device)
    ttt_model.load_state_dict(checkpoint['model_state_dict'])
    ttt_model = ttt_model.to(device)
    ttt_model.eval()
    return ttt_model


def load_blended_model(model_path, main_model, device):
    """Load the BlendedTTT model from a checkpoint"""
    print(f"Loading BlendedTTT model from {model_path}")
    from blended_ttt_model import BlendedTTT  # Original BlendedTTT model
    
    blended_model = BlendedTTT(
        img_size=64,
        patch_size=8,
        embed_dim=384,
        depth=8
    )
    checkpoint = torch.load(model_path, map_location=device)
    blended_model.load_state_dict(checkpoint['model_state_dict'])
    blended_model = blended_model.to(device)
    blended_model.eval()
    return blended_model


def load_baseline_model(model_path, device):
    """Load baseline ResNet18 model from checkpoint"""
    print(f"Loading baseline model from {model_path}")
    from baseline_models import SimpleResNet18  # Assuming this exists
    
    model = SimpleResNet18(num_classes=200)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_pretrained_model(model_path, device):
    """Load pretrained ResNet18 model from checkpoint"""
    print(f"Loading pretrained model from {model_path}")
    
    # Define PretrainedResNet18 here since it might not be in main_baselines.py
    class PretrainedResNet18(nn.Module):
        def __init__(self, num_classes=200):
            super(PretrainedResNet18, self).__init__()
            from torchvision import models
            self.resnet = models.resnet18(pretrained=True)
            
            # Modify first conv layer for 64x64 input
            old_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
            with torch.no_grad():
                old_weight = old_conv.weight
                new_weight = old_weight[:, :, 2:5, 2:5].clone()
                self.resnet.conv1.weight.copy_(new_weight)
            
            self.resnet.maxpool = nn.Identity()
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
            
        def forward(self, x):
            return self.resnet(x)
    
    model = PretrainedResNet18(num_classes=200)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def evaluate_all_model_combinations_with_3fc(dataset_path, severities, model_dir, args, device):
    """
    Extended version of evaluate_all_model_combinations that includes the new 3FC models:
    1. Main (not robust)
    2. Main robust  
    3. Healer + Main (not robust)
    4. Healer + Main robust
    5. TTT + Main (not robust)
    6. TTT + Main robust
    7. BlendedTTT + Main (not robust) 
    8. BlendedTTT + Main robust
    9. TTT3fc + Main (not robust)
    10. TTT3fc + Main robust
    11. BlendedTTT3fc + Main (not robust)
    12. BlendedTTT3fc + Main robust
    13. Baseline (ResNet18 from scratch)
    14. Pretrained (ResNet18 with ImageNet pretraining)
    """
    
    print("\n" + "="*100)
    print("🏆 COMPREHENSIVE MODEL EVALUATION - ALL COMBINATIONS INCLUDING 3FC MODELS")
    print("="*100)
    
    all_model_results = {}
    
    # Load all models (including the original ones)
    models = {}
    
    # 1. Load main models
    main_model_path = f"{model_dir}/bestmodel_main/best_model.pt"
    robust_model_path = f"{model_dir}/bestmodel_robust/best_model.pt"
    healer_model_path = f"{model_dir}/bestmodel_healer/best_model.pt"
    
    if os.path.exists(main_model_path):
        models['main'] = load_main_model(main_model_path, device)
        print("✅ Loaded: Main Model (not robust)")
    else:
        print("❌ Missing: Main Model")
        return None
        
    if os.path.exists(robust_model_path):
        models['main_robust'] = load_main_model(robust_model_path, device)
        print("✅ Loaded: Main Model (robust)")
    else:
        print("⚠️  Missing: Main Model (robust) - will skip robust combinations")
        
    if os.path.exists(healer_model_path):
        models['healer'] = load_healer_model(healer_model_path, device)
        print("✅ Loaded: Healer Model")
    else:
        print("❌ Missing: Healer Model")
        return None
    
    # 2. Load original TTT models
    if not getattr(args, 'exclude_ttt', False):
        ttt_model_path = f"{model_dir}/bestmodel_ttt/best_model.pt"
        if os.path.exists(ttt_model_path):
            models['ttt'] = load_ttt_model(ttt_model_path, models['main'], device)
            print("✅ Loaded: TTT Model (based on main)")
            
            if 'main_robust' in models:
                models['ttt_robust'] = load_ttt_model(ttt_model_path, models['main_robust'], device)
                print("✅ Loaded: TTT Model (based on robust)")
        else:
            print("⚠️  Missing: TTT Model - will skip TTT combinations")
    
    # 3. Load original BlendedTTT models  
    if not getattr(args, 'exclude_blended', False):
        blended_model_path = f"{model_dir}/bestmodel_blended/best_model.pt"
        if os.path.exists(blended_model_path):
            models['blended'] = load_blended_model(blended_model_path, models['main'], device)
            print("✅ Loaded: BlendedTTT Model (based on main)")
            
            if 'main_robust' in models:
                models['blended_robust'] = load_blended_model(blended_model_path, models['main_robust'], device)
                print("✅ Loaded: BlendedTTT Model (based on robust)")
        else:
            print("⚠️  Missing: BlendedTTT Model - will skip BlendedTTT combinations")
    
    # 4. Load NEW TTT3fc models
    if not getattr(args, 'exclude_ttt3fc', False):
        ttt3fc_model_path = f"{model_dir}/bestmodel_ttt3fc/best_model.pt"
        if os.path.exists(ttt3fc_model_path):
            models['ttt3fc'] = load_ttt3fc_model(ttt3fc_model_path, models['main'], device)
            print("✅ Loaded: TTT3fc Model (based on main)")
            
            if 'main_robust' in models:
                models['ttt3fc_robust'] = load_ttt3fc_model(ttt3fc_model_path, models['main_robust'], device)
                print("✅ Loaded: TTT3fc Model (based on robust)")
        else:
            print("⚠️  Missing: TTT3fc Model - will skip TTT3fc combinations")
    
    # 5. Load NEW BlendedTTT3fc models
    if not getattr(args, 'exclude_blended3fc', False):
        blended3fc_model_path = f"{model_dir}/bestmodel_blended3fc/best_model.pt"
        if os.path.exists(blended3fc_model_path):
            models['blended3fc'] = load_blended3fc_model(blended3fc_model_path, device)
            print("✅ Loaded: BlendedTTT3fc Model")
            
            # Note: BlendedTTT3fc doesn't depend on base model like TTT3fc does
            models['blended3fc_robust'] = models['blended3fc']  # Same model for both
            print("✅ Loaded: BlendedTTT3fc Model (compatible with robust)")
        else:
            print("⚠️  Missing: BlendedTTT3fc Model - will skip BlendedTTT3fc combinations")
    
    # 6. Load baseline models
    if getattr(args, 'compare_baseline', False):
        baseline_model_path = f"{model_dir}/bestmodel_resnet18_baseline/best_model.pt"
        if os.path.exists(baseline_model_path):
            models['baseline'] = load_baseline_model(baseline_model_path, device)
            print("✅ Loaded: Baseline ResNet18 (from scratch)")
        else:
            print("⚠️  Missing: Baseline ResNet18 - use --train_baseline to train it")
    
    if getattr(args, 'compare_pretrained', False):
        pretrained_model_path = f"{model_dir}/bestmodel_pretrained_resnet18/best_model.pt"
        if os.path.exists(pretrained_model_path):
            models['pretrained'] = load_pretrained_model(pretrained_model_path, device)
            print("✅ Loaded: Pretrained ResNet18 (ImageNet)")
        else:
            print("⚠️  Missing: Pretrained ResNet18 - use --train_pretrained to train it")
    
    print(f"\n📊 Evaluating {len(models)} model combinations on {len(severities)} severity levels...")
    
    # Define evaluation combinations (including new 3FC models)
    combinations = [
        # Original combinations
        ("Main", "main", None, "Main ViT (not robust)"),
        ("Main_Robust", "main_robust", None, "Main ViT (robust training)"),
        ("Healer+Main", "main", "healer", "Healer + Main ViT (not robust)"),
        ("Healer+Main_Robust", "main_robust", "healer", "Healer + Main ViT (robust)"),
        ("TTT+Main", "ttt", None, "TTT + Main ViT (not robust)"),
        ("TTT+Main_Robust", "ttt_robust", None, "TTT + Main ViT (robust)"),
        ("BlendedTTT+Main", "blended", None, "BlendedTTT + Main ViT (not robust)"),
        ("BlendedTTT+Main_Robust", "blended_robust", None, "BlendedTTT + Main ViT (robust)"),
        
        # NEW 3FC combinations
        ("TTT3fc+Main", "ttt3fc", None, "TTT3fc + Main ViT (not robust)"),
        ("TTT3fc+Main_Robust", "ttt3fc_robust", None, "TTT3fc + Main ViT (robust)"),
        ("BlendedTTT3fc", "blended3fc", None, "BlendedTTT3fc (standalone)"),
        ("BlendedTTT3fc_Robust", "blended3fc_robust", None, "BlendedTTT3fc (robust compatible)"),
        
        # Baseline combinations
        ("Baseline", "baseline", None, "ResNet18 (from scratch)"),
        ("Pretrained", "pretrained", None, "ResNet18 (ImageNet pretrained)"),
    ]
    
    # Evaluate each combination
    for combo_name, main_key, healer_key, description in combinations:
        if main_key not in models:
            print(f"⏭️  Skipping {combo_name}: {main_key} model not available")
            continue
            
        if healer_key and healer_key not in models:
            print(f"⏭️  Skipping {combo_name}: {healer_key} model not available")
            continue
            
        print(f"\n🔍 Evaluating: {description}")
        
        main_model = models[main_key]
        healer_model = models[healer_key] if healer_key else IdentityHealer().to(device)
        
        # Evaluate this combination
        results = evaluate_model_combination_3fc(main_model, healer_model, dataset_path, severities, device, main_key)
        
        all_model_results[combo_name] = {
            'results': results,
            'description': description
        }
    
    return all_model_results


def evaluate_model_combination_3fc(main_model, healer_model, dataset_path, severities, device, model_type):
    """Evaluate a specific model + healer combination (extended for 3FC models)"""
    results = {}
    
    # Standard validation transforms
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Evaluate on each severity level
    for severity in severities:
        print(f"    Severity {severity}...", end=" ")
        
        if severity == 0.0:
            # Clean data evaluation
            val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
            
            correct = 0
            total = 0
            
            main_model.eval()
            healer_model.eval()
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    # Apply healer if not identity
                    if not isinstance(healer_model, IdentityHealer):
                        healer_predictions = healer_model(images)
                        images = healer_model.apply_correction(images, healer_predictions)
                    
                    # Forward pass - handle different model types
                    if 'ttt3fc' in model_type.lower():
                        outputs, _ = main_model(images, use_base_model=False)  # TTT3fc models with own classification
                    elif 'blended3fc' in model_type.lower():
                        outputs, _ = main_model(images)  # BlendedTTT3fc models return tuple
                    elif 'ttt' in model_type.lower():
                        outputs, _ = main_model(images)  # TTT models return tuple
                    elif 'blended' in model_type.lower():
                        outputs, _ = main_model(images)  # BlendedTTT models return tuple
                    else:
                        outputs = main_model(images)  # Regular models
                    
                    _, predicted = torch.max(outputs, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            results[severity] = accuracy
            print(f"{accuracy:.4f}")
            
        else:
            # Transformed data evaluation
            ood_transform = ContinuousTransforms(severity=severity)
            ood_val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val, ood_transform=ood_transform)
            
            def collate_fn(batch):
                orig_imgs, trans_imgs, labels, params = zip(*batch)
                return torch.stack(orig_imgs), torch.stack(trans_imgs), torch.tensor(labels), params
            
            ood_val_loader = DataLoader(ood_val_dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate_fn)
            
            correct = 0
            total = 0
            
            main_model.eval()
            healer_model.eval()
            
            with torch.no_grad():
                for orig_images, trans_images, labels, params in ood_val_loader:
                    trans_images, labels = trans_images.to(device), labels.to(device)
                    
                    # Apply healer if not identity
                    if not isinstance(healer_model, IdentityHealer):
                        healer_predictions = healer_model(trans_images)
                        trans_images = healer_model.apply_correction(trans_images, healer_predictions)
                    
                    # Forward pass - handle different model types
                    if 'ttt3fc' in model_type.lower():
                        outputs, _ = main_model(trans_images, use_base_model=False)  # TTT3fc models with own classification
                    elif 'blended3fc' in model_type.lower():
                        outputs, _ = main_model(trans_images)  # BlendedTTT3fc models return tuple
                    elif 'ttt' in model_type.lower():
                        outputs, _ = main_model(trans_images)  # TTT models return tuple
                    elif 'blended' in model_type.lower():
                        outputs, _ = main_model(trans_images)  # BlendedTTT models return tuple
                    else:
                        outputs = main_model(trans_images)  # Regular models
                    
                    _, predicted = torch.max(outputs, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            results[severity] = accuracy
            print(f"{accuracy:.4f}")
    
    return results


def run_comprehensive_evaluation_with_3fc(args, device):
    """Run the comprehensive evaluation including 3FC models"""
    
    # Evaluate all combinations (including 3FC models)
    all_results = evaluate_all_model_combinations_with_3fc(
        args.dataset, args.severities, args.model_dir, args, device
    )
    
    if all_results is None:
        print("❌ Cannot run comprehensive evaluation - missing required models")
        return None
    
    # Print comprehensive results
    print_comprehensive_results_with_3fc(all_results, args.severities)
    
    # Generate plots for key comparisons
    create_comprehensive_plots_with_3fc(all_results, args.severities, args.visualize_dir)
    
    return all_results


def print_comprehensive_results_with_3fc(all_model_results, severities):
    """Print comprehensive comparison including 3FC models"""
    
    print("\n" + "="*130)
    print("🏆 COMPREHENSIVE RESULTS - ALL MODEL COMBINATIONS INCLUDING 3FC MODELS")
    print("="*130)
    
    # Create results table
    print(f"\n{'Model Combination':<35} {'Description':<40} ", end="")
    for severity in severities:
        if severity == 0.0:
            print(f"{'Clean':<8}", end="")
        else:
            print(f"{'S'+str(severity):<8}", end="")
    print()
    print("-" * 130)
    
    # Sort models by clean data performance
    model_items = [(name, data) for name, data in all_model_results.items()]
    model_items.sort(key=lambda x: x[1]['results'].get(0.0, 0), reverse=True)
    
    for name, data in model_items:
        results = data['results']
        description = data['description']
        
        print(f"{name:<35} {description:<40} ", end="")
        for severity in severities:
            if severity in results:
                acc = results[severity]
                print(f"{acc:.4f}  ", end="")
            else:
                print(f"{'--':<8}", end="")
        print()
    
    # Analysis section with focus on 3FC improvements
    print("\n" + "="*130)
    print("📊 ANALYSIS - INCLUDING 3FC MODEL IMPROVEMENTS")
    print("="*130)
    
    # Find best performers
    clean_best = max(model_items, key=lambda x: x[1]['results'].get(0.0, 0))
    print(f"🥇 Best Clean Data Performance: {clean_best[0]} ({clean_best[1]['results'][0.0]:.4f})")
    
    # Compare 3FC vs original models
    print(f"\n🔬 3FC MODEL IMPROVEMENTS:")
    
    # Compare BlendedTTT vs BlendedTTT3fc
    blended_results = [item for item in model_items if 'BlendedTTT+' in item[0] and '3fc' not in item[0]]
    blended3fc_results = [item for item in model_items if 'BlendedTTT3fc' in item[0]]
    
    if blended_results and blended3fc_results:
        blended_acc = blended_results[0][1]['results'].get(0.0, 0)
        blended3fc_acc = blended3fc_results[0][1]['results'].get(0.0, 0)
        improvement = blended3fc_acc - blended_acc
        print(f"📈 BlendedTTT3fc vs BlendedTTT: {improvement:.4f} improvement ({improvement*100:.1f} points)")
    
    # Compare TTT vs TTT3fc
    ttt_results = [item for item in model_items if 'TTT+' in item[0] and '3fc' not in item[0]]
    ttt3fc_results = [item for item in model_items if 'TTT3fc+' in item[0]]
    
    if ttt_results and ttt3fc_results:
        ttt_acc = ttt_results[0][1]['results'].get(0.0, 0)
        ttt3fc_acc = ttt3fc_results[0][1]['results'].get(0.0, 0)
        improvement = ttt3fc_acc - ttt_acc
        print(f"📈 TTT3fc vs TTT: {improvement:.4f} improvement ({improvement*100:.1f} points)")
    
    # Find most robust model (smallest drop from clean to worst severity)
    if len(severities) > 1:
        max_severity = max([s for s in severities if s > 0])
        robustness_scores = []
        
        for name, data in model_items:
            results = data['results']
            if 0.0 in results and max_severity in results:
                clean_acc = results[0.0]
                worst_acc = results[max_severity]
                drop = clean_acc - worst_acc
                drop_percent = (drop / clean_acc) * 100 if clean_acc > 0 else 100
                robustness_scores.append((name, drop_percent, worst_acc))
        
        if robustness_scores:
            most_robust = min(robustness_scores, key=lambda x: x[1])
            print(f"🛡️  Most Transform Robust: {most_robust[0]} ({most_robust[1]:.1f}% drop)")


def create_comprehensive_plots_with_3fc(all_results, severities, save_dir):
    """Create comprehensive comparison plots including 3FC models"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(16, 10))
    
    # Plot all model combinations with special highlighting for 3FC models
    for name, data in all_results.items():
        results = data['results']
        sev_list = sorted([s for s in severities if s in results])
        acc_list = [results[s] for s in sev_list]
        
        if len(sev_list) > 1:
            # Special styling for 3FC models
            if '3fc' in name.lower():
                plt.plot(sev_list, acc_list, 'o-', linewidth=3, markersize=8, 
                        label=name, alpha=0.9, linestyle='--')
            else:
                plt.plot(sev_list, acc_list, 'o-', linewidth=2, markersize=6, 
                        label=name, alpha=0.7)
    
    plt.xlabel('Transform Severity', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Comprehensive Model Comparison: All Combinations Including 3FC Models', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = f"{save_dir}/comprehensive_comparison_with_3fc.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"📊 Comprehensive comparison plot (with 3FC) saved to {save_path}")
    plt.close()
    
    # Create a separate plot for 3FC models only
    plt.figure(figsize=(12, 8))
    
    # Plot only 3FC models and their original counterparts
    comparison_models = {}
    for name, data in all_results.items():
        if '3fc' in name.lower() or 'BlendedTTT+' in name or 'TTT+' in name:
            comparison_models[name] = data
    
    for name, data in comparison_models.items():
        results = data['results']
        sev_list = sorted([s for s in severities if s in results])
        acc_list = [results[s] for s in sev_list]
        
        if len(sev_list) > 1:
            if '3fc' in name.lower():
                plt.plot(sev_list, acc_list, 'o-', linewidth=3, markersize=8, 
                        label=name + ' (3FC)', color='red' if 'BlendedTTT3fc' in name else 'blue')
            else:
                plt.plot(sev_list, acc_list, 'o-', linewidth=2, markersize=6, 
                        label=name + ' (Original)', alpha=0.7, linestyle=':')
    
    plt.xlabel('Transform Severity', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('3FC Models vs Original Models Comparison', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    save_path = f"{save_dir}/3fc_vs_original_comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"📊 3FC vs Original comparison plot saved to {save_path}")
    plt.close()


def train_3fc_models_if_missing(dataset_path, base_model=None, model_dir="./"):
    """Train 3FC models if they don't exist"""
    
    # Check and train BlendedTTT3fc
    blended3fc_path = f"{model_dir}/bestmodel_blended3fc/best_model.pt"
    if not os.path.exists(blended3fc_path):
        print("\n🔧 Training BlendedTTT3fc model...")
        train_blended_ttt3fc_model(base_model, dataset_path)
    else:
        print("✅ BlendedTTT3fc model already exists")
    
    # Check and train TTT3fc
    ttt3fc_path = f"{model_dir}/bestmodel_ttt3fc/best_model.pt"
    if not os.path.exists(ttt3fc_path):
        print("\n🔧 Training TTT3fc model...")
        train_ttt3fc_model(dataset_path, base_model)
    else:
        print("✅ TTT3fc model already exists")


def add_3fc_args_to_parser(parser):
    """Add 3FC-specific arguments to the argument parser"""
    parser.add_argument("--exclude_ttt3fc", action="store_true",
                      help="Whether to exclude TTT3fc model from the pipeline")
    parser.add_argument("--exclude_blended3fc", action="store_true",
                      help="Whether to exclude BlendedTTT3fc model from the pipeline")
    parser.add_argument("--train_3fc", action="store_true",
                      help="Train 3FC models (BlendedTTT3fc and TTT3fc)")
    parser.add_argument("--compare_3fc", action="store_true",
                      help="Include 3FC models in comprehensive comparison")
    return parser


def evaluate_3fc_models_only(dataset_path, severities, model_dir, device):
    """
    Evaluate only the 3FC models for focused comparison
    """
    print("\n" + "="*80)
    print("🔬 FOCUSED 3FC MODEL EVALUATION")
    print("="*80)
    
    # Load main and healer models (required)
    main_model_path = f"{model_dir}/bestmodel_main/best_model.pt"
    healer_model_path = f"{model_dir}/bestmodel_healer/best_model.pt"
    
    if not os.path.exists(main_model_path) or not os.path.exists(healer_model_path):
        print("❌ Missing required main or healer models")
        return None
    
    main_model = load_main_model(main_model_path, device)
    healer_model = load_healer_model(healer_model_path, device)
    
    # Run comprehensive 3FC evaluation
    results_3fc = evaluate_3fc_models_comprehensive(
        main_model=main_model,
        healer_model=healer_model,
        dataset_path=dataset_path,
        severities=severities,
        model_dir=model_dir,
        include_blended3fc=True,
        include_ttt3fc=True
    )
    
    return results_3fc


def log_3fc_comprehensive_results(all_results):
    """Log comprehensive 3FC results to wandb"""
    try:
        import wandb
        
        # Log overall comparison metrics
        wandb.log({"3fc_evaluation/total_models_evaluated": len(all_results)})
        
        # Log best performers
        if all_results:
            clean_performances = [(name, data['results'].get(0.0, 0)) 
                                for name, data in all_results.items()]
            best_clean = max(clean_performances, key=lambda x: x[1])
            wandb.log({
                "3fc_evaluation/best_clean_model": best_clean[0],
                "3fc_evaluation/best_clean_accuracy": best_clean[1]
            })
            
            # Log 3FC specific improvements
            blended_3fc = [(name, data['results'].get(0.0, 0)) 
                          for name, data in all_results.items() 
                          if 'BlendedTTT3fc' in name]
            ttt_3fc = [(name, data['results'].get(0.0, 0)) 
                      for name, data in all_results.items() 
                      if 'TTT3fc' in name]
            
            if blended_3fc:
                wandb.log({
                    "3fc_evaluation/blended3fc_best_accuracy": max(blended_3fc, key=lambda x: x[1])[1]
                })
            
            if ttt_3fc:
                wandb.log({
                    "3fc_evaluation/ttt3fc_best_accuracy": max(ttt_3fc, key=lambda x: x[1])[1]
                })
        
        # Log detailed results for each model and severity
        for model_name, model_data in all_results.items():
            results = model_data['results']
            for severity, accuracy in results.items():
                wandb.log({
                    f"3fc_detailed/{model_name}_s{severity}": accuracy,
                    "severity": severity,
                    "model": model_name
                })
                
    except Exception as e:
        print(f"Note: wandb logging failed: {e}")


# Instructions for integration:
"""
To integrate these 3FC models into your main_baselines.py:

1. Add the imports at the top of main_baselines.py:
   # Copy this entire file to your project directory, then:
   from main_baselines_3fc_integration import *

2. Modify the argument parser in main() by adding:
   parser = add_3fc_args_to_parser(parser)

3. In the training section, add:
   if args.train_3fc:
       train_3fc_models_if_missing(args.dataset, main_model, args.model_dir)

4. In the evaluation section, replace the call to run_comprehensive_evaluation with:
   all_results = run_comprehensive_evaluation_with_3fc(args, device)

5. For focused 3FC-only evaluation, you can also use:
   results_3fc = evaluate_3fc_models_only(args.dataset, args.severities, args.model_dir, device)

6. The new models will automatically be included in the comprehensive evaluation!

Example usage in main():
```python
def main():
    parser = argparse.ArgumentParser(description="Transform Healing with Vision Transformers")
    # ... existing arguments ...
    
    # Add 3FC arguments
    parser = add_3fc_args_to_parser(parser)
    args = parser.parse_args()
    
    # ... existing setup ...
    
    # Training phase
    if args.mode in ["train", "evaluate", "all"]:
        # ... existing training code ...
        
        # Train 3FC models if requested
        if args.train_3fc:
            train_3fc_models_if_missing(args.dataset, main_model, args.model_dir)
    
    # Evaluation phase  
    if args.mode in ["evaluate", "visualize", "all"]:
        print("\\n=== Comprehensive Evaluation With 3FC Models ===")
        
        # Use the extended evaluation that includes 3FC models
        all_results = run_comprehensive_evaluation_with_3fc(args, device)
        
        # Log comprehensive results
        log_3fc_comprehensive_results(all_results)
```
"""
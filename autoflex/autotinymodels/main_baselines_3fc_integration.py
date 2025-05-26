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
import argparse

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

# Set batch size based on LOCALHERE.TXT existence
LOCALHERE_PATH = os.path.join(os.path.dirname(__file__), "../../../LOCALHERE.TXT")
BATCH_SIZE = 128 if os.path.exists(LOCALHERE_PATH) else 250
print(f"Using batch size: {BATCH_SIZE} (LOCALHERE.TXT {'exists' if os.path.exists(LOCALHERE_PATH) else 'does not exist'})")


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
        ("TTT", "ttt", None, "TTT (Test-Time Training)"),
        ("TTT_Robust", "ttt_robust", None, "TTT (robust compatible)"),
        ("BlendedTTT", "blended", None, "BlendedTTT (standalone)"),
        ("BlendedTTT_Robust", "blended_robust", None, "BlendedTTT (robust compatible)"),
        
        # NEW 3FC combinations
        ("TTT3fc", "ttt3fc", None, "TTT3fc (Test-Time Training with 3FC)"),
        ("TTT3fc_Robust", "ttt3fc_robust", None, "TTT3fc (robust compatible)"),
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
    
    # Transforms without normalization
    transform_val_no_norm = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Normalization to apply separately
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Evaluate on each severity level
    for severity in severities:
        print(f"    Severity {severity}...", end=" ")
        
        if severity == 0.0:
            # Clean data evaluation
            val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val_no_norm)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            
            correct = 0
            total = 0
            
            main_model.eval()
            healer_model.eval()
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    # Normalize images
                    batch_size = images.size(0)
                    normalized_images = []
                    for i in range(batch_size):
                        normalized_images.append(normalize(images[i]))
                    images = torch.stack(normalized_images)
                    
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
            ood_val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val_no_norm, ood_transform=ood_transform)
            
            def collate_fn(batch):
                orig_imgs, trans_imgs, labels, params = zip(*batch)
                return torch.stack(orig_imgs), torch.stack(trans_imgs), torch.tensor(labels), params
            
            ood_val_loader = DataLoader(ood_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
            
            correct = 0
            total = 0
            
            main_model.eval()
            healer_model.eval()
            
            with torch.no_grad():
                for orig_images, trans_images, labels, params in ood_val_loader:
                    trans_images, labels = trans_images.to(device), labels.to(device)
                    
                    # Normalize transformed images
                    batch_size = trans_images.size(0)
                    normalized_images = []
                    for i in range(batch_size):
                        normalized_images.append(normalize(trans_images[i]))
                    trans_images = torch.stack(normalized_images)
                    
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


def train_all_models_if_missing(dataset_path, model_dir="./", args=None, device=None):
    """Train ALL models if they don't exist - comprehensive training pipeline"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import required training functions
    from new_new import train_main_model, train_healer_model
    from robust_training import train_main_model_robust
    from ttt_model import train_ttt_model
    from blended_ttt_training import train_blended_ttt_model
    
    print("\n" + "="*80)
    print("🚀 COMPREHENSIVE MODEL TRAINING PIPELINE")
    print("="*80)
    print("Checking and training all missing models...")
    
    models = {}
    
    # 1. Train Main Model (required for everything)
    main_model_path = f"{model_dir}/bestmodel_main/best_model.pt"
    if not os.path.exists(main_model_path):
        print("\n🔧 Training Main Classification Model...")
        main_model = train_main_model(dataset_path, model_dir=model_dir)
        models['main'] = main_model
    else:
        print("✅ Main model already exists")
        models['main'] = load_main_model(main_model_path, device)
    
    # 2. Train Robust Main Model
    robust_model_path = f"{model_dir}/bestmodel_robust/best_model.pt"
    if not os.path.exists(robust_model_path):
        print("\n🔧 Training Robust Main Classification Model...")
        robust_model = train_main_model_robust(dataset_path, severity=0.5, model_dir=model_dir)
        models['main_robust'] = robust_model
    else:
        print("✅ Robust main model already exists")
        models['main_robust'] = load_main_model(robust_model_path, device)
    
    # 3. Train Healer Model (required for healer combinations)
    healer_model_path = f"{model_dir}/bestmodel_healer/best_model.pt"
    if not os.path.exists(healer_model_path):
        print("\n🔧 Training Transformation Healer Model...")
        healer_model = train_healer_model(dataset_path, severity=0.5, model_dir=model_dir)
        models['healer'] = healer_model
    else:
        print("✅ Healer model already exists")
        models['healer'] = load_healer_model(healer_model_path, device)
    
    # 4. Train TTT Model (if not excluded)
    if not getattr(args, 'exclude_ttt', False):
        ttt_model_path = f"{model_dir}/bestmodel_ttt/best_model.pt"
        if not os.path.exists(ttt_model_path):
            print("\n🔧 Training Test-Time Training (TTT) Model...")
            ttt_model = train_ttt_model(dataset_path, base_model=models['main'], severity=0.5, model_dir=model_dir)
            models['ttt'] = ttt_model
        else:
            print("✅ TTT model already exists")
            models['ttt'] = load_ttt_model(ttt_model_path, models['main'], device)
    
    # 5. Train BlendedTTT Model (if not excluded)
    if not getattr(args, 'exclude_blended', False):
        blended_model_path = f"{model_dir}/bestmodel_blended/best_model.pt"
        if not os.path.exists(blended_model_path):
            print("\n🔧 Training BlendedTTT Model...")
            blended_model = train_blended_ttt_model(models['main'], dataset_path, model_dir=model_dir)
            models['blended'] = blended_model
        else:
            print("✅ BlendedTTT model already exists")
            models['blended'] = load_blended_model(blended_model_path, models['main'], device)
    
    # 6. Train NEW 3FC Models
    
    # Train BlendedTTT3fc
    if not getattr(args, 'exclude_blended3fc', False):
        blended3fc_path = f"{model_dir}/bestmodel_blended3fc/best_model.pt"
        # Ensure directory exists
        os.makedirs(f"{model_dir}/bestmodel_blended3fc", exist_ok=True)
        
        if not os.path.exists(blended3fc_path):
            print("\n🔧 Training BlendedTTT3fc Model...")
            train_blended_ttt3fc_model(models['main'], dataset_path, model_dir=model_dir)
            models['blended3fc'] = load_blended3fc_model(blended3fc_path, device)
        else:
            print("✅ BlendedTTT3fc model already exists")
            models['blended3fc'] = load_blended3fc_model(blended3fc_path, device)
    
    # Train TTT3fc
    if not getattr(args, 'exclude_ttt3fc', False):
        ttt3fc_path = f"{model_dir}/bestmodel_ttt3fc/best_model.pt"
        # Ensure directory exists
        os.makedirs(f"{model_dir}/bestmodel_ttt3fc", exist_ok=True)
        
        if not os.path.exists(ttt3fc_path):
            print("\n🔧 Training TTT3fc Model...")
            train_ttt3fc_model(dataset_path, models['main'], model_dir=model_dir)
            models['ttt3fc'] = load_ttt3fc_model(ttt3fc_path, models['main'], device)
        else:
            print("✅ TTT3fc model already exists")
            models['ttt3fc'] = load_ttt3fc_model(ttt3fc_path, models['main'], device)
    
    # 7. Train Baseline Models (if requested)
    
    # Train Baseline ResNet18
    if True:
        baseline_model_path = f"{model_dir}/bestmodel_resnet18_baseline/best_model.pt"
        if not os.path.exists(baseline_model_path):
            print("\n🔧 Training Baseline ResNet18 Model...")
            baseline_model = train_baseline_resnet18(dataset_path, model_dir=model_dir)
            models['baseline'] = baseline_model
        else:
            print("✅ Baseline ResNet18 model already exists")
            models['baseline'] = load_baseline_model(baseline_model_path, device)
    
    # Train Pretrained ResNet18
    if True:
        pretrained_model_path = f"{model_dir}/bestmodel_pretrained_resnet18/best_model.pt"
        if not os.path.exists(pretrained_model_path):
            print("\n🔧 Training Pretrained ResNet18 Model...")
            pretrained_model = train_pretrained_resnet18(dataset_path, model_dir=model_dir)
            models['pretrained'] = pretrained_model
        else:
            print("✅ Pretrained ResNet18 model already exists")
            models['pretrained'] = load_pretrained_model(pretrained_model_path, device)
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE MODEL TRAINING COMPLETED!")
    print("="*80)
    print(f"📁 All models saved in: {model_dir}")
    print(f"🎯 Total models available: {len(models)}")
    
    return models


def train_baseline_resnet18(dataset_path, model_dir="./"):
    """Train a ResNet18 baseline model with early stopping"""
    print("Training ResNet18 baseline model with early stopping...")
    
    # Import required modules
    from baseline_models import SimpleResNet18
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from new_new import TinyImageNetDataset
    import torch.optim as optim
    from tqdm import tqdm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleResNet18(num_classes=200).to(device)
    
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
    
    # Datasets and loaders
    train_dataset = TinyImageNetDataset(dataset_path, "train", transform_train)
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    best_val_acc = 0.0
    patience = 5
    early_stop_counter = 0
    epochs = 50
    
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
            early_stop_counter = 0
            save_path = f"{model_dir}/bestmodel_resnet18_baseline/best_model.pt"
            os.makedirs(f"{model_dir}/bestmodel_resnet18_baseline", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"  ✅ New best baseline model saved with val_acc: {val_acc:.4f}")
        else:
            early_stop_counter += 1
            
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        scheduler.step()
        print()
    
    print(f"Baseline ResNet18 training completed. Best validation accuracy: {best_val_acc:.4f}")
    return model


def train_pretrained_resnet18(dataset_path, model_dir="./"):
    """Train pretrained ResNet18 model with fine-tuning"""
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import os
    from tqdm import tqdm
    
    print("Training pretrained ResNet18 model (ImageNet → Tiny ImageNet)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define PretrainedResNet18 here
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
    
    model = PretrainedResNet18(num_classes=200).to(device)
    
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
    
    # Load datasets
    train_dataset = TinyImageNetDataset(dataset_path, "train", transform_train)
    val_dataset = TinyImageNetDataset(dataset_path, "val", transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Fine-tuning setup
    criterion = nn.CrossEntropyLoss()
    
    # Different learning rates for backbone vs classifier
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': 0.0001},
        {'params': classifier_params, 'lr': 0.001}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    epochs = 30
    best_val_acc = 0.0
    patience = 5
    early_stop_counter = 0
    
    print(f"Fine-tuning for {epochs} epochs with early stopping (patience={patience})...")
    
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
            os.makedirs(f"{model_dir}/bestmodel_pretrained_resnet18", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f"{model_dir}/bestmodel_pretrained_resnet18/best_model.pt")
            print(f"  ✅ New best pretrained model saved with val_acc: {val_acc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        scheduler.step()
        print()
    
    print(f"Fine-tuning completed. Best validation accuracy: {best_val_acc:.4f}")
    return model


def train_3fc_models_if_missing(dataset_path, base_model=None, model_dir="./"):
    """Train 3FC models if they don't exist (legacy function for backward compatibility)"""
    
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
        train_ttt3fc_model(dataset_path, base_model, model_dir=model_dir)
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


def main():
    """
    Main function to run comprehensive model training and evaluation including 3FC models
    """
    parser = argparse.ArgumentParser(description="3FC Models Integration and Comprehensive Evaluation")
    
    # Basic arguments
    parser.add_argument("--dataset", type=str, default="../../../tiny-imagenet-200",
                      help="Path to the dataset")
    parser.add_argument("--model_dir", type=str, default="../../newmodels",
                      help="Directory containing model checkpoints")
    parser.add_argument("--visualize_dir", type=str, default="./visualizations",
                      help="Directory to save visualization plots")
    parser.add_argument("--severities", type=float, nargs="+", 
                      default=[0.0, 0.25, 0.5, 0.75, 1.0],
                      help="Severity levels to evaluate")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, default="all", 
                      choices=["train", "evaluate", "visualize", "all"],
                      help="Mode of operation")
    
    # 3FC specific arguments
    parser = add_3fc_args_to_parser(parser)
    
    # Training flags
    parser.add_argument("--train_all", action="store_true", default=True,
                      help="Train all missing models automatically")
    parser.add_argument("--train_baseline", action="store_true", default=False,
                      help="Include baseline ResNet18 training")
    parser.add_argument("--train_pretrained", action="store_true", default=False,
                      help="Include pretrained ResNet18 training")
    
    # Comparison flags
    parser.add_argument("--compare_baseline", action="store_true", default=False,
                      help="Include baseline models in comparison")
    parser.add_argument("--compare_pretrained", action="store_true", default=False,
                      help="Include pretrained models in comparison")
    
    # Exclusion flags (keep original behavior for compatibility)
    parser.add_argument("--exclude_ttt", action="store_true", default=False,
                      help="Exclude original TTT models")
    parser.add_argument("--exclude_blended", action="store_true", default=False,
                      help="Exclude original BlendedTTT models")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create visualization directory
    os.makedirs(args.visualize_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("🚀 COMPREHENSIVE MODEL PIPELINE WITH 3FC INTEGRATION")
    print("="*80)
    print(f"📁 Dataset: {args.dataset}")
    print(f"📁 Model Directory: {args.model_dir}")
    print(f"📊 Severities: {args.severities}")
    print(f"🎯 Mode: {args.mode}")
    print(f"🔧 Auto-train missing models: {args.train_all}")
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset not found at {args.dataset}")
        print("Please ensure the dataset path is correct.")
        return
    
    # 🚀 TRAINING PHASE - Train ALL missing models automatically
    if args.mode in ["train", "all"] or args.train_all:
        print("\n=== COMPREHENSIVE MODEL TRAINING PHASE ===")
        
        # Train all models if missing (this is the new comprehensive function)
        trained_models = train_all_models_if_missing(
            dataset_path=args.dataset,
            model_dir=args.model_dir,
            args=args,
            device=device
        )
        
        print(f"\n✅ Training phase completed! {len(trained_models)} models ready.")
    
    # 📊 EVALUATION PHASE - Comprehensive evaluation including 3FC models
    if True:
        print("\n=== COMPREHENSIVE EVALUATION WITH 3FC MODELS ===")
        
        # Run the comprehensive evaluation that includes 3FC models
        all_results = run_comprehensive_evaluation_with_3fc(args, device)
        
        if all_results is not None:
            # Log results to wandb if available
            log_3fc_comprehensive_results(all_results)
            
            print("\n" + "="*80)
            print("✅ COMPREHENSIVE EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"📊 Results and plots saved to: {args.visualize_dir}")
            print(f"🔬 Total models evaluated: {len(all_results)}")
            
            # Print top performers
            if all_results:
                clean_performances = [(name, data['results'].get(0.0, 0)) 
                                    for name, data in all_results.items()]
                top_5 = sorted(clean_performances, key=lambda x: x[1], reverse=True)[:5]
                print("\n🏆 Top 5 Performers (Clean Data):")
                for i, (name, acc) in enumerate(top_5, 1):
                    medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
                    print(f"  {medal} {name}: {acc:.4f}")
                
                # Special focus on 3FC improvements
                print("\n🔬 3FC MODEL HIGHLIGHTS:")
                for name, data in all_results.items():
                    if '3fc' in name.lower():
                        acc = data['results'].get(0.0, 0)
                        print(f"  🆕 {name}: {acc:.4f}")
        else:
            print("\n❌ Evaluation failed - please check model availability")
    
    # 🎨 VISUALIZATION PHASE - Generate plots and comparisons
    if args.mode in ["visualize", "all"]:
        print("\n=== GENERATING VISUALIZATIONS ===")
        
        # The visualization plots are already generated in the evaluation phase
        # But we can add more specific visualizations here if needed
        
        print(f"📊 All visualization plots saved to: {args.visualize_dir}")
        print("🎨 Generated plots:")
        print(f"  • Comprehensive comparison: {args.visualize_dir}/comprehensive_comparison_with_3fc.png")
        print(f"  • 3FC vs Original models: {args.visualize_dir}/3fc_vs_original_comparison.png")
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("📋 Summary:")
    print("  ✅ All models trained (if missing)")
    print("  ✅ Comprehensive evaluation completed")
    print("  ✅ 3FC models integrated and compared")
    print("  ✅ Visualization plots generated")
    print(f"  📁 Results saved in: {args.visualize_dir}")


if __name__ == "__main__":
    main()

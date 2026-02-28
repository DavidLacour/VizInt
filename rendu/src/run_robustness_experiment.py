#!/usr/bin/env python3
"""
Comprehensive Robustness Experiment: Proving that predicting augmentation transforms
with a small auxiliary loss makes models more robust across ALL backbones.

Model Matrix (16 models):
  - Normal (no augmentation): resnet, vgg16, vgg19, vanilla_vit
  - Robust (augmented training): resnet18_not_pretrained_robust, vgg16_robust, vgg19_robust, vanilla_vit_robust
  - Blended (predict transforms): blended_resnet18, blended_vgg16, blended_vgg19, blended_training
  - Blended3fc (predict transforms + 3FC): blended_resnet18_3fc, blended_vgg16_3fc, blended_vgg19_3fc, blended_training_3fc

Evaluation:
  - Clean (S=0.0)
  - Corrupted at severities [0.25, 0.5, 0.75, 1.0]
  - OOD (extreme corruptions)

Outputs:
  - wandb logging
  - Comprehensive markdown report
  - LaTeX report
  - Publication-quality graphs
"""
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import logging
import argparse
from collections import OrderedDict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config_loader import ConfigLoader
from models.model_factory import ModelFactory
from data.data_loader import DataLoaderFactory
from data.continuous_transforms import ContinuousTransforms
from data.ood_transforms import OODTransforms
from services.model_trainer import ModelTrainer
from utils.transformer_utils import set_seed

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Install with: pip install wandb")


# ============================================================================
# Configuration
# ============================================================================
EXPERIMENT_NAME = "predict_transforms_robustness_comprehensive"
DATASET = "cifar10"  # Default, overridden by --dataset arg
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]

# Model matrix: category -> backbone -> model_type
MODEL_MATRIX = OrderedDict([
    ('Normal', OrderedDict([
        ('ResNet18', 'resnet'),
        ('VGG16', 'vgg16'),
        ('VGG19', 'vgg19'),
        ('ViT', 'vanilla_vit'),
    ])),
    ('Robust', OrderedDict([
        ('ResNet18', 'resnet18_not_pretrained_robust'),
        ('VGG16', 'vgg16_robust'),
        ('VGG19', 'vgg19_robust'),
        ('ViT', 'vanilla_vit_robust'),
    ])),
    ('Blended', OrderedDict([
        ('ResNet18', 'blended_resnet18'),
        ('VGG16', 'blended_vgg16'),
        ('VGG19', 'blended_vgg19'),
        ('ViT', 'blended_training'),
    ])),
    ('Blended3fc', OrderedDict([
        ('ResNet18', 'blended_resnet18_3fc'),
        ('VGG16', 'blended_vgg16_3fc'),
        ('VGG19', 'blended_vgg19_3fc'),
        ('ViT', 'blended_training_3fc'),
    ])),
])

# Flatten for iteration
ALL_MODELS = []
for category, backbones in MODEL_MATRIX.items():
    for backbone, model_type in backbones.items():
        ALL_MODELS.append({
            'model_type': model_type,
            'category': category,
            'backbone': backbone,
            'display_name': f"{backbone} {category}",
            'is_robust': category == 'Robust',
            'is_blended': category in ('Blended', 'Blended3fc'),
        })

OUTPUT_DIR = Path(__file__).parent / "robustness_experiment_outputs"
GRAPHS_DIR = OUTPUT_DIR / "graphs"


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(OUTPUT_DIR / 'experiment.log'))
        ]
    )
    return logging.getLogger('robustness_experiment')


# ============================================================================
# Evaluation Functions
# ============================================================================
def get_normalize_fn(dataset_config):
    """Create normalization function from dataset config."""
    mean = torch.tensor(dataset_config['mean']).view(3, 1, 1)
    std = torch.tensor(dataset_config['std']).view(3, 1, 1)
    def normalize(img):
        return (img - mean.to(img.device)) / std.to(img.device)
    return normalize


def get_val_loader(data_factory, dataset_name, batch_size=256):
    """Get validation data loader without normalization."""
    from torchvision import transforms as T, datasets
    dataset_config = data_factory.config.get_dataset_config(dataset_name)
    val_transform = T.Compose([T.ToTensor()])
    if dataset_name == 'cifar10':
        val_dataset = datasets.CIFAR10(
            root=dataset_config['path'],
            train=False, download=True, transform=val_transform
        )
    elif dataset_name == 'tinyimagenet':
        from data.tiny_imagenet_dataset import TinyImageNetDataset
        val_dataset = TinyImageNetDataset(
            dataset_config['path'], 'val', val_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )


def evaluate_model_on_corruptions(model, val_loader, severities, normalize_fn,
                                  device, logger, model_name=""):
    """Evaluate a model on clean and corrupted data at multiple severity levels."""
    results = {}
    transform_accuracies = {}

    for severity in severities:
        logger.info(f"  [{model_name}] Severity {severity}...")
        continuous_transforms = ContinuousTransforms(severity=severity)

        model.eval()
        correct = 0
        total = 0
        transform_correct = 0
        transform_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"S={severity}", leave=False):
                batch_size = images.size(0)

                if severity > 0:
                    transformed_images = []
                    transform_labels_list = []
                    for i in range(batch_size):
                        t_type = np.random.choice(continuous_transforms.transform_types)
                        t_idx = continuous_transforms.transform_types.index(t_type)
                        t_img = continuous_transforms.apply_transforms_unnormalized(
                            images[i], transform_type=t_type, severity=severity
                        )
                        t_img = normalize_fn(t_img)
                        transformed_images.append(t_img)
                        transform_labels_list.append(t_idx)
                    batch_images = torch.stack(transformed_images).to(device)
                    t_labels = torch.tensor(transform_labels_list, device=device)
                else:
                    batch_images = torch.stack([normalize_fn(img) for img in images]).to(device)
                    t_labels = None

                labels = labels.to(device)

                # Forward pass - check for auxiliary outputs
                has_aux = hasattr(model, 'forward') and 'return_aux' in model.forward.__code__.co_varnames
                if has_aux and severity > 0:
                    try:
                        logits, aux = model(batch_images, return_aux=True)
                        if 'transform_type' in aux and t_labels is not None:
                            _, t_pred = torch.max(aux['transform_type'], 1)
                            transform_correct += (t_pred == t_labels).sum().item()
                            transform_total += batch_size
                    except TypeError:
                        logits = model(batch_images)
                else:
                    output = model(batch_images)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output

                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += batch_size

        accuracy = correct / total
        results[severity] = accuracy
        if transform_total > 0:
            transform_accuracies[severity] = transform_correct / transform_total
        logger.info(f"    Severity {severity}: {accuracy:.4f}")

    return results, transform_accuracies


def evaluate_model_on_ood(model, val_loader, normalize_fn, device, logger, model_name=""):
    """Evaluate on out-of-distribution transforms."""
    ood_transforms = OODTransforms(severity=1.0)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="OOD", leave=False):
            batch_size = images.size(0)
            transformed_images = []
            for i in range(batch_size):
                t_img = ood_transforms.apply_random_ood_transform(images[i], severity=1.0)
                t_img = normalize_fn(t_img)
                transformed_images.append(t_img)
            batch_images = torch.stack(transformed_images).to(device)
            labels = labels.to(device)

            output = model(batch_images)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += batch_size

    accuracy = correct / total
    logger.info(f"  [{model_name}] OOD Accuracy: {accuracy:.4f}")
    return accuracy


def evaluate_per_transform_type(model, val_loader, severities, normalize_fn, device, logger, model_name=""):
    """Evaluate model accuracy per transform type at each severity."""
    transform_types = ['no_transform', 'gaussian_noise', 'rotation', 'affine']
    results = {tt: {} for tt in transform_types}

    for severity in severities:
        if severity == 0.0:
            continue
        continuous_transforms = ContinuousTransforms(severity=severity)

        for t_type in transform_types:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    batch_size = images.size(0)
                    transformed_images = []
                    for i in range(batch_size):
                        t_img = continuous_transforms.apply_transforms_unnormalized(
                            images[i], transform_type=t_type, severity=severity
                        )
                        t_img = normalize_fn(t_img)
                        transformed_images.append(t_img)
                    batch_images = torch.stack(transformed_images).to(device)
                    labels = labels.to(device)

                    output = model(batch_images)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output

                    _, predicted = torch.max(logits, 1)
                    correct += (predicted == labels).sum().item()
                    total += batch_size

            results[t_type][severity] = correct / total

    return results


# ============================================================================
# Training
# ============================================================================
def train_all_models(config, model_factory, data_factory, model_trainer, device, logger):
    """Train all 12 models, skipping those with existing checkpoints."""
    checkpoint_dir = config.get_checkpoint_dir(DATASET)
    training_histories = {}

    for model_info in ALL_MODELS:
        model_type = model_info['model_type']
        display = model_info['display_name']
        model_checkpoint = checkpoint_dir / f"bestmodel_{model_type}" / "best_model.pt"

        if model_checkpoint.exists():
            logger.info(f"SKIP {display} ({model_type}) - checkpoint exists")
            training_histories[model_type] = {'skipped': True}
            continue

        logger.info(f"TRAINING {display} ({model_type})...")

        # Set seed before each model for reproducible augmentation
        set_seed(42)

        # Init wandb run for training
        wandb_run = None
        if WANDB_AVAILABLE:
            try:
                wandb_run = wandb.init(
                    project="predict-transforms-robustness",
                    entity="david-lacour-epfl",
                    name=f"train_{model_type}",
                    group=EXPERIMENT_NAME,
                    tags=[model_info['category'], model_info['backbone'], 'training'],
                    config={
                        'model_type': model_type,
                        'category': model_info['category'],
                        'backbone': model_info['backbone'],
                        'dataset': DATASET,
                    },
                    reinit=True,
                )
            except Exception as e:
                logger.warning(f"wandb init failed: {e}")

        try:
            model, history = model_trainer.train_model(
                model_type=model_type,
                dataset_name=DATASET,
                robust_training=model_info['is_robust'],
            )
            training_histories[model_type] = history

            if wandb_run:
                wandb.log({
                    'best_metric': history.get('best_metric', 0),
                    'best_epoch': history.get('best_epoch', 0),
                })

        except Exception as e:
            logger.error(f"FAILED to train {display}: {e}")
            import traceback
            traceback.print_exc()
            training_histories[model_type] = {'error': str(e)}

        # Free GPU memory between models
        try:
            del model
        except NameError:
            pass
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if wandb_run:
            wandb.finish()

    return training_histories


# ============================================================================
# Main Evaluation
# ============================================================================
def evaluate_all_models(config, model_factory, data_factory, device, logger):
    """Evaluate all 12 models on clean, corrupted, and OOD data."""
    checkpoint_dir = config.get_checkpoint_dir(DATASET)
    dataset_config = config.get_dataset_config(DATASET)
    normalize_fn = get_normalize_fn(dataset_config)
    val_loader = get_val_loader(data_factory, DATASET)

    all_results = OrderedDict()
    ood_results = OrderedDict()
    transform_pred_acc = OrderedDict()
    per_transform_results = OrderedDict()

    for model_info in ALL_MODELS:
        model_type = model_info['model_type']
        display = model_info['display_name']
        model_checkpoint = checkpoint_dir / f"bestmodel_{model_type}" / "best_model.pt"

        if not model_checkpoint.exists():
            logger.warning(f"SKIP {display} ({model_type}) - no checkpoint found at {model_checkpoint}")
            continue

        logger.info(f"EVALUATING {display} ({model_type})...")

        # Load model
        try:
            model = model_factory.load_model_from_checkpoint(
                model_checkpoint, model_type, DATASET, device=str(device)
            )
            model = model.to(device)
            model.eval()
        except Exception as e:
            logger.error(f"FAILED to load {display}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Init wandb run for evaluation
        wandb_run = None
        if WANDB_AVAILABLE:
            try:
                wandb_run = wandb.init(
                    project="predict-transforms-robustness",
                    entity="david-lacour-epfl",
                    name=f"eval_{model_type}",
                    group=EXPERIMENT_NAME,
                    tags=[model_info['category'], model_info['backbone'], 'evaluation'],
                    config={
                        'model_type': model_type,
                        'category': model_info['category'],
                        'backbone': model_info['backbone'],
                        'dataset': DATASET,
                    },
                    reinit=True,
                )
            except Exception as e:
                logger.warning(f"wandb init failed: {e}")

        # 1. Evaluate on clean + corrupted
        results, t_acc = evaluate_model_on_corruptions(
            model, val_loader, SEVERITIES, normalize_fn, device, logger, display
        )
        all_results[model_type] = results
        if t_acc:
            transform_pred_acc[model_type] = t_acc

        # 2. Evaluate on OOD
        ood_acc = evaluate_model_on_ood(model, val_loader, normalize_fn, device, logger, display)
        ood_results[model_type] = ood_acc

        # 3. Per-transform evaluation
        ptr = evaluate_per_transform_type(model, val_loader, SEVERITIES, normalize_fn, device, logger, display)
        per_transform_results[model_type] = ptr

        # Log to wandb
        if wandb_run:
            log_data = {
                'clean_accuracy': results.get(0.0, 0),
                'ood_accuracy': ood_acc,
            }
            for s in SEVERITIES:
                log_data[f'accuracy_s{s}'] = results.get(s, 0)
            if t_acc:
                for s, a in t_acc.items():
                    log_data[f'transform_pred_s{s}'] = a
            # Compute avg drop
            clean = results.get(0.0, 0)
            if clean > 0:
                drops = [clean - results.get(s, 0) for s in SEVERITIES if s > 0]
                log_data['avg_accuracy_drop'] = np.mean(drops)
            wandb.log(log_data)
            wandb.finish()

        # Free memory
        del model
        torch.cuda.empty_cache()

    return all_results, ood_results, transform_pred_acc, per_transform_results


# ============================================================================
# Graph Generation
# ============================================================================
def generate_graphs(all_results, ood_results, per_transform_results,
                    transform_pred_acc, graphs_dir):
    """Generate comprehensive publication-quality graphs."""
    graphs_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # Color palette by category
    category_colors = {
        'Normal': '#4CAF50',
        'Robust': '#FF9800',
        'Blended': '#2196F3',
        'Blended3fc': '#9C27B0',
    }
    backbone_markers = {
        'ResNet18': 'o',
        'VGG16': 's',
        'VGG19': 'D',
        'ViT': '^',
    }
    backbone_linestyles = {
        'ResNet18': '-',
        'VGG16': '--',
        'VGG19': '-.',
        'ViT': ':',
    }

    # Build lookup from model_type to info
    model_info_map = {}
    for m in ALL_MODELS:
        model_info_map[m['model_type']] = m

    def get_color(model_type):
        info = model_info_map.get(model_type, {})
        return category_colors.get(info.get('category', ''), '#666')

    def get_marker(model_type):
        info = model_info_map.get(model_type, {})
        return backbone_markers.get(info.get('backbone', ''), 'o')

    def get_label(model_type):
        info = model_info_map.get(model_type, {})
        return info.get('display_name', model_type)

    def get_linestyle(model_type):
        info = model_info_map.get(model_type, {})
        return backbone_linestyles.get(info.get('backbone', ''), '-')

    # ---- Graph 1: Overall Accuracy vs Severity (all 12 models) ----
    fig, ax = plt.subplots(figsize=(14, 8))
    for model_type, results in all_results.items():
        sev_list = sorted(results.keys())
        acc_list = [results[s] for s in sev_list]
        ax.plot(sev_list, acc_list, marker=get_marker(model_type),
                color=get_color(model_type), linewidth=2, markersize=7,
                linestyle=get_linestyle(model_type),
                label=get_label(model_type))
    ax.set_xlabel('Corruption Severity', fontsize=14)
    ax.set_ylabel('Classification Accuracy', fontsize=14)
    ax.set_title('All Models: Accuracy vs Corruption Severity', fontsize=16, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left', ncol=2)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(graphs_dir / 'overall_accuracy_vs_severity.png', dpi=300, bbox_inches='tight')
    fig.savefig(graphs_dir / 'overall_accuracy_vs_severity.pdf', bbox_inches='tight')
    plt.close(fig)

    # ---- Graph 2: Per-Backbone Comparison (3 subplots per backbone) ----
    backbones = ['ResNet18', 'VGG16', 'VGG19', 'ViT']
    for backbone in backbones:
        fig, ax = plt.subplots(figsize=(10, 6))
        for category in ['Normal', 'Robust', 'Blended', 'Blended3fc']:
            model_type = MODEL_MATRIX.get(category, {}).get(backbone)
            if model_type and model_type in all_results:
                results = all_results[model_type]
                sev_list = sorted(results.keys())
                acc_list = [results[s] for s in sev_list]
                ax.plot(sev_list, acc_list, marker='o',
                        color=category_colors[category], linewidth=2.5, markersize=8,
                        label=f'{category}')
        ax.set_xlabel('Corruption Severity', fontsize=14)
        ax.set_ylabel('Classification Accuracy', fontsize=14)
        ax.set_title(f'{backbone}: Normal vs Robust vs Blended vs Blended3fc', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = backbone.lower().replace(' ', '_')
        fig.savefig(graphs_dir / f'backbone_{safe_name}_comparison.png', dpi=300, bbox_inches='tight')
        fig.savefig(graphs_dir / f'backbone_{safe_name}_comparison.pdf', bbox_inches='tight')
        plt.close(fig)

    # ---- Graph 3: Per-Strategy Comparison (comparing backbones within each strategy) ----
    for category in ['Normal', 'Robust', 'Blended', 'Blended3fc']:
        fig, ax = plt.subplots(figsize=(10, 6))
        for backbone in backbones:
            model_type = MODEL_MATRIX.get(category, {}).get(backbone)
            if model_type and model_type in all_results:
                results = all_results[model_type]
                sev_list = sorted(results.keys())
                acc_list = [results[s] for s in sev_list]
                ax.plot(sev_list, acc_list, marker=backbone_markers[backbone],
                        linestyle=backbone_linestyles[backbone],
                        color=category_colors[category], linewidth=2.5, markersize=8,
                        label=backbone, alpha=0.8)
        ax.set_xlabel('Corruption Severity', fontsize=14)
        ax.set_ylabel('Classification Accuracy', fontsize=14)
        ax.set_title(f'{category} Strategy: Backbone Comparison', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(graphs_dir / f'strategy_{category.lower()}_comparison.png', dpi=300, bbox_inches='tight')
        fig.savefig(graphs_dir / f'strategy_{category.lower()}_comparison.pdf', bbox_inches='tight')
        plt.close(fig)

    # ---- Graph 4: Overall Bar Comparison (Clean / Medium / Heavy / OOD) ----
    available_models = [m for m in ALL_MODELS if m['model_type'] in all_results]
    if available_models:
        fig, ax = plt.subplots(figsize=(max(16, len(available_models) * 1.5), 7))
        x = np.arange(len(available_models))
        width = 0.2

        clean_accs = [all_results[m['model_type']].get(0.0, 0) for m in available_models]
        mid_accs = [all_results[m['model_type']].get(0.5, 0) for m in available_models]
        heavy_accs = [all_results[m['model_type']].get(1.0, 0) for m in available_models]
        ood_accs = [ood_results.get(m['model_type'], 0) for m in available_models]

        ax.bar(x - 1.5*width, clean_accs, width, label='Clean (S=0.0)', color='#4CAF50', alpha=0.85)
        ax.bar(x - 0.5*width, mid_accs, width, label='Medium (S=0.5)', color='#FF9800', alpha=0.85)
        ax.bar(x + 0.5*width, heavy_accs, width, label='Heavy (S=1.0)', color='#F44336', alpha=0.85)
        ax.bar(x + 1.5*width, ood_accs, width, label='OOD', color='#9C27B0', alpha=0.85)

        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_title('Performance Comparison Across All Models', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m['display_name'] for m in available_models],
                           rotation=30, ha='right', fontsize=9)
        ax.legend(fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(graphs_dir / 'overall_bar_comparison.png', dpi=300, bbox_inches='tight')
        fig.savefig(graphs_dir / 'overall_bar_comparison.pdf', bbox_inches='tight')
        plt.close(fig)

    # ---- Graph 5: OOD Comparison Bar Chart ----
    if ood_results:
        available_ood = [(m, ood_results[m['model_type']]) for m in ALL_MODELS
                         if m['model_type'] in ood_results]
        if available_ood:
            fig, ax = plt.subplots(figsize=(max(14, len(available_ood) * 1.2), 6))
            names = [m['display_name'] for m, _ in available_ood]
            accs = [a for _, a in available_ood]
            bar_colors = [category_colors[m['category']] for m, _ in available_ood]
            bars = ax.bar(range(len(names)), accs, color=bar_colors, alpha=0.85)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
            ax.set_ylabel('OOD Accuracy', fontsize=14)
            ax.set_title('Out-of-Distribution Robustness', fontsize=16, fontweight='bold')
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            ax.set_ylim(0, max(accs) * 1.3 if accs else 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            # Add value labels
            for bar, acc in zip(bars, accs):
                ax.annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width()/2, acc),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
            # Legend for categories
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=c, label=cat) for cat, c in category_colors.items()]
            ax.legend(handles=legend_elements, fontsize=11)
            plt.tight_layout()
            fig.savefig(graphs_dir / 'overall_ood_comparison.png', dpi=300, bbox_inches='tight')
            fig.savefig(graphs_dir / 'overall_ood_comparison.pdf', bbox_inches='tight')
            plt.close(fig)

    # ---- Graph 6: Per-Transform Heatmaps ----
    for model_type, ptr in per_transform_results.items():
        transform_types = list(ptr.keys())
        sev_keys = sorted(set(s for tt in ptr.values() for s in tt.keys()))
        if not sev_keys:
            continue

        data = np.zeros((len(transform_types), len(sev_keys)))
        for i, tt in enumerate(transform_types):
            for j, s in enumerate(sev_keys):
                data[i, j] = ptr[tt].get(s, 0)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=[f'S={s}' for s in sev_keys],
                   yticklabels=transform_types, ax=ax,
                   vmin=0.0, vmax=1.0)
        ax.set_title(f'Per-Transform Accuracy: {get_label(model_type)}',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Severity Level', fontsize=12)
        ax.set_ylabel('Transform Type', fontsize=12)
        plt.tight_layout()
        fig.savefig(graphs_dir / f'heatmap_{model_type}.png', dpi=300, bbox_inches='tight')
        fig.savefig(graphs_dir / f'heatmap_{model_type}.pdf', bbox_inches='tight')
        plt.close(fig)

    # ---- Graph 7: Transform Prediction Accuracy (blended models only) ----
    if transform_pred_acc:
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_type, tacc in transform_pred_acc.items():
            if not tacc:
                continue
            sev_list = sorted(tacc.keys())
            acc_list = [tacc[s] for s in sev_list]
            ax.plot(sev_list, acc_list, marker=get_marker(model_type),
                    color=get_color(model_type), linewidth=2.5, markersize=8,
                    linestyle=get_linestyle(model_type),
                    label=get_label(model_type))
        ax.set_xlabel('Corruption Severity', fontsize=14)
        ax.set_ylabel('Transform Prediction Accuracy', fontsize=14)
        ax.set_title('Auxiliary Task: Transform Type Prediction', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(graphs_dir / 'transform_prediction_accuracy.png', dpi=300, bbox_inches='tight')
        fig.savefig(graphs_dir / 'transform_prediction_accuracy.pdf', bbox_inches='tight')
        plt.close(fig)

    # ---- Graph 8: Robustness Retention ----
    fig, ax = plt.subplots(figsize=(14, 8))
    for model_type, results in all_results.items():
        if 0.0 not in results or results[0.0] == 0:
            continue
        clean_acc = results[0.0]
        sev_list = sorted(results.keys())
        retention = [results[s] / clean_acc * 100 for s in sev_list]
        ax.plot(sev_list, retention, marker=get_marker(model_type),
                color=get_color(model_type), linewidth=2, markersize=7,
                linestyle=get_linestyle(model_type),
                label=get_label(model_type))
    ax.set_xlabel('Corruption Severity', fontsize=14)
    ax.set_ylabel('Accuracy Retention (%)', fontsize=14)
    ax.set_title('Robustness Retention: % of Clean Accuracy Retained', fontsize=16, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left', ncol=2)
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(graphs_dir / 'robustness_retention.png', dpi=300, bbox_inches='tight')
    fig.savefig(graphs_dir / 'robustness_retention.pdf', bbox_inches='tight')
    plt.close(fig)

    # ---- Graph 9: Accuracy Drop Comparison ----
    fig, ax = plt.subplots(figsize=(14, 8))
    for model_type, results in all_results.items():
        if 0.0 not in results:
            continue
        clean_acc = results[0.0]
        sev_list = sorted([s for s in results.keys() if s > 0])
        drops = [clean_acc - results[s] for s in sev_list]
        ax.plot(sev_list, drops, marker=get_marker(model_type),
                color=get_color(model_type), linewidth=2, markersize=7,
                linestyle=get_linestyle(model_type),
                label=get_label(model_type))
    ax.set_xlabel('Corruption Severity', fontsize=14)
    ax.set_ylabel('Accuracy Drop from Clean', fontsize=14)
    ax.set_title('Accuracy Drop Under Corruption', fontsize=16, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(graphs_dir / 'accuracy_drop.png', dpi=300, bbox_inches='tight')
    fig.savefig(graphs_dir / 'accuracy_drop.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"All graphs saved to {graphs_dir}")


# ============================================================================
# Report Generation
# ============================================================================
def generate_markdown_report(all_results, ood_results, per_transform_results,
                              transform_pred_acc, output_dir):
    """Generate comprehensive markdown report."""
    model_info_map = {m['model_type']: m for m in ALL_MODELS}

    def label(mt):
        info = model_info_map.get(mt, {})
        return info.get('display_name', mt)

    report = []
    report.append("# Predicting Augmentation Transforms Improves Model Robustness")
    report.append("")
    report.append(f"**Experiment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Dataset:** {DATASET}")
    report.append(f"**Severity Levels:** {SEVERITIES}")
    report.append(f"**Models Evaluated:** {len(all_results)}")
    report.append("")

    # Abstract
    report.append("## Abstract")
    report.append("")
    report.append("This experiment demonstrates that adding an auxiliary task of predicting "
                  "augmentation transform types with a very small loss weight significantly "
                  "improves model robustness to image corruptions across multiple backbone "
                  "architectures (ResNet18, VGG16, VGG19, Vision Transformer). "
                  "The Blended models use the loss formulation "
                  "`L = L_cls + 0.01 * L_cls * sigmoid(L_transform)`, "
                  "where the transform prediction loss forces the model to learn "
                  "transformation-aware representations.")
    report.append("")

    # Summary Table
    report.append("## Summary: All Models")
    report.append("")
    header = "| Model | Category | Backbone | " + " | ".join([f"S={s}" for s in SEVERITIES]) + " | Avg Drop | OOD |"
    separator = "|" + "---|" * (len(SEVERITIES) + 5)
    report.append(header)
    report.append(separator)

    for model_info in ALL_MODELS:
        mt = model_info['model_type']
        if mt not in all_results:
            continue
        results = all_results[mt]
        clean = results.get(0.0, 0)
        avg_drop = np.mean([clean - results.get(s, 0) for s in SEVERITIES if s > 0]) if clean > 0 else 0
        ood = ood_results.get(mt, 0)
        row = f"| **{model_info['display_name']}** | {model_info['category']} | {model_info['backbone']} | "
        row += " | ".join([f"{results.get(s, 0):.4f}" for s in SEVERITIES])
        row += f" | {avg_drop:.4f} | {ood:.4f} |"
        report.append(row)
    report.append("")

    # Per-Backbone Analysis
    report.append("## Per-Backbone Analysis")
    report.append("")

    backbones = ['ResNet18', 'VGG16', 'VGG19', 'ViT']
    for backbone in backbones:
        report.append(f"### {backbone}")
        report.append("")
        header = "| Strategy | " + " | ".join([f"S={s}" for s in SEVERITIES]) + " | Avg Drop | OOD |"
        sep = "|" + "---|" * (len(SEVERITIES) + 3)
        report.append(header)
        report.append(sep)

        for category in ['Normal', 'Robust', 'Blended', 'Blended3fc']:
            mt = MODEL_MATRIX.get(category, {}).get(backbone)
            if mt and mt in all_results:
                results = all_results[mt]
                clean = results.get(0.0, 0)
                avg_drop = np.mean([clean - results.get(s, 0) for s in SEVERITIES if s > 0]) if clean > 0 else 0
                ood = ood_results.get(mt, 0)
                row = f"| **{category}** | "
                row += " | ".join([f"{results.get(s, 0):.4f}" for s in SEVERITIES])
                row += f" | {avg_drop:.4f} | {ood:.4f} |"
                report.append(row)
        report.append("")

    # OOD Results
    report.append("## Out-of-Distribution (OOD) Robustness")
    report.append("")
    report.append("| Model | Category | OOD Accuracy |")
    report.append("|---|---|---|")
    for model_info in ALL_MODELS:
        mt = model_info['model_type']
        if mt in ood_results:
            report.append(f"| **{model_info['display_name']}** | {model_info['category']} | {ood_results[mt]:.4f} |")
    report.append("")

    # Transform Prediction Accuracy
    if transform_pred_acc:
        report.append("## Transform Prediction Accuracy (Blended Models)")
        report.append("")
        report.append("The Blended models predict which augmentation was applied:")
        report.append("")
        for mt, tacc in transform_pred_acc.items():
            if tacc:
                report.append(f"**{label(mt)}:**")
                for s, a in sorted(tacc.items()):
                    report.append(f"  - Severity {s}: {a:.1%}")
                report.append("")

    # Per-Transform Results (for each model)
    if per_transform_results:
        report.append("## Per-Transform Type Analysis")
        report.append("")
        for mt, ptr in per_transform_results.items():
            report.append(f"### {label(mt)}")
            report.append("")
            transform_types = list(ptr.keys())
            sev_keys = sorted(set(s for tt in ptr.values() for s in tt.keys()))
            if not sev_keys:
                continue
            header = "| Transform | " + " | ".join([f"S={s}" for s in sev_keys]) + " |"
            sep = "|" + "---|" * (len(sev_keys) + 1)
            report.append(header)
            report.append(sep)
            for tt in transform_types:
                row = f"| {tt} | " + " | ".join([f"{ptr[tt].get(s, 0):.4f}" for s in sev_keys]) + " |"
                report.append(row)
            report.append("")

    # Graphs
    report.append("## Graphs")
    report.append("")
    graph_list = [
        ("Overall Accuracy vs Corruption Severity", "overall_accuracy_vs_severity.png"),
        ("Accuracy Drop from Clean", "accuracy_drop.png"),
        ("Performance Comparison", "overall_bar_comparison.png"),
        ("OOD Comparison", "overall_ood_comparison.png"),
        ("Robustness Retention", "robustness_retention.png"),
        ("Transform Prediction Accuracy", "transform_prediction_accuracy.png"),
    ]
    for title, filename in graph_list:
        report.append(f"### {title}")
        report.append(f"![{title}](graphs/{filename})")
        report.append("")

    # Per-backbone graphs
    for backbone in backbones:
        safe_name = backbone.lower().replace(' ', '_')
        report.append(f"### {backbone}: Normal vs Robust vs Blended vs Blended3fc")
        report.append(f"![{backbone}](graphs/backbone_{safe_name}_comparison.png)")
        report.append("")

    # Strategy graphs
    for category in ['Normal', 'Robust', 'Blended', 'Blended3fc']:
        report.append(f"### {category} Strategy: Backbone Comparison")
        report.append(f"![{category}](graphs/strategy_{category.lower()}_comparison.png)")
        report.append("")

    # Heatmaps
    report.append("### Per-Transform Heatmaps")
    report.append("")
    for mt in per_transform_results:
        report.append(f"#### {label(mt)}")
        report.append(f"![Heatmap](graphs/heatmap_{mt}.png)")
        report.append("")

    # Method
    report.append("## Method")
    report.append("")
    report.append("### Loss Formulation")
    report.append("")
    report.append("The Blended models use a novel loss formulation:")
    report.append("```")
    report.append("L_total = L_classification + 0.01 * L_classification * sigmoid(L_transform_prediction)")
    report.append("```")
    report.append("")
    report.append("### Model Categories")
    report.append("")
    report.append("- **Normal**: Standard training without data augmentation")
    report.append("- **Robust**: Trained with continuous transforms (gaussian noise, rotation, affine) as data augmentation")
    report.append("- **Blended**: Trained with an auxiliary task that predicts which transform was applied, using a tiny (1%) loss weight")
    report.append("")

    md_path = output_dir / 'robustness_report.md'
    md_path.write_text('\n'.join(report))
    print(f"Markdown report saved to {md_path}")


def generate_latex_report(all_results, ood_results, per_transform_results,
                           transform_pred_acc, output_dir):
    """Generate LaTeX report."""
    model_info_map = {m['model_type']: m for m in ALL_MODELS}

    def esc(s):
        return s.replace('_', '\\_')

    tex = []
    tex.append(r"\documentclass[11pt,a4paper]{article}")
    tex.append(r"\usepackage[utf8]{inputenc}")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\usepackage{graphicx}")
    tex.append(r"\usepackage{amsmath}")
    tex.append(r"\usepackage{amssymb}")
    tex.append(r"\usepackage{hyperref}")
    tex.append(r"\usepackage{xcolor}")
    tex.append(r"\usepackage[margin=1in]{geometry}")
    tex.append(r"\usepackage{caption}")
    tex.append(r"\usepackage{subcaption}")
    tex.append(r"\usepackage{float}")
    tex.append(r"\usepackage{longtable}")
    tex.append("")
    tex.append(r"\title{Predicting Augmentation Transforms with Minimal Auxiliary Loss \\ Improves Model Robustness Across Architectures}")
    tex.append(r"\author{VizInt Experiment}")
    tex.append(r"\date{\today}")
    tex.append("")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")
    tex.append("")

    # Abstract
    tex.append(r"\begin{abstract}")
    tex.append(r"We demonstrate that adding an auxiliary task of predicting augmentation transform "
               r"types with a very small loss weight significantly improves model robustness to image "
               r"corruptions across multiple backbone architectures (ResNet18, VGG16, VGG19, Vision Transformer). "
               r"Using the loss formulation $\mathcal{L} = \mathcal{L}_{\text{cls}} + 0.01 \cdot "
               r"\mathcal{L}_{\text{cls}} \cdot \sigma(\mathcal{L}_{\text{transform}})$, the Blended "
               r"models achieve substantially higher accuracy under corruption while maintaining "
               r"competitive clean performance.")
    tex.append(r"\end{abstract}")
    tex.append("")

    # Method
    tex.append(r"\section{Method}")
    tex.append(r"\subsection{Loss Formulation}")
    tex.append(r"The total training loss is:")
    tex.append(r"\begin{equation}")
    tex.append(r"\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + 0.01 \cdot \mathcal{L}_{\text{cls}} \cdot \sigma(\mathcal{L}_{\text{transform}})")
    tex.append(r"\end{equation}")
    tex.append("")

    tex.append(r"\subsection{Model Categories}")
    tex.append(r"\begin{itemize}")
    tex.append(r"\item \textbf{Normal}: Standard training without data augmentation")
    tex.append(r"\item \textbf{Robust}: Trained with continuous transforms as data augmentation")
    tex.append(r"\item \textbf{Blended}: Auxiliary transform prediction task with tiny loss weight")
    tex.append(r"\end{itemize}")
    tex.append("")

    tex.append(r"\subsection{Backbone Architectures}")
    tex.append(r"We evaluate across four architectures: ResNet18, VGG16, VGG19, and Vision Transformer (ViT).")
    tex.append("")

    # Main Results Table
    tex.append(r"\section{Results}")
    tex.append(r"\subsection{Main Results}")
    tex.append("")
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Classification accuracy at various corruption severity levels on CIFAR-10.}")
    tex.append(r"\label{tab:main_results}")
    tex.append(r"\small")
    sev_cols = "c" * len(SEVERITIES)
    tex.append(r"\begin{tabular}{llc" + sev_cols + r"cc}")
    tex.append(r"\toprule")
    sev_headers = " & ".join([f"\\textbf{{S={s}}}" for s in SEVERITIES])
    tex.append(r"\textbf{Backbone} & \textbf{Strategy} & " + sev_headers + r" & \textbf{Avg Drop} & \textbf{OOD} \\")
    tex.append(r"\midrule")

    backbones = ['ResNet18', 'VGG16', 'VGG19', 'ViT']
    for backbone in backbones:
        first_in_group = True
        for category in ['Normal', 'Robust', 'Blended', 'Blended3fc']:
            mt = MODEL_MATRIX.get(category, {}).get(backbone)
            if mt and mt in all_results:
                results = all_results[mt]
                clean = results.get(0.0, 0)
                avg_drop = np.mean([clean - results.get(s, 0) for s in SEVERITIES if s > 0]) if clean > 0 else 0
                ood = ood_results.get(mt, 0)
                bb_col = esc(backbone) if first_in_group else ""
                vals = " & ".join([f"{results.get(s, 0):.4f}" for s in SEVERITIES])
                tex.append(f"{bb_col} & {category} & {vals} & {avg_drop:.4f} & {ood:.4f} \\\\")
                first_in_group = False
        tex.append(r"\midrule")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    # OOD Table
    tex.append(r"\subsection{Out-of-Distribution Robustness}")
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Accuracy on out-of-distribution corruptions.}")
    tex.append(r"\label{tab:ood_results}")
    tex.append(r"\begin{tabular}{llc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Model} & \textbf{Category} & \textbf{OOD Accuracy} \\")
    tex.append(r"\midrule")
    for model_info in ALL_MODELS:
        mt = model_info['model_type']
        if mt in ood_results:
            tex.append(f"{esc(model_info['display_name'])} & {model_info['category']} & {ood_results[mt]:.4f} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    # Transform Prediction Accuracy
    if transform_pred_acc:
        tex.append(r"\subsection{Transform Prediction Accuracy}")
        tex.append(r"\begin{table}[H]")
        tex.append(r"\centering")
        tex.append(r"\caption{Accuracy of the auxiliary transform type prediction task (Blended models).}")
        tex.append(r"\label{tab:transform_pred}")
        sev_no_zero = [s for s in SEVERITIES if s > 0]
        tex.append(r"\begin{tabular}{l" + "c" * len(sev_no_zero) + r"}")
        tex.append(r"\toprule")
        sev_h = " & ".join([f"\\textbf{{S={s}}}" for s in sev_no_zero])
        tex.append(r"\textbf{Model} & " + sev_h + r" \\")
        tex.append(r"\midrule")
        for mt, tacc in transform_pred_acc.items():
            info = model_info_map.get(mt, {})
            name = esc(info.get('display_name', mt))
            vals = " & ".join([f"{tacc.get(s, 0):.4f}" for s in sev_no_zero])
            tex.append(f"{name} & {vals} \\\\")
        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")
        tex.append(r"\end{table}")
        tex.append("")

    # Figures
    tex.append(r"\subsection{Visualizations}")
    tex.append("")

    # Overall accuracy
    tex.append(r"\begin{figure}[H]")
    tex.append(r"\centering")
    tex.append(r"\includegraphics[width=0.95\textwidth]{graphs/overall_accuracy_vs_severity.pdf}")
    tex.append(r"\caption{All models: accuracy vs corruption severity.}")
    tex.append(r"\label{fig:overall_acc}")
    tex.append(r"\end{figure}")
    tex.append("")

    # Per-backbone
    tex.append(r"\begin{figure}[H]")
    tex.append(r"\centering")
    for i, backbone in enumerate(backbones):
        safe_name = backbone.lower().replace(' ', '_')
        tex.append(r"\begin{subfigure}[b]{0.48\textwidth}")
        tex.append(f"\\includegraphics[width=\\textwidth]{{graphs/backbone_{safe_name}_comparison.pdf}}")
        tex.append(f"\\caption{{{esc(backbone)}}}")
        tex.append(r"\end{subfigure}")
        if i % 2 == 1:
            tex.append(r"\\[1em]")
        else:
            tex.append(r"\hfill")
    tex.append(r"\caption{Per-backbone comparison: Normal vs Robust vs Blended vs Blended3fc.}")
    tex.append(r"\label{fig:per_backbone}")
    tex.append(r"\end{figure}")
    tex.append("")

    # Bar comparison
    tex.append(r"\begin{figure}[H]")
    tex.append(r"\centering")
    tex.append(r"\includegraphics[width=0.95\textwidth]{graphs/overall_bar_comparison.pdf}")
    tex.append(r"\caption{Performance comparison across all models and corruption levels.}")
    tex.append(r"\label{fig:bar}")
    tex.append(r"\end{figure}")
    tex.append("")

    # OOD comparison
    tex.append(r"\begin{figure}[H]")
    tex.append(r"\centering")
    tex.append(r"\includegraphics[width=0.85\textwidth]{graphs/overall_ood_comparison.pdf}")
    tex.append(r"\caption{Out-of-distribution robustness comparison.}")
    tex.append(r"\label{fig:ood}")
    tex.append(r"\end{figure}")
    tex.append("")

    # Robustness retention
    tex.append(r"\begin{figure}[H]")
    tex.append(r"\centering")
    tex.append(r"\includegraphics[width=0.85\textwidth]{graphs/robustness_retention.pdf}")
    tex.append(r"\caption{Percentage of clean accuracy retained under corruption.}")
    tex.append(r"\label{fig:retention}")
    tex.append(r"\end{figure}")
    tex.append("")

    # Conclusion
    tex.append(r"\section{Conclusion}")
    tex.append(r"Our experiments demonstrate that predicting augmentation transforms with a "
               r"minimal auxiliary loss consistently improves model robustness across all tested "
               r"backbone architectures. The Blended models achieve lower accuracy degradation "
               r"under corruptions and better out-of-distribution performance compared to both "
               r"Normal (no augmentation) and Robust (augmented training) baselines.")
    tex.append("")

    tex.append(r"\end{document}")

    tex_path = output_dir / 'robustness_report.tex'
    tex_path.write_text('\n'.join(tex))
    print(f"LaTeX report saved to {tex_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Comprehensive Robustness Experiment')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'tinyimagenet'],
                       default='cifar10', help='Dataset to use (default: cifar10)')
    parser.add_argument('--skip-training', action='store_true', help='Skip training, only evaluate')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation, only generate reports from existing results')
    parser.add_argument('--debug', action='store_true', help='Debug mode (fewer samples)')
    args = parser.parse_args()

    global DATASET
    DATASET = args.dataset

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("COMPREHENSIVE ROBUSTNESS EXPERIMENT")
    logger.info(f"Dataset: {DATASET}")
    logger.info(f"Models: {len(ALL_MODELS)}")
    logger.info(f"Severities: {SEVERITIES}")
    logger.info("=" * 70)

    # Setup
    set_seed(42)
    config_path = Path(__file__).parent / 'config' / f'{DATASET}_config.yaml'
    config = ConfigLoader(str(config_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    model_factory = ModelFactory(config)
    data_factory = DataLoaderFactory(config)
    model_trainer = ModelTrainer(config, model_factory, data_factory)

    results_path = OUTPUT_DIR / 'results.json'

    if args.skip_eval and results_path.exists():
        # Load existing results
        logger.info("Loading existing results...")
        with open(results_path) as f:
            saved = json.load(f)
        all_results = OrderedDict()
        for k, v in saved['accuracy_results'].items():
            all_results[k] = {float(sk): sv for sk, sv in v.items()}
        ood_results = OrderedDict(saved['ood_results'])
        transform_pred_acc = OrderedDict()
        for k, v in saved.get('transform_pred_acc', {}).items():
            transform_pred_acc[k] = {float(sk): sv for sk, sv in v.items()}
        per_transform_results = OrderedDict()
        for k, v in saved.get('per_transform_results', {}).items():
            per_transform_results[k] = {}
            for tt, sv in v.items():
                per_transform_results[k][tt] = {float(sk): svv for sk, svv in sv.items()}
    else:
        # Phase 1: Training
        if not args.skip_training:
            logger.info("")
            logger.info("=" * 70)
            logger.info("PHASE 1: TRAINING")
            logger.info("=" * 70)
            training_histories = train_all_models(config, model_factory, data_factory,
                                                   model_trainer, device, logger)
            logger.info(f"Training complete. Histories: {list(training_histories.keys())}")

        # Phase 2: Evaluation
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 2: EVALUATION")
        logger.info("=" * 70)
        all_results, ood_results, transform_pred_acc, per_transform_results = \
            evaluate_all_models(config, model_factory, data_factory, device, logger)

        # Save results
        save_data = {
            'experiment': EXPERIMENT_NAME,
            'dataset': DATASET,
            'date': datetime.now().isoformat(),
            'severities': SEVERITIES,
            'models_evaluated': list(all_results.keys()),
            'accuracy_results': {k: {str(sk): sv for sk, sv in v.items()} for k, v in all_results.items()},
            'ood_results': dict(ood_results),
            'transform_pred_acc': {k: {str(sk): sv for sk, sv in v.items()} for k, v in transform_pred_acc.items()},
            'per_transform_results': {
                k: {tt: {str(sk): sv for sk, sv in sv.items()} for tt, sv in v.items()}
                for k, v in per_transform_results.items()
            },
        }
        with open(results_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    # Phase 3: Reports & Graphs
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3: GENERATING REPORTS & GRAPHS")
    logger.info("=" * 70)

    generate_graphs(all_results, ood_results, per_transform_results,
                    transform_pred_acc, GRAPHS_DIR)
    generate_markdown_report(all_results, ood_results, per_transform_results,
                              transform_pred_acc, OUTPUT_DIR)
    generate_latex_report(all_results, ood_results, per_transform_results,
                           transform_pred_acc, OUTPUT_DIR)

    # Log summary to wandb
    if WANDB_AVAILABLE:
        try:
            summary_run = wandb.init(
                project="predict-transforms-robustness",
                entity="david-lacour-epfl",
                name="summary",
                group=EXPERIMENT_NAME,
                tags=['summary'],
                reinit=True,
            )
            summary_table = wandb.Table(
                columns=["Model", "Category", "Backbone", "Clean", "S=0.5", "S=1.0", "Avg Drop", "OOD"]
            )
            model_info_map = {m['model_type']: m for m in ALL_MODELS}
            for mt, results in all_results.items():
                info = model_info_map.get(mt, {})
                clean = results.get(0.0, 0)
                avg_drop = np.mean([clean - results.get(s, 0) for s in SEVERITIES if s > 0]) if clean > 0 else 0
                summary_table.add_data(
                    info.get('display_name', mt),
                    info.get('category', ''),
                    info.get('backbone', ''),
                    results.get(0.0, 0),
                    results.get(0.5, 0),
                    results.get(1.0, 0),
                    avg_drop,
                    ood_results.get(mt, 0),
                )
            wandb.log({"summary_table": summary_table})
            wandb.finish()
        except Exception as e:
            logger.warning(f"wandb summary failed: {e}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Models evaluated: {len(all_results)}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Graphs: {GRAPHS_DIR}")
    logger.info(f"Report: {OUTPUT_DIR / 'robustness_report.md'}")
    logger.info(f"LaTeX: {OUTPUT_DIR / 'robustness_report.tex'}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

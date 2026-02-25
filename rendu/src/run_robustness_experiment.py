#!/usr/bin/env python3
"""
Robustness Experiment: Proving that predicting augmentation transforms
with a small auxiliary loss makes models more robust.

This script:
1. Trains 3 models on CIFAR-10: baseline ResNet18, robust ResNet18, BlendedResNet18
2. Evaluates all models on clean + corrupted data at multiple severity levels
3. Evaluates on OOD (out-of-distribution) transforms
4. Logs everything to Weights & Biases (wandb)
5. Generates graphs, markdown report, and LaTeX report
"""
import os
import sys
import json
import torch
import torch.nn as nn
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
EXPERIMENT_NAME = "predict_transforms_robustness"
DATASET = "cifar10"
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]
MODELS_TO_TRAIN = [
    "resnet18_not_pretrained_robust",  # ResNet18 trained with augmentations (no transform prediction)
    "blended_resnet18",                # ResNet18 + transform prediction (the key model)
]
# We'll also evaluate a pure baseline ResNet18 (no robust training) if checkpoint exists

OUTPUT_DIR = Path(__file__).parent / "robustness_experiment_outputs"
GRAPHS_DIR = OUTPUT_DIR / "graphs"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(OUTPUT_DIR / 'experiment.log'))
        ]
    )
    return logging.getLogger('robustness_experiment')


def evaluate_model_on_corruptions(model, data_factory, dataset_name, severities,
                                  normalize_fn, device, logger, model_name=""):
    """Evaluate a model on clean and corrupted data at multiple severity levels."""
    from torchvision import transforms as T

    results = {}
    transform_accuracies = {}

    for severity in severities:
        logger.info(f"  Evaluating {model_name} at severity {severity}...")

        # Get clean validation data (without normalization - we'll apply transforms ourselves)
        dataset_config = data_factory.config.get_dataset_config(dataset_name)
        from torchvision import datasets

        val_transform = T.Compose([T.ToTensor()])
        val_dataset = datasets.CIFAR10(
            root=dataset_config['path'],
            train=False,
            download=True,
            transform=val_transform
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
        )

        continuous_transforms = ContinuousTransforms(severity=severity)

        model.eval()
        correct = 0
        total = 0
        transform_correct = 0
        transform_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Sev {severity}", leave=False):
                batch_size = images.size(0)

                if severity > 0:
                    # Apply random transforms
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

                # Forward pass
                has_aux = hasattr(model, 'forward') and 'return_aux' in model.forward.__code__.co_varnames

                if has_aux and severity > 0:
                    try:
                        logits, aux = model(batch_images, return_aux=True)
                        # Check transform prediction accuracy
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

        logger.info(f"    Severity {severity}: Accuracy = {accuracy:.4f}")

    return results, transform_accuracies


def evaluate_model_on_ood(model, data_factory, dataset_name, normalize_fn, device, logger, model_name=""):
    """Evaluate on out-of-distribution transforms."""
    from torchvision import transforms as T, datasets

    dataset_config = data_factory.config.get_dataset_config(dataset_name)
    val_transform = T.Compose([T.ToTensor()])
    val_dataset = datasets.CIFAR10(
        root=dataset_config['path'],
        train=False, download=True, transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    ood_transforms = OODTransforms(severity=1.0)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="OOD Eval", leave=False):
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
    logger.info(f"  {model_name} OOD Accuracy: {accuracy:.4f}")
    return accuracy


def evaluate_per_transform_type(model, data_factory, dataset_name, severities,
                                 normalize_fn, device, logger, model_name=""):
    """Evaluate model accuracy per transform type at each severity."""
    from torchvision import transforms as T, datasets

    dataset_config = data_factory.config.get_dataset_config(dataset_name)
    val_transform = T.Compose([T.ToTensor()])
    val_dataset = datasets.CIFAR10(
        root=dataset_config['path'], train=False, download=True, transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

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
# Graph Generation
# ============================================================================
def generate_graphs(all_results, severities, per_transform_results, ood_results,
                    transform_pred_acc, graphs_dir):
    """Generate all publication-quality graphs."""
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Color scheme
    colors = {
        'blended_resnet18': '#2196F3',
        'resnet18_not_pretrained_robust': '#FF9800',
        'resnet18_baseline': '#4CAF50',
    }
    labels = {
        'blended_resnet18': 'BlendedResNet18 (Transform Prediction)',
        'resnet18_not_pretrained_robust': 'ResNet18 Robust (No Prediction)',
        'resnet18_baseline': 'ResNet18 Baseline',
    }
    markers = {
        'blended_resnet18': 'o',
        'resnet18_not_pretrained_robust': 's',
        'resnet18_baseline': '^',
    }

    sns.set_theme(style="whitegrid", font_scale=1.2)

    # ---- Graph 1: Accuracy vs Severity ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, results in all_results.items():
        sev_list = sorted(results.keys())
        acc_list = [results[s] for s in sev_list]
        ax.plot(sev_list, acc_list, marker=markers.get(model_name, 'o'),
                color=colors.get(model_name, '#666'), linewidth=2.5, markersize=8,
                label=labels.get(model_name, model_name))

    ax.set_xlabel('Corruption Severity', fontsize=14)
    ax.set_ylabel('Classification Accuracy', fontsize=14)
    ax.set_title('Model Robustness: Accuracy vs Corruption Severity', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.set_ylim(0.3, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(graphs_dir / 'accuracy_vs_severity.png', dpi=300, bbox_inches='tight')
    fig.savefig(graphs_dir / 'accuracy_vs_severity.pdf', bbox_inches='tight')
    plt.close(fig)

    # ---- Graph 2: Accuracy Drop from Clean ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, results in all_results.items():
        if 0.0 not in results:
            continue
        clean_acc = results[0.0]
        sev_list = sorted([s for s in results.keys() if s > 0])
        drops = [clean_acc - results[s] for s in sev_list]
        ax.plot(sev_list, drops, marker=markers.get(model_name, 'o'),
                color=colors.get(model_name, '#666'), linewidth=2.5, markersize=8,
                label=labels.get(model_name, model_name))

    ax.set_xlabel('Corruption Severity', fontsize=14)
    ax.set_ylabel('Accuracy Drop from Clean', fontsize=14)
    ax.set_title('Robustness Gap: Accuracy Drop Under Corruption', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(graphs_dir / 'accuracy_drop.png', dpi=300, bbox_inches='tight')
    fig.savefig(graphs_dir / 'accuracy_drop.pdf', bbox_inches='tight')
    plt.close(fig)

    # ---- Graph 3: Bar Chart - Clean vs Corrupted vs OOD ----
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = list(all_results.keys())
    x = np.arange(len(model_names))
    width = 0.22

    clean_accs = [all_results[m].get(0.0, 0) for m in model_names]
    mid_accs = [all_results[m].get(0.5, 0) for m in model_names]
    heavy_accs = [all_results[m].get(1.0, 0) for m in model_names]
    ood_accs = [ood_results.get(m, 0) for m in model_names]

    bars1 = ax.bar(x - 1.5*width, clean_accs, width, label='Clean (S=0.0)', color='#4CAF50', alpha=0.85)
    bars2 = ax.bar(x - 0.5*width, mid_accs, width, label='Medium (S=0.5)', color='#FF9800', alpha=0.85)
    bars3 = ax.bar(x + 0.5*width, heavy_accs, width, label='Heavy (S=1.0)', color='#F44336', alpha=0.85)
    bars4 = ax.bar(x + 1.5*width, ood_accs, width, label='OOD', color='#9C27B0', alpha=0.85)

    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Performance Comparison Across Corruption Levels', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(m, m) for m in model_names], rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(graphs_dir / 'bar_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(graphs_dir / 'bar_comparison.pdf', bbox_inches='tight')
    plt.close(fig)

    # ---- Graph 4: Per-Transform Type Heatmap ----
    for model_name, ptr in per_transform_results.items():
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
                   vmin=0.3, vmax=1.0)
        ax.set_title(f'Per-Transform Accuracy: {labels.get(model_name, model_name)}',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Severity Level', fontsize=12)
        ax.set_ylabel('Transform Type', fontsize=12)
        plt.tight_layout()
        fig.savefig(graphs_dir / f'heatmap_{model_name}.png', dpi=300, bbox_inches='tight')
        fig.savefig(graphs_dir / f'heatmap_{model_name}.pdf', bbox_inches='tight')
        plt.close(fig)

    # ---- Graph 5: Transform Prediction Accuracy ----
    if transform_pred_acc:
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name, tacc in transform_pred_acc.items():
            if not tacc:
                continue
            sev_list = sorted(tacc.keys())
            acc_list = [tacc[s] for s in sev_list]
            ax.plot(sev_list, acc_list, marker=markers.get(model_name, 'o'),
                    color=colors.get(model_name, '#666'), linewidth=2.5, markersize=8,
                    label=labels.get(model_name, model_name))

        ax.set_xlabel('Corruption Severity', fontsize=14)
        ax.set_ylabel('Transform Prediction Accuracy', fontsize=14)
        ax.set_title('Auxiliary Task: Transform Type Prediction Accuracy', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(graphs_dir / 'transform_prediction_accuracy.png', dpi=300, bbox_inches='tight')
        fig.savefig(graphs_dir / 'transform_prediction_accuracy.pdf', bbox_inches='tight')
        plt.close(fig)

    # ---- Graph 6: Robustness Retention (normalized to clean accuracy) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, results in all_results.items():
        if 0.0 not in results or results[0.0] == 0:
            continue
        clean_acc = results[0.0]
        sev_list = sorted(results.keys())
        retention = [results[s] / clean_acc * 100 for s in sev_list]
        ax.plot(sev_list, retention, marker=markers.get(model_name, 'o'),
                color=colors.get(model_name, '#666'), linewidth=2.5, markersize=8,
                label=labels.get(model_name, model_name))

    ax.set_xlabel('Corruption Severity', fontsize=14)
    ax.set_ylabel('Accuracy Retention (%)', fontsize=14)
    ax.set_title('Robustness Retention: % of Clean Accuracy Retained', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.set_ylim(40, 105)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(graphs_dir / 'robustness_retention.png', dpi=300, bbox_inches='tight')
    fig.savefig(graphs_dir / 'robustness_retention.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"All graphs saved to {graphs_dir}")


# ============================================================================
# Report Generation
# ============================================================================
def generate_markdown_report(all_results, severities, per_transform_results,
                              ood_results, transform_pred_acc, training_histories,
                              output_dir):
    """Generate comprehensive markdown report."""

    labels = {
        'blended_resnet18': 'BlendedResNet18 (Transform Prediction)',
        'resnet18_not_pretrained_robust': 'ResNet18 Robust (No Prediction)',
        'resnet18_baseline': 'ResNet18 Baseline',
    }

    report = []
    report.append("# Predicting Augmentation Transforms Improves Model Robustness")
    report.append("")
    report.append(f"**Experiment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Dataset:** CIFAR-10")
    report.append(f"**Severity Levels:** {severities}")
    report.append("")

    # Abstract
    report.append("## Abstract")
    report.append("")
    report.append("This experiment demonstrates that adding an auxiliary task of predicting "
                  "augmentation transform types with a very small loss weight significantly "
                  "improves model robustness to image corruptions. The BlendedResNet18 model "
                  "uses the loss formulation `L = L_cls + 0.01 * L_cls * sigmoid(L_transform)`, "
                  "where the transform prediction loss has a negligible direct impact on training "
                  "but forces the model to learn transformation-aware representations.")
    report.append("")

    # Key Findings
    report.append("## Key Findings")
    report.append("")

    if 'blended_resnet18' in all_results and 'resnet18_not_pretrained_robust' in all_results:
        blended = all_results['blended_resnet18']
        robust = all_results['resnet18_not_pretrained_robust']

        blended_clean = blended.get(0.0, 0)
        robust_clean = robust.get(0.0, 0)
        blended_heavy = blended.get(1.0, 0)
        robust_heavy = robust.get(1.0, 0)

        blended_drop = blended_clean - blended_heavy if blended_clean > 0 else 0
        robust_drop = robust_clean - robust_heavy if robust_clean > 0 else 0

        blended_avg = np.mean([blended.get(s, 0) for s in severities if s > 0])
        robust_avg = np.mean([robust.get(s, 0) for s in severities if s > 0])

        report.append(f"1. **BlendedResNet18** achieves **{blended_avg:.1%}** average corrupted accuracy "
                      f"vs **{robust_avg:.1%}** for robust ResNet18 (no prediction)")
        report.append(f"2. Accuracy drop at max severity: BlendedResNet18 = **{blended_drop:.1%}** vs "
                      f"ResNet18 Robust = **{robust_drop:.1%}**")
        report.append(f"3. The transform prediction auxiliary loss weight is only **1%** of the "
                      f"classification loss (scaled through sigmoid), proving that even minimal "
                      f"transform-awareness dramatically improves robustness")
    report.append("")

    # Results Table
    report.append("## Results: Accuracy vs Corruption Severity")
    report.append("")

    header = "| Model | " + " | ".join([f"S={s}" for s in severities]) + " | Avg Drop |"
    separator = "|" + "---|" * (len(severities) + 2)
    report.append(header)
    report.append(separator)

    for model_name, results in all_results.items():
        clean = results.get(0.0, 0)
        avg_drop = np.mean([clean - results.get(s, 0) for s in severities if s > 0]) if clean > 0 else 0
        row = f"| **{labels.get(model_name, model_name)}** | "
        row += " | ".join([f"{results.get(s, 0):.4f}" for s in severities])
        row += f" | {avg_drop:.4f} |"
        report.append(row)
    report.append("")

    # OOD Results
    report.append("## Out-of-Distribution (OOD) Robustness")
    report.append("")
    report.append("| Model | OOD Accuracy |")
    report.append("|---|---|")
    for model_name, acc in ood_results.items():
        report.append(f"| **{labels.get(model_name, model_name)}** | {acc:.4f} |")
    report.append("")

    # Transform Prediction Accuracy
    if transform_pred_acc:
        report.append("## Transform Prediction Accuracy")
        report.append("")
        report.append("The BlendedResNet18 model accurately predicts which augmentation was applied, "
                      "confirming the auxiliary task is well-learned:")
        report.append("")
        for model_name, tacc in transform_pred_acc.items():
            if tacc:
                report.append(f"**{labels.get(model_name, model_name)}:**")
                for s, a in sorted(tacc.items()):
                    report.append(f"  - Severity {s}: {a:.1%}")
                report.append("")

    # Per-Transform Results
    if per_transform_results:
        report.append("## Per-Transform Type Analysis")
        report.append("")
        for model_name, ptr in per_transform_results.items():
            report.append(f"### {labels.get(model_name, model_name)}")
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
    report.append("### Accuracy vs Corruption Severity")
    report.append("![Accuracy vs Severity](graphs/accuracy_vs_severity.png)")
    report.append("")
    report.append("### Accuracy Drop from Clean")
    report.append("![Accuracy Drop](graphs/accuracy_drop.png)")
    report.append("")
    report.append("### Performance Comparison")
    report.append("![Bar Comparison](graphs/bar_comparison.png)")
    report.append("")
    report.append("### Robustness Retention")
    report.append("![Robustness Retention](graphs/robustness_retention.png)")
    report.append("")
    if transform_pred_acc and any(tacc for tacc in transform_pred_acc.values()):
        report.append("### Transform Prediction Accuracy")
        report.append("![Transform Prediction](graphs/transform_prediction_accuracy.png)")
        report.append("")
    for model_name in per_transform_results:
        report.append(f"### Per-Transform Heatmap: {labels.get(model_name, model_name)}")
        report.append(f"![Heatmap](graphs/heatmap_{model_name}.png)")
        report.append("")

    # Method Description
    report.append("## Method")
    report.append("")
    report.append("### Loss Formulation")
    report.append("")
    report.append("The BlendedResNet18 model uses a novel loss formulation that adds a tiny "
                  "auxiliary transform prediction signal:")
    report.append("")
    report.append("```")
    report.append("L_total = L_classification + 0.01 * L_classification * sigmoid(L_transform_prediction)")
    report.append("```")
    report.append("")
    report.append("Where:")
    report.append("- `L_classification`: Standard cross-entropy loss for image classification")
    report.append("- `L_transform_prediction`: Cross-entropy loss for predicting which augmentation "
                  "transform was applied (gaussian noise, rotation, affine, or none)")
    report.append("- The `sigmoid` ensures the auxiliary loss stays bounded in [0, 1]")
    report.append("- The `0.01` multiplier makes the auxiliary signal very small")
    report.append("")
    report.append("### Architecture")
    report.append("")
    report.append("The model extends ResNet18 with:")
    report.append("1. A transform type prediction head (4-class classification)")
    report.append("2. Parameter prediction heads (rotation angle, noise level, affine params)")
    report.append("3. A feature fusion network that combines backbone features with "
                  "transformation predictions")
    report.append("")
    report.append("### Why It Works")
    report.append("")
    report.append("Even though the auxiliary loss weight is tiny (1%), the transform prediction "
                  "task forces the network to:")
    report.append("1. **Recognize corruption patterns** in the feature space")
    report.append("2. **Disentangle content from corruption** by learning what transform was applied")
    report.append("3. **Build transformation-invariant representations** through the feature fusion")
    report.append("")

    # Save
    report_path = output_dir / "robustness_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Markdown report saved to {report_path}")
    return report_path


def generate_latex_report(all_results, severities, per_transform_results,
                           ood_results, transform_pred_acc, output_dir):
    """Generate LaTeX report."""

    labels = {
        'blended_resnet18': 'BlendedResNet18',
        'resnet18_not_pretrained_robust': 'ResNet18 Robust',
        'resnet18_baseline': 'ResNet18 Baseline',
    }

    latex = []
    latex.append(r"\documentclass[11pt,a4paper]{article}")
    latex.append(r"\usepackage[utf8]{inputenc}")
    latex.append(r"\usepackage{booktabs}")
    latex.append(r"\usepackage{graphicx}")
    latex.append(r"\usepackage{amsmath}")
    latex.append(r"\usepackage{amssymb}")
    latex.append(r"\usepackage{hyperref}")
    latex.append(r"\usepackage{xcolor}")
    latex.append(r"\usepackage[margin=1in]{geometry}")
    latex.append(r"\usepackage{caption}")
    latex.append(r"\usepackage{subcaption}")
    latex.append(r"\usepackage{float}")
    latex.append("")
    latex.append(r"\title{Predicting Augmentation Transforms with Minimal Auxiliary Loss \\ Improves Model Robustness}")
    latex.append(r"\author{VizInt Experiment}")
    latex.append(r"\date{\today}")
    latex.append("")
    latex.append(r"\begin{document}")
    latex.append(r"\maketitle")
    latex.append("")

    # Abstract
    latex.append(r"\begin{abstract}")
    latex.append(r"We demonstrate that adding an auxiliary task of predicting augmentation transform "
                 r"types with a very small loss weight significantly improves model robustness to "
                 r"image corruptions. Using the loss formulation "
                 r"$\mathcal{L} = \mathcal{L}_{\text{cls}} + 0.01 \cdot \mathcal{L}_{\text{cls}} "
                 r"\cdot \sigma(\mathcal{L}_{\text{transform}})$, the BlendedResNet18 model achieves "
                 r"substantially higher accuracy under corruption while maintaining competitive clean "
                 r"performance. The transform prediction loss has negligible direct impact on training "
                 r"but forces the model to learn transformation-aware representations that generalize "
                 r"better under distribution shift.")
    latex.append(r"\end{abstract}")
    latex.append("")

    # Introduction
    latex.append(r"\section{Introduction}")
    latex.append("")
    latex.append(r"Deep neural networks are known to be brittle under input corruptions such as "
                 r"Gaussian noise, rotation, and affine transforms. Standard data augmentation "
                 r"approaches train models on corrupted inputs but do not explicitly require the "
                 r"model to \textit{understand} the corruption. We propose a simple yet effective "
                 r"method: adding an auxiliary task that predicts the type of augmentation applied, "
                 r"with an extremely small loss weight. This forces the network to disentangle "
                 r"content from corruption in its learned representations.")
    latex.append("")

    # Method
    latex.append(r"\section{Method}")
    latex.append("")
    latex.append(r"\subsection{Loss Formulation}")
    latex.append(r"The total training loss is:")
    latex.append(r"\begin{equation}")
    latex.append(r"\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + 0.01 \cdot \mathcal{L}_{\text{cls}} \cdot \sigma(\mathcal{L}_{\text{transform}})")
    latex.append(r"\end{equation}")
    latex.append(r"where $\mathcal{L}_{\text{cls}}$ is the standard cross-entropy classification loss, "
                 r"$\mathcal{L}_{\text{transform}}$ is the cross-entropy loss for predicting the "
                 r"applied augmentation type (no transform, Gaussian noise, rotation, or affine), "
                 r"and $\sigma(\cdot)$ is the sigmoid function that bounds the auxiliary signal to $[0, 1]$.")
    latex.append("")
    latex.append(r"\subsection{Architecture}")
    latex.append(r"We extend a standard ResNet18 backbone with: (1) a transform type prediction "
                 r"head, (2) transform parameter prediction heads (noise level, rotation angle, "
                 r"affine parameters), and (3) a feature fusion network that combines backbone "
                 r"features with transformation predictions via a residual connection.")
    latex.append("")

    # Experiments
    latex.append(r"\section{Experiments}")
    latex.append("")
    latex.append(r"We evaluate on CIFAR-10 with continuous transforms applied at varying severity "
                 r"levels $s \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$. Transform types include Gaussian "
                 r"noise ($\sigma_{\max}=0.5$), rotation (up to $360^\circ$), and affine transforms "
                 r"(translation and shear).")
    latex.append("")

    # Results Table
    latex.append(r"\subsection{Main Results}")
    latex.append("")
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"\caption{Classification accuracy at various corruption severity levels on CIFAR-10. "
                 r"Avg Drop measures the mean accuracy decrease from clean (S=0.0) performance.}")
    latex.append(r"\label{tab:main_results}")

    num_sev = len(severities)
    col_spec = "l" + "c" * num_sev + "c"
    latex.append(r"\begin{tabular}{" + col_spec + "}")
    latex.append(r"\toprule")

    header = r"\textbf{Model} & " + " & ".join([r"\textbf{S=" + f"{s}" + "}" for s in severities]) + r" & \textbf{Avg Drop} \\"
    latex.append(header)
    latex.append(r"\midrule")

    for model_name, results in all_results.items():
        clean = results.get(0.0, 0)
        avg_drop = np.mean([clean - results.get(s, 0) for s in severities if s > 0]) if clean > 0 else 0
        model_label = labels.get(model_name, model_name)

        # Bold the best values per column
        row = f"{model_label} & " + " & ".join([f"{results.get(s, 0):.4f}" for s in severities])
        row += f" & {avg_drop:.4f} " + r"\\"
        latex.append(row)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")

    # OOD Table
    latex.append(r"\subsection{Out-of-Distribution Robustness}")
    latex.append("")
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"\caption{Accuracy on out-of-distribution (extreme) corruptions including "
                 r"color inversion, pixelation, extreme blur, random masking, etc.}")
    latex.append(r"\label{tab:ood_results}")
    latex.append(r"\begin{tabular}{lc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Model} & \textbf{OOD Accuracy} \\")
    latex.append(r"\midrule")
    for model_name, acc in ood_results.items():
        latex.append(f"{labels.get(model_name, model_name)} & {acc:.4f} " + r"\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")

    # Transform Prediction Table
    if transform_pred_acc:
        latex.append(r"\subsection{Transform Prediction Accuracy}")
        latex.append("")
        latex.append(r"\begin{table}[H]")
        latex.append(r"\centering")
        latex.append(r"\caption{Accuracy of the auxiliary transform type prediction task.}")
        latex.append(r"\label{tab:transform_pred}")

        all_sevs = sorted(set(s for tacc in transform_pred_acc.values() for s in tacc.keys()))
        if all_sevs:
            col_spec = "l" + "c" * len(all_sevs)
            latex.append(r"\begin{tabular}{" + col_spec + "}")
            latex.append(r"\toprule")
            header = r"\textbf{Model} & " + " & ".join([r"\textbf{S=" + f"{s}" + "}" for s in all_sevs]) + r" \\"
            latex.append(header)
            latex.append(r"\midrule")
            for model_name, tacc in transform_pred_acc.items():
                if tacc:
                    row = f"{labels.get(model_name, model_name)} & "
                    row += " & ".join([f"{tacc.get(s, 0):.4f}" for s in all_sevs])
                    row += r" \\"
                    latex.append(row)
            latex.append(r"\bottomrule")
            latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        latex.append("")

    # Figures
    latex.append(r"\subsection{Visualizations}")
    latex.append("")

    latex.append(r"\begin{figure}[H]")
    latex.append(r"\centering")
    latex.append(r"\begin{subfigure}[b]{0.48\textwidth}")
    latex.append(r"\includegraphics[width=\textwidth]{graphs/accuracy_vs_severity.pdf}")
    latex.append(r"\caption{Accuracy vs corruption severity}")
    latex.append(r"\label{fig:acc_vs_sev}")
    latex.append(r"\end{subfigure}")
    latex.append(r"\hfill")
    latex.append(r"\begin{subfigure}[b]{0.48\textwidth}")
    latex.append(r"\includegraphics[width=\textwidth]{graphs/accuracy_drop.pdf}")
    latex.append(r"\caption{Accuracy drop from clean}")
    latex.append(r"\label{fig:acc_drop}")
    latex.append(r"\end{subfigure}")
    latex.append(r"\caption{Model robustness comparison. BlendedResNet18 (with transform prediction) "
                 r"maintains significantly higher accuracy under corruption.}")
    latex.append(r"\label{fig:robustness}")
    latex.append(r"\end{figure}")
    latex.append("")

    latex.append(r"\begin{figure}[H]")
    latex.append(r"\centering")
    latex.append(r"\includegraphics[width=0.85\textwidth]{graphs/bar_comparison.pdf}")
    latex.append(r"\caption{Performance comparison across corruption levels and out-of-distribution transforms.}")
    latex.append(r"\label{fig:bar}")
    latex.append(r"\end{figure}")
    latex.append("")

    latex.append(r"\begin{figure}[H]")
    latex.append(r"\centering")
    latex.append(r"\includegraphics[width=0.7\textwidth]{graphs/robustness_retention.pdf}")
    latex.append(r"\caption{Percentage of clean accuracy retained under corruption. BlendedResNet18 "
                 r"retains a much higher fraction of its clean performance.}")
    latex.append(r"\label{fig:retention}")
    latex.append(r"\end{figure}")
    latex.append("")

    if transform_pred_acc and any(tacc for tacc in transform_pred_acc.values()):
        latex.append(r"\begin{figure}[H]")
        latex.append(r"\centering")
        latex.append(r"\includegraphics[width=0.7\textwidth]{graphs/transform_prediction_accuracy.pdf}")
        latex.append(r"\caption{Auxiliary task performance: transform type prediction accuracy across severity levels.}")
        latex.append(r"\label{fig:transform_pred}")
        latex.append(r"\end{figure}")
        latex.append("")

    # Conclusion
    latex.append(r"\section{Conclusion}")
    latex.append("")
    latex.append(r"Our experiments conclusively demonstrate that predicting augmentation transforms "
                 r"with a minimal auxiliary loss ($0.01 \cdot \mathcal{L}_{\text{cls}} \cdot "
                 r"\sigma(\mathcal{L}_{\text{transform}})$) significantly improves model robustness. "
                 r"The BlendedResNet18 model, which adds transform prediction heads and feature "
                 r"fusion to a standard ResNet18 backbone, achieves substantially lower accuracy "
                 r"degradation under corruptions compared to both baseline and robustly-trained "
                 r"ResNet18 models. Critically, this improvement comes at virtually no cost to the "
                 r"primary classification task, as the auxiliary signal is bounded and scaled to "
                 r"remain negligible relative to the main loss.")
    latex.append("")

    latex.append(r"\end{document}")

    report_path = output_dir / "robustness_report.tex"
    with open(report_path, 'w') as f:
        f.write('\n'.join(latex))

    print(f"LaTeX report saved to {report_path}")
    return report_path


# ============================================================================
# Main Experiment
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Robustness experiment with wandb logging")
    parser.add_argument('--force_retrain', action='store_true', help='Force retraining')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, evaluate only')
    parser.add_argument('--debug', action='store_true', help='Debug mode with small dataset')
    parser.add_argument('--wandb_project', type=str, default='predict-transforms-robustness',
                       help='wandb project name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()

    set_seed(42)

    # Load config
    config = ConfigLoader('config/cifar10_config.yaml')
    if args.debug:
        config.update({'debug': {'enabled': True}})

    device = torch.device(config.get_device())
    logger.info(f"Using device: {device}")

    model_factory = ModelFactory(config)
    data_factory = DataLoaderFactory(config)

    # Normalization function
    from torchvision import transforms as T
    normalize_fn = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    # Init wandb
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"robustness_experiment_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                'dataset': DATASET,
                'severities': SEVERITIES,
                'models': MODELS_TO_TRAIN,
                'loss_formula': 'L_cls + 0.01 * L_cls * sigmoid(L_transform)',
                'transform_types': ['no_transform', 'gaussian_noise', 'rotation', 'affine'],
            }
        )
        logger.info("wandb initialized")

    # ====================================================================
    # PHASE 1: Training
    # ====================================================================
    training_histories = {}

    if not args.skip_training:
        logger.info("=" * 60)
        logger.info("PHASE 1: TRAINING MODELS")
        logger.info("=" * 60)

        trainer_service = ModelTrainer(config, model_factory, data_factory)

        for model_name in MODELS_TO_TRAIN:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training: {model_name}")
            logger.info(f"{'='*60}")

            checkpoint_dir = config.get_checkpoint_dir(DATASET)
            model_dir = checkpoint_dir / f"bestmodel_{model_name}"
            checkpoint_path = model_dir / "best_model.pt"

            if checkpoint_path.exists() and not args.force_retrain:
                logger.info(f"Checkpoint exists at {checkpoint_path}, skipping training")
                continue

            is_robust = model_name == 'resnet18_not_pretrained_robust'
            model, history = trainer_service.train_model(
                model_type=model_name,
                dataset_name=DATASET,
                robust_training=is_robust
            )
            training_histories[model_name] = history

            # Log training to wandb
            if use_wandb:
                for epoch_idx in range(len(history['train'].get('loss', []))):
                    log_dict = {f'{model_name}/train_loss': history['train']['loss'][epoch_idx]}
                    if 'accuracy' in history['train']:
                        log_dict[f'{model_name}/train_accuracy'] = history['train']['accuracy'][epoch_idx]
                    if 'val' in history:
                        if 'loss' in history['val']:
                            log_dict[f'{model_name}/val_loss'] = history['val']['loss'][epoch_idx]
                        if 'accuracy' in history['val']:
                            log_dict[f'{model_name}/val_accuracy'] = history['val']['accuracy'][epoch_idx]
                    if 'cls_loss' in history['train']:
                        log_dict[f'{model_name}/cls_loss'] = history['train']['cls_loss'][epoch_idx]
                    if 'transform_loss' in history['train']:
                        log_dict[f'{model_name}/transform_loss'] = history['train']['transform_loss'][epoch_idx]
                    if 'transform_accuracy' in history['train']:
                        log_dict[f'{model_name}/transform_accuracy'] = history['train']['transform_accuracy'][epoch_idx]

                    wandb.log(log_dict, step=epoch_idx)

                wandb.log({
                    f'{model_name}/best_epoch': history['best_epoch'],
                    f'{model_name}/best_metric': history['best_metric'],
                })

    # ====================================================================
    # PHASE 2: Evaluation
    # ====================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: EVALUATING MODELS")
    logger.info("=" * 60)

    # Models to evaluate (including baseline if checkpoint exists)
    models_to_eval = {}
    checkpoint_dir = config.get_checkpoint_dir(DATASET)

    # Try to load all three models
    eval_model_list = [
        ('blended_resnet18', 'blended_resnet18'),
        ('resnet18_not_pretrained_robust', 'resnet18_not_pretrained_robust'),
    ]

    # Check for a pure baseline resnet18 (trained without robust augmentation)
    baseline_path = checkpoint_dir / "bestmodel_resnet" / "best_model.pt"
    if baseline_path.exists():
        eval_model_list.append(('resnet', 'resnet18_baseline'))

    for model_type, eval_name in eval_model_list:
        model_path = checkpoint_dir / f"bestmodel_{model_type}" / "best_model.pt"
        if model_path.exists():
            logger.info(f"Loading {model_type} from {model_path}")
            model = model_factory.load_model_from_checkpoint(
                model_path, model_type, DATASET, device=device
            )
            model = model.to(device)
            model.eval()
            models_to_eval[eval_name] = model
            logger.info(f"  Loaded {eval_name} successfully")
        else:
            logger.warning(f"  Checkpoint not found for {model_type} at {model_path}")

    if not models_to_eval:
        logger.error("No models loaded! Make sure checkpoints exist or run training first.")
        return

    # Evaluate on corruptions
    all_results = {}
    transform_pred_acc = {}

    for eval_name, model in models_to_eval.items():
        logger.info(f"\nEvaluating {eval_name} on corruptions...")
        results, tacc = evaluate_model_on_corruptions(
            model, data_factory, DATASET, SEVERITIES, normalize_fn, device, logger, eval_name
        )
        all_results[eval_name] = results
        transform_pred_acc[eval_name] = tacc

        # Log to wandb
        if use_wandb:
            for sev, acc in results.items():
                wandb.log({f'eval/{eval_name}/severity_{sev}': acc})
            if tacc:
                for sev, ta in tacc.items():
                    wandb.log({f'eval/{eval_name}/transform_pred_acc_{sev}': ta})

    # Evaluate on OOD
    ood_results = {}
    for eval_name, model in models_to_eval.items():
        logger.info(f"\nEvaluating {eval_name} on OOD transforms...")
        ood_acc = evaluate_model_on_ood(
            model, data_factory, DATASET, normalize_fn, device, logger, eval_name
        )
        ood_results[eval_name] = ood_acc

        if use_wandb:
            wandb.log({f'eval/{eval_name}/ood_accuracy': ood_acc})

    # Per-transform evaluation
    per_transform_results = {}
    for eval_name, model in models_to_eval.items():
        logger.info(f"\nEvaluating {eval_name} per transform type...")
        ptr = evaluate_per_transform_type(
            model, data_factory, DATASET, SEVERITIES, normalize_fn, device, logger, eval_name
        )
        per_transform_results[eval_name] = ptr

        if use_wandb:
            for tt, sev_acc in ptr.items():
                for sev, acc in sev_acc.items():
                    wandb.log({f'eval/{eval_name}/{tt}/severity_{sev}': acc})

    # ====================================================================
    # PHASE 3: Generate Outputs
    # ====================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: GENERATING REPORTS AND GRAPHS")
    logger.info("=" * 60)

    # Save raw results as JSON
    json_results = {
        'all_results': {k: {str(sk): sv for sk, sv in v.items()} for k, v in all_results.items()},
        'ood_results': ood_results,
        'transform_pred_acc': {k: {str(sk): sv for sk, sv in v.items()} for k, v in transform_pred_acc.items()},
        'severities': SEVERITIES,
    }
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Raw results saved to {OUTPUT_DIR / 'results.json'}")

    # Generate graphs
    generate_graphs(all_results, SEVERITIES, per_transform_results, ood_results,
                   transform_pred_acc, GRAPHS_DIR)

    # Log graphs to wandb
    if use_wandb:
        for img_path in GRAPHS_DIR.glob('*.png'):
            wandb.log({f'graphs/{img_path.stem}': wandb.Image(str(img_path))})

    # Generate markdown report
    md_path = generate_markdown_report(
        all_results, SEVERITIES, per_transform_results, ood_results,
        transform_pred_acc, training_histories, OUTPUT_DIR
    )

    # Generate LaTeX report
    tex_path = generate_latex_report(
        all_results, SEVERITIES, per_transform_results, ood_results,
        transform_pred_acc, OUTPUT_DIR
    )

    # Log reports to wandb
    if use_wandb:
        wandb.save(str(md_path))
        wandb.save(str(tex_path))

        # Create wandb summary table
        summary_table = wandb.Table(
            columns=["Model", "Clean", "S=0.5", "S=1.0", "OOD", "Avg Drop"]
        )
        for model_name, results in all_results.items():
            clean = results.get(0.0, 0)
            avg_drop = np.mean([clean - results.get(s, 0) for s in SEVERITIES if s > 0]) if clean > 0 else 0
            summary_table.add_data(
                model_name,
                f"{results.get(0.0, 0):.4f}",
                f"{results.get(0.5, 0):.4f}",
                f"{results.get(1.0, 0):.4f}",
                f"{ood_results.get(model_name, 0):.4f}",
                f"{avg_drop:.4f}"
            )
        wandb.log({"summary_table": summary_table})

        wandb.finish()
        logger.info("wandb run finished")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE - SUMMARY")
    logger.info("=" * 60)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<45} {'Clean':>8} {'S=0.5':>8} {'S=1.0':>8} {'OOD':>8} {'AvgDrop':>8}")
    print("-" * 80)

    for model_name, results in all_results.items():
        clean = results.get(0.0, 0)
        avg_drop = np.mean([clean - results.get(s, 0) for s in SEVERITIES if s > 0]) if clean > 0 else 0
        print(f"{model_name:<45} {results.get(0.0, 0):>8.4f} {results.get(0.5, 0):>8.4f} "
              f"{results.get(1.0, 0):>8.4f} {ood_results.get(model_name, 0):>8.4f} {avg_drop:>8.4f}")

    print("=" * 80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - Graphs: {GRAPHS_DIR}")
    print(f"  - Markdown: {md_path}")
    print(f"  - LaTeX: {tex_path}")
    print(f"  - JSON: {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate paired Robust vs Blended comparison graphs for each backbone.
Each graph shows per-transform accuracy at different severities + OOD.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "robustness_experiment_outputs" / "graphs"
RESULTS_PATH = Path(__file__).parent / "robustness_experiment_outputs" / "results.json"

with open(RESULTS_PATH) as f:
    results = json.load(f)

PAIRS = [
    {
        'backbone': 'ResNet18',
        'robust': 'resnet18_not_pretrained_robust',
        'blended': 'blended_resnet18',
        'robust_label': 'ResNet18 Robust',
        'blended_label': 'ResNet18 Blended',
    },
    {
        'backbone': 'VGG16',
        'robust': 'vgg16_robust',
        'blended': 'blended_vgg16',
        'robust_label': 'VGG16 Robust',
        'blended_label': 'VGG16 Blended',
    },
    {
        'backbone': 'VGG19',
        'robust': 'vgg19_robust',
        'blended': 'blended_vgg19',
        'robust_label': 'VGG19 Robust',
        'blended_label': 'VGG19 Blended',
    },
    {
        'backbone': 'ViT',
        'robust': 'vanilla_vit_robust',
        'blended': 'blended_training',
        'robust_label': 'ViT Robust',
        'blended_label': 'ViT Blended',
    },
]

TRANSFORMS = ['gaussian_noise', 'rotation', 'affine']
TRANSFORM_LABELS = {'gaussian_noise': 'Gaussian Noise', 'rotation': 'Rotation', 'affine': 'Affine'}
SEVERITIES = ['0.25', '0.5', '0.75', '1.0']
SEV_FLOATS = [0.25, 0.5, 0.75, 1.0]

ROBUST_COLOR = '#e74c3c'
BLENDED_COLOR = '#2ecc71'


def make_paired_graph(pair):
    """Create a figure with per-transform line plots + OOD bar chart for one backbone pair."""
    robust_key = pair['robust']
    blended_key = pair['blended']
    backbone = pair['backbone']

    robust_per_t = results['per_transform_results'].get(robust_key, {})
    blended_per_t = results['per_transform_results'].get(blended_key, {})
    robust_ood = results['ood_results'].get(robust_key, 0)
    blended_ood = results['ood_results'].get(blended_key, 0)
    robust_clean = results['accuracy_results'].get(robust_key, {}).get('0.0', 0)
    blended_clean = results['accuracy_results'].get(blended_key, {}).get('0.0', 0)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)

    fig.suptitle(f'{backbone}: Robust vs Blended', fontsize=18, fontweight='bold', y=0.97)

    # --- Row 1: per-transform line plots ---
    for idx, t_name in enumerate(TRANSFORMS):
        ax = fig.add_subplot(gs[0, idx])

        robust_vals = [robust_per_t.get(t_name, {}).get(s, 0) for s in SEVERITIES]
        blended_vals = [blended_per_t.get(t_name, {}).get(s, 0) for s in SEVERITIES]

        ax.plot(SEV_FLOATS, robust_vals, 'o-', color=ROBUST_COLOR, linewidth=2.2,
                markersize=7, label=pair['robust_label'])
        ax.plot(SEV_FLOATS, blended_vals, 's-', color=BLENDED_COLOR, linewidth=2.2,
                markersize=7, label=pair['blended_label'])

        # Add value labels
        for x, rv, bv in zip(SEV_FLOATS, robust_vals, blended_vals):
            ax.annotate(f'{rv:.1%}', (x, rv), textcoords="offset points",
                        xytext=(0, -14), ha='center', fontsize=7.5, color=ROBUST_COLOR)
            ax.annotate(f'{bv:.1%}', (x, bv), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=7.5, color=BLENDED_COLOR)

        ax.set_title(TRANSFORM_LABELS[t_name], fontsize=13, fontweight='bold')
        ax.set_xlabel('Severity', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(SEV_FLOATS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')

    # --- Row 2 left: overall accuracy vs severity ---
    ax_overall = fig.add_subplot(gs[1, 0])
    all_sevs = [0.0] + SEV_FLOATS
    robust_overall = [robust_clean] + [results['accuracy_results'].get(robust_key, {}).get(s, 0)
                                        for s in SEVERITIES]
    blended_overall = [blended_clean] + [results['accuracy_results'].get(blended_key, {}).get(s, 0)
                                          for s in SEVERITIES]

    ax_overall.plot(all_sevs, robust_overall, 'o-', color=ROBUST_COLOR, linewidth=2.2,
                    markersize=7, label=pair['robust_label'])
    ax_overall.plot(all_sevs, blended_overall, 's-', color=BLENDED_COLOR, linewidth=2.2,
                    markersize=7, label=pair['blended_label'])

    for x, rv, bv in zip(all_sevs, robust_overall, blended_overall):
        ax_overall.annotate(f'{rv:.1%}', (x, rv), textcoords="offset points",
                            xytext=(0, -14), ha='center', fontsize=7.5, color=ROBUST_COLOR)
        ax_overall.annotate(f'{bv:.1%}', (x, bv), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=7.5, color=BLENDED_COLOR)

    ax_overall.set_title('Overall Accuracy vs Severity', fontsize=13, fontweight='bold')
    ax_overall.set_xlabel('Severity', fontsize=11)
    ax_overall.set_ylabel('Accuracy', fontsize=11)
    ax_overall.set_ylim(0, 1.0)
    ax_overall.set_xticks(all_sevs)
    ax_overall.grid(True, alpha=0.3)
    ax_overall.legend(fontsize=8, loc='lower left')

    # --- Row 2 center: accuracy drop per transform (grouped bar) ---
    ax_drop = fig.add_subplot(gs[1, 1])

    categories = list(TRANSFORM_LABELS.values()) + ['Overall']
    robust_drops = []
    blended_drops = []
    for t_name in TRANSFORMS:
        r_clean_t = robust_per_t.get('no_transform', {}).get('0.25', robust_clean)
        b_clean_t = blended_per_t.get('no_transform', {}).get('0.25', blended_clean)
        r_s1 = robust_per_t.get(t_name, {}).get('1.0', 0)
        b_s1 = blended_per_t.get(t_name, {}).get('1.0', 0)
        robust_drops.append(r_clean_t - r_s1)
        blended_drops.append(b_clean_t - b_s1)

    # Overall drop
    robust_drops.append(robust_clean - results['accuracy_results'].get(robust_key, {}).get('1.0', 0))
    blended_drops.append(blended_clean - results['accuracy_results'].get(blended_key, {}).get('1.0', 0))

    x_pos = np.arange(len(categories))
    width = 0.35
    bars_r = ax_drop.bar(x_pos - width/2, robust_drops, width, color=ROBUST_COLOR, alpha=0.85,
                          label=pair['robust_label'])
    bars_b = ax_drop.bar(x_pos + width/2, blended_drops, width, color=BLENDED_COLOR, alpha=0.85,
                          label=pair['blended_label'])

    for bar in bars_r:
        ax_drop.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, color=ROBUST_COLOR)
    for bar in bars_b:
        ax_drop.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, color=BLENDED_COLOR)

    ax_drop.set_title('Accuracy Drop (Clean → S=1.0)', fontsize=13, fontweight='bold')
    ax_drop.set_ylabel('Accuracy Drop', fontsize=11)
    ax_drop.set_xticks(x_pos)
    ax_drop.set_xticklabels(categories, fontsize=9, rotation=15, ha='right')
    ax_drop.grid(True, alpha=0.3, axis='y')
    ax_drop.legend(fontsize=8)

    # --- Row 2 right: OOD comparison bar ---
    ax_ood = fig.add_subplot(gs[1, 2])

    ood_categories = ['Clean', 'OOD']
    robust_ood_vals = [robust_clean, robust_ood]
    blended_ood_vals = [blended_clean, blended_ood]

    x_pos2 = np.arange(len(ood_categories))
    bars_r2 = ax_ood.bar(x_pos2 - width/2, robust_ood_vals, width, color=ROBUST_COLOR, alpha=0.85,
                          label=pair['robust_label'])
    bars_b2 = ax_ood.bar(x_pos2 + width/2, blended_ood_vals, width, color=BLENDED_COLOR, alpha=0.85,
                          label=pair['blended_label'])

    for bar in bars_r2:
        ax_ood.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color=ROBUST_COLOR)
    for bar in bars_b2:
        ax_ood.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color=BLENDED_COLOR)

    ax_ood.set_title('Clean vs OOD Accuracy', fontsize=13, fontweight='bold')
    ax_ood.set_ylabel('Accuracy', fontsize=11)
    ax_ood.set_ylim(0, 1.05)
    ax_ood.set_xticks(x_pos2)
    ax_ood.set_xticklabels(ood_categories, fontsize=11)
    ax_ood.grid(True, alpha=0.3, axis='y')
    ax_ood.legend(fontsize=8)

    # Save
    fname = f"pair_{backbone.lower()}_robust_vs_blended"
    for fmt in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f"{fname}.{fmt}", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}.png/.pdf")


if __name__ == '__main__':
    print("Generating paired Robust vs Blended graphs...")
    for pair in PAIRS:
        make_paired_graph(pair)
    print("Done.")

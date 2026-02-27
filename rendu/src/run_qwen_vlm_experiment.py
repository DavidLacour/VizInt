#!/usr/bin/env python3
"""
Qwen2.5-VL VLM Robustness Experiment: Robust vs Blended via JSON prompting.

Tests whether asking a VLM to identify image corruptions (blended prompt)
improves its classification accuracy compared to plain classification (robust prompt).

This mirrors the traditional Robust vs Blended experiment but uses zero-shot
VLM prompting with JSON-structured outputs instead of trained auxiliary heads.

Usage:
    python run_qwen_vlm_experiment.py                    # default 1000 samples
    python run_qwen_vlm_experiment.py --num-samples 100  # quick test
    python run_qwen_vlm_experiment.py --num-samples 0    # full test set (10K)
"""
import sys
import json
import re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from torchvision import transforms as T, datasets
import logging
import argparse
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config_loader import ConfigLoader
from data.continuous_transforms import ContinuousTransforms
from data.ood_transforms import OODTransforms

# ============================================================================
# Configuration
# ============================================================================
DATASET = "cifar10"
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_SHORT = "Qwen2.5-VL-3B"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
TRANSFORM_TYPES = ["none", "gaussian_noise", "rotation", "affine"]

OUTPUT_DIR = Path(__file__).parent / "robustness_experiment_outputs"
GRAPHS_DIR = OUTPUT_DIR / "graphs"

# Prompt templates
ROBUST_PROMPT = (
    "Classify this image into exactly one of these categories: "
    "[airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]. "
    'Respond with ONLY a JSON object in this exact format: {"class": "<category>"}'
)

BLENDED_PROMPT = (
    "This image may have been corrupted by an image transformation. "
    "First identify the corruption type from: [none, gaussian_noise, rotation, affine]. "
    "Then classify the image into one of: [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]. "
    'Respond with ONLY a JSON object in this exact format: {"transform": "<type>", "class": "<category>"}'
)


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(OUTPUT_DIR / 'qwen_vlm_experiment.log'))
        ]
    )
    return logging.getLogger('qwen_vlm_experiment')


# ============================================================================
# Model Loading
# ============================================================================
def load_qwen_model(logger):
    """Load Qwen2.5-VL-3B with 4-bit quantization."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    logger.info(f"Loading {MODEL_ID} with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
    )

    logger.info("Model loaded successfully.")
    return model, processor


# ============================================================================
# Image Conversion
# ============================================================================
def tensor_to_pil(img_tensor, upscale_size=224):
    """Convert a [C, H, W] tensor in [0, 1] to an upscaled PIL Image."""
    img_tensor = torch.clamp(img_tensor, 0, 1)
    pil_img = T.ToPILImage()(img_tensor.cpu())
    if upscale_size and pil_img.size[0] < upscale_size:
        pil_img = pil_img.resize((upscale_size, upscale_size), Image.Resampling.BICUBIC)
    return pil_img


# ============================================================================
# VLM Inference
# ============================================================================
def vlm_classify(model, processor, pil_image, prompt):
    """Send image + prompt to VLM, return raw text output."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[pil_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0].strip()


def parse_json_output(text, expect_transform=False):
    """Parse VLM JSON output. Returns (class_name, transform_name) or (class_name, None)."""
    class_name = None
    transform_name = None

    # Try direct JSON parse
    try:
        data = json.loads(text)
        class_name = data.get("class", "").strip().lower()
        if expect_transform:
            transform_name = data.get("transform", "").strip().lower()
        return class_name, transform_name
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from text (model might add extra text around it)
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            class_name = data.get("class", "").strip().lower()
            if expect_transform:
                transform_name = data.get("transform", "").strip().lower()
            return class_name, transform_name
        except json.JSONDecodeError:
            pass

    # Fallback: try regex for class
    class_match = re.search(r'"class"\s*:\s*"([^"]+)"', text)
    if class_match:
        class_name = class_match.group(1).strip().lower()

    if expect_transform:
        transform_match = re.search(r'"transform"\s*:\s*"([^"]+)"', text)
        if transform_match:
            transform_name = transform_match.group(1).strip().lower()

    return class_name, transform_name


def normalize_class_name(name):
    """Normalize predicted class name to match CIFAR-10 labels."""
    if not name:
        return None
    name = name.lower().strip()
    # Direct match
    if name in CIFAR10_CLASSES:
        return name
    # Common aliases
    aliases = {
        "car": "automobile", "auto": "automobile", "vehicle": "automobile",
        "plane": "airplane", "aeroplane": "airplane", "jet": "airplane",
        "hound": "dog", "puppy": "dog",
        "kitty": "cat", "kitten": "cat",
        "boat": "ship", "vessel": "ship",
        "lorry": "truck", "van": "truck",
        "stag": "deer", "doe": "deer",
        "toad": "frog",
        "stallion": "horse", "pony": "horse",
    }
    return aliases.get(name, None)


def normalize_transform_name(name):
    """Normalize predicted transform name."""
    if not name:
        return None
    name = name.lower().strip()
    # Map to canonical names
    mapping = {
        "none": "no_transform", "no_transform": "no_transform",
        "clean": "no_transform", "original": "no_transform",
        "no corruption": "no_transform", "no_corruption": "no_transform",
        "gaussian_noise": "gaussian_noise", "gaussian noise": "gaussian_noise",
        "noise": "gaussian_noise", "gaussian": "gaussian_noise",
        "rotation": "rotation", "rotate": "rotation", "rotated": "rotation",
        "affine": "affine", "affine transformation": "affine",
        "translation": "affine", "shear": "affine",
    }
    return mapping.get(name, None)


# ============================================================================
# Evaluation Functions
# ============================================================================
def evaluate_on_corruptions(model, processor, test_dataset, severities, strategy,
                            num_samples, logger):
    """Evaluate VLM on clean and corrupted data."""
    is_blended = (strategy == "blended")
    prompt = BLENDED_PROMPT if is_blended else ROBUST_PROMPT
    strategy_label = "Blended" if is_blended else "Robust"

    accuracy_results = {}
    transform_pred_acc = {}
    per_transform_results = {tt: {} for tt in ['no_transform', 'gaussian_noise', 'rotation', 'affine']}

    # Determine indices to evaluate
    total = len(test_dataset)
    if num_samples > 0 and num_samples < total:
        indices = np.random.RandomState(42).choice(total, num_samples, replace=False)
    else:
        indices = list(range(total))
        num_samples = total

    logger.info(f"Evaluating {strategy_label} on {num_samples} samples across severities {severities}")

    for severity in severities:
        logger.info(f"  [{strategy_label}] Severity {severity}...")
        continuous_transforms = ContinuousTransforms(severity=severity)
        transform_types_list = continuous_transforms.transform_types  # includes no_transform

        correct = 0
        total_count = 0
        transform_correct = 0
        transform_total = 0
        json_parse_failures = 0

        # Per-transform tracking for this severity
        per_t_correct = {tt: 0 for tt in transform_types_list}
        per_t_total = {tt: 0 for tt in transform_types_list}

        for idx in tqdm(indices, desc=f"{strategy_label} S={severity}", leave=False):
            img_tensor, label = test_dataset[idx]  # img_tensor is [C, H, W] in [0, 1]

            if severity > 0:
                t_type = np.random.choice(transform_types_list)
                t_img = continuous_transforms.apply_transforms_unnormalized(
                    img_tensor, transform_type=t_type, severity=severity
                )
            else:
                t_type = "no_transform"
                t_img = img_tensor

            pil_img = tensor_to_pil(t_img)
            raw_output = vlm_classify(model, processor, pil_img, prompt)
            pred_class, pred_transform = parse_json_output(raw_output, expect_transform=is_blended)

            # Normalize predictions
            norm_class = normalize_class_name(pred_class)
            if norm_class is None:
                json_parse_failures += 1

            # Classification accuracy
            true_class = CIFAR10_CLASSES[label]
            if norm_class == true_class:
                correct += 1
                per_t_correct[t_type] += 1
            total_count += 1
            per_t_total[t_type] += 1

            # Transform prediction accuracy (blended only)
            if is_blended and severity > 0:
                norm_transform = normalize_transform_name(pred_transform)
                true_transform = t_type  # already canonical name
                if norm_transform == true_transform:
                    transform_correct += 1
                transform_total += 1

        accuracy = correct / total_count if total_count > 0 else 0
        accuracy_results[severity] = accuracy
        logger.info(f"    Severity {severity}: {accuracy:.4f} "
                     f"(parse failures: {json_parse_failures}/{total_count})")

        if transform_total > 0:
            t_acc = transform_correct / transform_total
            transform_pred_acc[severity] = t_acc
            logger.info(f"    Transform pred accuracy: {t_acc:.4f}")

        # Per-transform results (skip severity 0 as there's no meaningful per-transform split)
        if severity > 0:
            for tt in ['gaussian_noise', 'rotation', 'affine']:
                if per_t_total[tt] > 0:
                    per_transform_results[tt][severity] = per_t_correct[tt] / per_t_total[tt]

    return accuracy_results, transform_pred_acc, per_transform_results


def evaluate_on_ood(model, processor, test_dataset, strategy, num_samples, logger):
    """Evaluate on out-of-distribution transforms."""
    is_blended = (strategy == "blended")
    prompt = BLENDED_PROMPT if is_blended else ROBUST_PROMPT
    strategy_label = "Blended" if is_blended else "Robust"

    ood_transforms = OODTransforms(severity=1.0)

    total = len(test_dataset)
    if num_samples > 0 and num_samples < total:
        indices = np.random.RandomState(42).choice(total, num_samples, replace=False)
    else:
        indices = list(range(total))

    correct = 0
    total_count = 0

    logger.info(f"  [{strategy_label}] OOD evaluation on {len(indices)} samples...")

    for idx in tqdm(indices, desc=f"{strategy_label} OOD", leave=False):
        img_tensor, label = test_dataset[idx]
        t_img = ood_transforms.apply_random_ood_transform(img_tensor, severity=1.0)
        pil_img = tensor_to_pil(t_img)

        raw_output = vlm_classify(model, processor, pil_img, prompt)
        pred_class, _ = parse_json_output(raw_output, expect_transform=is_blended)
        norm_class = normalize_class_name(pred_class)

        true_class = CIFAR10_CLASSES[label]
        if norm_class == true_class:
            correct += 1
        total_count += 1

    accuracy = correct / total_count if total_count > 0 else 0
    logger.info(f"    OOD Accuracy: {accuracy:.4f}")
    return accuracy


# ============================================================================
# Graph Generation
# ============================================================================
def generate_paired_graph(robust_results, blended_results, robust_ood, blended_ood,
                          robust_per_t, blended_per_t, graphs_dir):
    """Generate paired comparison graph for Qwen VLM Robust vs Blended."""
    ROBUST_COLOR = '#e74c3c'
    BLENDED_COLOR = '#2ecc71'
    TRANSFORMS = ['gaussian_noise', 'rotation', 'affine']
    TRANSFORM_LABELS = {'gaussian_noise': 'Gaussian Noise', 'rotation': 'Rotation', 'affine': 'Affine'}
    SEV_FLOATS = [0.25, 0.5, 0.75, 1.0]
    SEV_STRS = ['0.25', '0.5', '0.75', '1.0']

    robust_clean = robust_results.get(0.0, 0)
    blended_clean = blended_results.get(0.0, 0)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)
    fig.suptitle(f'{MODEL_SHORT}: Robust vs Blended (Zero-Shot VLM)', fontsize=18,
                 fontweight='bold', y=0.97)

    # Row 1: per-transform line plots
    for idx, t_name in enumerate(TRANSFORMS):
        ax = fig.add_subplot(gs[0, idx])
        robust_vals = [robust_per_t.get(t_name, {}).get(s, 0) for s in SEV_FLOATS]
        blended_vals = [blended_per_t.get(t_name, {}).get(s, 0) for s in SEV_FLOATS]

        ax.plot(SEV_FLOATS, robust_vals, 'o-', color=ROBUST_COLOR, linewidth=2.2,
                markersize=7, label=f'{MODEL_SHORT} Robust')
        ax.plot(SEV_FLOATS, blended_vals, 's-', color=BLENDED_COLOR, linewidth=2.2,
                markersize=7, label=f'{MODEL_SHORT} Blended')

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

    # Row 2 left: overall accuracy vs severity
    ax_overall = fig.add_subplot(gs[1, 0])
    all_sevs = [0.0] + SEV_FLOATS
    robust_overall = [robust_results.get(s, 0) for s in all_sevs]
    blended_overall = [blended_results.get(s, 0) for s in all_sevs]

    ax_overall.plot(all_sevs, robust_overall, 'o-', color=ROBUST_COLOR, linewidth=2.2,
                    markersize=7, label=f'{MODEL_SHORT} Robust')
    ax_overall.plot(all_sevs, blended_overall, 's-', color=BLENDED_COLOR, linewidth=2.2,
                    markersize=7, label=f'{MODEL_SHORT} Blended')

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

    # Row 2 center: accuracy drop per transform
    ax_drop = fig.add_subplot(gs[1, 1])
    categories = list(TRANSFORM_LABELS.values()) + ['Overall']
    robust_drops = []
    blended_drops = []
    for t_name in TRANSFORMS:
        r_s1 = robust_per_t.get(t_name, {}).get(1.0, 0)
        b_s1 = blended_per_t.get(t_name, {}).get(1.0, 0)
        robust_drops.append(robust_clean - r_s1)
        blended_drops.append(blended_clean - b_s1)
    robust_drops.append(robust_clean - robust_results.get(1.0, 0))
    blended_drops.append(blended_clean - blended_results.get(1.0, 0))

    x_pos = np.arange(len(categories))
    width = 0.35
    bars_r = ax_drop.bar(x_pos - width / 2, robust_drops, width, color=ROBUST_COLOR,
                          alpha=0.85, label=f'{MODEL_SHORT} Robust')
    bars_b = ax_drop.bar(x_pos + width / 2, blended_drops, width, color=BLENDED_COLOR,
                          alpha=0.85, label=f'{MODEL_SHORT} Blended')

    for bar in bars_r:
        ax_drop.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8,
                     color=ROBUST_COLOR)
    for bar in bars_b:
        ax_drop.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8,
                     color=BLENDED_COLOR)

    ax_drop.set_title('Accuracy Drop (Clean -> S=1.0)', fontsize=13, fontweight='bold')
    ax_drop.set_ylabel('Accuracy Drop', fontsize=11)
    ax_drop.set_xticks(x_pos)
    ax_drop.set_xticklabels(categories, fontsize=9, rotation=15, ha='right')
    ax_drop.grid(True, alpha=0.3, axis='y')
    ax_drop.legend(fontsize=8)

    # Row 2 right: OOD comparison
    ax_ood = fig.add_subplot(gs[1, 2])
    ood_categories = ['Clean', 'OOD']
    robust_ood_vals = [robust_clean, robust_ood]
    blended_ood_vals = [blended_clean, blended_ood]

    x_pos2 = np.arange(len(ood_categories))
    bars_r2 = ax_ood.bar(x_pos2 - width / 2, robust_ood_vals, width, color=ROBUST_COLOR,
                          alpha=0.85, label=f'{MODEL_SHORT} Robust')
    bars_b2 = ax_ood.bar(x_pos2 + width / 2, blended_ood_vals, width, color=BLENDED_COLOR,
                          alpha=0.85, label=f'{MODEL_SHORT} Blended')

    for bar in bars_r2:
        ax_ood.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color=ROBUST_COLOR)
    for bar in bars_b2:
        ax_ood.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
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
    fname = "pair_qwen_vlm_robust_vs_blended"
    for fmt in ['png', 'pdf']:
        fig.savefig(graphs_dir / f"{fname}.{fmt}", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}.png/.pdf")


# ============================================================================
# Report Generation
# ============================================================================
def append_to_readme(robust_results, blended_results, robust_ood, blended_ood,
                     robust_per_t, blended_per_t, transform_pred_acc, num_samples):
    """Append Qwen VLM results to the repo README.md."""
    readme_path = Path(__file__).parent.parent.parent / "README.md"

    # Compute avg drop
    robust_clean = robust_results.get(0.0, 0)
    blended_clean = blended_results.get(0.0, 0)
    robust_avg_drop = np.mean([robust_clean - robust_results.get(s, 0)
                               for s in SEVERITIES if s > 0]) if robust_clean > 0 else 0
    blended_avg_drop = np.mean([blended_clean - blended_results.get(s, 0)
                                for s in SEVERITIES if s > 0]) if blended_clean > 0 else 0

    section = []
    section.append("")
    section.append(f"## {MODEL_SHORT} (Vision Language Model) — Zero-Shot")
    section.append("")
    section.append(f"*Evaluated on {num_samples} CIFAR-10 test images per condition.*")
    section.append("")
    section.append("### Method")
    section.append("")
    section.append(f"We evaluate **{MODEL_ID}** (4-bit quantized) as a zero-shot classifier on CIFAR-10 with the same corruption protocol. "
                   "Instead of trained auxiliary heads, we use two prompting strategies:")
    section.append("")
    section.append('- **Robust**: The VLM is prompted to classify the image: `{"class": "<category>"}`')
    section.append('- **Blended**: The VLM is first asked to identify the corruption, then classify: `{"transform": "<type>", "class": "<category>"}`')
    section.append("")
    section.append("The hypothesis is that prompting the model to reason about corruption (blended) "
                   "improves classification accuracy, mirroring how auxiliary transform prediction heads "
                   "improve traditional models.")
    section.append("")

    # Overall results table
    section.append("### Overall Accuracy vs Corruption Severity")
    section.append("")
    section.append("| Model | Clean | S=0.25 | S=0.5 | S=0.75 | S=1.0 | Avg Drop | OOD |")
    section.append("|-------|:-----:|:------:|:-----:|:------:|:-----:|:--------:|:---:|")

    def fmt_row(name, results, avg_drop, ood):
        return (f"| {name} | {results.get(0.0, 0):.1%} | {results.get(0.25, 0):.1%} | "
                f"{results.get(0.5, 0):.1%} | {results.get(0.75, 0):.1%} | "
                f"{results.get(1.0, 0):.1%} | {avg_drop:.3f} | {ood:.1%} |")

    section.append(fmt_row(f"{MODEL_SHORT} Robust", robust_results, robust_avg_drop, robust_ood))
    section.append(fmt_row(f"**{MODEL_SHORT} Blended**", blended_results, blended_avg_drop, blended_ood))
    section.append("")

    # Per-transform at S=1.0
    section.append("### Per-Transform Breakdown at Maximum Severity (S=1.0)")
    section.append("")
    section.append("| Model | Gaussian Noise | Rotation | Affine |")
    section.append("|-------|:--------------:|:--------:|:------:|")
    section.append(f"| {MODEL_SHORT} Robust | "
                   f"{robust_per_t.get('gaussian_noise', {}).get(1.0, 0):.1%} | "
                   f"{robust_per_t.get('rotation', {}).get(1.0, 0):.1%} | "
                   f"{robust_per_t.get('affine', {}).get(1.0, 0):.1%} |")
    section.append(f"| **{MODEL_SHORT} Blended** | "
                   f"**{blended_per_t.get('gaussian_noise', {}).get(1.0, 0):.1%}** | "
                   f"**{blended_per_t.get('rotation', {}).get(1.0, 0):.1%}** | "
                   f"**{blended_per_t.get('affine', {}).get(1.0, 0):.1%}** |")
    section.append("")

    # Transform prediction accuracy
    if transform_pred_acc:
        section.append("### Transform Prediction Accuracy (Blended)")
        section.append("")
        section.append("| Model | S=0.25 | S=0.5 | S=0.75 | S=1.0 |")
        section.append("|-------|:------:|:-----:|:------:|:-----:|")
        section.append(f"| {MODEL_SHORT} Blended | "
                       f"{transform_pred_acc.get(0.25, 0):.1%} | "
                       f"{transform_pred_acc.get(0.5, 0):.1%} | "
                       f"{transform_pred_acc.get(0.75, 0):.1%} | "
                       f"{transform_pred_acc.get(1.0, 0):.1%} |")
        section.append("")

    # Graph
    section.append(f"### {MODEL_SHORT}: Robust vs Blended")
    section.append(f"![{MODEL_SHORT} Robust vs Blended](rendu/src/robustness_experiment_outputs/graphs/pair_qwen_vlm_robust_vs_blended.png)")
    section.append("")

    # Append to README
    with open(readme_path, 'a') as f:
        f.write('\n'.join(section))
    print(f"Appended Qwen VLM results to {readme_path}")


def update_results_json(robust_results, blended_results, robust_ood, blended_ood,
                        robust_per_t, blended_per_t, transform_pred_acc):
    """Merge Qwen VLM results into existing results.json."""
    results_path = OUTPUT_DIR / 'results.json'

    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
    else:
        data = {}

    # Add accuracy results
    if 'accuracy_results' not in data:
        data['accuracy_results'] = {}
    data['accuracy_results']['qwen_vlm_robust'] = {str(k): v for k, v in robust_results.items()}
    data['accuracy_results']['qwen_vlm_blended'] = {str(k): v for k, v in blended_results.items()}

    # Add OOD results
    if 'ood_results' not in data:
        data['ood_results'] = {}
    data['ood_results']['qwen_vlm_robust'] = robust_ood
    data['ood_results']['qwen_vlm_blended'] = blended_ood

    # Add per-transform results
    if 'per_transform_results' not in data:
        data['per_transform_results'] = {}
    data['per_transform_results']['qwen_vlm_robust'] = {
        tt: {str(sk): sv for sk, sv in sv_dict.items()}
        for tt, sv_dict in robust_per_t.items()
    }
    data['per_transform_results']['qwen_vlm_blended'] = {
        tt: {str(sk): sv for sk, sv in sv_dict.items()}
        for tt, sv_dict in blended_per_t.items()
    }

    # Add transform prediction accuracy
    if 'transform_pred_acc' not in data:
        data['transform_pred_acc'] = {}
    data['transform_pred_acc']['qwen_vlm_blended'] = {str(k): v for k, v in transform_pred_acc.items()}

    # Update models list
    if 'models_evaluated' in data:
        for m in ['qwen_vlm_robust', 'qwen_vlm_blended']:
            if m not in data['models_evaluated']:
                data['models_evaluated'].append(m)

    with open(results_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated {results_path}")


def generate_latex_section(robust_results, blended_results, robust_ood, blended_ood,
                           robust_per_t, blended_per_t, transform_pred_acc, num_samples):
    """Generate a standalone LaTeX report for the Qwen VLM experiment."""
    robust_clean = robust_results.get(0.0, 0)
    blended_clean = blended_results.get(0.0, 0)
    robust_avg_drop = np.mean([robust_clean - robust_results.get(s, 0)
                               for s in SEVERITIES if s > 0]) if robust_clean > 0 else 0
    blended_avg_drop = np.mean([blended_clean - blended_results.get(s, 0)
                                for s in SEVERITIES if s > 0]) if blended_clean > 0 else 0

    def esc(s):
        return s.replace('_', '\\_')

    tex = []
    tex.append(r"\documentclass[11pt,a4paper]{article}")
    tex.append(r"\usepackage[utf8]{inputenc}")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\usepackage{graphicx}")
    tex.append(r"\usepackage{amsmath}")
    tex.append(r"\usepackage{hyperref}")
    tex.append(r"\usepackage[margin=1in]{geometry}")
    tex.append(r"\usepackage{float}")
    tex.append("")
    tex.append(r"\title{Qwen2.5-VL-3B: Zero-Shot Robust vs Blended Prompting \\ on CIFAR-10 Corruptions}")
    tex.append(r"\author{VizInt Experiment}")
    tex.append(r"\date{\today}")
    tex.append("")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")
    tex.append("")

    tex.append(r"\begin{abstract}")
    tex.append(f"We evaluate {esc(MODEL_ID)} (4-bit quantized) as a zero-shot classifier "
               r"on CIFAR-10 with image corruptions. Two prompting strategies are compared: "
               r"\textbf{Robust} (classify only) and \textbf{Blended} (identify corruption then classify). "
               f"Evaluated on {num_samples} images per condition.")
    tex.append(r"\end{abstract}")
    tex.append("")

    # Method
    tex.append(r"\section{Method}")
    tex.append(r"The Robust prompt asks the VLM to classify into CIFAR-10 categories via JSON output. "
               r"The Blended prompt additionally asks the model to identify the corruption type "
               r"(none, gaussian\_noise, rotation, affine) before classifying, "
               r"mirroring how auxiliary transform prediction heads improve traditional models.")
    tex.append("")

    # Results table
    tex.append(r"\section{Results}")
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Classification accuracy at various corruption severities.}")
    tex.append(r"\begin{tabular}{lccccccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Model} & \textbf{Clean} & \textbf{S=0.25} & \textbf{S=0.5} & \textbf{S=0.75} & \textbf{S=1.0} & \textbf{Avg Drop} & \textbf{OOD} \\")
    tex.append(r"\midrule")
    for name, results, avg_drop, ood in [
        (f"{esc(MODEL_SHORT)} Robust", robust_results, robust_avg_drop, robust_ood),
        (f"{esc(MODEL_SHORT)} Blended", blended_results, blended_avg_drop, blended_ood),
    ]:
        vals = " & ".join([f"{results.get(s, 0):.4f}" for s in SEVERITIES])
        tex.append(f"{name} & {vals} & {avg_drop:.4f} & {ood:.4f} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    # Per-transform table
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Per-transform accuracy at S=1.0.}")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Model} & \textbf{Gaussian Noise} & \textbf{Rotation} & \textbf{Affine} \\")
    tex.append(r"\midrule")
    for name, per_t in [
        (f"{esc(MODEL_SHORT)} Robust", robust_per_t),
        (f"{esc(MODEL_SHORT)} Blended", blended_per_t),
    ]:
        vals = " & ".join([f"{per_t.get(tt, {}).get(1.0, 0):.4f}" for tt in ['gaussian_noise', 'rotation', 'affine']])
        tex.append(f"{name} & {vals} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    # Graph
    tex.append(r"\begin{figure}[H]")
    tex.append(r"\centering")
    tex.append(r"\includegraphics[width=0.95\textwidth]{graphs/pair_qwen_vlm_robust_vs_blended.pdf}")
    tex.append(r"\caption{Qwen2.5-VL-3B: Robust vs Blended zero-shot prompting comparison.}")
    tex.append(r"\end{figure}")
    tex.append("")

    tex.append(r"\end{document}")

    tex_path = OUTPUT_DIR / 'qwen_vlm_report.tex'
    tex_path.write_text('\n'.join(tex))
    print(f"LaTeX report saved to {tex_path}")
    return tex_path


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Qwen VLM Robustness Experiment')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of test images to evaluate (0 = full test set)')
    parser.add_argument('--skip-robust', action='store_true', help='Skip robust evaluation')
    parser.add_argument('--skip-blended', action='store_true', help='Skip blended evaluation')
    parser.add_argument('--skip-ood', action='store_true', help='Skip OOD evaluation')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info(f"QWEN VLM ROBUSTNESS EXPERIMENT: {MODEL_ID}")
    logger.info(f"Dataset: {DATASET}")
    logger.info(f"Samples: {args.num_samples if args.num_samples > 0 else 'full test set'}")
    logger.info(f"Severities: {SEVERITIES}")
    logger.info("=" * 70)

    # Load model
    model, processor = load_qwen_model(logger)

    # Load CIFAR-10 test set (unnormalized, just ToTensor)
    config_path = Path(__file__).parent / 'config' / f'{DATASET}_config.yaml'
    config = ConfigLoader(str(config_path))
    dataset_config = config.get_dataset_config(DATASET)

    test_dataset = datasets.CIFAR10(
        root=dataset_config['path'],
        train=False, download=True,
        transform=T.Compose([T.ToTensor()])
    )

    num_samples = args.num_samples if args.num_samples > 0 else len(test_dataset)

    # Results storage
    robust_results = {}
    blended_results = {}
    robust_ood = 0
    blended_ood = 0
    robust_per_t = {}
    blended_per_t = {}
    blended_transform_pred = {}

    # Check for intermediate results
    intermediate_path = OUTPUT_DIR / 'qwen_vlm_intermediate.json'

    # Evaluate Robust
    if not args.skip_robust:
        logger.info("")
        logger.info("=" * 70)
        logger.info("ROBUST EVALUATION (classify only)")
        logger.info("=" * 70)
        robust_results, _, robust_per_t = evaluate_on_corruptions(
            model, processor, test_dataset, SEVERITIES, "robust", num_samples, logger
        )
        if not args.skip_ood:
            robust_ood = evaluate_on_ood(model, processor, test_dataset, "robust", num_samples, logger)

        # Save intermediate
        intermediate = {
            'robust_results': {str(k): v for k, v in robust_results.items()},
            'robust_ood': robust_ood,
            'robust_per_t': {tt: {str(sk): sv for sk, sv in d.items()} for tt, d in robust_per_t.items()},
        }
        with open(intermediate_path, 'w') as f:
            json.dump(intermediate, f, indent=2)
        logger.info(f"Intermediate results saved to {intermediate_path}")

    # Evaluate Blended
    if not args.skip_blended:
        logger.info("")
        logger.info("=" * 70)
        logger.info("BLENDED EVALUATION (identify transform + classify)")
        logger.info("=" * 70)
        blended_results, blended_transform_pred, blended_per_t = evaluate_on_corruptions(
            model, processor, test_dataset, SEVERITIES, "blended", num_samples, logger
        )
        if not args.skip_ood:
            blended_ood = evaluate_on_ood(model, processor, test_dataset, "blended", num_samples, logger)

    # Load intermediate if we skipped robust
    if args.skip_robust and intermediate_path.exists():
        with open(intermediate_path) as f:
            intermediate = json.load(f)
        robust_results = {float(k): v for k, v in intermediate['robust_results'].items()}
        robust_ood = intermediate['robust_ood']
        robust_per_t = {tt: {float(sk): sv for sk, sv in d.items()}
                        for tt, d in intermediate['robust_per_t'].items()}

    # Free model memory
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    # Generate outputs
    logger.info("")
    logger.info("=" * 70)
    logger.info("GENERATING REPORTS & GRAPHS")
    logger.info("=" * 70)

    # Graph
    generate_paired_graph(robust_results, blended_results, robust_ood, blended_ood,
                          robust_per_t, blended_per_t, GRAPHS_DIR)

    # Update results.json
    update_results_json(robust_results, blended_results, robust_ood, blended_ood,
                        robust_per_t, blended_per_t, blended_transform_pred)

    # Append to README
    append_to_readme(robust_results, blended_results, robust_ood, blended_ood,
                     robust_per_t, blended_per_t, blended_transform_pred, num_samples)

    # Generate LaTeX
    tex_path = generate_latex_section(robust_results, blended_results, robust_ood, blended_ood,
                                      robust_per_t, blended_per_t, blended_transform_pred, num_samples)

    # Try to compile PDF
    import subprocess
    try:
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-output-directory', str(OUTPUT_DIR),
             str(tex_path)],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            logger.info(f"PDF compiled: {OUTPUT_DIR / 'qwen_vlm_report.pdf'}")
        else:
            logger.warning(f"pdflatex failed: {result.stderr[:500]}")
    except FileNotFoundError:
        logger.warning("pdflatex not found, skipping PDF compilation")
    except subprocess.TimeoutExpired:
        logger.warning("pdflatex timed out")

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Robust  - Clean: {robust_results.get(0.0, 0):.4f}, S=1.0: {robust_results.get(1.0, 0):.4f}, OOD: {robust_ood:.4f}")
    logger.info(f"Blended - Clean: {blended_results.get(0.0, 0):.4f}, S=1.0: {blended_results.get(1.0, 0):.4f}, OOD: {blended_ood:.4f}")
    logger.info(f"Graph: {GRAPHS_DIR / 'pair_qwen_vlm_robust_vs_blended.png'}")
    logger.info(f"README updated: {Path(__file__).parent.parent.parent / 'README.md'}")


if __name__ == '__main__':
    main()

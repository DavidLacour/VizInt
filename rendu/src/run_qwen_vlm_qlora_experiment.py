#!/usr/bin/env python3
"""
Qwen2.5-VL QLoRA Fine-Tuning Experiment: Robust vs Blended.

Fine-tunes Qwen2.5-VL-3B with QLoRA adapters on CIFAR-10:
  - Robust: classify only, trained with augmented images
  - Blended: classify + predict transform type, with 0.95/0.05 loss weighting

Evaluates on the full 10K CIFAR-10 test set.

Usage:
    python run_qwen_vlm_qlora_experiment.py
    python run_qwen_vlm_qlora_experiment.py --train-samples 5000 --epochs 3
    python run_qwen_vlm_qlora_experiment.py --skip-training  # eval only from checkpoints
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
from PIL import Image
from torchvision import transforms as T, datasets
import logging
import argparse
import gc
import random

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
MODEL_SHORT = "Qwen2.5-VL-3B-QLoRA"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
TRANSFORM_TYPES_CANON = ["no_transform", "gaussian_noise", "rotation", "affine"]

OUTPUT_DIR = Path(__file__).parent / "robustness_experiment_outputs"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
CHECKPOINT_DIR = OUTPUT_DIR / "qwen_qlora_checkpoints"

ROBUST_PROMPT = (
    "Classify this image into exactly one of these categories: "
    "[airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]. "
    'Respond with ONLY a JSON object: {"class": "<category>"}'
)

BLENDED_PROMPT = (
    "This image may have been corrupted. First identify the corruption type from: "
    "[none, gaussian_noise, rotation, affine]. Then classify into one of: "
    "[airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]. "
    'Respond with ONLY a JSON object: {"transform": "<type>", "class": "<category>"}'
)


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(OUTPUT_DIR / 'qwen_qlora_experiment.log'))
        ]
    )
    return logging.getLogger('qwen_qlora_experiment')


# ============================================================================
# Model Loading
# ============================================================================
def load_base_model(logger):
    """Load Qwen2.5-VL-3B with 4-bit quantization for QLoRA."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    logger.info(f"Loading {MODEL_ID} with 4-bit quantization for QLoRA...")

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
    # Ensure pad token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    logger.info("Base model loaded.")
    return model, processor


def apply_qlora(model, logger):
    """Apply QLoRA adapters including output projection (lm_head)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head",
        ],
        modules_to_save=None,
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"QLoRA applied: {trainable:,} trainable / {total:,} total params "
                f"({100 * trainable / total:.2f}%)")
    return model


# ============================================================================
# Image Helpers
# ============================================================================
def tensor_to_pil(img_tensor, upscale_size=224):
    """Convert [C, H, W] tensor in [0,1] to upscaled PIL Image."""
    img_tensor = torch.clamp(img_tensor, 0, 1)
    pil_img = T.ToPILImage()(img_tensor.cpu())
    if upscale_size and pil_img.size[0] < upscale_size:
        pil_img = pil_img.resize((upscale_size, upscale_size), Image.Resampling.BICUBIC)
    return pil_img


# ============================================================================
# Training
# ============================================================================
def create_training_example(processor, pil_image, prompt, target_json_str):
    """Create a single training example with input_ids and labels."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": target_json_str},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    inputs = processor(
        text=[text],
        images=[pil_image],
        padding=False,
        return_tensors="pt",
    )

    # Create labels: mask input tokens (set to -100), only compute loss on assistant response
    input_ids = inputs["input_ids"][0]
    labels = input_ids.clone()

    # Find the assistant response start - tokenize just the prompt part
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False,
                                                 add_generation_prompt=True)
    prompt_inputs = processor(
        text=[prompt_text],
        images=[pil_image],
        padding=False,
        return_tensors="pt",
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]

    # Mask everything before the assistant response
    labels[:prompt_len] = -100

    inputs["labels"] = labels.unsqueeze(0)
    return inputs


def compute_standard_loss(logits, labels):
    """Compute standard causal LM cross-entropy loss (for robust training)."""
    import torch.nn.functional as F

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def compute_blended_loss(logits, labels, input_ids, processor):
    """Compute blended loss matching traditional experiment formula:
    loss = cls_loss + cls_loss * 0.01 * sigmoid(transform_loss)

    Separates LM loss into class tokens and transform tokens,
    then combines using the same formula as blended_trainer.py line 101.
    """
    import torch.nn.functional as F

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute per-token loss
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none',
        ignore_index=-100,
    ).view(shift_labels.shape)

    valid_mask = (shift_labels != -100).float()

    # Try to separate class vs transform tokens
    class_mask = torch.zeros_like(valid_mask)
    transform_mask = torch.zeros_like(valid_mask)

    try:
        label_tokens = shift_labels[0]
        valid_positions = (label_tokens != -100).nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            response_token_ids = label_tokens[valid_positions]
            response_text = processor.tokenizer.decode(response_token_ids, skip_special_tokens=True)

            class_match = re.search(r'"class"\s*:\s*"([^"]+)"', response_text)
            transform_match = re.search(r'"transform"\s*:\s*"([^"]+)"', response_text)

            if class_match:
                class_value = class_match.group(1)
                class_token_ids = processor.tokenizer.encode(class_value, add_special_tokens=False)
                for i in range(len(valid_positions) - len(class_token_ids) + 1):
                    match = True
                    for j, tid in enumerate(class_token_ids):
                        if i + j >= len(valid_positions) or label_tokens[valid_positions[i + j]] != tid:
                            match = False
                            break
                    if match:
                        for j in range(len(class_token_ids)):
                            class_mask[0, valid_positions[i + j]] = 1.0
                        break

            if transform_match:
                transform_value = transform_match.group(1)
                transform_token_ids = processor.tokenizer.encode(transform_value,
                                                                  add_special_tokens=False)
                for i in range(len(valid_positions) - len(transform_token_ids) + 1):
                    match = True
                    for j, tid in enumerate(transform_token_ids):
                        if i + j >= len(valid_positions) or label_tokens[valid_positions[i + j]] != tid:
                            match = False
                            break
                    if match:
                        for j in range(len(transform_token_ids)):
                            transform_mask[0, valid_positions[i + j]] = 1.0
                        break
    except Exception:
        pass

    # Compute separate losses for class and transform tokens
    class_token_count = class_mask.sum().clamp(min=1)
    transform_token_count = transform_mask.sum().clamp(min=1)

    cls_loss = (per_token_loss * class_mask).sum() / class_token_count
    transform_loss = (per_token_loss * transform_mask).sum() / transform_token_count

    # If we couldn't separate tokens, fall back to standard loss
    if class_mask.sum() == 0:
        return per_token_loss.sum() / valid_mask.sum().clamp(min=1)

    # Apply the same formula as traditional blended_trainer.py:
    # loss = cls_loss + cls_loss * 0.01 * sigmoid(transform_loss)
    loss = cls_loss + cls_loss * 0.01 * torch.sigmoid(transform_loss)

    # Also include loss on structural tokens (JSON braces, keys, etc.)
    structural_mask = valid_mask - class_mask - transform_mask
    structural_mask = structural_mask.clamp(min=0)
    if structural_mask.sum() > 0:
        structural_loss = (per_token_loss * structural_mask).sum() / structural_mask.sum()
        loss = loss + 0.1 * structural_loss  # Small weight for JSON structure

    return loss


def train_model(model, processor, train_dataset, strategy, num_train_samples, epochs,
                logger, grad_accum=8, lr=2e-4):
    """Train QLoRA model on CIFAR-10 with transforms.

    Both Robust and Blended use the SAME augmentation for fair comparison:
      - 100% of images are augmented
      - Severity sampled from U[0, 1.0]
      - All transform types including no_transform

    The only difference is the output format and loss:
      - Robust: {"class": "..."} — standard LM loss
      - Blended: {"transform": "...", "class": "..."} — blended loss formula
        loss = cls_loss + cls_loss * 0.01 * sigmoid(transform_loss)
    """
    is_blended = (strategy == "blended")
    prompt = BLENDED_PROMPT if is_blended else ROBUST_PROMPT
    strategy_label = "Blended" if is_blended else "Robust"

    logger.info(f"Training {strategy_label} QLoRA model...")
    logger.info(f"  Samples: {num_train_samples}, Epochs: {epochs}, LR: {lr}")
    logger.info(f"  Grad accumulation: {grad_accum}")
    logger.info(f"  Augmentation: 100% probability, severity U[0, 1.0] (same for both)")
    if is_blended:
        logger.info(f"  Loss: cls_loss + cls_loss * 0.01 * sigmoid(transform_loss)")
    else:
        logger.info(f"  Loss: standard LM cross-entropy")

    total = len(train_dataset)
    if num_train_samples > 0 and num_train_samples < total:
        indices = np.random.RandomState(42).choice(total, num_train_samples, replace=False)
    else:
        indices = list(range(total))
        num_train_samples = total

    # Optimizer — AdamW matching traditional experiment
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.05  # Match traditional: weight_decay=0.05
    )

    # Cosine schedule with warmup (matching traditional CosineAnnealingLR)
    total_steps = (len(indices) * epochs) // grad_accum
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps
    )

    continuous_transforms = ContinuousTransforms(severity=1.0)
    transform_types = continuous_transforms.transform_types  # includes no_transform

    model.train()
    global_step = 0
    running_loss = 0.0
    step_i = 0

    for epoch in range(epochs):
        epoch_indices = list(indices)
        random.shuffle(epoch_indices)

        logger.info(f"  Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()

        pbar = tqdm(epoch_indices, desc=f"{strategy_label} Epoch {epoch + 1}", leave=False)
        for step_i, idx in enumerate(pbar):
            img_tensor, label = train_dataset[idx]
            true_class = CIFAR10_CLASSES[label]

            # Same augmentation for both strategies: 100%, severity U[0, 1.0]
            t_type = np.random.choice(transform_types)
            severity = np.random.uniform(0.0, 1.0)
            t_img = continuous_transforms.apply_transforms_unnormalized(
                img_tensor, transform_type=t_type, severity=severity
            )
            pil_img = tensor_to_pil(t_img)

            # Build target JSON
            if is_blended:
                t_name = "none" if t_type == "no_transform" else t_type
                target_json = json.dumps({"transform": t_name, "class": true_class})
            else:
                target_json = json.dumps({"class": true_class})

            try:
                inputs = create_training_example(processor, pil_img, prompt, target_json)
            except Exception:
                continue

            device = model.device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                labels=None,
            )

            # Loss computation
            if is_blended:
                # Blended: cls_loss + cls_loss * 0.01 * sigmoid(transform_loss)
                loss = compute_blended_loss(
                    outputs.logits, inputs["labels"], inputs["input_ids"], processor
                )
            else:
                # Robust: standard LM cross-entropy
                loss = compute_standard_loss(outputs.logits, inputs["labels"])

            loss = loss / grad_accum
            loss.backward()

            running_loss += loss.item()

            if (step_i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = running_loss
                running_loss = 0.0
                pbar.set_postfix(loss=f"{avg_loss:.4f}", step=global_step)

        logger.info(f"    Epoch {epoch + 1} done. Global steps: {global_step}")

    # Final optimizer step for remaining gradients
    if (step_i + 1) % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()

    logger.info(f"Training complete. Total optimizer steps: {global_step}")
    return model


def save_adapter(model, save_path, logger):
    """Save QLoRA adapter weights."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    logger.info(f"Adapter saved to {save_path}")


def load_adapter(base_model, adapter_path, logger):
    """Load QLoRA adapter weights onto base model."""
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    logger.info(f"Adapter loaded from {adapter_path}")
    return model


# ============================================================================
# Inference
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
    inputs = processor(text=[text], images=[pil_image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()


def parse_json_output(text, expect_transform=False):
    class_name = None
    transform_name = None
    try:
        data = json.loads(text)
        class_name = data.get("class", "").strip().lower()
        if expect_transform:
            transform_name = data.get("transform", "").strip().lower()
        return class_name, transform_name
    except json.JSONDecodeError:
        pass
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
    class_match = re.search(r'"class"\s*:\s*"([^"]+)"', text)
    if class_match:
        class_name = class_match.group(1).strip().lower()
    if expect_transform:
        transform_match = re.search(r'"transform"\s*:\s*"([^"]+)"', text)
        if transform_match:
            transform_name = transform_match.group(1).strip().lower()
    return class_name, transform_name


def normalize_class_name(name):
    if not name:
        return None
    name = name.lower().strip()
    if name in CIFAR10_CLASSES:
        return name
    aliases = {
        "car": "automobile", "auto": "automobile", "vehicle": "automobile",
        "plane": "airplane", "aeroplane": "airplane", "jet": "airplane",
        "hound": "dog", "puppy": "dog", "kitty": "cat", "kitten": "cat",
        "boat": "ship", "vessel": "ship", "lorry": "truck", "van": "truck",
        "stag": "deer", "doe": "deer", "toad": "frog",
        "stallion": "horse", "pony": "horse",
    }
    return aliases.get(name, None)


def normalize_transform_name(name):
    if not name:
        return None
    name = name.lower().strip()
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
# Evaluation (same as zero-shot script but uses full 10K)
# ============================================================================
def evaluate_on_corruptions(model, processor, test_dataset, severities, strategy,
                            num_samples, logger):
    is_blended = (strategy == "blended")
    prompt = BLENDED_PROMPT if is_blended else ROBUST_PROMPT
    strategy_label = "Blended" if is_blended else "Robust"

    accuracy_results = {}
    transform_pred_acc = {}
    per_transform_results = {tt: {} for tt in TRANSFORM_TYPES_CANON}

    total = len(test_dataset)
    if num_samples > 0 and num_samples < total:
        indices = np.random.RandomState(42).choice(total, num_samples, replace=False)
    else:
        indices = list(range(total))
        num_samples = total

    logger.info(f"Evaluating {strategy_label} on {num_samples} samples")

    for severity in severities:
        logger.info(f"  [{strategy_label}] Severity {severity}...")
        continuous_transforms = ContinuousTransforms(severity=severity)
        transform_types_list = continuous_transforms.transform_types

        correct = 0
        total_count = 0
        transform_correct = 0
        transform_total = 0
        json_parse_failures = 0
        per_t_correct = {tt: 0 for tt in transform_types_list}
        per_t_total = {tt: 0 for tt in transform_types_list}

        for idx in tqdm(indices, desc=f"{strategy_label} S={severity}", leave=False):
            img_tensor, label = test_dataset[idx]
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

            norm_class = normalize_class_name(pred_class)
            if norm_class is None:
                json_parse_failures += 1

            true_class = CIFAR10_CLASSES[label]
            if norm_class == true_class:
                correct += 1
                per_t_correct[t_type] += 1
            total_count += 1
            per_t_total[t_type] += 1

            if is_blended and severity > 0:
                norm_transform = normalize_transform_name(pred_transform)
                if norm_transform == t_type:
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

        if severity > 0:
            for tt in ['gaussian_noise', 'rotation', 'affine']:
                if per_t_total[tt] > 0:
                    per_transform_results[tt][severity] = per_t_correct[tt] / per_t_total[tt]

    return accuracy_results, transform_pred_acc, per_transform_results


def evaluate_on_ood(model, processor, test_dataset, strategy, num_samples, logger):
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
# Graph + Report Generation (same style as zero-shot)
# ============================================================================
def generate_paired_graph(robust_results, blended_results, robust_ood, blended_ood,
                          robust_per_t, blended_per_t, graphs_dir):
    ROBUST_COLOR = '#e74c3c'
    BLENDED_COLOR = '#2ecc71'
    TRANSFORMS = ['gaussian_noise', 'rotation', 'affine']
    TRANSFORM_LABELS = {'gaussian_noise': 'Gaussian Noise', 'rotation': 'Rotation', 'affine': 'Affine'}
    SEV_FLOATS = [0.25, 0.5, 0.75, 1.0]

    robust_clean = robust_results.get(0.0, 0)
    blended_clean = blended_results.get(0.0, 0)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)
    fig.suptitle(f'{MODEL_SHORT}: Robust vs Blended (QLoRA Fine-Tuned)', fontsize=18,
                 fontweight='bold', y=0.97)

    for idx, t_name in enumerate(TRANSFORMS):
        ax = fig.add_subplot(gs[0, idx])
        robust_vals = [robust_per_t.get(t_name, {}).get(s, 0) for s in SEV_FLOATS]
        blended_vals = [blended_per_t.get(t_name, {}).get(s, 0) for s in SEV_FLOATS]
        ax.plot(SEV_FLOATS, robust_vals, 'o-', color=ROBUST_COLOR, linewidth=2.2, markersize=7,
                label=f'{MODEL_SHORT} Robust')
        ax.plot(SEV_FLOATS, blended_vals, 's-', color=BLENDED_COLOR, linewidth=2.2, markersize=7,
                label=f'{MODEL_SHORT} Blended')
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

    ax_overall = fig.add_subplot(gs[1, 0])
    all_sevs = [0.0] + SEV_FLOATS
    robust_overall = [robust_results.get(s, 0) for s in all_sevs]
    blended_overall = [blended_results.get(s, 0) for s in all_sevs]
    ax_overall.plot(all_sevs, robust_overall, 'o-', color=ROBUST_COLOR, linewidth=2.2, markersize=7,
                    label=f'{MODEL_SHORT} Robust')
    ax_overall.plot(all_sevs, blended_overall, 's-', color=BLENDED_COLOR, linewidth=2.2, markersize=7,
                    label=f'{MODEL_SHORT} Blended')
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

    ax_drop = fig.add_subplot(gs[1, 1])
    categories = list(TRANSFORM_LABELS.values()) + ['Overall']
    robust_drops = []
    blended_drops = []
    for t_name in TRANSFORMS:
        robust_drops.append(robust_clean - robust_per_t.get(t_name, {}).get(1.0, 0))
        blended_drops.append(blended_clean - blended_per_t.get(t_name, {}).get(1.0, 0))
    robust_drops.append(robust_clean - robust_results.get(1.0, 0))
    blended_drops.append(blended_clean - blended_results.get(1.0, 0))
    x_pos = np.arange(len(categories))
    width = 0.35
    bars_r = ax_drop.bar(x_pos - width / 2, robust_drops, width, color=ROBUST_COLOR, alpha=0.85,
                          label=f'{MODEL_SHORT} Robust')
    bars_b = ax_drop.bar(x_pos + width / 2, blended_drops, width, color=BLENDED_COLOR, alpha=0.85,
                          label=f'{MODEL_SHORT} Blended')
    for bar in bars_r:
        ax_drop.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, color=ROBUST_COLOR)
    for bar in bars_b:
        ax_drop.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, color=BLENDED_COLOR)
    ax_drop.set_title('Accuracy Drop (Clean -> S=1.0)', fontsize=13, fontweight='bold')
    ax_drop.set_ylabel('Accuracy Drop', fontsize=11)
    ax_drop.set_xticks(x_pos)
    ax_drop.set_xticklabels(categories, fontsize=9, rotation=15, ha='right')
    ax_drop.grid(True, alpha=0.3, axis='y')
    ax_drop.legend(fontsize=8)

    ax_ood = fig.add_subplot(gs[1, 2])
    x_pos2 = np.arange(2)
    bars_r2 = ax_ood.bar(x_pos2 - width / 2, [robust_clean, robust_ood], width,
                          color=ROBUST_COLOR, alpha=0.85, label=f'{MODEL_SHORT} Robust')
    bars_b2 = ax_ood.bar(x_pos2 + width / 2, [blended_clean, blended_ood], width,
                          color=BLENDED_COLOR, alpha=0.85, label=f'{MODEL_SHORT} Blended')
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
    ax_ood.set_xticklabels(['Clean', 'OOD'], fontsize=11)
    ax_ood.grid(True, alpha=0.3, axis='y')
    ax_ood.legend(fontsize=8)

    fname = "pair_qwen_qlora_robust_vs_blended"
    for fmt in ['png', 'pdf']:
        fig.savefig(graphs_dir / f"{fname}.{fmt}", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}.png/.pdf")


def append_to_readme(robust_results, blended_results, robust_ood, blended_ood,
                     robust_per_t, blended_per_t, transform_pred_acc, num_eval_samples):
    readme_path = Path(__file__).parent.parent.parent / "README.md"

    robust_clean = robust_results.get(0.0, 0)
    blended_clean = blended_results.get(0.0, 0)
    robust_avg_drop = np.mean([robust_clean - robust_results.get(s, 0)
                               for s in SEVERITIES if s > 0]) if robust_clean > 0 else 0
    blended_avg_drop = np.mean([blended_clean - blended_results.get(s, 0)
                                for s in SEVERITIES if s > 0]) if blended_clean > 0 else 0

    section = []
    section.append("")
    section.append(f"## {MODEL_SHORT} (QLoRA Fine-Tuned)")
    section.append("")
    section.append(f"*Evaluated on {num_eval_samples} CIFAR-10 test images per condition.*")
    section.append("")
    section.append("### Method")
    section.append("")
    section.append(f"We fine-tune **{MODEL_ID}** (4-bit quantized) using QLoRA adapters "
                   "(LoRA r=16, alpha=32) targeting attention layers + output projection (lm_head). "
                   "Both models use the same augmentation (100% probability, severity U[0,1.0]) for fair comparison:")
    section.append("")
    section.append('- **Robust**: Fine-tuned to classify augmented images: `{"class": "<category>"}` — standard LM loss')
    section.append('- **Blended**: Fine-tuned to identify corruption AND classify: '
                   '`{"transform": "<type>", "class": "<category>"}` — '
                   'loss = `cls_loss + cls_loss * 0.01 * sigmoid(transform_loss)`')
    section.append("")

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

    section.append(f"### {MODEL_SHORT}: Robust vs Blended")
    section.append(f"![{MODEL_SHORT} Robust vs Blended](rendu/src/robustness_experiment_outputs/graphs/pair_qwen_qlora_robust_vs_blended.png)")
    section.append("")

    with open(readme_path, 'a') as f:
        f.write('\n'.join(section))
    print(f"Appended QLoRA results to {readme_path}")


def update_results_json(robust_results, blended_results, robust_ood, blended_ood,
                        robust_per_t, blended_per_t, transform_pred_acc):
    results_path = OUTPUT_DIR / 'results.json'
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
    else:
        data = {}

    if 'accuracy_results' not in data:
        data['accuracy_results'] = {}
    data['accuracy_results']['qwen_qlora_robust'] = {str(k): v for k, v in robust_results.items()}
    data['accuracy_results']['qwen_qlora_blended'] = {str(k): v for k, v in blended_results.items()}

    if 'ood_results' not in data:
        data['ood_results'] = {}
    data['ood_results']['qwen_qlora_robust'] = robust_ood
    data['ood_results']['qwen_qlora_blended'] = blended_ood

    if 'per_transform_results' not in data:
        data['per_transform_results'] = {}
    data['per_transform_results']['qwen_qlora_robust'] = {
        tt: {str(sk): sv for sk, sv in d.items()} for tt, d in robust_per_t.items()
    }
    data['per_transform_results']['qwen_qlora_blended'] = {
        tt: {str(sk): sv for sk, sv in d.items()} for tt, d in blended_per_t.items()
    }

    if 'transform_pred_acc' not in data:
        data['transform_pred_acc'] = {}
    data['transform_pred_acc']['qwen_qlora_blended'] = {str(k): v for k, v in transform_pred_acc.items()}

    with open(results_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated {results_path}")


def generate_latex_report(robust_results, blended_results, robust_ood, blended_ood,
                          robust_per_t, blended_per_t, transform_pred_acc, num_eval):
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
    tex.append(r"\title{Qwen2.5-VL-3B QLoRA: Robust vs Blended Fine-Tuning \\ on CIFAR-10 Corruptions}")
    tex.append(r"\author{VizInt Experiment}")
    tex.append(r"\date{\today}")
    tex.append("")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")
    tex.append("")
    tex.append(r"\begin{abstract}")
    tex.append(f"We fine-tune {esc(MODEL_ID)} using QLoRA (4-bit, LoRA r=16) on CIFAR-10 with augmentation. "
               r"Two strategies are compared: \textbf{Robust} (classify only) and \textbf{Blended} "
               r"(predict corruption then classify, with 95\%/5\% loss weighting). "
               f"Evaluated on {num_eval} test images per condition.")
    tex.append(r"\end{abstract}")
    tex.append("")

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

    tex.append(r"\begin{figure}[H]")
    tex.append(r"\centering")
    tex.append(r"\includegraphics[width=0.95\textwidth]{graphs/pair_qwen_qlora_robust_vs_blended.pdf}")
    tex.append(r"\caption{Qwen2.5-VL-3B QLoRA: Robust vs Blended fine-tuning comparison.}")
    tex.append(r"\end{figure}")
    tex.append("")
    tex.append(r"\end{document}")

    tex_path = OUTPUT_DIR / 'qwen_qlora_report.tex'
    tex_path.write_text('\n'.join(tex))
    print(f"LaTeX report saved to {tex_path}")
    return tex_path


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Qwen VLM QLoRA Robustness Experiment')
    parser.add_argument('--train-samples', type=int, default=5000,
                        help='Number of training images (0 = full train set)')
    parser.add_argument('--eval-samples', type=int, default=0,
                        help='Number of test images for eval (0 = full 10K)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--grad-accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--skip-training', action='store_true', help='Skip training, load adapters')
    parser.add_argument('--skip-robust', action='store_true')
    parser.add_argument('--skip-blended', action='store_true')
    parser.add_argument('--skip-ood', action='store_true')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info(f"QWEN VLM QLoRA EXPERIMENT: {MODEL_ID}")
    logger.info(f"Train samples: {args.train_samples or 'full'}, Epochs: {args.epochs}")
    logger.info(f"Eval samples: {args.eval_samples or 'full 10K'}")
    logger.info(f"Blended loss: cls_loss + cls_loss * 0.01 * sigmoid(transform_loss)")
    logger.info("=" * 70)

    # Load datasets
    config_path = Path(__file__).parent / 'config' / f'{DATASET}_config.yaml'
    config = ConfigLoader(str(config_path))
    dataset_config = config.get_dataset_config(DATASET)

    train_dataset = datasets.CIFAR10(
        root=dataset_config['path'], train=True, download=True,
        transform=T.Compose([T.ToTensor()])
    )
    test_dataset = datasets.CIFAR10(
        root=dataset_config['path'], train=False, download=True,
        transform=T.Compose([T.ToTensor()])
    )

    num_eval = args.eval_samples if args.eval_samples > 0 else len(test_dataset)

    robust_results = {}
    blended_results = {}
    robust_ood = 0.0
    blended_ood = 0.0
    robust_per_t = {}
    blended_per_t = {}
    blended_transform_pred = {}

    # ---- ROBUST ----
    if not args.skip_robust:
        logger.info("\n" + "=" * 70)
        logger.info("ROBUST MODEL")
        logger.info("=" * 70)

        model, processor = load_base_model(logger)
        robust_adapter_path = CHECKPOINT_DIR / "robust_adapter"

        if not args.skip_training:
            model = apply_qlora(model, logger)
            model = train_model(model, processor, train_dataset, "robust",
                                args.train_samples, args.epochs, logger,
                                grad_accum=args.grad_accum, lr=args.lr)
            save_adapter(model, robust_adapter_path, logger)
        else:
            model = load_adapter(model, robust_adapter_path, logger)

        model.eval()
        robust_results, _, robust_per_t = evaluate_on_corruptions(
            model, processor, test_dataset, SEVERITIES, "robust", num_eval, logger
        )
        if not args.skip_ood:
            robust_ood = evaluate_on_ood(model, processor, test_dataset, "robust", num_eval, logger)

        # Save intermediate
        intermediate = {
            'robust_results': {str(k): v for k, v in robust_results.items()},
            'robust_ood': robust_ood,
            'robust_per_t': {tt: {str(sk): sv for sk, sv in d.items()} for tt, d in robust_per_t.items()},
        }
        with open(OUTPUT_DIR / 'qwen_qlora_intermediate.json', 'w') as f:
            json.dump(intermediate, f, indent=2)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- BLENDED ----
    if not args.skip_blended:
        logger.info("\n" + "=" * 70)
        logger.info("BLENDED MODEL (cls_loss + cls_loss * 0.01 * sigmoid(transform_loss))")
        logger.info("=" * 70)

        model, processor = load_base_model(logger)
        blended_adapter_path = CHECKPOINT_DIR / "blended_adapter"

        if not args.skip_training:
            model = apply_qlora(model, logger)
            model = train_model(model, processor, train_dataset, "blended",
                                args.train_samples, args.epochs, logger,
                                grad_accum=args.grad_accum, lr=args.lr)
            save_adapter(model, blended_adapter_path, logger)
        else:
            model = load_adapter(model, blended_adapter_path, logger)

        model.eval()
        blended_results, blended_transform_pred, blended_per_t = evaluate_on_corruptions(
            model, processor, test_dataset, SEVERITIES, "blended", num_eval, logger
        )
        if not args.skip_ood:
            blended_ood = evaluate_on_ood(model, processor, test_dataset, "blended", num_eval, logger)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Load intermediate if robust was skipped
    if args.skip_robust:
        intermediate_path = OUTPUT_DIR / 'qwen_qlora_intermediate.json'
        if intermediate_path.exists():
            with open(intermediate_path) as f:
                intermediate = json.load(f)
            robust_results = {float(k): v for k, v in intermediate['robust_results'].items()}
            robust_ood = intermediate['robust_ood']
            robust_per_t = {tt: {float(sk): sv for sk, sv in d.items()}
                            for tt, d in intermediate['robust_per_t'].items()}

    # ---- REPORTS ----
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING REPORTS & GRAPHS")
    logger.info("=" * 70)

    generate_paired_graph(robust_results, blended_results, robust_ood, blended_ood,
                          robust_per_t, blended_per_t, GRAPHS_DIR)
    update_results_json(robust_results, blended_results, robust_ood, blended_ood,
                        robust_per_t, blended_per_t, blended_transform_pred)
    append_to_readme(robust_results, blended_results, robust_ood, blended_ood,
                     robust_per_t, blended_per_t, blended_transform_pred, num_eval)
    tex_path = generate_latex_report(robust_results, blended_results, robust_ood, blended_ood,
                                     robust_per_t, blended_per_t, blended_transform_pred, num_eval)

    import subprocess
    try:
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-output-directory', str(OUTPUT_DIR),
             str(tex_path)],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            logger.info(f"PDF compiled: {OUTPUT_DIR / 'qwen_qlora_report.pdf'}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("pdflatex not available or timed out")

    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Robust  - Clean: {robust_results.get(0.0, 0):.4f}, S=1.0: {robust_results.get(1.0, 0):.4f}, OOD: {robust_ood:.4f}")
    logger.info(f"Blended - Clean: {blended_results.get(0.0, 0):.4f}, S=1.0: {blended_results.get(1.0, 0):.4f}, OOD: {blended_ood:.4f}")


if __name__ == '__main__':
    main()

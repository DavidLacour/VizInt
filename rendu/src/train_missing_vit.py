#!/usr/bin/env python3
"""
Train the 2 missing ViT models (vanilla_vit_robust and blended_training)
with reduced batch size to avoid CUDA OOM.
"""
import sys
import gc
import torch
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.config_loader import ConfigLoader
from models.model_factory import ModelFactory
from data.data_loader import DataLoaderFactory
from services.model_trainer import ModelTrainer
from utils.transformer_utils import set_seed

# Setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger('train_missing_vit')

DATASET = "cifar10"
config_path = str(Path(__file__).parent / 'config' / f'{DATASET}_config.yaml')

# Models to train
MODELS_TO_TRAIN = [
    {'model_type': 'vanilla_vit_robust', 'display': 'ViT Robust', 'is_robust': True, 'is_blended': False},
    {'model_type': 'blended_training', 'display': 'ViT Blended', 'is_robust': False, 'is_blended': True},
]

def main():
    set_seed(42)

    # Set memory allocation config to reduce fragmentation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    config = ConfigLoader(config_path)
    checkpoint_dir = config.get_checkpoint_dir(DATASET)

    for model_info in MODELS_TO_TRAIN:
        model_type = model_info['model_type']
        display = model_info['display']
        checkpoint_path = checkpoint_dir / f"bestmodel_{model_type}" / "best_model.pt"

        if checkpoint_path.exists():
            logger.info(f"SKIP {display} ({model_type}) - checkpoint exists")
            continue

        logger.info(f"=== TRAINING {display} ({model_type}) ===")

        # Aggressively clear GPU memory before each model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Reduce batch size for ViT models to fit in 8GB
        original_batch_size = config.get('training.batch_size', 300)
        config.config['training']['batch_size'] = 128
        logger.info(f"Reduced batch_size from {original_batch_size} to 128 for ViT training")

        try:
            model_factory = ModelFactory(config)
            data_factory = DataLoaderFactory(config)
            model_trainer = ModelTrainer(config, model_factory, data_factory)

            model, history = model_trainer.train_model(
                model_type=model_type,
                dataset_name=DATASET,
                robust_training=model_info['is_robust'],
            )

            logger.info(f"SUCCESS: {display} - Best metric: {history['best_metric']:.4f} at epoch {history['best_epoch'] + 1}")

            # Clean up
            del model
            del model_factory
            del data_factory
            del model_trainer

        except Exception as e:
            logger.error(f"FAILED to train {display}: {e}")
            import traceback
            traceback.print_exc()

        # Aggressively clean up
        gc.collect()
        torch.cuda.empty_cache()

        # Restore batch size
        config.config['training']['batch_size'] = original_batch_size

    logger.info("Done training missing ViT models.")

if __name__ == '__main__':
    main()

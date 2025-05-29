"""
Model trainer service for handling the complete training pipeline
"""
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

from src.config.config_loader import ConfigLoader
from src.models.model_factory import ModelFactory
from src.data.data_loader import DataLoaderFactory
from src.trainers.classification_trainer import ClassificationTrainer


class ModelTrainer:
    """Service class for training models"""
    
    def __init__(self, 
                 config: ConfigLoader,
                 model_factory: ModelFactory,
                 data_factory: DataLoaderFactory):
        """
        Initialize model trainer
        
        Args:
            config: Configuration loader
            model_factory: Model factory instance
            data_factory: Data loader factory instance
        """
        self.config = config
        self.model_factory = model_factory
        self.data_factory = data_factory
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device(config.get_device())
        
    def train_model(self,
                   model_type: str,
                   dataset_name: str,
                   base_model: Optional[nn.Module] = None,
                   robust_training: bool = False) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Train a model
        
        Args:
            model_type: Type of model to train
            dataset_name: Name of dataset
            base_model: Base model for TTT-based models
            robust_training: Whether to use robust training
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        self.logger.info(f"Starting training for {model_type} on {dataset_name}")
        
        # Handle special model dependencies
        if model_type in ['ttt', 'ttt_robust', 'ttt3fc', 'ttt3fc_robust']:
            if base_model is None:
                # Load base model
                base_model = self._load_base_model(dataset_name, robust=(model_type.endswith('_robust')))
        
        # Create model
        model = self.model_factory.create_model(model_type, dataset_name, base_model)
        model = model.to(self.device)
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Created {model_type} model with {num_params:,} trainable parameters")
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(model_type, dataset_name)
        
        # Create trainer
        trainer = self._create_trainer(model_type, model, robust_training)
        
        # Get checkpoint directory
        checkpoint_dir = self._get_checkpoint_dir(model_type, dataset_name)
        
        # Train model
        history = trainer.train(train_loader, val_loader, checkpoint_dir)
        
        self.logger.info(f"Training completed for {model_type}")
        self.logger.info(f"Best metric: {history['best_metric']:.4f} at epoch {history['best_epoch'] + 1}")
        
        return model, history
    
    def _load_base_model(self, dataset_name: str, robust: bool = False) -> nn.Module:
        """Load base model for TTT-based models"""
        model_type = 'vanilla_vit_robust' if robust else 'vanilla_vit'
        checkpoint_dir = self.config.get_checkpoint_dir(dataset_name)
        checkpoint_path = checkpoint_dir / f"bestmodel_{model_type}" / "best_model.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Base model not found at {checkpoint_path}. "
                f"Please train the {model_type} model first."
            )
        
        return self.model_factory.load_model_from_checkpoint(
            checkpoint_path, model_type, dataset_name, device=self.device
        )
    
    def _create_data_loaders(self, model_type: str, dataset_name: str):
        """Create appropriate data loaders for model type"""
        # Determine if normalization is needed
        # For healer and transformation-aware models, we might need unnormalized data
        if model_type in ['healer', 'ttt', 'ttt3fc', 'blended', 'blended3fc', 'blended_resnet18']:
            # These models handle normalization internally
            with_normalization = False
        else:
            with_normalization = True
        
        # Create loaders
        train_loader, val_loader = self.data_factory.create_data_loaders(
            dataset_name,
            with_normalization=with_normalization,
            with_augmentation=True
        )
        
        return train_loader, val_loader
    
    def _create_trainer(self, model_type: str, model: nn.Module, robust_training: bool):
        """Create appropriate trainer for model type"""
        # Get model-specific training config
        if model_type in ['ttt', 'ttt_robust', 'ttt3fc', 'ttt3fc_robust']:
            training_config = self.config.get('training.ttt', {})
        elif model_type in ['blended', 'blended_robust', 'blended3fc', 'blended3fc_robust', 'blended_resnet18']:
            training_config = self.config.get('training.blended', {})
        else:
            training_config = self.config.get('training', {})
        
        # Merge with general training config
        general_config = self.config.get('training', {})
        merged_config = {**general_config, **training_config}
        
        # Create trainer based on model type
        if model_type in ['healer', 'ttt', 'ttt3fc', 'ttt_resnet18', 'blended', 'blended3fc', 'blended_resnet18']:
            # These models need special trainers
            return self._create_specialized_trainer(model_type, model, merged_config)
        else:
            # Standard classification trainer
            return ClassificationTrainer(
                model=model,
                config={'training': merged_config},
                device=str(self.device),
                robust_training=robust_training
            )
    
    def _create_specialized_trainer(self, model_type: str, model: nn.Module, config: Dict[str, Any]):
        """Create specialized trainers for specific model types"""
        # Import specialized trainers as needed
        if model_type == 'healer':
            # Healer should always use HealerTrainer for transformation prediction
            from trainers.healer_trainer import HealerTrainer
            return HealerTrainer(
                model=model,
                config={'training': config},
                device=str(self.device)
            )
        elif model_type in ['ttt', 'ttt3fc', 'ttt_resnet18']:
            from trainers.ttt_trainer import TTTTrainer
            return TTTTrainer(
                model=model,
                config={'training': config},
                device=str(self.device)
            )
        elif model_type in ['blended', 'blended3fc', 'blended_resnet18']:
            from trainers.blended_trainer import BlendedTrainer
            return BlendedTrainer(
                model=model,
                config={'training': config},
                device=str(self.device)
            )
        else:
            # Fallback to classification trainer
            return ClassificationTrainer(
                model=model,
                config={'training': config},
                device=str(self.device)
            )
    
    def _get_checkpoint_dir(self, model_type: str, dataset_name: str) -> Path:
        """Get checkpoint directory for model"""
        base_dir = self.config.get_checkpoint_dir(dataset_name)
        return base_dir / f"bestmodel_{model_type}"
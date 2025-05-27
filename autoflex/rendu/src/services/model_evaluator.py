"""
Model evaluator service for comprehensive model evaluation
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from src.config.config_loader import ConfigLoader
from src.models.model_factory import ModelFactory
from src.data.data_loader import DataLoaderFactory


class ModelEvaluator:
    """Service class for evaluating models"""
    
    def __init__(self,
                 config: ConfigLoader,
                 model_factory: ModelFactory,
                 data_factory: DataLoaderFactory):
        """
        Initialize model evaluator
        
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
        
    def evaluate_all_combinations(self,
                                 dataset_name: str,
                                 severities: List[float],
                                 model_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate all model combinations
        
        Args:
            dataset_name: Name of dataset
            severities: List of severity levels
            model_types: Specific model types to evaluate (None for all)
            
        Returns:
            Dictionary of evaluation results
        """
        # Get model combinations from config
        combinations = self.config.get('model_combinations', [])
        
        # Load all available models
        available_models = self._load_available_models(dataset_name, model_types)
        
        # Evaluate each combination
        results = {}
        for combo in combinations:
            combo_name = combo['name']
            main_model_type = combo['main_model']
            healer_model_type = combo.get('healer_model')
            description = combo['description']
            
            # Skip if models not available or not in requested types
            if model_types:
                base_type = main_model_type.split('_')[0]
                if base_type not in model_types:
                    continue
            
            if main_model_type not in available_models:
                self.logger.warning(f"Skipping {combo_name}: {main_model_type} not available")
                continue
                
            if healer_model_type and healer_model_type not in available_models:
                self.logger.warning(f"Skipping {combo_name}: {healer_model_type} not available")
                continue
            
            self.logger.info(f"Evaluating: {description}")
            
            # Get models
            main_model = available_models[main_model_type]
            healer_model = available_models.get(healer_model_type) if healer_model_type else None
            
            # Evaluate combination
            combo_results = self._evaluate_model_combination(
                main_model=main_model,
                healer_model=healer_model,
                dataset_name=dataset_name,
                severities=severities,
                model_type=main_model_type
            )
            
            results[combo_name] = {
                'results': combo_results,
                'description': description,
                'main_model': main_model_type,
                'healer_model': healer_model_type
            }
        
        return results
    
    def _load_available_models(self, 
                              dataset_name: str,
                              model_types: Optional[List[str]] = None) -> Dict[str, nn.Module]:
        """Load all available models from checkpoints"""
        available_models = {}
        checkpoint_dir = self.config.get_checkpoint_dir(dataset_name)
        
        # Define all possible model types
        all_model_types = [
            'main', 'main_robust', 'healer', 'ttt', 'ttt_robust',
            'ttt3fc', 'ttt3fc_robust', 'blended', 'blended_robust',
            'blended3fc', 'blended3fc_robust', 'baseline', 'pretrained'
        ]
        
        # Filter by requested types
        if model_types:
            types_to_load = []
            for mt in all_model_types:
                base_type = mt.split('_')[0]
                if base_type in model_types:
                    types_to_load.append(mt)
        else:
            types_to_load = all_model_types
        
        # Load base model first (needed for TTT models)
        base_model = None
        base_model_robust = None
        
        for model_type in ['main', 'main_robust']:
            if model_type in types_to_load:
                checkpoint_path = checkpoint_dir / f"bestmodel_{model_type}" / "best_model.pt"
                if checkpoint_path.exists():
                    model = self.model_factory.load_model_from_checkpoint(
                        checkpoint_path, model_type, dataset_name, device=self.device
                    )
                    available_models[model_type] = model
                    if model_type == 'main':
                        base_model = model
                    else:
                        base_model_robust = model
        
        # Load other models
        for model_type in types_to_load:
            if model_type in available_models:
                continue
                
            checkpoint_path = checkpoint_dir / f"bestmodel_{model_type}" / "best_model.pt"
            if checkpoint_path.exists():
                # Determine base model for TTT variants
                if model_type in ['ttt', 'ttt3fc']:
                    bm = base_model
                elif model_type in ['ttt_robust', 'ttt3fc_robust']:
                    bm = base_model_robust
                else:
                    bm = None
                
                try:
                    model = self.model_factory.load_model_from_checkpoint(
                        checkpoint_path, model_type, dataset_name, base_model=bm, device=self.device
                    )
                    available_models[model_type] = model
                    self.logger.info(f"Loaded {model_type} model")
                except Exception as e:
                    self.logger.error(f"Failed to load {model_type}: {e}")
        
        return available_models
    
    def _evaluate_model_combination(self,
                                   main_model: nn.Module,
                                   healer_model: Optional[nn.Module],
                                   dataset_name: str,
                                   severities: List[float],
                                   model_type: str) -> Dict[float, float]:
        """Evaluate a specific model combination"""
        results = {}
        
        # Get normalization transform
        normalize = self.data_factory.get_normalization_transform(dataset_name)
        
        for severity in severities:
            if severity == 0.0:
                # Clean data evaluation
                _, val_loader = self.data_factory.create_data_loaders(
                    dataset_name, with_normalization=True, with_augmentation=False
                )
                accuracy = self._evaluate_clean_data(
                    main_model, healer_model, val_loader, model_type
                )
            else:
                # OOD data evaluation
                accuracy = self._evaluate_ood_data(
                    main_model, healer_model, dataset_name, severity, model_type, normalize
                )
            
            results[severity] = accuracy
            self.logger.info(f"  Severity {severity}: {accuracy:.4f}")
        
        return results
    
    def _evaluate_clean_data(self,
                            main_model: nn.Module,
                            healer_model: Optional[nn.Module],
                            val_loader,
                            model_type: str) -> float:
        """Evaluate on clean validation data"""
        main_model.eval()
        if healer_model:
            healer_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating clean data", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Apply healer if available
                if healer_model:
                    healer_predictions = healer_model(images)
                    images = healer_model.apply_correction(images, healer_predictions)
                
                # Get predictions
                outputs = self._get_model_outputs(main_model, images, model_type)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def _evaluate_ood_data(self,
                          main_model: nn.Module,
                          healer_model: Optional[nn.Module],
                          dataset_name: str,
                          severity: float,
                          model_type: str,
                          normalize) -> float:
        """Evaluate on OOD data with transformations"""
        from new_new import ContinuousTransforms
        
        main_model.eval()
        if healer_model:
            healer_model.eval()
        
        # Create OOD transform
        ood_transform = ContinuousTransforms(severity=severity)
        
        # Get validation loader without normalization
        _, val_loader = self.data_factory.create_data_loaders(
            dataset_name, with_normalization=False, with_augmentation=False
        )
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluating severity {severity}", leave=False):
                if dataset_name == 'tinyimagenet' and len(batch) == 4:
                    # Handle OOD loader format
                    orig_images, trans_images, labels, params = batch
                    images = trans_images.to(self.device)
                else:
                    # Standard format
                    images, labels = batch
                    images = images.to(self.device)
                    
                    # Apply transformations
                    batch_size = images.size(0)
                    transformed_images = []
                    
                    for i in range(batch_size):
                        # Apply random transformation
                        transform_type = np.random.choice(ood_transform.transform_types)
                        transformed_img, _ = ood_transform.apply_transforms_unnormalized(
                            images[i], transform_type=transform_type, severity=severity
                        )
                        # Normalize after transformation
                        transformed_img = normalize(transformed_img)
                        transformed_images.append(transformed_img)
                    
                    images = torch.stack(transformed_images)
                
                labels = labels.to(self.device)
                
                # Apply healer if available
                if healer_model:
                    healer_predictions = healer_model(images)
                    images = healer_model.apply_correction(images, healer_predictions)
                
                # Get predictions
                outputs = self._get_model_outputs(main_model, images, model_type)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def _get_model_outputs(self, model: nn.Module, images: torch.Tensor, model_type: str) -> torch.Tensor:
        """Get model outputs handling different model types"""
        # Handle different model output formats
        if 'ttt' in model_type or 'blended' in model_type:
            # These models return tuple (class_logits, aux_outputs)
            outputs, _ = model(images)
            return outputs
        else:
            # Standard models return just logits
            return model(images)
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a formatted table"""
        if not results:
            self.logger.warning("No results to print")
            return
        
        # Get severities
        first_result = next(iter(results.values()))
        severities = sorted(first_result['results'].keys())
        
        # Print header
        print("\n" + "="*100)
        print("EVALUATION RESULTS")
        print("="*100)
        
        # Print table header
        header = f"{'Model':<30} {'Description':<40}"
        for sev in severities:
            if sev == 0.0:
                header += f" {'Clean':>8}"
            else:
                header += f" {f'S{sev}':>8}"
        print(header)
        print("-"*100)
        
        # Sort by clean accuracy
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['results'].get(0.0, 0),
            reverse=True
        )
        
        # Print results
        for name, data in sorted_results:
            row = f"{name:<30} {data['description']:<40}"
            for sev in severities:
                acc = data['results'].get(sev, 0)
                row += f" {acc:>8.4f}"
            print(row)
        
        print("="*100)
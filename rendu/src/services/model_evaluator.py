"""
Model evaluator service for comprehensive model evaluation
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from config.config_loader import ConfigLoader
from models.model_factory import ModelFactory
from data.data_loader import DataLoaderFactory


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
                                 model_types: Optional[List[str]] = None,
                                 include_ood: bool = True) -> Dict[str, Any]:
        """
        Evaluate all model combinations
        
        Args:
            dataset_name: Name of dataset
            severities: List of severity levels
            model_types: Specific model types to evaluate (None for all)
            include_ood: Whether to include OOD evaluation
            
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
                # Check if the main model type is in the requested list
                # Also check without _robust suffix for robust variants
                model_type_to_check = main_model_type.replace('_robust', '')
                if model_type_to_check not in model_types:
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
                model_type=main_model_type,
                include_ood=include_ood
            )
            
            results[combo_name] = {
                'results': combo_results['accuracies'],
                'transform_accuracies': combo_results['transform_accuracies'],
                'ood_results': combo_results.get('ood_accuracies', {}),
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
        # Use debug checkpoint directory if in debug mode
        checkpoint_dir = self.config.get_checkpoint_dir(dataset_name, use_debug_dir=True)
        
        self.logger.info(f"Loading models from checkpoint directory: {checkpoint_dir}")
        
        # Define all possible model types
        all_model_types = [
            'vanilla_vit', 'vanilla_vit_robust', 'healer', 'ttt', 'ttt_robust',
            'ttt3fc', 'blended_training', 'blended_training_3fc', 
            'resnet', 'resnet_pretrained', 'blended_resnet18', 'ttt_resnet18', 'healer_resnet18'
        ]
        
        # Filter by requested types
        if model_types:
            types_to_load = []
            for mt in all_model_types:
                # Check exact match first
                if mt in model_types:
                    types_to_load.append(mt)
                # Also check for base type match (e.g., 'ttt' matches 'ttt_robust')
                elif mt.replace('_robust', '') in model_types:
                    types_to_load.append(mt)
            self.logger.debug(f"Requested model types: {model_types}")
            self.logger.debug(f"Types to load: {types_to_load}")
        else:
            types_to_load = all_model_types
        
        # Load base model first (needed for TTT models)
        base_model = None
        base_model_robust = None
        
        for model_type in ['vanilla_vit', 'vanilla_vit_robust']:
            if model_type in types_to_load:
                checkpoint_path = checkpoint_dir / f"bestmodel_{model_type}" / "best_model.pt"
                self.logger.debug(f"Looking for {model_type} at {checkpoint_path}")
                if checkpoint_path.exists():
                    try:
                        model = self.model_factory.load_model_from_checkpoint(
                            checkpoint_path, model_type, dataset_name, device=self.device
                        )
                        available_models[model_type] = model
                        self.logger.info(f"Loaded {model_type} model")
                        if model_type == 'vanilla_vit':
                            base_model = model
                        else:
                            base_model_robust = model
                    except Exception as e:
                        self.logger.error(f"Failed to load {model_type}: {e}")
        
        # Load other models
        for model_type in types_to_load:
            if model_type in available_models:
                continue
                
            checkpoint_path = checkpoint_dir / f"bestmodel_{model_type}" / "best_model.pt"
            if checkpoint_path.exists():
                # Determine base model for TTT variants
                if model_type in ['ttt', 'ttt3fc']:
                    bm = base_model
                elif model_type in ['ttt_robust']:
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
                                   model_type: str,
                                   include_ood: bool = True) -> Dict[str, Any]:
        """Evaluate a specific model combination"""
        results = {}
        transform_accuracies = {}
        ood_accuracies = {}
        
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
                # Robustness data evaluation
                accuracy, transform_metrics = self._evaluate_robustness_data(
                    main_model, healer_model, dataset_name, severity, model_type, normalize
                )
                if transform_metrics is not None:
                    transform_accuracies[severity] = transform_metrics
            
            results[severity] = accuracy
            self.logger.info(f"  Severity {severity}: {accuracy:.4f}")
        
        # Evaluate OOD performance with funky transforms if requested
        if include_ood:
            self.logger.info("  Evaluating OOD performance with funky transforms...")
            ood_accuracy = self._evaluate_ood_data(
                main_model, healer_model, dataset_name, model_type, normalize
            )
            ood_accuracies['funky_ood'] = ood_accuracy
            self.logger.info(f"  Funky OOD: {ood_accuracy:.4f}")
        
        return {
            'accuracies': results,
            'transform_accuracies': transform_accuracies,
            'ood_accuracies': ood_accuracies
        }
    
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
                    try:
                        # Get healer predictions
                        predictions, _ = healer_model(images, return_reconstruction=False, return_logits=False)
                        # Apply corrections using the predicted transformations
                        images = healer_model.apply_correction(images, predictions)
                    except Exception as e:
                        self.logger.warning(f"Healer failed to process images: {e}. Skipping healer for this batch.")
                
                # Get predictions
                outputs = self._get_model_outputs(main_model, images, model_type)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def _compute_param_errors(self, predictions, ground_truth_transforms, ground_truth_params, param_errors):
        """Helper to compute parameter prediction errors"""
        # For simplicity, let's track overall MAE for each parameter type
        if 'rotation_angle' in predictions:
            for i, true_type in enumerate(ground_truth_transforms):
                if true_type.item() == 2:  # rotation
                    # Find how many rotations we've seen so far
                    rotation_count = sum(1 for j in range(i) if ground_truth_transforms[j].item() == 2)
                    if rotation_count < len(ground_truth_params['rotation']):
                        pred_val = predictions['rotation_angle'][i].item()
                        true_val = ground_truth_params['rotation'][rotation_count]
                        param_errors['rotation'].append(abs(pred_val - true_val))
        
        if 'noise_std' in predictions:
            for i, true_type in enumerate(ground_truth_transforms):
                if true_type.item() == 1:  # gaussian_noise
                    noise_count = sum(1 for j in range(i) if ground_truth_transforms[j].item() == 1)
                    if noise_count < len(ground_truth_params['noise']):
                        pred_val = predictions['noise_std'][i].item()
                        true_val = ground_truth_params['noise'][noise_count]
                        param_errors['noise'].append(abs(pred_val - true_val))
        
        # Similar for translation if needed
        return param_errors
    
    def _evaluate_robustness_data(self,
                          main_model: nn.Module,
                          healer_model: Optional[nn.Module],
                          dataset_name: str,
                          severity: float,
                          model_type: str,
                          normalize) -> Tuple[float, Optional[float]]:
        """Evaluate on robustness data with transformations"""
        from src.data.continuous_transforms import ContinuousTransforms
        
        main_model.eval()
        if healer_model:
            healer_model.eval()
        
        # Create robustness transform
        transforms_for_robustness = ContinuousTransforms(severity=severity)
        
        # Get validation loader without normalization
        _, val_loader = self.data_factory.create_data_loaders(
            dataset_name, with_normalization=False, with_augmentation=False
        )
        
        correct = 0
        total = 0
        
        # Track transformation prediction accuracy
        transform_correct = 0
        transform_total = 0
        
        # Track per-transform-type accuracy
        transform_type_correct = {i: 0 for i in range(5)}  # 0: none, 1: noise, 2: rotation, 3: translate, 4: scale
        transform_type_total = {i: 0 for i in range(5)}
        
        # Track parameter prediction errors (for regression heads)
        param_errors = {
            'rotation': [],
            'noise': [],
            'translate_x': [],
            'translate_y': []
        }
        
        # Store ground truth parameters for tracking
        ground_truth_params = {
            'rotation': [],
            'noise': [],
            'translate_x': [],
            'translate_y': []
        }
        
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
                    
                    # Apply transformations and track ground truth
                    batch_size = images.size(0)
                    transformed_images = []
                    ground_truth_transforms = []
                    
                    for i in range(batch_size):
                        # Apply random transformation
                        transform_type = np.random.choice(transforms_for_robustness.transform_types)
                        # Convert transform type string to index
                        transform_idx = transforms_for_robustness.transform_types.index(transform_type)
                        ground_truth_transforms.append(transform_idx)
                        
                        # Apply transformation and get parameters
                        transformed_img, params = transforms_for_robustness.apply_transforms_unnormalized(
                            images[i], transform_type=transform_type, severity=severity, return_params=True
                        )
                        
                        # Store ground truth parameters
                        if transform_type == 'rotate' and 'angle' in params:
                            ground_truth_params['rotation'].append(params['angle'])
                        elif transform_type == 'gaussian_noise' and 'noise_std' in params:
                            ground_truth_params['noise'].append(params['noise_std'])
                        elif transform_type == 'translate':
                            ground_truth_params['translate_x'].append(params.get('translate_x', 0.0))
                            ground_truth_params['translate_y'].append(params.get('translate_y', 0.0))
                        
                        # Normalize after transformation
                        transformed_img = normalize(transformed_img)
                        transformed_images.append(transformed_img)
                    
                    images = torch.stack(transformed_images)
                    ground_truth_transforms = torch.tensor(ground_truth_transforms, dtype=torch.long).to(self.device)
                
                labels = labels.to(self.device)
                
                # Apply healer if available and track its prediction accuracy
                if healer_model:
                    try:
                        # Get healer predictions
                        predictions, _ = healer_model(images, return_reconstruction=False, return_logits=False)
                        
                        # Track healer's transform prediction accuracy
                        if 'transform_type_logits' in predictions:
                            predicted_transforms = torch.argmax(predictions['transform_type_logits'], dim=1)
                            if 'ground_truth_transforms' in locals():
                                transform_correct += (predicted_transforms == ground_truth_transforms).sum().item()
                                transform_total += ground_truth_transforms.size(0)
                                
                                # Track per-transform-type accuracy
                                for i in range(len(ground_truth_transforms)):
                                    true_type = ground_truth_transforms[i].item()
                                    pred_type = predicted_transforms[i].item()
                                    transform_type_total[true_type] = transform_type_total.get(true_type, 0) + 1
                                    if true_type == pred_type:
                                        transform_type_correct[true_type] = transform_type_correct.get(true_type, 0) + 1
                                
                                # Track parameter prediction errors for this batch
                                batch_param_errors = self._compute_param_errors(
                                    predictions, ground_truth_transforms, ground_truth_params, param_errors
                                )
                        
                        # Apply corrections using the predicted transformations
                        images = healer_model.apply_correction(images, predictions)
                    except Exception as e:
                        self.logger.warning(f"Healer failed to process images: {e}. Skipping healer for this batch.")
                
                # Get predictions and track transform predictions for all models
                if 'ttt' in model_type or 'blended' in model_type:
                    if 'ttt' in model_type:
                        outputs, aux_outputs = main_model(images)
                    else:  # blended
                        outputs, aux_outputs = main_model(images, return_aux=True)
                    
                    # Track transform prediction accuracy for TTT/Blended models
                    if aux_outputs and 'transform_type' in aux_outputs and 'ground_truth_transforms' in locals():
                        predicted_transforms = torch.argmax(aux_outputs['transform_type'], dim=1)
                        transform_correct += (predicted_transforms == ground_truth_transforms).sum().item()
                        transform_total += ground_truth_transforms.size(0)
                        
                        # Track per-transform-type accuracy
                        for i in range(len(ground_truth_transforms)):
                            true_type = ground_truth_transforms[i].item()
                            pred_type = predicted_transforms[i].item()
                            transform_type_total[true_type] = transform_type_total.get(true_type, 0) + 1
                            if true_type == pred_type:
                                transform_type_correct[true_type] = transform_type_correct.get(true_type, 0) + 1
                        
                        # Track parameter errors for TTT/Blended
                        if aux_outputs and 'ground_truth_params' in locals():
                            self._compute_param_errors(aux_outputs, ground_truth_transforms, ground_truth_params, param_errors)
                elif 'healer_resnet18' in model_type:
                    # HealerWrapper models support returning auxiliary outputs
                    outputs, aux_outputs = main_model(images, return_aux=True)
                    
                    # Track transform prediction accuracy for HealerResNet18
                    if aux_outputs and 'transform_type' in aux_outputs and 'ground_truth_transforms' in locals():
                        predicted_transforms = torch.argmax(aux_outputs['transform_type'], dim=1)
                        transform_correct += (predicted_transforms == ground_truth_transforms).sum().item()
                        transform_total += ground_truth_transforms.size(0)
                        
                        # Track per-transform-type accuracy
                        for i in range(len(ground_truth_transforms)):
                            true_type = ground_truth_transforms[i].item()
                            pred_type = predicted_transforms[i].item()
                            transform_type_total[true_type] = transform_type_total.get(true_type, 0) + 1
                            if true_type == pred_type:
                                transform_type_correct[true_type] = transform_type_correct.get(true_type, 0) + 1
                        
                        # Track parameter errors for HealerResNet18
                        if aux_outputs and 'ground_truth_params' in locals():
                            self._compute_param_errors(aux_outputs, ground_truth_transforms, ground_truth_params, param_errors)
                else:
                    outputs = self._get_model_outputs(main_model, images, model_type)
                
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate transform prediction accuracy if available
        transform_accuracy = None
        per_type_accuracy = {}
        param_mae = {}
        
        if transform_total > 0:
            transform_accuracy = transform_correct / transform_total
            self.logger.info(f"  Transform prediction accuracy: {transform_accuracy:.4f}")
            
            # Calculate per-transform-type accuracy
            transform_names = ['none', 'gaussian_noise', 'rotate', 'translate', 'scale']
            for i, name in enumerate(transform_names):
                if i in transform_type_total and transform_type_total[i] > 0:
                    acc = transform_type_correct[i] / transform_type_total[i]
                    per_type_accuracy[name] = acc
                    self.logger.info(f"    {name}: {acc:.4f} ({transform_type_correct[i]}/{transform_type_total[i]})")
            
            # Calculate parameter prediction MAE
            self.logger.info("  Parameter prediction errors (MAE):")
            for param_name, errors in param_errors.items():
                if len(errors) > 0:
                    mae = sum(errors) / len(errors)
                    param_mae[param_name] = mae
                    self.logger.info(f"    {param_name}: {mae:.4f}")
        
        # Return overall accuracy and detailed transform metrics
        transform_metrics = {
            'overall': transform_accuracy,
            'per_type': per_type_accuracy,
            'param_mae': param_mae
        } if transform_accuracy is not None else None
        
        return correct / total, transform_metrics
    
    def _evaluate_ood_data(self,
                          main_model: nn.Module,
                          healer_model: Optional[nn.Module],
                          dataset_name: str,
                          model_type: str,
                          normalize) -> float:
        """Evaluate on OOD data with extreme transformations"""
        from data.ood_transforms import OODTransforms
        
        main_model.eval()
        if healer_model:
            healer_model.eval()
        
        # Create OOD transforms with high severity
        ood_transforms = OODTransforms(severity=1.0)
        
        # Get validation loader without normalization
        _, val_loader = self.data_factory.create_data_loaders(
            dataset_name, with_normalization=False, with_augmentation=False
        )
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating OOD data", leave=False):
                if dataset_name == 'tinyimagenet' and len(batch) == 4:
                    # Handle OOD loader format
                    orig_images, _, labels, _ = batch
                    images = orig_images.to(self.device)
                else:
                    # Standard format
                    images, labels = batch
                    images = images.to(self.device)
                
                labels = labels.to(self.device)
                
                # Apply random OOD transformations to each image
                batch_size = images.size(0)
                ood_images = []
                
                for i in range(batch_size):
                    # Apply random funky transform
                    transformed_img = ood_transforms.apply_random_ood_transform(
                        images[i], severity=1.0
                    )
                    # Normalize after transformation
                    transformed_img = normalize(transformed_img)
                    ood_images.append(transformed_img)
                
                images = torch.stack(ood_images)
                
                # Apply healer if available
                if healer_model:
                    try:
                        # Get healer predictions
                        predictions, _ = healer_model(images, return_reconstruction=False, return_logits=False)
                        # Apply corrections using the predicted transformations
                        images = healer_model.apply_correction(images, predictions)
                    except Exception as e:
                        self.logger.warning(f"Healer failed to process OOD images: {e}. Skipping healer for this batch.")
                
                # Get predictions
                outputs = self._get_model_outputs(main_model, images, model_type)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def _get_model_outputs(self, model: nn.Module, images: torch.Tensor, model_type: str) -> torch.Tensor:
        """Get model outputs handling different model types"""
        # Handle different model output formats
        if 'ttt' in model_type:
            # TTT models return tuple (class_logits, aux_outputs)
            outputs, _ = model(images)
            return outputs
        elif 'blended' in model_type:
            # Blended models need return_aux=True to return tuple
            outputs, _ = model(images, return_aux=True)
            return outputs
        elif 'healer_resnet18' in model_type:
            # HealerWrapper models can return aux outputs with return_aux=True
            # For now, just get the logits
            return model(images)
        else:
            # Standard models return just logits
            return model(images)
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a formatted table"""
        if not results:
            self.logger.warning("No results to print")
            return
        
        # Get dataset name from config
        dataset_config = self.config.get('dataset', {})
        dataset_name = dataset_config.get('name', 'Unknown').upper()
        if dataset_name == 'TINYIMAGENET':
            dataset_name = 'TinyImagenet200'
        elif dataset_name == 'CIFAR10':
            dataset_name = 'CIFAR-10'
        
        # Get severities
        first_result = next(iter(results.values()))
        severities = sorted(first_result['results'].keys())
        
        # Print header
        print("\n" + "="*141)
        print(f"🏆 COMPREHENSIVE RESULTS - {dataset_name}")
        print("="*141)
        
        # Print table header for continuous robustness
        header = f"{'Model Combination':<35} {'Description':<50}"
        header += f" {'Clean':>8}"
        for sev in severities[1:]:
            header += f" {f'S{sev}':>8}"
        
        print(header)
        print("-" * 141)
        
        # Sort by clean accuracy
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['results'].get(0.0, 0),
            reverse=True
        )
        
        # Print continuous robustness results
        for name, data in sorted_results:
            row = f"{name:<35} {data['description']:<50}"
            for sev in severities:
                acc = data['results'].get(sev, 0)
                row += f" {acc:>8.4f}"
            print(row)
        
        print("=" * 141)
        
        # Print analysis section
        print("\n" + "="*141)
        print("📊 ANALYSIS")
        print("="*141)
        
        # Find best clean data performance
        best_clean = max(sorted_results, key=lambda x: x[1]['results'].get(0.0, 0))
        print(f"🥇 Best Clean Data Performance: {best_clean[0]} ({best_clean[1]['results'][0.0]:.4f})")
        
        # Find most robust model (smallest drop from clean to worst severity)
        most_robust = None
        smallest_drop = float('inf')
        
        for name, data in sorted_results:
            clean_acc = data['results'].get(0.0, 0)
            worst_acc = data['results'].get(max(severities), 0)
            drop = clean_acc - worst_acc
            
            if drop < smallest_drop and clean_acc > 0.1:  # Ignore models with very low clean accuracy
                smallest_drop = drop
                most_robust = (name, data, drop)
        
        if most_robust:
            print(f"🛡️  Most Transform Robust: {most_robust[0]} ({most_robust[2]:.1%} drop)")
        
        # Find best continuous robustness performance (separate from OOD)
        # This will be analyzed later in the OOD section
        
        # Print transformation robustness summary
        print("\n" + "="*141)
        print("📊 TRANSFORMATION ROBUSTNESS SUMMARY")
        print("="*141)
        
        # Print header for continuous robustness summary
        header = f"{'Model':<35}"
        for sev in severities:
            if sev == 0.0:
                header += f" {'Sev 0.0':>10}"
            else:
                header += f" {f'Sev {sev}':>10}"
        header += f" {'Avg Drop':>10}"
        print(header)
        print("-" * 141)
        
        # Print robustness data
        for name, data in sorted_results:
            row = f"{name:<35}"
            clean_acc = data['results'].get(0.0, 0)
            
            for sev in severities:
                acc = data['results'].get(sev, 0)
                row += f" {acc:>10.4f}"
            
            # Calculate average drop
            if clean_acc > 0:
                drops = [(clean_acc - data['results'].get(sev, 0)) / clean_acc 
                        for sev in severities[1:]]
                avg_drop = sum(drops) / len(drops) if drops else 0
                row += f" {avg_drop:>10.4f}"
            else:
                row += f" {'N/A':>10}"
            
            print(row)
        
        # Print healer evaluation if applicable
        healer_results = [(name, data) for name, data in sorted_results 
                         if 'Healer' in name and '+' in name]
        
        if healer_results:
            print("\n" + "="*141)
            print("🔍 HEALER GUIDANCE EVALUATION")
            print("="*141)
            
            for healer_name, healer_data in healer_results:
                # Find corresponding non-healer model
                base_model_name = healer_name.replace('Healer+', '').replace('_Robust', '')
                base_results = None
                
                for name, data in sorted_results:
                    if name == base_model_name or name == base_model_name + '_Robust':
                        base_results = data
                        break
                
                if base_results:
                    print(f"\n🔍 Evaluating {healer_name}...")
                    for sev in severities[1:]:
                        base_acc = base_results['results'].get(sev, 0)
                        healer_acc = healer_data['results'].get(sev, 0)
                        improvement = healer_acc - base_acc
                        
                        print(f"    Severity {sev}: Original: {base_acc:.4f}, "
                              f"Healed: {healer_acc:.4f}, Improvement: {improvement:+.4f}")
        
        # Print transformation prediction accuracy section
        models_with_transform_pred = [(name, data) for name, data in sorted_results 
                                      if data.get('transform_accuracies')]
        
        if models_with_transform_pred:
            print("\n" + "="*141)
            print("🎯 TRANSFORMATION PREDICTION ACCURACY")
            print("="*141)
            
            # Print header
            header = f"{'Model':<35}"
            for sev in severities[1:]:  # Skip 0.0 as no transforms on clean data
                header += f" {f'Sev {sev}':>10}"
            header += f" {'Average':>10}"
            print(header)
            print("-"*141)
            
            # Print transform prediction accuracies
            for name, data in models_with_transform_pred:
                row = f"{name:<35}"
                transform_accs = []
                
                for sev in severities[1:]:
                    if sev in data['transform_accuracies']:
                        metrics = data['transform_accuracies'][sev]
                        # Handle both old format (float) and new format (dict)
                        if isinstance(metrics, dict):
                            acc = metrics.get('overall', 0)
                        else:
                            acc = metrics
                        row += f" {acc:>10.4f}"
                        transform_accs.append(acc)
                    else:
                        row += f" {'N/A':>10}"
                
                # Calculate average
                if transform_accs:
                    avg_acc = sum(transform_accs) / len(transform_accs)
                    row += f" {avg_acc:>10.4f}"
                else:
                    row += f" {'N/A':>10}"
                
                print(row)
            
            # Print detailed per-transform-type accuracy
            print("\n" + "="*141)
            print("📊 DETAILED TRANSFORM TYPE PREDICTION ACCURACY")
            print("="*141)
            
            for name, data in models_with_transform_pred:
                has_detailed = False
                for sev in severities[1:]:
                    if sev in data['transform_accuracies'] and isinstance(data['transform_accuracies'][sev], dict):
                        if 'per_type' in data['transform_accuracies'][sev] and data['transform_accuracies'][sev]['per_type']:
                            has_detailed = True
                            break
                
                if has_detailed:
                    print(f"\n{name}:")
                    print("-" * 80)
                    
                    # Collect all transform types
                    all_types = set()
                    for sev in severities[1:]:
                        if sev in data['transform_accuracies'] and isinstance(data['transform_accuracies'][sev], dict):
                            if 'per_type' in data['transform_accuracies'][sev]:
                                all_types.update(data['transform_accuracies'][sev]['per_type'].keys())
                    
                    # Print header
                    header = f"  {'Transform Type':<20}"
                    for sev in severities[1:]:
                        header += f" {f'Sev {sev}':>12}"
                    print(header)
                    print("  " + "-" * 78)
                    
                    # Print per-type accuracies
                    for transform_type in sorted(all_types):
                        row = f"  {transform_type:<20}"
                        for sev in severities[1:]:
                            if (sev in data['transform_accuracies'] and 
                                isinstance(data['transform_accuracies'][sev], dict) and
                                'per_type' in data['transform_accuracies'][sev] and
                                transform_type in data['transform_accuracies'][sev]['per_type']):
                                acc = data['transform_accuracies'][sev]['per_type'][transform_type]
                                row += f" {acc:>12.4f}"
                            else:
                                row += f" {'N/A':>12}"
                        print(row)
            
            # Print parameter prediction accuracy
            print("\n" + "="*141)
            print("📏 PARAMETER PREDICTION ACCURACY (Mean Absolute Error)")
            print("="*141)
            
            for name, data in models_with_transform_pred:
                has_param_data = False
                for sev in severities[1:]:
                    if (sev in data['transform_accuracies'] and 
                        isinstance(data['transform_accuracies'][sev], dict) and
                        'param_mae' in data['transform_accuracies'][sev] and 
                        data['transform_accuracies'][sev]['param_mae']):
                        has_param_data = True
                        break
                
                if has_param_data:
                    print(f"\n{name}:")
                    print("-" * 80)
                    
                    # Collect all parameter types
                    all_params = set()
                    for sev in severities[1:]:
                        if (sev in data['transform_accuracies'] and 
                            isinstance(data['transform_accuracies'][sev], dict) and
                            'param_mae' in data['transform_accuracies'][sev]):
                            all_params.update(data['transform_accuracies'][sev]['param_mae'].keys())
                    
                    # Print header
                    header = f"  {'Parameter':<20}"
                    for sev in severities[1:]:
                        header += f" {f'Sev {sev}':>12}"
                    header += f" {'Average':>12}"
                    print(header)
                    print("  " + "-" * 90)
                    
                    # Print MAE for each parameter
                    for param in sorted(all_params):
                        row = f"  {param:<20}"
                        param_maes = []
                        
                        for sev in severities[1:]:
                            if (sev in data['transform_accuracies'] and 
                                isinstance(data['transform_accuracies'][sev], dict) and
                                'param_mae' in data['transform_accuracies'][sev] and
                                param in data['transform_accuracies'][sev]['param_mae']):
                                mae = data['transform_accuracies'][sev]['param_mae'][param]
                                row += f" {mae:>12.4f}"
                                param_maes.append(mae)
                            else:
                                row += f" {'N/A':>12}"
                        
                        # Calculate average
                        if param_maes:
                            avg_mae = sum(param_maes) / len(param_maes)
                            row += f" {avg_mae:>12.4f}"
                        else:
                            row += f" {'N/A':>12}"
                        
                        print(row)
        
        # Add separate OOD evaluation section
        self._print_ood_evaluation_section(sorted_results)
        
        print("\n" + "="*141)
    
    def _print_ood_evaluation_section(self, sorted_results):
        """Print separate OOD evaluation section with funky transforms"""
        # Check if we have OOD results (including 0.0 values)
        has_ood = any('ood_results' in data and data['ood_results'] and 'funky_ood' in data['ood_results'] for _, data in sorted_results)
        
        if not has_ood:
            return
        
        print("\n" + "="*141)
        print("🚀 OUT-OF-DISTRIBUTION (FUNKY TRANSFORMS) EVALUATION")
        print("="*141)
        print("This section evaluates model performance on extreme, funky transformations")
        print("including color inversion, pixelation, extreme blur, masking, etc.")
        print("-"*141)
        
        # Print OOD header
        header = f"{'Model Combination':<35} {'Description':<50} {'Funky OOD':>12}"
        print(header)
        print("-" * 141)
        
        # Sort by OOD performance
        ood_sorted_results = sorted(
            sorted_results,
            key=lambda x: x[1].get('ood_results', {}).get('funky_ood', 0),
            reverse=True
        )
        
        # Print OOD results
        for name, data in ood_sorted_results:
            ood_acc = data.get('ood_results', {}).get('funky_ood', 0)
            row = f"{name:<35} {data['description']:<50} {ood_acc:>12.4f}"
            print(row)
        
        print("=" * 141)
        
        # Print OOD analysis
        print("\n📊 OOD ANALYSIS")
        print("-" * 50)
        
        # Best OOD performance
        best_ood = ood_sorted_results[0] if ood_sorted_results else None
        if best_ood:
            best_ood_acc = best_ood[1].get('ood_results', {}).get('funky_ood', 0)
            print(f"🥇 Best Funky OOD Performance: {best_ood[0]} ({best_ood_acc:.4f})")
        
        # Compare OOD vs Clean performance
        print(f"\n🔍 OOD vs Clean Performance Gap:")
        for name, data in ood_sorted_results[:5]:  # Top 5 models
            clean_acc = data['results'].get(0.0, 0)
            ood_acc = data.get('ood_results', {}).get('funky_ood', 0)
            gap = clean_acc - ood_acc
            gap_percent = (gap / clean_acc * 100) if clean_acc > 0 else 0
            print(f"    {name}: Clean {clean_acc:.4f} → OOD {ood_acc:.4f} (Gap: {gap:.4f}, {gap_percent:.1f}%)")
        
        # Rank OOD robustness
        print(f"\n🏆 OOD Robustness Ranking:")
        for i, (name, data) in enumerate(ood_sorted_results[:5], 1):
            ood_acc = data.get('ood_results', {}).get('funky_ood', 0)
            print(f"    {i}. {name}: {ood_acc:.4f}")
        
        print("\n" + "="*141)
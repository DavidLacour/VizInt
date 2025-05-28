"""
Configuration loader module for managing experiment configurations
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigLoader:
    """Handles loading and accessing configuration settings"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to config file. If None, uses default config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Process environment variables
        config = self._process_env_vars(config)
        
        return config
    
    def _process_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace environment variable references in config"""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith("$"):
                    env_var = value[1:]
                    config[key] = os.environ.get(env_var, value)
                elif isinstance(value, dict):
                    config[key] = self._process_env_vars(value)
        return config
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=log_level,
            format=log_format
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key
        
        Args:
            key: Dot-separated configuration key (e.g., 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset-specific configuration"""
        # For new config format, dataset config is at root level
        dataset_config = self.config.get('dataset', {})
        
        # Handle both old and new config formats
        if dataset_config and dataset_config.get('name', '').lower() == dataset_name.lower():
            return dataset_config
        
        # Fallback to old format
        datasets = self.config.get('datasets', {})
        if dataset_name in datasets:
            return datasets[dataset_name]
            
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        models = self.config.get('models', {})
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        return models[model_name]
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.config.get('evaluation', {})
    
    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug mode configuration"""
        return self.config.get('debug', {})
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get('debug.enabled', False)
    
    def get_batch_size(self, mode: str = 'training') -> int:
        """
        Get batch size based on mode and debug settings
        
        Args:
            mode: 'training' or 'evaluation'
            
        Returns:
            Batch size
        """
        if self.is_debug_mode():
            return self.get('debug.batch_size', 3)
        
        if mode == 'training':
            return self.get('training.batch_size', 128)
        else:
            return self.get('evaluation.batch_size', 128)
    
    def get_num_epochs(self) -> int:
        """Get number of epochs based on debug mode"""
        if self.is_debug_mode():
            return self.get('debug.epochs', 2)
        return self.get('training.epochs', 100)
    
    def get_checkpoint_dir(self, dataset_name: str, use_debug_dir: bool = True) -> Path:
        """Get checkpoint directory for a dataset
        
        Args:
            dataset_name: Name of the dataset
            use_debug_dir: Whether to use debug checkpoint directory in debug mode
                          (default: True for training, can be False for evaluation)
        
        Returns:
            Path to checkpoint directory
        """
        paths = self.config.get('paths', {})
        
        # Determine the checkpoint directory path
        checkpoint_path = None
        
        # Handle new config format
        checkpoint_dir = paths.get('checkpoint_dir')
        if checkpoint_dir and isinstance(checkpoint_dir, str):
            checkpoint_path = Path(checkpoint_dir)
        
        # Handle old config format with per-dataset dirs
        elif isinstance(checkpoint_dir, dict) and dataset_name in checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir[dataset_name])
        
        # Check debug mode only if use_debug_dir is True
        if self.is_debug_mode() and use_debug_dir:
            debug_dir = self.get('debug.checkpoint_dir')
            if debug_dir:
                checkpoint_path = Path(debug_dir)
        
        # Default to current directory if no path found
        if checkpoint_path is None:
            checkpoint_path = Path('./checkpoints') / dataset_name
        
        # Create directory if it doesn't exist
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        return checkpoint_path
    
    def get_device(self) -> str:
        """Get device configuration"""
        import torch
        
        device_config = self.get('general.device', 'auto')
        
        if device_config == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return device_config
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
    
    def save(self, path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        return f"ConfigLoader(config_path='{self.config_path}')"
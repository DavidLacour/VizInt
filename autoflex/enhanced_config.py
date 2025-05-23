"""
Enhanced training configuration system with model-specific settings
and automatic hyperparameter optimization suggestions.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import torch

@dataclass
class ModelConfig:
    """Configuration for a specific model type"""
    epochs: int
    learning_rate: float
    batch_size: int = None  # None = auto-detect
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    patience: int = 3
    dropout: float = 0.1
    optimizer: str = "adamw"
    scheduler: str = "cosine_warmup"
    
    # Model-specific parameters
    severity: float = 1.0  # For healer models
    adaptation_steps: int = 10  # For TTT models
    adaptation_lr: float = 1e-4  # For TTT models

@dataclass
class BackboneConfig:
    """Configuration for backbone-specific settings"""
    preferred_batch_size: int = None
    memory_efficient: bool = False
    compile_model: bool = False  # PyTorch 2.0 compilation
    gradient_checkpointing: bool = False

class TrainingConfigManager:
    """Manages training configurations for different model and backbone combinations"""
    
    def __init__(self, config_file: str = "training_configs.json"):
        self.config_file = Path(config_file)
        self.configs = self._load_configs()
    
    def _get_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default configurations for all model types"""
        return {
            "model_configs": {
                "classification": asdict(ModelConfig(
                    epochs=50,
                    learning_rate=1e-4,
                    batch_size=64,  # Fixed batch size
                    weight_decay=0.05,
                    warmup_steps=1000,
                    patience=5,  # Increased patience for classification
                    dropout=0.1
                )),
                "healer": asdict(ModelConfig(
                    epochs=15,
                    learning_rate=5e-5,
                    batch_size=32,  # Fixed batch size for healer
                    weight_decay=0.01,
                    warmup_steps=500,
                    patience=3,  # Lower patience for healer
                    dropout=0.1,
                    severity=1.0
                )),
                "ttt": asdict(ModelConfig(
                    epochs=10,
                    learning_rate=1e-4,
                    batch_size=32,  # Fixed batch size
                    weight_decay=0.01,
                    warmup_steps=200,
                    patience=3,
                    dropout=0.1,
                    adaptation_steps=5,
                    adaptation_lr=1e-4
                )),
                "blended_ttt": asdict(ModelConfig(
                    epochs=20,
                    learning_rate=1e-4,
                    batch_size=64,  # Fixed batch size
                    weight_decay=0.01,
                    warmup_steps=800,
                    patience=4,
                    dropout=0.1,
                    adaptation_steps=3,
                    adaptation_lr=5e-5
                ))
            },
            "backbone_configs": {
                "vit_small": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,  # Disable compilation for now
                    gradient_checkpointing=False
                )),
                "vit_base": asdict(BackboneConfig(
                    preferred_batch_size=32,
                    memory_efficient=True,
                    compile_model=False,
                    gradient_checkpointing=True
                )),
                "resnet18": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False
                )),
                "resnet50": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False
                )),
                "vgg16": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False
                )),
                "deit_small": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False
                )),
                "swin_small": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False
                ))
            },
            "hardware_configs": {
                "gpu_memory_thresholds": {
                    "low": 4,    # GB
                    "medium": 8,  # GB
                    "high": 16    # GB
                },
                "batch_size_multipliers": {
                    "low": 0.5,
                    "medium": 1.0,
                    "high": 1.5
                }
            },
            "early_stopping_configs": {
                "classification": {
                    "patience": 5,
                    "min_delta": 1e-4,
                    "monitor": "val_acc",
                    "mode": "max",
                    "restore_best_weights": True,
                    "cleanup_checkpoints": True
                },
                "healer": {
                    "patience": 3,
                    "min_delta": 1e-5,
                    "monitor": "val_loss",
                    "mode": "min",
                    "restore_best_weights": True,
                    "cleanup_checkpoints": True
                },
                "ttt": {
                    "patience": 3,
                    "min_delta": 1e-4,
                    "monitor": "val_loss",
                    "mode": "min",
                    "restore_best_weights": True,
                    "cleanup_checkpoints": True
                },
                "blended_ttt": {
                    "patience": 4,
                    "min_delta": 1e-4,
                    "monitor": "val_acc",
                    "mode": "max",
                    "restore_best_weights": True,
                    "cleanup_checkpoints": True
                }
            }
        }
    
    def _load_configs(self) -> Dict[str, Any]:
        """Load configurations from file or create defaults"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            configs = self._get_default_configs()
            self._save_configs(configs)
            return configs
    
    def _save_configs(self, configs: Dict[str, Any]):
        """Save configurations to file"""
        with open(self.config_file, 'w') as f:
            json.dump(configs, f, indent=2)
    
    def get_model_config(self, model_type: str, backbone_name: str = None) -> Dict[str, Any]:
        """Get configuration for a specific model type and backbone"""
        # Start with base model config
        base_config = self.configs["model_configs"].get(model_type, {})
        
        # Apply backbone-specific modifications if available
        if backbone_name:
            backbone_config = self.configs["backbone_configs"].get(backbone_name, {})
            
            # Use backbone's preferred batch size if model doesn't specify one
            if base_config.get("batch_size") is None and backbone_config.get("preferred_batch_size"):
                base_config = base_config.copy()
                base_config["batch_size"] = backbone_config["preferred_batch_size"]
        
        return base_config
    
    def get_backbone_config(self, backbone_name: str) -> Dict[str, Any]:
        """Get backbone-specific configuration"""
        return self.configs["backbone_configs"].get(backbone_name, {})
    
    def adjust_config_for_hardware(self, config: Dict[str, Any], backbone_name: str) -> Dict[str, Any]:
        """Adjust configuration based on available hardware"""
        if not torch.cuda.is_available():
            # CPU adjustments
            config = config.copy()
            config["batch_size"] = min(config.get("batch_size", 32), 16)
            config["compile_model"] = False
            return config
        
        # Get GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Determine memory tier
        hardware_configs = self.configs["hardware_configs"]
        thresholds = hardware_configs["gpu_memory_thresholds"]
        multipliers = hardware_configs["batch_size_multipliers"]
        
        if gpu_memory_gb < thresholds["low"]:
            memory_tier = "low"
        elif gpu_memory_gb < thresholds["medium"]:
            memory_tier = "medium"
        else:
            memory_tier = "high"
        
        # Adjust batch size
        config = config.copy()
        if config.get("batch_size"):
            original_batch_size = config["batch_size"]
            multiplier = multipliers[memory_tier]
            config["batch_size"] = max(1, int(original_batch_size * multiplier))
        
        # Adjust other settings based on memory
        backbone_config = self.get_backbone_config(backbone_name)
        if memory_tier == "low":
            config["gradient_checkpointing"] = True
            config["memory_efficient"] = True
        else:
            config["gradient_checkpointing"] = backbone_config.get("gradient_checkpointing", False)
            config["memory_efficient"] = backbone_config.get("memory_efficient", False)
        
        return config
    
    def get_training_config(self, model_type: str, backbone_name: str) -> Dict[str, Any]:
        """Get complete training configuration for model type and backbone"""
        # Get base configuration
        config = self.get_model_config(model_type, backbone_name)
        
        # Add backbone-specific settings
        backbone_config = self.get_backbone_config(backbone_name)
        config.update({
            "compile_model": backbone_config.get("compile_model", False),
            "gradient_checkpointing": backbone_config.get("gradient_checkpointing", False),
            "memory_efficient": backbone_config.get("memory_efficient", False)
        })
        
        # Adjust for hardware
        config = self.adjust_config_for_hardware(config, backbone_name)
        
        return config
    
    def update_config(self, model_type: str, backbone_name: str = None, **kwargs):
        """Update configuration for a model type or specific backbone"""
        if backbone_name:
            # Update backbone-specific config
            if backbone_name not in self.configs["backbone_configs"]:
                self.configs["backbone_configs"][backbone_name] = {}
            self.configs["backbone_configs"][backbone_name].update(kwargs)
        else:
            # Update model type config
            if model_type not in self.configs["model_configs"]:
                self.configs["model_configs"][model_type] = {}
            self.configs["model_configs"][model_type].update(kwargs)
        
        self._save_configs(self.configs)
    
    def print_config_summary(self):
        """Print a summary of all configurations"""
        print("📋 Training Configuration Summary")
        print("=" * 50)
        
        print("\n🤖 Model Configurations:")
        for model_type, config in self.configs["model_configs"].items():
            print(f"  {model_type}:")
            for key, value in config.items():
                print(f"    {key}: {value}")
        
        print("\n🏗️  Backbone Configurations:")
        for backbone_name, config in self.configs["backbone_configs"].items():
            print(f"  {backbone_name}:")
            for key, value in config.items():
                print(f"    {key}: {value}")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\n💾 Hardware: GPU with {gpu_memory:.1f} GB memory")
        else:
            print(f"\n💾 Hardware: CPU only")

# Global config manager instance
config_manager = TrainingConfigManager()

def get_training_config(model_type: str, backbone_name: str) -> Dict[str, Any]:
    """Convenience function to get training configuration"""
    return config_manager.get_training_config(model_type, backbone_name)

def print_recommended_configs():
    """Print recommended configurations for different scenarios"""
    print("🎯 Recommended Configurations")
    print("=" * 50)
    
    recommendations = {
        "Fast Prototyping": {
            "models": ["classification"],
            "backbones": ["resnet18", "vit_small"],
            "epochs": 10,
            "note": "Quick training for initial experiments"
        },
        "Best Accuracy": {
            "models": ["classification", "healer"],
            "backbones": ["vit_base", "swin_small"],
            "epochs": 50,
            "note": "Longer training with larger models"
        },
        "Memory Constrained": {
            "models": ["classification"],
            "backbones": ["resnet18", "vit_small"],
            "epochs": 30,
            "note": "For GPUs with <8GB memory",
            "batch_size": 32
        },
        "Research Comparison": {
            "models": ["classification", "healer", "ttt"],
            "backbones": ["vit_small", "resnet50", "deit_small"],
            "epochs": 30,
            "note": "Compare different approaches"
        }
    }
    
    for scenario, config in recommendations.items():
        print(f"\n📊 {scenario}:")
        print(f"  Models: {', '.join(config['models'])}")
        print(f"  Backbones: {', '.join(config['backbones'])}")
        print(f"  Epochs: {config['epochs']}")
        if 'batch_size' in config:
            print(f"  Batch Size: {config['batch_size']}")
        print(f"  Note: {config['note']}")

if __name__ == "__main__":
    # Example usage
    config_manager.print_config_summary()
    print("\n")
    print_recommended_configs()
    
    # Example of getting specific configuration
    print(f"\n🔍 Example - Classification with ViT Small:")
    config = get_training_config("classification", "vit_small")
    for key, value in config.items():
        print(f"  {key}: {value}")
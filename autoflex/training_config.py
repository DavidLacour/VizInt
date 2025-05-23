"""
Enhanced training configuration system with model-specific settings
and automatic hyperparameter optimization suggestions.
Now supports both pretrained and non-pretrained backbone variants.
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
    is_pretrained: bool = True  # Whether the backbone uses pretrained weights

class TrainingConfigManager:
    """Manages training configurations for different model and backbone combinations"""
    
    def __init__(self, config_file: str = "training_configs_enhanced.json"):
        self.config_file = Path(config_file)
        self.configs = self._load_configs()
    
    def _get_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default configurations for all model types"""
        return {
            "model_configs": {
                "classification": asdict(ModelConfig(
                    epochs=50,
                    learning_rate=1e-4,
                    batch_size=None,
                    weight_decay=0.05,
                    warmup_steps=1000,
                    patience=3,
                    dropout=0.1
                )),
                "healer": asdict(ModelConfig(
                    epochs=15,
                    learning_rate=5e-5,
                    batch_size=50,
                    weight_decay=0.01,
                    warmup_steps=500,
                    patience=3,
                    dropout=0.1,
                    severity=1.0
                )),
                "ttt": asdict(ModelConfig(
                    epochs=10,
                    learning_rate=1e-4,
                    batch_size=32,
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
                    batch_size=None,
                    weight_decay=0.01,
                    warmup_steps=800,
                    patience=3,
                    dropout=0.1,
                    adaptation_steps=3,
                    adaptation_lr=5e-5
                ))
            },
            "backbone_configs": {
                # Custom ViT models (no pretrained versions)
                "vit_small": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,  # Disable compilation for stability
                    gradient_checkpointing=False,
                    is_pretrained=False
                )),
                "vit_base": asdict(BackboneConfig(
                    preferred_batch_size=32,
                    memory_efficient=True,
                    compile_model=False,
                    gradient_checkpointing=True,
                    is_pretrained=False
                )),
                
                # ResNet models - pretrained versions
                "resnet18_pretrained": asdict(BackboneConfig(
                    preferred_batch_size=128,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False,
                    is_pretrained=True
                )),
                "resnet50_pretrained": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False,
                    is_pretrained=True
                )),
                
                # ResNet models - scratch versions (may need different training configs)
                "resnet18_scratch": asdict(BackboneConfig(
                    preferred_batch_size=128,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False,
                    is_pretrained=False
                )),
                "resnet50_scratch": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False,
                    is_pretrained=False
                )),
                
                # VGG models - pretrained versions
                "vgg16_pretrained": asdict(BackboneConfig(
                    preferred_batch_size=32,
                    memory_efficient=True,
                    compile_model=False,
                    gradient_checkpointing=True,
                    is_pretrained=True
                )),
                
                # VGG models - scratch versions
                "vgg16_scratch": asdict(BackboneConfig(
                    preferred_batch_size=32,
                    memory_efficient=True,
                    compile_model=False,
                    gradient_checkpointing=True,
                    is_pretrained=False
                )),
                
                # Timm models - pretrained versions
                "deit_small_pretrained": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False,
                    is_pretrained=True
                )),
                "swin_small_pretrained": asdict(BackboneConfig(
                    preferred_batch_size=48,
                    memory_efficient=True,
                    compile_model=False,
                    gradient_checkpointing=True,
                    is_pretrained=True
                )),
                
                # Timm models - scratch versions
                "deit_small_scratch": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False,
                    is_pretrained=False
                )),
                "swin_small_scratch": asdict(BackboneConfig(
                    preferred_batch_size=48,
                    memory_efficient=True,
                    compile_model=False,
                    gradient_checkpointing=True,
                    is_pretrained=False
                )),
                
                # Backward compatibility - original names (pretrained versions)
                "resnet18": asdict(BackboneConfig(
                    preferred_batch_size=128,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False,
                    is_pretrained=True
                )),
                "resnet50": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False,
                    is_pretrained=True
                )),
                "vgg16": asdict(BackboneConfig(
                    preferred_batch_size=32,
                    memory_efficient=True,
                    compile_model=False,
                    gradient_checkpointing=True,
                    is_pretrained=True
                )),
                "deit_small": asdict(BackboneConfig(
                    preferred_batch_size=64,
                    memory_efficient=False,
                    compile_model=False,
                    gradient_checkpointing=False,
                    is_pretrained=True
                )),
                "swin_small": asdict(BackboneConfig(
                    preferred_batch_size=48,
                    memory_efficient=True,
                    compile_model=False,
                    gradient_checkpointing=True,
                    is_pretrained=True
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
            "pretrained_adjustments": {
                # Different training strategies for pretrained vs scratch models
                "scratch_models": {
                    "learning_rate_multiplier": 1.5,  # Higher LR for training from scratch
                    "epochs_multiplier": 1.2,         # More epochs for scratch training
                    "warmup_steps_multiplier": 1.5,   # More warmup for scratch training
                    "weight_decay_multiplier": 0.8    # Less weight decay for scratch
                },
                "pretrained_models": {
                    "learning_rate_multiplier": 1.0,  # Standard LR for fine-tuning
                    "epochs_multiplier": 1.0,         # Standard epochs
                    "warmup_steps_multiplier": 1.0,   # Standard warmup
                    "weight_decay_multiplier": 1.0    # Standard weight decay
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
    
    def _is_scratch_model(self, backbone_name: str) -> bool:
        """Determine if a backbone is trained from scratch"""
        return ('scratch' in backbone_name or 
                backbone_name in ['vit_small', 'vit_base'] or
                self.configs["backbone_configs"].get(backbone_name, {}).get("is_pretrained", True) == False)
    
    def get_model_config(self, model_type: str, backbone_name: str = None) -> Dict[str, Any]:
        """Get configuration for a specific model type and backbone"""
        # Start with base model config
        base_config = self.configs["model_configs"].get(model_type, {}).copy()
        
        # Apply backbone-specific modifications if available
        if backbone_name:
            backbone_config = self.configs["backbone_configs"].get(backbone_name, {})
            
            # Use backbone's preferred batch size if model doesn't specify one
            if base_config.get("batch_size") is None and backbone_config.get("preferred_batch_size"):
                base_config["batch_size"] = backbone_config["preferred_batch_size"]
            
            # Apply pretrained vs scratch adjustments
            if self._is_scratch_model(backbone_name):
                adjustments = self.configs["pretrained_adjustments"]["scratch_models"]
                print(f"🔨 Applying scratch model adjustments for {backbone_name}")
            else:
                adjustments = self.configs["pretrained_adjustments"]["pretrained_models"]
                print(f"✅ Applying pretrained model adjustments for {backbone_name}")
            
            # Apply multipliers
            base_config["learning_rate"] = base_config.get("learning_rate", 1e-4) * adjustments["learning_rate_multiplier"]
            base_config["epochs"] = int(base_config.get("epochs", 50) * adjustments["epochs_multiplier"])
            base_config["warmup_steps"] = int(base_config.get("warmup_steps", 1000) * adjustments["warmup_steps_multiplier"])
            base_config["weight_decay"] = base_config.get("weight_decay", 0.01) * adjustments["weight_decay_multiplier"]
        
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
            "memory_efficient": backbone_config.get("memory_efficient", False),
            "is_pretrained": backbone_config.get("is_pretrained", True)
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
        print("📋 Enhanced Training Configuration Summary")
        print("=" * 60)
        
        print("\n🤖 Model Configurations:")
        for model_type, config in self.configs["model_configs"].items():
            print(f"  {model_type}:")
            for key, value in config.items():
                print(f"    {key}: {value}")
        
        print("\n🏗️  Backbone Configurations:")
        pretrained_count = 0
        scratch_count = 0
        custom_count = 0
        
        for backbone_name, config in self.configs["backbone_configs"].items():
            is_pretrained = config.get("is_pretrained", True)
            if backbone_name in ['vit_small', 'vit_base']:
                backbone_type = "🔧 CUSTOM"
                custom_count += 1
            elif is_pretrained:
                backbone_type = "✅ PRETRAINED"
                pretrained_count += 1
            else:
                backbone_type = "🔨 SCRATCH"
                scratch_count += 1
                
            print(f"  {backbone_name} ({backbone_type}):")
            for key, value in config.items():
                print(f"    {key}: {value}")
        
        print(f"\n📊 Summary:")
        print(f"  Custom models: {custom_count}")
        print(f"  Pretrained models: {pretrained_count}")
        print(f"  Scratch models: {scratch_count}")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\n💾 Hardware: GPU with {gpu_memory:.1f} GB memory")
        else:
            print(f"\n💾 Hardware: CPU only")
    
    def compare_pretrained_vs_scratch(self, model_type: str = "classification"):
        """Compare configurations for pretrained vs scratch models"""
        print(f"🔍 Pretrained vs Scratch Comparison for {model_type}")
        print("=" * 60)
        
        # Find matching pairs
        pairs = []
        scratch_models = [name for name in self.configs["backbone_configs"].keys() if 'scratch' in name]
        for scratch in scratch_models:
            pretrained = scratch.replace('_scratch', '_pretrained')
            if pretrained in self.configs["backbone_configs"]:
                pairs.append((pretrained, scratch))
        
        for pretrained, scratch in pairs:
            print(f"\n📊 {pretrained.replace('_pretrained', '').upper()}:")
            
            pretrained_config = self.get_training_config(model_type, pretrained)
            scratch_config = self.get_training_config(model_type, scratch)
            
            # Compare key metrics
            metrics = ['learning_rate', 'epochs', 'warmup_steps', 'weight_decay']
            for metric in metrics:
                p_val = pretrained_config.get(metric, 'N/A')
                s_val = scratch_config.get(metric, 'N/A')
                print(f"  {metric:<15}: Pretrained={p_val:<8} | Scratch={s_val}")

# Global config manager instance
config_manager = TrainingConfigManager()

def get_training_config(model_type: str, backbone_name: str) -> Dict[str, Any]:
    """Convenience function to get training configuration"""
    return config_manager.get_training_config(model_type, backbone_name)

def print_recommended_configs():
    """Print recommended configurations for different scenarios"""
    print("🎯 Enhanced Recommended Configurations")
    print("=" * 60)
    
    recommendations = {
        "Fast Prototyping": {
            "models": ["classification"],
            "backbones": ["resnet18_pretrained", "vit_small"],
            "epochs": 10,
            "note": "Quick training for initial experiments"
        },
        "Pretrained vs Scratch Comparison": {
            "models": ["classification"],
            "backbones": ["resnet50_pretrained", "resnet50_scratch", "deit_small_pretrained", "deit_small_scratch"],
            "epochs": 30,
            "note": "Compare pretrained initialization vs training from scratch"
        },
        "Best Accuracy (Pretrained)": {
            "models": ["classification", "healer"],
            "backbones": ["swin_small_pretrained", "deit_small_pretrained"],
            "epochs": 50,
            "note": "Leverage pretrained weights for best performance"
        },
        "Best Accuracy (Scratch)": {
            "models": ["classification", "healer"],
            "backbones": ["swin_small_scratch", "deit_small_scratch"],
            "epochs": 60,
            "note": "Train large models from scratch"
        },
        "Memory Constrained": {
            "models": ["classification"],
            "backbones": ["resnet18_pretrained", "resnet18_scratch"],
            "epochs": 30,
            "note": "For GPUs with <8GB memory",
            "batch_size": 32
        },
        "Research Comparison": {
            "models": ["classification", "healer", "ttt"],
            "backbones": ["vit_small", "resnet50_pretrained", "resnet50_scratch"],
            "epochs": 30,
            "note": "Compare different approaches and initialization strategies"
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
    print(f"\n🔍 Example Configurations:")
    
    # Compare pretrained vs scratch
    examples = [
        ("classification", "resnet50_pretrained"),
        ("classification", "resnet50_scratch"),
        ("classification", "vit_small"),
    ]
    
    for model_type, backbone in examples:
        config = get_training_config(model_type, backbone)
        is_pretrained = "✅ PRETRAINED" if config.get("is_pretrained", True) else "🔨 SCRATCH"
        if backbone in ['vit_small', 'vit_base']:
            is_pretrained = "🔧 CUSTOM"
        
        print(f"\n{backbone} ({is_pretrained}):")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print(f"\n📊 Pretrained vs Scratch Training Differences:")
    config_manager.compare_pretrained_vs_scratch("classification")

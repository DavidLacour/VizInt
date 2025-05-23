#!/usr/bin/env python3
"""
Setup script for debug environment.
Checks dependencies and provides installation instructions for conda/pyenv.
"""

import sys
import subprocess
import importlib
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires Python 3.8+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        else:
            version = "unknown"
        print(f"✅ {package_name} ({version})")
        return True
    except ImportError:
        print(f"❌ {package_name} (not installed)")
        return False

def get_conda_install_command():
    """Get conda installation command for all dependencies"""
    packages = [
        "pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia",
        "numpy scipy scikit-learn",
        "pillow",
        "tqdm",
        "wandb -c conda-forge"
    ]
    
    commands = []
    for package in packages:
        commands.append(f"conda install {package}")
    
    return commands

def get_pip_install_command():
    """Get pip installation command for all dependencies"""
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "numpy scipy scikit-learn",
        "Pillow",
        "tqdm",
        "wandb"
    ]
    
    commands = []
    for package in packages:
        commands.append(f"pip install {package}")
    
    return commands

def check_dataset_path(dataset_path="../tiny-imagenet-200"):
    """Check if dataset exists"""
    path = Path(dataset_path)
    if path.exists():
        # Check for key directories
        train_dir = path / "train"
        val_dir = path / "val"
        if train_dir.exists() and val_dir.exists():
            print(f"✅ Dataset found at {path.absolute()}")
            print(f"   📁 Train classes: {len(list(train_dir.glob('*')))}")
            print(f"   📁 Val images: {len(list(val_dir.glob('**/*.JPEG')))}")
            return True
        else:
            print(f"❌ Dataset directory exists but missing train/val folders")
            return False
    else:
        print(f"❌ Dataset not found at {path.absolute()}")
        return False

def main():
    print("🔧 Debug Environment Setup Check")
    print("=" * 50)
    
    # Check Python version
    print("\n📋 Python Environment:")
    python_ok = check_python_version()
    print(f"   Platform: {platform.system()} {platform.machine()}")
    print(f"   Environment: {sys.prefix}")
    
    # Check required packages
    print("\n📦 Required Packages:")
    required_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("Pillow", "PIL"),
        ("tqdm", "tqdm"),
        ("wandb", "wandb")
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    # Check GPU availability
    print("\n🖥️  Hardware:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ CUDA GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"   Available GPUs: {gpu_count}")
        else:
            print("⚠️  No CUDA GPU detected (will use CPU)")
    except ImportError:
        print("❌ Cannot check GPU (torch not installed)")
    
    # Check dataset
    print("\n📂 Dataset:")
    dataset_ok = check_dataset_path()
    
    # Provide installation instructions if needed
    if missing_packages:
        print(f"\n📥 Installation Instructions:")
        print("=" * 50)
        
        print("\n🐍 For Conda environments:")
        conda_commands = get_conda_install_command()
        for i, cmd in enumerate(conda_commands, 1):
            print(f"  {i}. {cmd}")
        
        print("\n📦 For pip/pyenv environments:")
        pip_commands = get_pip_install_command()
        for i, cmd in enumerate(pip_commands, 1):
            print(f"  {i}. {cmd}")
        
        print(f"\n💡 Missing packages: {', '.join(missing_packages)}")
    
    if not dataset_ok:
        print(f"\n📂 Dataset Setup:")
        print("=" * 50)
        print("1. Download tiny-imagenet-200 from:")
        print("   http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        print("2. Extract to parent directory:")
        print("   unzip tiny-imagenet-200.zip -d ../")
        print("3. Or update dataset path in debug_train.py")
    
    # Final status
    print(f"\n🎯 Setup Status:")
    print("=" * 50)
    
    if python_ok and not missing_packages and dataset_ok:
        print("✅ Ready for debug training!")
        print("\n🚀 Quick start:")
        print("   python debug_train.py")
        print("   python debug_train.py --backbone resnet18")
        print("   python debug_train.py --test-all")
    else:
        issues = []
        if not python_ok:
            issues.append("Python version")
        if missing_packages:
            issues.append("missing packages")
        if not dataset_ok:
            issues.append("dataset")
        
        print(f"❌ Issues found: {', '.join(issues)}")
        print("   Please fix the issues above before running debug training.")

if __name__ == "__main__":
    main()
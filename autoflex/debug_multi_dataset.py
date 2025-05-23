#!/usr/bin/env python3
"""
Multi-dataset debug training script.
Tests different datasets with debug mode for quick validation.
"""

import argparse
import torch
import os
import sys
from pathlib import Path
from debug_config import debug_config, setup_debug_environment
from flexible_models import create_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

class MultiDatasetLoader:
    """Handles loading different dataset types"""
    
    @staticmethod
    def detect_dataset_type(dataset_path: str):
        """Auto-detect dataset type based on structure"""
        path = Path(dataset_path)
        
        # Check for tiny-imagenet-200 structure
        if (path / "train").exists() and (path / "val").exists() and (path / "wnids.txt").exists():
            return "tiny_imagenet"
        
        # Check for hybrid_small_imagenetc structure  
        if (path / "dataset_summary.json").exists():
            return "hybrid_imagenetc"
        
        # Check for laionc_small structure
        if (path / "laion_dataset_summary.json").exists():
            return "laionc_small"
        
        # Check for CIFAR-10/100 structure
        if (path / "cifar-10-python.tar.gz").exists():
            return "cifar10"
        if (path / "cifar-100-python.tar.gz").exists():
            return "cifar100"
        
        # Check for Fashion-MNIST structure
        if (path / "FashionMNIST").exists() or any(f.name.startswith("train") and f.name.endswith("ubyte") for f in path.iterdir()):
            return "fashion_mnist"
        
        # Check for Oxford-IIIT Pet Dataset structure
        if (path / "oxford-iiit-pet").exists() or (path / "images").exists() and (path / "annotations").exists():
            return "oxford_pets"
        
        # Check for Food-101 structure
        if (path / "food-101").exists() or ((path / "images").exists() and (path / "meta").exists()):
            return "food101"
        
        # Check for ImageNet validation structure (single folder with images)
        if path.is_dir() and any(f.suffix.lower() in ['.jpeg', '.jpg', '.png'] for f in path.iterdir() if f.is_file()):
            return "imagenet_val"
        
        # Check for directory with subdirectories (class folders)
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if len(subdirs) > 10:  # Likely class-based structure
            return "class_folders"
        
        return "unknown"

class SimpleImageDataset(Dataset):
    """Simple dataset for debug mode with limited samples"""
    
    def __init__(self, image_paths, labels, transform=None, limit=None):
        self.image_paths = image_paths[:limit] if limit else image_paths
        self.labels = labels[:limit] if limit else labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                dummy_image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label

def load_tiny_imagenet(dataset_path: str, is_train: bool = True, limit: int = None, img_size: int = 224):
    """Load Tiny ImageNet dataset"""
    from new_new import TinyImageNetDataset
    
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Force square resize
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Force square resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = TinyImageNetDataset(
        root_dir=dataset_path,
        split='train' if is_train else 'val',
        transform=transform
    )
    
    if limit and len(dataset) > limit:
        from torch.utils.data import Subset
        indices = list(range(min(limit, len(dataset))))
        dataset = Subset(dataset, indices)
    
    return dataset

def load_hybrid_imagenetc(dataset_path: str, is_train: bool = True, limit: int = None, img_size: int = 224):
    """Load hybrid small ImageNet-C dataset"""
    path = Path(dataset_path)
    
    # Load metadata
    if is_train:
        labels_file = path / "train_labels.csv"
        metadata_file = path / "train_metadata.json"
    else:
        labels_file = path / "val_labels.csv"
        metadata_file = path / "val_metadata.json"
    
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    # Read labels
    df = pd.read_csv(labels_file)
    
    # Fix column names if needed
    if 'filename' in df.columns and 'image_path' not in df.columns:
        df['image_path'] = df['filename']
    
    # Create transforms  
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Force square resize
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Force square resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Build image paths and labels
    image_paths = []
    labels = []
    
    # Create label mapping from corruption types
    unique_corruption_types = df['corruption_type'].unique()
    corruption_to_label = {corruption: idx for idx, corruption in enumerate(unique_corruption_types)}
    
    # Check for subdirectories (train/val/test structure)
    split_dir = 'train' if is_train else 'val'
    subdir_path = path / split_dir
    
    for _, row in df.iterrows():
        # Try main directory first, then subdirectory
        image_path = path / row['image_path']
        if not image_path.exists():
            image_path = subdir_path / row['image_path']
        
        if image_path.exists():
            image_paths.append(str(image_path))
            labels.append(corruption_to_label[row['corruption_type']])
    
    print(f"Found {len(image_paths)} images in {dataset_path}")
    
    return SimpleImageDataset(image_paths, labels, transform, limit)

def load_cifar(dataset_path: str, is_cifar100: bool = False, is_train: bool = True, limit: int = None, img_size: int = 224):
    """Load CIFAR-10 or CIFAR-100 dataset"""
    import torchvision.datasets as datasets
    
    # Create transforms for CIFAR (32x32 -> img_size)
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load CIFAR dataset
    if is_cifar100:
        dataset = datasets.CIFAR100(dataset_path, train=is_train, transform=transform, download=False)
    else:
        dataset = datasets.CIFAR10(dataset_path, train=is_train, transform=transform, download=False)
    
    # Apply limit if specified
    if limit and len(dataset) > limit:
        from torch.utils.data import Subset
        indices = list(range(min(limit, len(dataset))))
        dataset = Subset(dataset, indices)
    
    return dataset

def load_fashion_mnist(dataset_path: str, is_train: bool = True, limit: int = None, img_size: int = 224):
    """Load Fashion-MNIST dataset (grayscale -> RGB conversion)"""
    import torchvision.datasets as datasets
    
    # Create transforms for Fashion-MNIST (28x28 grayscale -> img_size RGB)
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load Fashion-MNIST dataset
    dataset = datasets.FashionMNIST(dataset_path, train=is_train, transform=transform, download=False)
    
    # Apply limit if specified
    if limit and len(dataset) > limit:
        from torch.utils.data import Subset
        indices = list(range(min(limit, len(dataset))))
        dataset = Subset(dataset, indices)
    
    return dataset

def load_oxford_pets(dataset_path: str, is_train: bool = True, limit: int = None, img_size: int = 224):
    """Load Oxford-IIIT Pet Dataset (37 pet breeds)"""
    import torchvision.datasets as datasets
    
    # Create transforms for Oxford Pets
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load Oxford Pets dataset
    split = "trainval" if is_train else "test"
    dataset = datasets.OxfordIIITPet(dataset_path, split=split, transform=transform, download=False)
    
    # Apply limit if specified
    if limit and len(dataset) > limit:
        from torch.utils.data import Subset
        indices = list(range(min(limit, len(dataset))))
        dataset = Subset(dataset, indices)
    
    return dataset

def load_food101(dataset_path: str, is_train: bool = True, limit: int = None, img_size: int = 224):
    """Load Food-101 Dataset (101 food categories)"""
    import torchvision.datasets as datasets
    
    # Create transforms for Food-101
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load Food-101 dataset
    split = "train" if is_train else "test"
    dataset = datasets.Food101(dataset_path, split=split, transform=transform, download=False)
    
    # Apply limit if specified
    if limit and len(dataset) > limit:
        from torch.utils.data import Subset
        indices = list(range(min(limit, len(dataset))))
        dataset = Subset(dataset, indices)
    
    return dataset

def load_imagenet_val(dataset_path: str, limit: int = None, img_size: int = 224):
    """Load ImageNet validation directory"""
    path = Path(dataset_path)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Force square resize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(path.glob(f"*{ext}")))
        image_paths.extend(list(path.glob(f"*{ext.upper()}")))
    
    # Create dummy labels (for debug mode we don't need real labels)
    labels = list(range(len(image_paths)))
    
    print(f"Found {len(image_paths)} images in {dataset_path}")
    
    return SimpleImageDataset([str(p) for p in image_paths], labels, transform, limit)

def debug_train_multi_dataset(
    dataset_path: str,
    dataset_type: str = None,
    backbone_name: str = "vit_small",
    verbose: bool = True
):
    """Debug training with multiple dataset types"""
    
    setup_debug_environment()
    
    if verbose:
        debug_config.print_debug_info()
        print(f"\n🎯 Testing dataset: {dataset_path}")
        print(f"🎯 Using backbone: {backbone_name}")
    
    # Auto-detect dataset type if not provided
    if dataset_type is None:
        dataset_type = MultiDatasetLoader.detect_dataset_type(dataset_path)
        print(f"🔍 Detected dataset type: {dataset_type}")
    
    # Determine image size based on backbone
    if backbone_name in ['vit_small', 'vit_base'] or any(name in backbone_name for name in ['deit', 'swin']):
        img_size = 64
    else:
        img_size = 224
    
    print(f"📐 Using image size: {img_size}x{img_size}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    
    # Determine number of classes based on dataset type
    if dataset_type == "cifar10":
        num_classes = 10
    elif dataset_type == "cifar100":
        num_classes = 100
    elif dataset_type == "fashion_mnist":
        num_classes = 10  # Fashion-MNIST has 10 classes
    elif dataset_type == "oxford_pets":
        num_classes = 37  # Oxford-IIIT Pet Dataset has 37 pet breeds
    elif dataset_type == "food101":
        num_classes = 101  # Food-101 has 101 food categories
    else:
        num_classes = 200  # Default for other datasets
    
    # Create model
    try:
        model = create_model('classification', backbone_name, num_classes=num_classes)
        model.to(device)
        print(f"✅ Model created: {backbone_name} (classes: {num_classes})")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False
    
    # Load datasets based on type
    try:
        if dataset_type == "tiny_imagenet":
            train_dataset = load_tiny_imagenet(dataset_path, is_train=True, limit=debug_config.num_samples_train, img_size=img_size)
            val_dataset = load_tiny_imagenet(dataset_path, is_train=False, limit=debug_config.num_samples_val, img_size=img_size)
        
        elif dataset_type == "hybrid_imagenetc":
            train_dataset = load_hybrid_imagenetc(dataset_path, is_train=True, limit=debug_config.num_samples_train, img_size=img_size)
            val_dataset = load_hybrid_imagenetc(dataset_path, is_train=False, limit=debug_config.num_samples_val, img_size=img_size)
        
        elif dataset_type == "laionc_small":
            # LAION-C small is for additional testing, only has val/test data
            # Use val set and split it for debug training
            full_dataset = load_hybrid_imagenetc(dataset_path, is_train=False, limit=debug_config.num_samples_train + debug_config.num_samples_val, img_size=img_size)
            
            # Split dataset for debug purposes
            train_size = debug_config.num_samples_train
            val_size = debug_config.num_samples_val
            
            from torch.utils.data import Subset
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, train_size + val_size))
            
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
        
        elif dataset_type == "cifar10":
            # CIFAR-10 dataset (10 classes)
            train_dataset = load_cifar(dataset_path, is_cifar100=False, is_train=True, limit=debug_config.num_samples_train, img_size=img_size)
            val_dataset = load_cifar(dataset_path, is_cifar100=False, is_train=False, limit=debug_config.num_samples_val, img_size=img_size)
        
        elif dataset_type == "cifar100":
            # CIFAR-100 dataset (100 classes)
            train_dataset = load_cifar(dataset_path, is_cifar100=True, is_train=True, limit=debug_config.num_samples_train, img_size=img_size)
            val_dataset = load_cifar(dataset_path, is_cifar100=True, is_train=False, limit=debug_config.num_samples_val, img_size=img_size)
        
        elif dataset_type == "fashion_mnist":
            # Fashion-MNIST dataset (10 classes, grayscale -> RGB)
            train_dataset = load_fashion_mnist(dataset_path, is_train=True, limit=debug_config.num_samples_train, img_size=img_size)
            val_dataset = load_fashion_mnist(dataset_path, is_train=False, limit=debug_config.num_samples_val, img_size=img_size)
        
        elif dataset_type == "oxford_pets":
            # Oxford-IIIT Pet Dataset (37 pet breeds)
            train_dataset = load_oxford_pets(dataset_path, is_train=True, limit=debug_config.num_samples_train, img_size=img_size)
            val_dataset = load_oxford_pets(dataset_path, is_train=False, limit=debug_config.num_samples_val, img_size=img_size)
        
        elif dataset_type == "food101":
            # Food-101 Dataset (101 food categories)
            train_dataset = load_food101(dataset_path, is_train=True, limit=debug_config.num_samples_train, img_size=img_size)
            val_dataset = load_food101(dataset_path, is_train=False, limit=debug_config.num_samples_val, img_size=img_size)
        
        elif dataset_type == "imagenet_val":
            # For single directory datasets, split into train/val
            full_dataset = load_imagenet_val(dataset_path, limit=debug_config.num_samples_train + debug_config.num_samples_val, img_size=img_size)
            
            # Split dataset
            train_size = debug_config.num_samples_train
            val_size = debug_config.num_samples_val
            
            from torch.utils.data import Subset
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, train_size + val_size))
            
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
        
        else:
            print(f"❌ Unsupported dataset type: {dataset_type}")
            return False
        
        print(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        **debug_config.get_dataloader_config()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=debug_config.batch_size,
        num_workers=debug_config.num_workers,
        pin_memory=debug_config.pin_memory,
        shuffle=False
    )
    
    # Setup optimizer and loss
    config = debug_config.get_model_config('classification')
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\n🚀 Starting debug training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        print(f"\n📈 Epoch {epoch + 1}/{config['epochs']}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (data, targets) in enumerate(train_pbar):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            for data, targets in val_pbar:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"🎯 Epoch {epoch + 1} Results:")
        print(f"   Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"   Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
    
    print(f"\n✅ Debug training completed successfully!")
    print(f"🎉 Final validation accuracy: {val_acc:.2f}%")
    print(f"📂 Dataset: {dataset_type} at {dataset_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Multi-dataset debug training")
    parser.add_argument("--dataset", type=str, default="../tiny-imagenet-200",
                        help="Path to dataset")
    parser.add_argument("--type", type=str, choices=['tiny_imagenet', 'hybrid_imagenetc', 'imagenet_val', 'auto'],
                        default='auto', help="Dataset type (auto-detect if not specified)")
    parser.add_argument("--backbone", type=str, default="vit_small",
                        help="Backbone to train")
    parser.add_argument("--test-all-datasets", action="store_true",
                        help="Test all available datasets")
    
    args = parser.parse_args()
    
    if args.test_all_datasets:
        # Test all available datasets
        datasets_to_test = [
            ("../tiny-imagenet-200", "tiny_imagenet"),
            ("../hybrid_small_imagenetc", "hybrid_imagenetc"),
            ("../val12", "imagenet_val"),
            ("../laionc_small", "hybrid_imagenetc"),  # Similar structure
        ]
        
        print("🧪 Testing all available datasets in DEBUG MODE")
        print("=" * 60)
        
        results = {}
        for dataset_path, dataset_type in datasets_to_test:
            if Path(dataset_path).exists():
                print(f"\n🔍 Testing: {dataset_path} ({dataset_type})")
                try:
                    success = debug_train_multi_dataset(dataset_path, dataset_type, args.backbone, verbose=False)
                    results[dataset_path] = "✅ PASS" if success else "❌ FAIL"
                except Exception as e:
                    results[dataset_path] = f"❌ ERROR: {str(e)[:50]}..."
                print(f"Result: {results[dataset_path]}")
            else:
                results[dataset_path] = "⚠️ NOT FOUND"
                print(f"\n⚠️ Dataset not found: {dataset_path}")
        
        # Print summary
        print(f"\n📊 Multi-Dataset Test Summary")
        print("=" * 60)
        for dataset, result in results.items():
            print(f"{Path(dataset).name:<25}: {result}")
    
    else:
        # Test single dataset
        dataset_type = args.type if args.type != 'auto' else None
        debug_train_multi_dataset(args.dataset, dataset_type, args.backbone)

if __name__ == "__main__":
    main()
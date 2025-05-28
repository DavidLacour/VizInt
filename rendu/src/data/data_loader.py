"""
Data loader module for handling different datasets
"""
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import logging
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.config_loader import ConfigLoader
from src.data.tiny_imagenet_dataset import TinyImageNetDataset
from src.data.continuous_transforms import ContinuousTransforms


class DataLoaderFactory:
    """Factory class for creating data loaders"""
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize data loader factory
        
        Args:
            config_loader: Configuration loader instance
        """
        self.config = config_loader
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def create_data_loaders(self, 
                           dataset_name: str,
                           with_normalization: bool = True,
                           with_augmentation: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation data loaders
        
        Args:
            dataset_name: Name of dataset ('cifar10' or 'tinyimagenet')
            with_normalization: Whether to apply normalization
            with_augmentation: Whether to apply data augmentation for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        dataset_config = self.config.get_dataset_config(dataset_name)
        
        if dataset_name.lower() == 'cifar10':
            return self._create_cifar10_loaders(dataset_config, with_normalization, with_augmentation)
        elif dataset_name.lower() == 'tinyimagenet':
            return self._create_tinyimagenet_loaders(dataset_config, with_normalization, with_augmentation)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _create_cifar10_loaders(self, 
                               dataset_config: Dict[str, Any],
                               with_normalization: bool,
                               with_augmentation: bool) -> Tuple[DataLoader, DataLoader]:
        """Create CIFAR-10 data loaders"""
        # Get transforms
        transform_train = self._get_cifar10_transform(
            dataset_config, 
            train=True, 
            with_normalization=with_normalization,
            with_augmentation=with_augmentation
        )
        transform_val = self._get_cifar10_transform(
            dataset_config, 
            train=False, 
            with_normalization=with_normalization,
            with_augmentation=False
        )
        
        # Create datasets
        train_dataset = datasets.CIFAR10(
            root=dataset_config['path'],
            train=True,
            download=True,
            transform=transform_train
        )
        
        val_dataset = datasets.CIFAR10(
            root=dataset_config['path'],
            train=False,
            download=True,
            transform=transform_val
        )
        
        # Apply debug mode if enabled
        if self.config.is_debug_mode():
            num_samples = self.config.get('debug.num_samples', 30)
            train_dataset = Subset(train_dataset, range(min(num_samples, len(train_dataset))))
            val_dataset = Subset(val_dataset, range(min(num_samples // 3, len(val_dataset))))
            self.logger.info(f"Debug mode: Using {len(train_dataset)} train and {len(val_dataset)} val samples")
        
        # Create data loaders
        batch_size = self.config.get_batch_size('training')
        
        # Handle debug mode settings
        if self.config.is_debug_mode():
            num_workers = self.config.get('debug.num_workers', 0)
            pin_memory = False
        else:
            num_workers = self.config.get('general.num_workers', 4)
            pin_memory = self.config.get('general.pin_memory', True)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get_batch_size('evaluation'),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        self.logger.info(f"Created CIFAR-10 loaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return train_loader, val_loader
    
    def _create_tinyimagenet_loaders(self, 
                                    dataset_config: Dict[str, Any],
                                    with_normalization: bool,
                                    with_augmentation: bool) -> Tuple[DataLoader, DataLoader]:
        """Create TinyImageNet data loaders"""
        # Get transforms
        transform_train = self._get_tinyimagenet_transform(
            dataset_config, 
            train=True, 
            with_normalization=with_normalization,
            with_augmentation=with_augmentation
        )
        transform_val = self._get_tinyimagenet_transform(
            dataset_config, 
            train=False, 
            with_normalization=with_normalization,
            with_augmentation=False
        )
        
        # Create datasets
        train_dataset = TinyImageNetDataset(
            dataset_config['path'],
            "train",
            transform_train
        )
        
        val_dataset = TinyImageNetDataset(
            dataset_config['path'],
            "val",
            transform_val
        )
        
        # Apply debug mode if enabled
        if self.config.is_debug_mode():
            num_samples = self.config.get('debug.num_samples', 30)
            train_indices = np.random.choice(len(train_dataset), min(num_samples, len(train_dataset)), replace=False)
            val_indices = np.random.choice(len(val_dataset), min(num_samples // 3, len(val_dataset)), replace=False)
            train_dataset = Subset(train_dataset, train_indices)
            val_dataset = Subset(val_dataset, val_indices)
            self.logger.info(f"Debug mode: Using {len(train_dataset)} train and {len(val_dataset)} val samples")
        
        # Create data loaders
        batch_size = self.config.get_batch_size('training')
        
        # Handle debug mode settings
        if self.config.is_debug_mode():
            num_workers = self.config.get('debug.num_workers', 0)
            pin_memory = False
        else:
            num_workers = self.config.get('general.num_workers', 4)
            pin_memory = self.config.get('general.pin_memory', True)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get_batch_size('evaluation'),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        self.logger.info(f"Created TinyImageNet loaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return train_loader, val_loader
    
    def _get_cifar10_transform(self, 
                              dataset_config: Dict[str, Any],
                              train: bool,
                              with_normalization: bool,
                              with_augmentation: bool) -> transforms.Compose:
        """Get CIFAR-10 transforms"""
        transform_list = []
        
        # Data augmentation for training
        if train and with_augmentation:
            transform_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        if with_normalization:
            transform_list.append(
                transforms.Normalize(
                    mean=dataset_config['mean'],
                    std=dataset_config['std']
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _get_tinyimagenet_transform(self, 
                                   dataset_config: Dict[str, Any],
                                   train: bool,
                                   with_normalization: bool,
                                   with_augmentation: bool) -> transforms.Compose:
        """Get TinyImageNet transforms"""
        transform_list = []
        
        # Data augmentation for training
        if train and with_augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        if with_normalization:
            transform_list.append(
                transforms.Normalize(
                    mean=dataset_config['mean'],
                    std=dataset_config['std']
                )
            )
        
        return transforms.Compose(transform_list)
    
    def create_ood_loader(self,
                         dataset_name: str,
                         severity: float,
                         split: str = 'val') -> DataLoader:
        """
        Create OOD (Out-of-Distribution) data loader with transformations
        
        Args:
            dataset_name: Name of dataset
            severity: Transformation severity
            split: Data split ('train' or 'val')
            
        Returns:
            OOD data loader
        """
        dataset_config = self.config.get_dataset_config(dataset_name)
        
        # Create base transform without normalization
        if dataset_name.lower() == 'cifar10':
            base_transform = self._get_cifar10_transform(
                dataset_config, train=False, with_normalization=False, with_augmentation=False
            )
        else:
            base_transform = self._get_tinyimagenet_transform(
                dataset_config, train=False, with_normalization=False, with_augmentation=False
            )
        
        # Create OOD transform
        ood_transform = ContinuousTransforms(severity=severity)
        
        # Create dataset with OOD transforms
        if dataset_name.lower() == 'cifar10':
            dataset = datasets.CIFAR10(
                root=dataset_config['path'],
                train=(split == 'train'),
                download=True,
                transform=base_transform
            )
        else:
            dataset = TinyImageNetDataset(
                dataset_config['path'],
                split,
                base_transform,
                ood_transform=ood_transform
            )
        
        # Create data loader
        batch_size = self.config.get_batch_size('evaluation')
        
        if dataset_name.lower() == 'tinyimagenet':
            # Custom collate function for OOD data
            def collate_fn(batch):
                orig_imgs, trans_imgs, labels, params = zip(*batch)
                return torch.stack(orig_imgs), torch.stack(trans_imgs), torch.tensor(labels), params
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.get('general.num_workers', 4),
                collate_fn=collate_fn
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.get('general.num_workers', 4)
            )
        
        return loader
    
    def get_normalization_transform(self, dataset_name: str) -> transforms.Normalize:
        """Get normalization transform for a dataset"""
        dataset_config = self.config.get_dataset_config(dataset_name)
        return transforms.Normalize(
            mean=dataset_config['mean'],
            std=dataset_config['std']
        )
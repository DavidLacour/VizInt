"""
ResNet model implementations
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any
from .base_model import ClassificationModel


class ResNetBaseline(ClassificationModel):
    """
    ResNet18 baseline model trained from scratch
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ResNet baseline model
        
        Args:
            config: Model configuration containing:
                - num_classes: Number of output classes
                - img_size: Input image size
        """
        num_classes = config['num_classes']
        super().__init__(config, num_classes)
        
        self.img_size = config['img_size']
        
        # use ResNet18 from torchvision
        self.model = models.resnet18(pretrained=False, num_classes=num_classes)
        
        if self.img_size == 32:  # CIFAR-10
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()
        elif self.img_size == 64:  # TinyImageNet
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, 512) for ResNet18
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        return self.model(x)


class ResNetPretrained(ClassificationModel):
    """
    ResNet18 model with ImageNet pretrained weights
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pretrained ResNet model
        
        Args:
            config: Model configuration containing:
                - num_classes: Number of output classes
                - img_size: Input image size
        """
        num_classes = config['num_classes']
        super().__init__(config, num_classes)
        
        self.img_size = config['img_size']
        
        self.model = models.resnet18(pretrained=True)
        
        if self.img_size == 32:  # CIFAR-10
            pretrained_conv1 = self.model.conv1.weight.data.clone()
            
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
            with torch.no_grad():
                self.model.conv1.weight.data = pretrained_conv1[:, :, 2:5, 2:5]
            
            self.model.maxpool = nn.Identity()
            
        elif self.img_size == 64:  # TinyImageNet
            pretrained_conv1 = self.model.conv1.weight.data.clone()
            
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
            with torch.no_grad():
                self.model.conv1.weight.data = pretrained_conv1[:, :, 2:5, 2:5]
            
            self.model.maxpool = nn.Identity()
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, 512) for ResNet18
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        return self.model(x)
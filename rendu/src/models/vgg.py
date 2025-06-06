"""
VGG model implementation for CIFAR-10 and TinyImageNet
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any
from .base_model import ClassificationModel


class VGG(ClassificationModel):
    """VGG model implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        num_classes = config['num_classes']
        super().__init__(config, num_classes)
        
        self.img_size = config['img_size']
        self.pretrained = config.get('pretrained', False)
        self.vgg_type = config.get('vgg_type', 'vgg16')
        
        # Create VGG model
        if self.vgg_type == 'vgg11':
            vgg_fn = models.vgg11
        elif self.vgg_type == 'vgg13':
            vgg_fn = models.vgg13
        elif self.vgg_type == 'vgg16':
            vgg_fn = models.vgg16
        elif self.vgg_type == 'vgg19':
            vgg_fn = models.vgg19
        else:
            raise ValueError(f"Unknown VGG type: {self.vgg_type}")
        
        if self.pretrained:
            self.vgg = vgg_fn(weights='DEFAULT')
        else:
            self.vgg = vgg_fn(weights=None)
        
        # Get feature extractor (everything except classifier)
        self.features = self.vgg.features
        self.avgpool = self.vgg.avgpool
        
        # Adapt for smaller images if needed
        if self.img_size == 32:  # CIFAR-10
            # For 32x32 images, the output of conv layers will be smaller
            # VGG expects 224x224, but we have 32x32
            # After all conv layers, the feature map size will be 1x1
            # So we need to adjust the adaptive pooling
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.img_size == 64:  # TinyImageNet
            # For 64x64 images, adjust pooling
            self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Calculate feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.img_size, self.img_size)
            features = self.features(dummy_input)
            features = self.avgpool(features)
            feature_dim = features.view(1, -1).size(1)
        
        # Create new classifier for our number of classes
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights for new classifier
        if not self.pretrained:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the model"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Pass through first part of classifier to get features
        # (everything except the final classification layer)
        for layer in self.classifier[:-1]:
            x = layer(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG16(VGG):
    """VGG16 model"""
    def __init__(self, config: Dict[str, Any]):
        config['vgg_type'] = 'vgg16'
        super().__init__(config)


class VGG19(VGG):
    """VGG19 model"""
    def __init__(self, config: Dict[str, Any]):
        config['vgg_type'] = 'vgg19'
        super().__init__(config)
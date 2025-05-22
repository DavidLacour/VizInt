import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, VGG16_Weights
from abc import ABC, abstractmethod
import timm
from transformer_utils import LayerNorm, TransformerTrunk
from vit_implementation import PatchEmbed

class BackboneBase(ABC):
    """Abstract base class for all backbones"""
    
    @abstractmethod
    def forward_features(self, x):
        """Extract features from input"""
        pass
    
    @abstractmethod
    def get_feature_dim(self):
        """Return the dimension of extracted features"""
        pass
    
    @abstractmethod
    def get_num_patches(self):
        """Return number of patches (for transformer-based models)"""
        pass

class ViTBackbone(nn.Module, BackboneBase):
    """Custom ViT backbone (your original implementation)"""
    
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=384, depth=8, head_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_resnet_stem=True
        )
        
        # Learnable cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.transformer = TransformerTrunk(
            dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            mlp_ratio=4.0,
            use_bias=False
        )
        
        # Normalization
        self.norm = LayerNorm(embed_dim, bias=False)
    
    def forward_features(self, x):
        B = x.shape[0]
        
        # Extract patches
        x = self.patch_embed(x)
        
        # Add cls token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed.expand(B, -1, -1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Apply normalization
        x = self.norm(x)
        
        return x
    
    def get_feature_dim(self):
        return self.embed_dim
    
    def get_num_patches(self):
        return self.patch_embed.num_patches

class ResNetBackbone(nn.Module, BackboneBase):
    """ResNet backbone"""
    
    def __init__(self, model_name='resnet50', pretrained=True, feature_dim=2048):
        super().__init__()
        self.feature_dim = feature_dim
        
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add global average pooling if not present
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward_features(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x
    
    def get_feature_dim(self):
        return self.feature_dim
    
    def get_num_patches(self):
        return 0  # CNN doesn't have patches

class VGGBackbone(nn.Module, BackboneBase):
    """VGG backbone"""
    
    def __init__(self, model_name='vgg16', pretrained=True):
        super().__init__()
        
        if model_name == 'vgg11':
            self.backbone = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == 'vgg13':
            self.backbone = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == 'vgg16':
            self.backbone = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == 'vgg19':
            self.backbone = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported VGG model: {model_name}")
        
        # Use only the feature extractor
        self.features = self.backbone.features
        self.global_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.feature_dim = 512 * 7 * 7
    
    def forward_features(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x
    
    def get_feature_dim(self):
        return self.feature_dim
    
    def get_num_patches(self):
        return 0  # CNN doesn't have patches

class TimmBackbone(nn.Module, BackboneBase):
    """Backbone using timm library (supports many vision transformers and CNNs)"""
    
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, img_size=64):
        super().__init__()
        
        # Create model without classification head
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,  # Remove classification head
            img_size=img_size
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            dummy_output = self.backbone(dummy_input)
            self.feature_dim = dummy_output.shape[1]
        
        self.model_name = model_name
    
    def forward_features(self, x):
        return self.backbone(x)
    
    def get_feature_dim(self):
        return self.feature_dim
    
    def get_num_patches(self):
        # Try to get number of patches for transformer models
        if hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone.patch_embed, 'num_patches'):
            return self.backbone.patch_embed.num_patches
        return 0

class BackboneFactory:
    """Factory for creating different backbones"""
    
    @staticmethod
    def create_backbone(backbone_type, **kwargs):
        """
        Create a backbone based on the specified type.
        
        Args:
            backbone_type: Type of backbone ('vit', 'resnet', 'vgg', 'timm')
            **kwargs: Additional arguments for the backbone
            
        Returns:
            backbone: The created backbone instance
        """
        if backbone_type == 'vit':
            return ViTBackbone(**kwargs)
        elif backbone_type == 'resnet':
            return ResNetBackbone(**kwargs)
        elif backbone_type == 'vgg':
            return VGGBackbone(**kwargs)
        elif backbone_type == 'timm':
            return TimmBackbone(**kwargs)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
    
    @staticmethod
    def get_supported_backbones():
        """Return list of supported backbone types"""
        return ['vit', 'resnet', 'vgg', 'timm']

# Example usage and configuration
BACKBONE_CONFIGS = {
    'vit_small': {
        'type': 'vit',
        'img_size': 64,
        'patch_size': 8,
        'embed_dim': 384,
        'depth': 6,
        'head_dim': 64
    },
    'vit_base': {
        'type': 'vit',
        'img_size': 64,
        'patch_size': 8,
        'embed_dim': 768,
        'depth': 12,
        'head_dim': 64
    },
    'resnet50': {
        'type': 'resnet',
        'model_name': 'resnet50',
        'pretrained': True
    },
    'resnet18': {
        'type': 'resnet',
        'model_name': 'resnet18',
        'pretrained': True
    },
    'vgg16': {
        'type': 'vgg',
        'model_name': 'vgg16',
        'pretrained': True
    },
    'deit_small': {
        'type': 'timm',
        'model_name': 'deit_small_patch16_224',
        'pretrained': True,
        'img_size': 64
    },
    'swin_small': {
        'type': 'timm',
        'model_name': 'swin_small_patch4_window7_224',
        'pretrained': True,
        'img_size': 64
    }
}
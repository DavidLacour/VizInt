import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone_factory import BackboneFactory, BACKBONE_CONFIGS

class FlexibleClassificationModel(nn.Module):
    """Classification model that can use different backbones"""
    
    def __init__(self, backbone_config, num_classes=200, dropout=0.1):
        super().__init__()
        
        # Create backbone
        backbone_type = backbone_config.pop('type')
        self.backbone = BackboneFactory.create_backbone(backbone_type, **backbone_config)
        
        # Classification head
        feature_dim = self.backbone.get_feature_dim()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        # Handle different feature shapes
        if len(features.shape) == 3:  # Transformer output [B, N, D]
            features = features[:, 0]  # Use CLS token
        elif len(features.shape) == 2:  # CNN output [B, D]
            pass  # Already flattened
        else:
            features = torch.flatten(features, 1)
        
        return self.classifier(features)

class FlexibleTransformationHealer(nn.Module):
    """Transformation healer that can use different backbones"""
    
    def __init__(self, backbone_config, dropout=0.1):
        super().__init__()
        
        # Create backbone
        backbone_type = backbone_config.pop('type')
        self.backbone = BackboneFactory.create_backbone(backbone_type, **backbone_config)
        
        # Get feature dimension
        feature_dim = self.backbone.get_feature_dim()
        
        # Transformation prediction heads
        self.transform_type_head = nn.Linear(feature_dim, 4)
        
        # Severity heads for each transform type
        self.severity_noise_head = nn.Linear(feature_dim, 1)   
        self.severity_rotation_head = nn.Linear(feature_dim, 1)
        self.severity_affine_head = nn.Linear(feature_dim, 1)
        
        # Specific parameter heads for each transform type
        self.rotation_head = nn.Linear(feature_dim, 1)        # Rotation angle
        self.noise_head = nn.Linear(feature_dim, 1)           # Noise std
        self.affine_head = nn.Linear(feature_dim, 4)          # Affine params
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        
        # Handle different feature shapes
        if len(features.shape) == 3:  # Transformer output [B, N, D]
            features = features[:, 0]  # Use CLS token
        elif len(features.shape) == 2:  # CNN output [B, D]
            pass  # Already flattened
        else:
            features = torch.flatten(features, 1)
        
        features = self.dropout(features)
        
        # Predict transform type probabilities
        transform_type_logits = self.transform_type_head(features)
        
        # Predict transform-specific severities
        severity_noise = torch.sigmoid(self.severity_noise_head(features))
        severity_rotation = torch.sigmoid(self.severity_rotation_head(features))
        severity_affine = torch.sigmoid(self.severity_affine_head(features))
        
        # Predict various parameters
        rotation_angle = torch.tanh(self.rotation_head(features)) * 180.0
        noise_std = torch.sigmoid(self.noise_head(features)) * 0.5
        
        # Affine transformation parameters
        affine_params = self.affine_head(features)  # [B, 4]
        translate_x = torch.tanh(affine_params[:, 0:1]) * 0.1
        translate_y = torch.tanh(affine_params[:, 1:2]) * 0.1
        shear_x = torch.tanh(affine_params[:, 2:3]) * 15.0
        shear_y = torch.tanh(affine_params[:, 3:4]) * 15.0
        
        return {
            'transform_type_logits': transform_type_logits,
            'severity_noise': severity_noise,
            'severity_rotation': severity_rotation,
            'severity_affine': severity_affine,
            'rotation_angle': rotation_angle,
            'noise_std': noise_std,
            'translate_x': translate_x,
            'translate_y': translate_y,
            'shear_x': shear_x,
            'shear_y': shear_y
        }
    
    def apply_correction(self, transformed_images, predictions):
        """Apply inverse transformations to correct distorted images"""
        # This method remains the same as your original implementation
        device = transformed_images.device
        batch_size = transformed_images.shape[0]
        
        # Get the predicted transform types
        transform_type_logits = predictions['transform_type_logits']
        transform_types = torch.argmax(transform_type_logits, dim=1)  # [B]
        
        # Initialize corrected images as a clone of transformed images
        corrected_images = transformed_images.clone()
        
        # Process each image in the batch (implementation same as original)
        # ... (rest of your apply_correction method)
        
        return corrected_images

class FlexibleBlendedTTT(nn.Module):
    """Blended TTT model that can use different backbones"""
    
    def __init__(self, backbone_config, num_classes=200, dropout=0.1):
        super().__init__()
        
        # Create backbone
        backbone_type = backbone_config.pop('type')
        self.backbone = BackboneFactory.create_backbone(backbone_type, **backbone_config)
        
        # Get feature dimension
        feature_dim = self.backbone.get_feature_dim()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Transformation prediction heads (auxiliary task)
        self.transform_type_head = nn.Linear(feature_dim, 4)
        self.severity_noise_head = nn.Linear(feature_dim, 1)   
        self.severity_rotation_head = nn.Linear(feature_dim, 1)      
        self.severity_affine_head = nn.Linear(feature_dim, 1)   
        self.rotation_head = nn.Linear(feature_dim, 1)
        self.noise_head = nn.Linear(feature_dim, 1)
        self.affine_head = nn.Linear(feature_dim, 4)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, aux_only=False):
        features = self.backbone.forward_features(x)
        
        # Handle different feature shapes
        if len(features.shape) == 3:  # Transformer output [B, N, D]
            features = features[:, 0]  # Use CLS token
        elif len(features.shape) == 2:  # CNN output [B, D]
            pass  # Already flattened
        else:
            features = torch.flatten(features, 1)
        
        features = self.dropout(features)
        
        # Auxiliary task predictions
        aux_outputs = {
            'transform_type_logits': self.transform_type_head(features),
            'severity_noise': torch.sigmoid(self.severity_noise_head(features)),
            'severity_rotation': torch.sigmoid(self.severity_rotation_head(features)),
            'severity_affine': torch.sigmoid(self.severity_affine_head(features)),
            'rotation_angle': torch.tanh(self.rotation_head(features)) * 180.0,
            'noise_std': torch.sigmoid(self.noise_head(features)) * 0.5,
            'translate_x': torch.tanh(self.affine_head(features)[:, 0:1]) * 0.1,
            'translate_y': torch.tanh(self.affine_head(features)[:, 1:2]) * 0.1,
            'shear_x': torch.tanh(self.affine_head(features)[:, 2:3]) * 15.0,
            'shear_y': torch.tanh(self.affine_head(features)[:, 3:4]) * 15.0
        }
        
        if aux_only:
            return aux_outputs
        
        # Classification logits
        logits = self.classifier(features)
        
        return logits, aux_outputs

class FlexibleTestTimeTrainer(nn.Module):
    """TTT model that can use different backbones"""
    
    def __init__(self, base_model, aux_backbone_config, adaptation_steps=10, adaptation_lr=1e-4):
        super().__init__()
        self.base_model = base_model
        self.adaptation_steps = adaptation_steps
        self.adaptation_lr = adaptation_lr
        
        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Create auxiliary backbone for self-supervised task
        backbone_type = aux_backbone_config.pop('type')
        self.aux_backbone = BackboneFactory.create_backbone(backbone_type, **aux_backbone_config)
        
        # Self-supervised task head
        feature_dim = self.aux_backbone.get_feature_dim()
        self.transform_head = nn.Linear(feature_dim, 4)  # no_transform, gaussian_noise, rotation, affine
        
        # Store the original base model to reset when needed
        from copy import deepcopy
        self.original_base_model = deepcopy(base_model)
    
    def forward(self, x, aux_only=False):
        # Extract features using auxiliary backbone
        features = self.aux_backbone.forward_features(x)
        
        # Handle different feature shapes
        if len(features.shape) == 3:  # Transformer output [B, N, D]
            features = features[:, 0]  # Use CLS token
        elif len(features.shape) == 2:  # CNN output [B, D]
            pass  # Already flattened
        else:
            features = torch.flatten(features, 1)
        
        # Predict transformation
        transform_logits = self.transform_head(features)
        
        if aux_only:
            return transform_logits
        
        # Run the base model for classification
        with torch.no_grad():
            logits = self.base_model(x)
        
        return logits, transform_logits
    
    def adapt(self, x, transform_labels=None, reset=False):
        """Adapt the model using the self-supervised task"""
        # Reset the base model if requested
        if reset:
            self.base_model.load_state_dict(self.original_base_model.state_dict())
        
        # Enable gradients for the base model during adaptation
        for param in self.base_model.parameters():
            param.requires_grad = True
        
        # Create a temporary optimizer for adaptation
        optimizer = torch.optim.Adam(
            list(self.parameters()) + list(self.base_model.parameters()),
            lr=self.adaptation_lr
        )
        
        # Adaptation loop
        self.train()
        self.base_model.train()
        
        for _ in range(self.adaptation_steps):
            # Forward pass
            logits, transform_logits = self(x)
            
            # Self-supervised loss
            if transform_labels is not None:
                aux_loss = F.cross_entropy(transform_logits, transform_labels)
            else:
                probs = F.softmax(transform_logits, dim=1)
                aux_loss = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            aux_loss.backward()
            optimizer.step()
        
        # Evaluate with adapted model
        self.eval()
        self.base_model.eval()
        with torch.no_grad():
            adapted_logits = self.base_model(x)
        
        # Restore gradients state
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        return adapted_logits

def create_model(model_type, backbone_name, num_classes=200, **kwargs):
    """
    Factory function to create different models with specified backbones.
    
    Args:
        model_type: Type of model ('classification', 'healer', 'blended_ttt', 'ttt')
        backbone_name: Name of backbone configuration from BACKBONE_CONFIGS
        num_classes: Number of classes for classification
        **kwargs: Additional model-specific arguments
        
    Returns:
        model: The created model instance
    """
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from {list(BACKBONE_CONFIGS.keys())}")
    
    backbone_config = BACKBONE_CONFIGS[backbone_name].copy()
    
    if model_type == 'classification':
        return FlexibleClassificationModel(backbone_config, num_classes, **kwargs)
    elif model_type == 'healer':
        return FlexibleTransformationHealer(backbone_config, **kwargs)
    elif model_type == 'blended_ttt':
        return FlexibleBlendedTTT(backbone_config, num_classes, **kwargs)
    elif model_type == 'ttt':
        # For TTT, we need a base model and aux backbone
        base_model = kwargs.pop('base_model')
        aux_backbone_name = kwargs.pop('aux_backbone_name', backbone_name)
        aux_backbone_config = BACKBONE_CONFIGS[aux_backbone_name].copy()
        return FlexibleTestTimeTrainer(base_model, aux_backbone_config, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Example usage:
def example_usage():
    """Examples of how to use the flexible models"""
    
    print("Creating different models with different backbones:")
    
    # Classification model with ResNet50
    resnet_classifier = create_model('classification', 'resnet50', num_classes=200)
    print(f"ResNet50 classifier: {resnet_classifier}")
    
    # Classification model with ViT
    vit_classifier = create_model('classification', 'vit_small', num_classes=200)
    print(f"ViT classifier: {vit_classifier}")
    
    # Healer model with VGG16
    vgg_healer = create_model('healer', 'vgg16')
    print(f"VGG16 healer: {vgg_healer}")
    
    # Blended TTT with Swin Transformer
    swin_blended = create_model('blended_ttt', 'swin_small', num_classes=200)
    print(f"Swin Blended TTT: {swin_blended}")
    
    # Test with dummy input
    dummy_input = torch.randn(2, 3, 64, 64)
    
    # Test ResNet classifier
    with torch.no_grad():
        output = resnet_classifier(dummy_input)
        print(f"ResNet output shape: {output.shape}")
    
    # Test ViT classifier
    with torch.no_grad():
        output = vit_classifier(dummy_input)
        print(f"ViT output shape: {output.shape}")

if __name__ == "__main__":
    example_usage()
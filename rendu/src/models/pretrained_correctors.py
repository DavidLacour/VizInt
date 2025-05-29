"""
Pre-trained corrector models for image correction
Includes UNet with ResNet encoder and image-to-image transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Tuple
from .base_model import BaseModel
from .vit_implementation import PatchEmbed
from utils.transformer_utils import LayerNorm, TransformerTrunk


class PretrainedUNetCorrector(BaseModel):
    """UNet corrector with pre-trained ResNet encoder"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize UNet corrector with pre-trained ResNet encoder
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        self.img_size = config['img_size']
        self.in_channels = config.get('in_channels', 3)
        self.out_channels = config.get('out_channels', 3)
        self.use_residual = config.get('use_residual', True)
        
        # Load pre-trained ResNet18 as encoder
        self.encoder = models.resnet18(pretrained=True)
        
        # Modify first conv for different input sizes if needed
        if self.img_size in [32, 64]:
            pretrained_conv1 = self.encoder.conv1.weight.data.clone()
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            with torch.no_grad():
                self.encoder.conv1.weight.data = pretrained_conv1[:, :, 2:5, 2:5]
            self.encoder.maxpool = nn.Identity()
        
        # Remove classifier
        self.encoder.fc = nn.Identity()
        
        # Get feature dimensions from ResNet layers
        # ResNet18: [64, 64, 128, 256, 512]
        self.encoder_dims = [64, 64, 128, 256, 512]
        
        # Decoder layers
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.out_channels, kernel_size=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Encode input through ResNet encoder
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple of (bottleneck_features, skip_connections)
        """
        skip_connections = []
        
        # Initial conv
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        skip_connections.append(x)  # 64 channels
        
        x = self.encoder.maxpool(x)
        
        # ResNet layers
        x = self.encoder.layer1(x)
        skip_connections.append(x)  # 64 channels
        
        x = self.encoder.layer2(x)
        skip_connections.append(x)  # 128 channels
        
        x = self.encoder.layer3(x)
        skip_connections.append(x)  # 256 channels
        
        x = self.encoder.layer4(x)
        # x is the bottleneck (512 channels)
        
        return x, skip_connections
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of UNet corrector
        
        Args:
            x: Input tensor (corrupted image) [B, C, H, W]
            
        Returns:
            Corrected image [B, C, H, W]
        """
        input_img = x
        
        # Encoder
        bottleneck, skip_connections = self.encode(x)
        
        # Decoder with skip connections
        x = self.decoder4(bottleneck, skip_connections[3])  # 256
        x = self.decoder3(x, skip_connections[2])           # 128
        x = self.decoder2(x, skip_connections[1])           # 64
        x = self.decoder1(x, skip_connections[0])           # 64
        
        # Final output
        output = self.final_conv(x)
        
        # Residual connection
        if self.use_residual:
            output = output + input_img
            
        return torch.clamp(output, -1, 1)
    
    def correct_image(self, x: torch.Tensor) -> torch.Tensor:
        """Correct a transformed/corrupted image"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class DecoderBlock(nn.Module):
    """Decoder block for UNet"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ImageToImageTransformer(BaseModel):
    """Image-to-image transformer for correction using Vision Transformer architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image-to-image transformer corrector
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        self.img_size = config['img_size']
        self.patch_size = config.get('patch_size', 8)
        self.in_channels = config.get('in_channels', 3)
        self.out_channels = config.get('out_channels', 3)
        self.embed_dim = config.get('embed_dim', 768)
        self.depth = config.get('depth', 12)
        self.head_dim = config.get('head_dim', 64)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.use_residual = config.get('use_residual', True)
        
        # Input patch embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.embed_dim,
            use_resnet_stem=True  # Use ResNet stem for better feature extraction
        )
        
        self.num_patches = self.patch_embed.num_patches
        
        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.transformer = TransformerTrunk(
            dim=self.embed_dim,
            depth=self.depth,
            head_dim=self.head_dim,
            mlp_ratio=self.mlp_ratio,
            use_bias=False
        )
        
        # Output normalization
        self.norm = LayerNorm(self.embed_dim, bias=False)
        
        # Patch reconstruction head
        self.patch_reconstruct = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.patch_size * self.patch_size * self.out_channels)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize patch reconstruction head
        nn.init.normal_(self.patch_reconstruct[0].weight, std=0.02)
        nn.init.normal_(self.patch_reconstruct[2].weight, std=0.02)
        nn.init.zeros_(self.patch_reconstruct[0].bias)
        nn.init.zeros_(self.patch_reconstruct[2].bias)
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches
        
        Args:
            imgs: [B, C, H, W]
            
        Returns:
            patches: [B, num_patches, patch_size^2 * C]
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        h_patches = H // p
        w_patches = W // p
        
        # Reshape to patches
        x = imgs.reshape(B, C, h_patches, p, w_patches, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, h_patches, w_patches, p, p, C]
        x = x.reshape(B, h_patches * w_patches, p * p * C)
        
        return x
    
    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images
        
        Args:
            patches: [B, num_patches, patch_size^2 * C]
            
        Returns:
            imgs: [B, C, H, W]
        """
        B, num_patches, patch_dim = patches.shape
        C = self.out_channels
        p = self.patch_size
        h_patches = w_patches = int(num_patches ** 0.5)
        
        # Reshape patches to image
        x = patches.reshape(B, h_patches, w_patches, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, h_patches, p, w_patches, p]
        x = x.reshape(B, C, h_patches * p, w_patches * p)
        
        return x
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of image-to-image transformer
        
        Args:
            x: Input tensor (corrupted image) [B, C, H, W]
            
        Returns:
            Corrected image [B, C, H, W]
        """
        input_img = x
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer
        x = self.transformer(x)
        
        # Apply normalization
        x = self.norm(x)
        
        # Reconstruct patches
        patch_outputs = self.patch_reconstruct(x)  # [B, num_patches, patch_size^2 * C]
        
        # Convert patches back to image
        output = self.unpatchify(patch_outputs)  # [B, C, H, W]
        
        # Apply tanh activation for output range
        output = torch.tanh(output)
        
        # Residual connection
        if self.use_residual:
            output = output + input_img
            
        return torch.clamp(output, -1, 1)
    
    def correct_image(self, x: torch.Tensor) -> torch.Tensor:
        """Correct a transformed/corrupted image"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class HybridCorrector(BaseModel):
    """Hybrid corrector combining transformer and convolutional approaches"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hybrid corrector
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        self.img_size = config['img_size']
        self.use_transformer = config.get('use_transformer', True)
        self.use_cnn = config.get('use_cnn', True)
        self.fusion_weight = config.get('fusion_weight', 0.5)
        
        # Transformer branch
        if self.use_transformer:
            transformer_config = config.copy()
            transformer_config['embed_dim'] = config.get('embed_dim', 384)
            transformer_config['depth'] = config.get('depth', 6)
            self.transformer = ImageToImageTransformer(transformer_config)
        
        # CNN branch (UNet)
        if self.use_cnn:
            self.cnn = PretrainedUNetCorrector(config)
        
        # Fusion layer
        if self.use_transformer and self.use_cnn:
            self.fusion = nn.Sequential(
                nn.Conv2d(6, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=1),
                nn.Tanh()
            )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of hybrid corrector
        
        Args:
            x: Input tensor (corrupted image) [B, C, H, W]
            
        Returns:
            Corrected image [B, C, H, W]
        """
        outputs = []
        
        if self.use_transformer:
            transformer_out = self.transformer(x)
            outputs.append(transformer_out)
        
        if self.use_cnn:
            cnn_out = self.cnn(x)
            outputs.append(cnn_out)
        
        if len(outputs) == 1:
            return outputs[0]
        elif len(outputs) == 2:
            # Learnable fusion
            combined = torch.cat(outputs, dim=1)
            return self.fusion(combined)
        else:
            return x  # Fallback
    
    def correct_image(self, x: torch.Tensor) -> torch.Tensor:
        """Correct a transformed/corrupted image"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
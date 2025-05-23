import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class ConvStem(nn.Module):
    """
    Convolutional stem for hybrid CNN-Transformer architectures.
    Extracts local features before transformer processing.
    """
    def __init__(self, in_chans=3, embed_dim=768, patch_size=16, kernel_size=7):
        super().__init__()
        self.patch_size = patch_size
        
        # Multi-stage convolutional stem
        self.conv1 = nn.Conv2d(in_chans, embed_dim // 4, kernel_size=kernel_size, 
                              stride=2, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(embed_dim // 4)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, 
                              stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim // 2)
        self.act2 = nn.GELU()
        
        self.conv3 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, 
                              stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.act3 = nn.GELU()
        
        # Final patch projection
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size // 8, 
                             stride=patch_size // 8)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.proj(x)
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution for efficient feature extraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                  stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.act(self.bn(x))
        return x


class ConvMLP(nn.Module):
    """
    Convolutional MLP for processing patch tokens with local connectivity.
    """
    def __init__(self, dim, hidden_dim, patch_size=14, drop=0.):
        super().__init__()
        self.patch_size = patch_size
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DepthwiseSeparableConv(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        
        # Check if we have spatial tokens (excluding cls token)
        if N == self.patch_size * self.patch_size + 1:
            # Separate cls token and spatial tokens
            cls_token = x[:, 0:1, :]  # (B, 1, C)
            spatial_tokens = x[:, 1:, :]  # (B, patch_size^2, C)
            
            # Process spatial tokens with convolution
            x_spatial = self.fc1(spatial_tokens)
            x_spatial = self.drop(x_spatial)
            
            # Reshape for convolution
            x_spatial = x_spatial.transpose(1, 2).view(B, -1, self.patch_size, self.patch_size)
            x_spatial = self.dwconv(x_spatial)
            x_spatial = x_spatial.flatten(2).transpose(1, 2)
            
            x_spatial = self.act(x_spatial)
            x_spatial = self.fc2(x_spatial)
            x_spatial = self.drop(x_spatial)
            
            # Process cls token with standard MLP
            cls_token = self.fc1(cls_token)
            cls_token = self.drop(cls_token)
            cls_token = self.act(cls_token)
            cls_token = self.fc2(cls_token)
            cls_token = self.drop(cls_token)
            
            # Combine cls token and spatial tokens
            x = torch.cat([cls_token, x_spatial], dim=1)
        else:
            # Fallback to standard MLP if spatial structure is unclear
            x = self.fc1(x)
            x = self.drop(x)
            x = self.act(x)
            x = self.fc2(x)
            x = self.drop(x)
        
        return x


class HybridAttention(nn.Module):
    """
    Hybrid attention combining global attention with local convolution.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., 
                 patch_size=14, local_kernel_size=3):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.patch_size = patch_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Local convolution for value processing
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=local_kernel_size, 
                                   padding=local_kernel_size // 2, groups=dim)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

        # Standard attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x_global = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Local convolution for spatial modeling
        if N == self.patch_size * self.patch_size + 1:
            # Process only spatial tokens with convolution
            v_spatial = x[:, 1:, :].transpose(1, 2).view(B, C, self.patch_size, self.patch_size)
            v_local = self.local_conv(v_spatial).flatten(2).transpose(1, 2)
            
            # Combine with cls token
            cls_token = x_global[:, 0:1, :]
            x_local = torch.cat([cls_token, v_local], dim=1)
        else:
            x_local = x_global  # Fallback if spatial structure is unclear
        
        # Combine global and local features
        x = 0.5 * x_global + 0.5 * x_local
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HybridBlock(nn.Module):
    """
    Hybrid transformer block combining CNN and transformer components.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, patch_size=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = HybridAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop, patch_size=patch_size
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMLP(
            dim=dim, 
            hidden_dim=mlp_hidden_dim, 
            patch_size=patch_size, 
            drop=drop
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CNNTransformerHybrid(nn.Module):
    """
    Hybrid CNN-Transformer architecture.
    CNN extracts local features, Transformer models global relationships.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., use_conv_stem=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.patch_size = patch_size
        
        # Patch embedding with optional conv stem
        if use_conv_stem:
            self.patch_embed = ConvStem(in_chans, embed_dim, patch_size)
            num_patches = (img_size // patch_size) ** 2
        else:
            self.patch_embed = nn.Conv2d(in_chans, embed_dim, 
                                        kernel_size=patch_size, stride=patch_size)
            num_patches = (img_size // patch_size) ** 2
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            HybridBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                patch_size=int(math.sqrt(num_patches))
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Feature pyramid for multi-scale processing
        self.feature_pyramid = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1) for _ in range(3)
        ])
        
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.head, nn.Linear):
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)

    def forward_features(self, x):
        B = x.shape[0]
        
        # Patch embedding
        if hasattr(self.patch_embed, 'conv1'):  # ConvStem
            x = self.patch_embed(x)
        else:  # Standard patch embedding
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply hybrid blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x[:, 0])  # Use cls token for classification
        return x


class ResNetTransformerHybrid(nn.Module):
    """
    Hybrid combining ResNet backbone with Transformer head.
    """
    def __init__(self, img_size=224, num_classes=1000, embed_dim=768, 
                 depth=6, num_heads=12, resnet_layers=[2, 2, 2, 2]):
        super().__init__()
        self.embed_dim = embed_dim
        
        # ResNet backbone (simplified)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, resnet_layers[0])
        self.layer2 = self._make_layer(64, 128, resnet_layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, resnet_layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, resnet_layers[3], stride=2)
        
        # Projection to transformer dimension
        self.feature_proj = nn.Linear(512, embed_dim)
        
        # Transformer layers
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim * 4,
                dropout=0.1, 
                activation='gelu'
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def _make_layer(self, in_planes, planes, num_blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Convert to sequence format
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.feature_proj(x)  # (B, H*W, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer
        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.transpose(0, 1)  # (batch, seq_len, embed_dim)
        
        # Classification
        x = self.norm(x)
        x = self.head(x[:, 0])  # Use cls token
        
        return x


if __name__ == "__main__":
    # Test hybrid architectures
    batch_size = 2
    img_size = 224
    test_input = torch.randn(batch_size, 3, img_size, img_size)
    
    # Test ConvStem
    conv_stem = ConvStem()
    stem_output = conv_stem(test_input)
    print(f"ConvStem output shape: {stem_output.shape}")
    
    # Test HybridBlock
    hybrid_block = HybridBlock(dim=768, num_heads=12)
    block_input = torch.randn(batch_size, 197, 768)  # ViT input format
    block_output = hybrid_block(block_input)
    print(f"HybridBlock output shape: {block_output.shape}")
    
    # Test CNNTransformerHybrid
    hybrid_model = CNNTransformerHybrid(embed_dim=384, depth=6, num_heads=6)
    hybrid_output = hybrid_model(test_input)
    print(f"CNNTransformerHybrid output shape: {hybrid_output.shape}")
    
    # Test ResNetTransformerHybrid
    resnet_transformer = ResNetTransformerHybrid(embed_dim=384, depth=4, num_heads=6)
    resnet_output = resnet_transformer(test_input)
    print(f"ResNetTransformerHybrid output shape: {resnet_output.shape}")
    
    print("All hybrid architecture tests passed!")
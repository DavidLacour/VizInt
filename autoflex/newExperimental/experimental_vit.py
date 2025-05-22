import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union
from einops import rearrange

# Import experimental components
from fourier_attention import FourierAttention, FNetBlock
from linear_attention import LinearAttention, EfficientLinearAttention
from vision_mamba import VisionMambaBlock, BidirectionalMamba
from kan_transformer import KANAttention, KANMLP
from hybrid_architectures import ConvStem, HybridAttention


class ExperimentalAttention(nn.Module):
    """
    Unified experimental attention that can switch between different mechanisms.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., 
                 attention_type='standard', **kwargs):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == 'fourier':
            self.attn = FourierAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop, **kwargs)
        elif attention_type == 'linear':
            self.attn = LinearAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop, **kwargs)
        elif attention_type == 'elfatt':
            self.attn = EfficientLinearAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        elif attention_type == 'kan':
            self.attn = KANAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop, **kwargs)
        elif attention_type == 'hybrid':
            self.attn = HybridAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop, **kwargs)
        else:  # standard
            self.attn = StandardAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop)

    def forward(self, x):
        return self.attn(x)


class StandardAttention(nn.Module):
    """
    Standard multi-head self-attention for comparison.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ExperimentalMLP(nn.Module):
    """
    Experimental MLP that can use different implementations.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0., mlp_type='standard', **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp_type = mlp_type
        
        if mlp_type == 'kan':
            self.mlp = KANMLP(in_features, hidden_features, out_features, act_layer, drop, **kwargs)
        else:  # standard
            self.mlp = StandardMLP(in_features, hidden_features, out_features, act_layer, drop)

    def forward(self, x):
        return self.mlp(x)


class StandardMLP(nn.Module):
    """
    Standard MLP implementation.
    """
    def __init__(self, in_features, hidden_features, out_features, act_layer, drop):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ExperimentalBlock(nn.Module):
    """
    Experimental transformer block that can use different attention and MLP types.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 attention_type='standard', mlp_type='standard', 
                 use_mamba=False, **kwargs):
        super().__init__()
        self.use_mamba = use_mamba
        
        if use_mamba:
            # Use Mamba block instead of attention + MLP
            self.mamba_block = VisionMambaBlock(
                d_model=dim, 
                mlp_ratio=mlp_ratio, 
                drop=drop, 
                **kwargs
            )
        else:
            # Standard transformer structure with experimental components
            self.norm1 = norm_layer(dim)
            self.attn = ExperimentalAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop, 
                attention_type=attention_type, **kwargs
            )
            
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = ExperimentalMLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                mlp_type=mlp_type,
                **kwargs
            )

    def forward(self, x):
        if self.use_mamba:
            return self.mamba_block(x)
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class AdaptivePositionEmbedding(nn.Module):
    """
    Adaptive position embedding that can switch between different types.
    """
    def __init__(self, num_patches, embed_dim, pos_type='learnable', img_size=224, patch_size=16):
        super().__init__()
        self.pos_type = pos_type
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        if pos_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        elif pos_type == 'sincos':
            pos_embed = self.get_2d_sincos_pos_embed(embed_dim, int(num_patches**0.5))
            self.register_buffer('pos_embed', pos_embed)
        elif pos_type == 'rope':
            # Rotary position embedding implementation would go here
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        """Generate 2D sinusoidal position embeddings."""
        grid_h = torch.arange(grid_size)
        grid_w = torch.arange(grid_size)
        grid = torch.meshgrid(grid_w, grid_h, indexing='ij')
        grid = torch.stack(grid, dim=0)  # 2, grid_size, grid_size
        
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        # Add cls token
        cls_pos = torch.zeros(1, embed_dim)
        pos_embed = torch.cat([cls_pos, pos_embed], dim=0)
        return pos_embed.unsqueeze(0)

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0
        
        # Use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        
        emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
        return emb

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)

        emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb

    def forward(self, x):
        if self.pos_embed is not None:
            B = x.shape[0]
            return x + self.pos_embed.expand(B, -1, -1)
        return x


class ExperimentalVisionTransformer(nn.Module):
    """
    Experimental Vision Transformer with multiple cutting-edge techniques.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        pos_type='learnable',  # 'learnable', 'sincos', 'rope'
        attention_type='standard',  # 'standard', 'fourier', 'linear', 'elfatt', 'kan', 'hybrid'
        mlp_type='standard',  # 'standard', 'kan'
        use_conv_stem=False,
        use_mamba_blocks=None,  # None, list of block indices, or 'all'
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Patch embedding
        if use_conv_stem:
            self.patch_embed = ConvStem(in_chans, embed_dim, patch_size)
        else:
            self.patch_embed = nn.Conv2d(in_chans, embed_dim, 
                                        kernel_size=patch_size, stride=patch_size)
        
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed_layer = AdaptivePositionEmbedding(
            num_patches, embed_dim, pos_type, img_size, patch_size
        )
        
        # Determine which blocks use Mamba
        if use_mamba_blocks == 'all':
            mamba_blocks = list(range(depth))
        elif isinstance(use_mamba_blocks, list):
            mamba_blocks = use_mamba_blocks
        else:
            mamba_blocks = []
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ExperimentalBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                attention_type=attention_type,
                mlp_type=mlp_type,
                use_mamba=(i in mamba_blocks),
                patch_size=int(math.sqrt(num_patches)),
                **kwargs
            )
            for i in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if hasattr(self.pos_embed_layer, 'pos_embed') and self.pos_embed_layer.pos_embed is not None:
            if self.pos_embed_layer.pos_embed.requires_grad:
                nn.init.trunc_normal_(self.pos_embed_layer.pos_embed, std=0.02)
        
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
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = self.pos_embed_layer(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x[:, 0])  # Use cls token for classification
        return x


def create_experimental_vit(
    architecture_type='fourier',
    img_size=224,
    num_classes=1000,
    **kwargs
):
    """
    Factory function to create different experimental ViT variants.
    """
    configs = {
        'fourier': {
            'attention_type': 'fourier',
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'fourier_scale': 1.0
        },
        'elfatt': {
            'attention_type': 'elfatt',
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12
        },
        'mamba': {
            'attention_type': 'standard',
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'use_mamba_blocks': 'all',
            'd_state': 16,
            'd_conv': 4,
            'expand': 2
        },
        'kan': {
            'attention_type': 'kan',
            'mlp_type': 'kan',
            'embed_dim': 384,  # Smaller for efficiency
            'depth': 6,
            'num_heads': 6,
            'grid_size': 5,
            'spline_order': 3
        },
        'hybrid': {
            'attention_type': 'hybrid',
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'use_conv_stem': True
        },
        'mixed': {
            'attention_type': 'fourier',
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'use_mamba_blocks': [8, 9, 10, 11],  # Last 4 blocks
            'use_conv_stem': True,
            'pos_type': 'sincos'
        }
    }
    
    config = configs.get(architecture_type, configs['fourier'])
    config.update(kwargs)
    
    return ExperimentalVisionTransformer(
        img_size=img_size,
        num_classes=num_classes,
        **config
    )


if __name__ == "__main__":
    # Test different experimental architectures
    batch_size = 2
    img_size = 224
    test_input = torch.randn(batch_size, 3, img_size, img_size)
    
    architectures = ['fourier', 'elfatt', 'mamba', 'kan', 'hybrid', 'mixed']
    
    for arch in architectures:
        try:
            print(f"\nTesting {arch} architecture:")
            model = create_experimental_vit(
                architecture_type=arch,
                img_size=img_size,
                num_classes=1000
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            output = model(test_input)
            print(f"  Output shape: {output.shape}")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  ✓ {arch} test passed!")
            
        except Exception as e:
            print(f"  ✗ {arch} test failed: {e}")
    
    print("\nExperimental ViT testing completed!")
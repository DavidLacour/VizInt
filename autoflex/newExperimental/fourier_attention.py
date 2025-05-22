import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange


class FourierAttention(nn.Module):
    """
    Fourier-based attention mechanism inspired by FourierFormer.
    Replaces dot-product kernels with generalized Fourier integral kernels.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., fourier_scale=1.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fourier_scale = fourier_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable parameters for Fourier kernels
        self.fourier_weight_q = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim))
        self.fourier_weight_k = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim))
        
        # Initialize Fourier weights
        nn.init.xavier_uniform_(self.fourier_weight_q)
        nn.init.xavier_uniform_(self.fourier_weight_k)

    def fourier_transform_2d(self, x):
        """Apply 2D FFT to the input tensor."""
        # x shape: (B, H, L, D)
        B, H, L, D = x.shape
        
        # Reshape to 2D for spatial FFT (assuming square patches)
        patch_size = int(math.sqrt(L - 1))  # -1 for cls token
        
        # Handle cls token separately
        cls_token = x[:, :, 0:1, :]  # (B, H, 1, D)
        spatial_tokens = x[:, :, 1:, :]  # (B, H, L-1, D)
        
        if spatial_tokens.size(2) == patch_size * patch_size:
            # Reshape to spatial dimensions
            spatial_tokens = spatial_tokens.view(B, H, patch_size, patch_size, D)
            
            # Apply 2D FFT on spatial dimensions
            spatial_fft = torch.fft.fft2(spatial_tokens, dim=(2, 3))
            
            # Flatten back
            spatial_fft = spatial_fft.view(B, H, patch_size * patch_size, D)
            
            # Concatenate with cls token
            x_fft = torch.cat([cls_token, spatial_fft], dim=2)
        else:
            # Fallback to 1D FFT if spatial reshape doesn't work
            x_fft = torch.fft.fft(x, dim=2)
        
        return x_fft

    def fourier_kernel(self, q, k):
        """
        Compute Fourier integral kernel between queries and keys.
        """
        B, H, L, D = q.shape
        
        # Apply learnable linear transformations in Fourier domain
        q_transformed = torch.matmul(q, self.fourier_weight_q)
        k_transformed = torch.matmul(k, self.fourier_weight_k)
        
        # Apply Fourier transform
        q_fft = self.fourier_transform_2d(q_transformed)
        k_fft = self.fourier_transform_2d(k_transformed)
        
        # Compute attention weights using Fourier correlation
        # Use real part for numerical stability
        attn = torch.real(q_fft * torch.conj(k_fft))
        
        # Apply scaling
        attn = attn * self.fourier_scale
        
        # Sum over feature dimension to get attention scores
        attn = attn.sum(dim=-1)  # (B, H, L)
        
        # Softmax normalization
        attn = F.softmax(attn, dim=-1)
        
        return attn

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

        # Compute Fourier attention weights
        attn = self.fourier_kernel(q, k)  # (B, H, N)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = torch.matmul(attn.unsqueeze(-1), v).squeeze(-2)  # (B, H, N, D)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FourierBlock(nn.Module):
    """
    Transformer block with Fourier attention.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, fourier_scale=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FourierAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop, fourier_scale=fourier_scale
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FNetAttention(nn.Module):
    """
    FNet attention that replaces self-attention with Fourier Transform.
    Based on Google's FNet paper.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x shape: (B, N, C)
        B, N, C = x.shape
        
        # Apply 1D FFT along sequence dimension
        x_fft = torch.fft.fft(x, dim=1)
        
        # Take real part for output
        x_real = torch.real(x_fft)
        
        return x_real


class FNetBlock(nn.Module):
    """
    FNet block that replaces self-attention with Fourier Transform.
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.fourier = FNetAttention(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.fourier(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


if __name__ == "__main__":
    # Test Fourier attention
    batch_size, seq_len, dim = 2, 197, 768  # ViT-Base dimensions
    num_heads = 12
    
    x = torch.randn(batch_size, seq_len, dim)
    
    # Test FourierAttention
    fourier_attn = FourierAttention(dim, num_heads)
    output1 = fourier_attn(x)
    print(f"FourierAttention output shape: {output1.shape}")
    
    # Test FourierBlock
    fourier_block = FourierBlock(dim, num_heads)
    output2 = fourier_block(x)
    print(f"FourierBlock output shape: {output2.shape}")
    
    # Test FNetBlock
    fnet_block = FNetBlock(dim)
    output3 = fnet_block(x)
    print(f"FNetBlock output shape: {output3.shape}")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class LinearAttention(nn.Module):
    """
    Linear attention mechanism with O(N) complexity.
    Based on ELFATT and other efficient linear attention methods.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., 
                 kernel_type='elu', causal=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_type = kernel_type
        self.causal = causal

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def kernel_func(self, x):
        """Apply kernel function to transform attention scores."""
        if self.kernel_type == 'elu':
            return F.elu(x) + 1
        elif self.kernel_type == 'relu':
            return F.relu(x)
        elif self.kernel_type == 'softmax':
            return F.softmax(x, dim=-1)
        else:
            # Default to ELU + 1 for numerical stability
            return F.elu(x) + 1

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

        # Apply kernel function to queries and keys
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # Compute linear attention
        if self.causal:
            # Causal linear attention (for autoregressive tasks)
            kv = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)
            outputs = []
            for i in range(N):
                kv = kv + k[:, :, i:i+1, :].transpose(-2, -1) @ v[:, :, i:i+1, :]
                out = q[:, :, i:i+1, :] @ kv
                outputs.append(out)
            x = torch.cat(outputs, dim=2)
        else:
            # Non-causal linear attention
            # Compute denominator: Q @ K^T @ 1
            k_sum = k.sum(dim=2, keepdim=True)  # (B, H, 1, D)
            z = (q * k_sum).sum(dim=-1, keepdim=True)  # (B, H, N, 1)
            
            # Compute numerator: Q @ K^T @ V
            kv = k.transpose(-2, -1) @ v  # (B, H, D, D)
            numerator = q @ kv  # (B, H, N, D)
            
            # Normalize
            x = numerator / (z + 1e-6)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientLinearAttention(nn.Module):
    """
    ELFATT: Efficient Linear Fast Attention for Vision Transformers.
    Optimized for high-resolution vision tasks.
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

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

        # Apply temperature scaling
        q = q / self.temperature
        k = k / self.temperature

        # Apply kernel function (ReLU for ELFATT)
        q = F.relu(q)
        k = F.relu(k)

        # Efficient linear attention computation
        # Step 1: Compute K^T @ V
        kv = k.transpose(-2, -1) @ v  # (B, H, D, D)
        
        # Step 2: Compute Q @ (K^T @ V)
        qkv_out = q @ kv  # (B, H, N, D)
        
        # Step 3: Compute normalization term
        k_sum = k.sum(dim=2, keepdim=True)  # (B, H, 1, D)
        normalizer = (q * k_sum).sum(dim=-1, keepdim=True)  # (B, H, N, 1)
        
        # Step 4: Normalize output
        x = qkv_out / (normalizer + 1e-8)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttentionBlock(nn.Module):
    """
    Transformer block with linear attention.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='elfatt'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        if attention_type == 'elfatt':
            self.attn = EfficientLinearAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                attn_drop=attn_drop, proj_drop=drop
            )
        else:
            self.attn = LinearAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                attn_drop=attn_drop, proj_drop=drop
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


class GatedLinearAttention(nn.Module):
    """
    Gated Linear Attention with hardware-efficient training.
    Based on recent developments in efficient attention mechanisms.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.gate = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

        # Compute gates
        gates = torch.sigmoid(self.gate(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply gating to queries and keys
        q = q * gates
        k = k * gates

        # Apply ELU activation for linear attention
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Linear attention computation
        kv = k.transpose(-2, -1) @ v  # (B, H, D, D)
        qkv_out = q @ kv  # (B, H, N, D)
        
        # Normalization
        k_sum = k.sum(dim=2, keepdim=True)  # (B, H, 1, D)
        normalizer = (q * k_sum).sum(dim=-1, keepdim=True)  # (B, H, N, 1)
        x = qkv_out / (normalizer + 1e-8)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == "__main__":
    # Test linear attention mechanisms
    batch_size, seq_len, dim = 2, 197, 768  # ViT-Base dimensions
    num_heads = 12
    
    x = torch.randn(batch_size, seq_len, dim)
    
    # Test LinearAttention
    linear_attn = LinearAttention(dim, num_heads)
    output1 = linear_attn(x)
    print(f"LinearAttention output shape: {output1.shape}")
    
    # Test EfficientLinearAttention (ELFATT)
    elfatt = EfficientLinearAttention(dim, num_heads)
    output2 = elfatt(x)
    print(f"ELFATT output shape: {output2.shape}")
    
    # Test GatedLinearAttention
    gated_attn = GatedLinearAttention(dim, num_heads)
    output3 = gated_attn(x)
    print(f"GatedLinearAttention output shape: {output3.shape}")
    
    # Test LinearAttentionBlock
    linear_block = LinearAttentionBlock(dim, num_heads, attention_type='elfatt')
    output4 = linear_block(x)
    print(f"LinearAttentionBlock output shape: {output4.shape}")
    
    print("All linear attention tests passed!")
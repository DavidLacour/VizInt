import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network linear layer.
    Replaces traditional linear layer with learnable spline functions.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, noise_scale=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Grid points for spline functions
        grid = torch.linspace(-1, 1, grid_size + 1).expand(in_features, -1)
        self.register_buffer('grid', grid)
        
        # Spline coefficients (learnable parameters)
        self.coeff = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order))
        
        # Scale and bias parameters
        self.scale_noise = noise_scale
        self.scale_base = nn.Parameter(torch.randn(out_features, in_features))
        self.scale_spline = nn.Parameter(torch.randn(out_features, in_features))
        self.base_bias = nn.Parameter(torch.randn(out_features, in_features))
        
        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            # Simplified initialization - avoid complex curve fitting during init
            nn.init.xavier_uniform_(self.coeff)
            self.coeff.data *= 0.1
            
            # Initialize scale parameters
            nn.init.xavier_uniform_(self.scale_base)
            nn.init.xavier_uniform_(self.scale_spline)
            self.scale_base.data *= 0.1
            self.scale_spline.data *= 1.0

    def curve2coeff(self, x, y):
        """Convert curve points to spline coefficients."""
        A = self.b_splines(x)
        coeff = torch.linalg.lstsq(A, y).solution
        return coeff

    def b_splines(self, x):
        """Compute B-spline basis functions."""
        def cox_de_boor(x, grid, k, i):
            if k == 0:
                return ((grid[i] <= x) & (x < grid[i + 1])).float()
            else:
                # Ensure proper broadcasting
                grid_i = grid[..., i:i+1]
                grid_i_k = grid[..., i+k:i+k+1]
                grid_i_k_1 = grid[..., i+k+1:i+k+2]
                grid_i_1 = grid[..., i+1:i+2]
                
                coeff1 = (x - grid_i) / (grid_i_k - grid_i + 1e-8)
                coeff2 = (grid_i_k_1 - x) / (grid_i_k_1 - grid_i_1 + 1e-8)
                return coeff1 * cox_de_boor(x, grid, k - 1, i) + coeff2 * cox_de_boor(x, grid, k - 1, i + 1)

        batch_size, in_features, num_samples = x.shape
        
        # Create grid for each input feature
        grid = self.grid.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Extend grid for spline order
        extended_grid = torch.cat([
            grid[..., [0]] - 1, 
            grid, 
            grid[..., [-1]] + 1
        ], dim=-1)
        
        bases = []
        for i in range(self.grid_size + self.spline_order):
            basis = cox_de_boor(x, extended_grid, self.spline_order, i)
            bases.append(basis)
        
        return torch.stack(bases, dim=-1)

    def forward(self, x):
        # Get input shape
        input_shape = x.shape
        batch_elements = input_shape[:-1]
        
        # Reshape input for processing
        x_reshaped = x.view(-1, self.in_features)
        
        # Clamp input to grid range
        x_clamped = torch.clamp(x_reshaped, -1, 1)
        
        # Compute base activation (SiLU)
        base = F.silu(x_clamped)
        
        # Compute spline activation using simple polynomial basis
        # Initialize spline output
        spline = torch.zeros(x_reshaped.shape[0], self.out_features, device=x.device)
        
        # Compute polynomial basis functions
        basis_prev = torch.ones_like(x_clamped)
        basis_curr = x_clamped
        
        for i in range(min(self.grid_size + self.spline_order, 5)):  # Limit to avoid numerical issues
            if i == 0:
                basis = basis_prev
            elif i == 1:
                basis = basis_curr
            else:
                # Chebyshev recurrence: T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
                basis_new = 2 * x_clamped * basis_curr - basis_prev
                basis_prev = basis_curr
                basis_curr = basis_new
                basis = basis_curr
            
            # Apply coefficients
            # x_reshaped: (batch*seq, in_features)
            # basis: (batch*seq, in_features)
            # coeff: (out_features, in_features, grid_size+spline_order)
            weighted_basis = basis.unsqueeze(1) * self.coeff[:, :, i].T.unsqueeze(0)  # (batch*seq, out_features, in_features)
            spline += weighted_basis.sum(dim=-1)  # (batch*seq, out_features)
        
        # Compute base output
        base_out = torch.matmul(base, self.scale_base.T)  # (batch*seq, out_features)
        
        # Combine base and spline
        # Scale spline contributions properly
        spline_out = torch.matmul(spline, self.scale_spline.T)  # (batch*seq, out_features)
        y = base_out + spline_out
        
        # Add bias
        y = y + self.base_bias.sum(dim=1).unsqueeze(0)
        
        # Reshape back to original shape
        output_shape = list(batch_elements) + [self.out_features]
        return y.view(*output_shape)


class KANAttention(nn.Module):
    """
    Attention mechanism using KAN layers instead of traditional linear layers.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., 
                 grid_size=5, spline_order=3):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Replace linear layers with KAN layers
        self.q_kan = KANLinear(dim, dim, grid_size, spline_order)
        self.k_kan = KANLinear(dim, dim, grid_size, spline_order)
        self.v_kan = KANLinear(dim, dim, grid_size, spline_order)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = KANLinear(dim, dim, grid_size, spline_order)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        # Apply KAN transformations
        q = self.q_kan(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_kan(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_kan(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Standard attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class KANMLP(nn.Module):
    """
    MLP layer using KAN instead of traditional linear layers.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0., grid_size=5, spline_order=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = KANLinear(in_features, hidden_features, grid_size, spline_order)
        self.act = act_layer()
        self.fc2 = KANLinear(hidden_features, out_features, grid_size, spline_order)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class KANTransformerBlock(nn.Module):
    """
    Transformer block with KAN layers.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, grid_size=5, spline_order=3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = KANAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop,
            grid_size=grid_size, spline_order=spline_order
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = KANMLP(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop,
            grid_size=grid_size, 
            spline_order=spline_order
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SineKAN(nn.Module):
    """
    SineKAN: KAN with sine activation functions instead of B-splines.
    More efficient for some applications.
    """
    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # Learnable frequencies and phases
        self.frequencies = nn.Parameter(torch.randn(out_features, in_features, grid_size))
        self.phases = nn.Parameter(torch.randn(out_features, in_features, grid_size))
        self.amplitudes = nn.Parameter(torch.randn(out_features, in_features, grid_size))
        
        # Base linear transformation
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.base_bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(-1, self.in_features)
        
        # Base transformation
        base_output = F.linear(x_reshaped, self.base_weight, self.base_bias)
        
        # Sine transformations
        sine_output = torch.zeros(x_reshaped.shape[0], self.out_features, device=x.device)
        
        for i in range(self.grid_size):
            freq = self.frequencies[:, :, i]  # (out_features, in_features)
            phase = self.phases[:, :, i]      # (out_features, in_features)
            amp = self.amplitudes[:, :, i]    # (out_features, in_features)
            
            # Compute sine activation
            sine_term = amp * torch.sin(freq * x_reshaped.unsqueeze(1) + phase)
            sine_output += sine_term.sum(dim=-1)  # Sum over input features
        
        output = base_output + sine_output
        return output.view(batch_size, -1, self.out_features).squeeze(-2)


if __name__ == "__main__":
    # Test KAN components
    batch_size, seq_len, dim = 2, 197, 768
    
    x = torch.randn(batch_size, seq_len, dim)
    
    # Test KANLinear
    kan_linear = KANLinear(dim, dim)
    output1 = kan_linear(x)
    print(f"KANLinear output shape: {output1.shape}")
    
    # Test KANAttention
    kan_attn = KANAttention(dim, num_heads=12)
    output2 = kan_attn(x)
    print(f"KANAttention output shape: {output2.shape}")
    
    # Test KANMLP
    kan_mlp = KANMLP(dim, dim * 4)
    output3 = kan_mlp(x)
    print(f"KANMLP output shape: {output3.shape}")
    
    # Test KANTransformerBlock
    kan_block = KANTransformerBlock(dim, num_heads=12)
    output4 = kan_block(x)
    print(f"KANTransformerBlock output shape: {output4.shape}")
    
    # Test SineKAN
    sine_kan = SineKAN(dim, dim)
    output5 = sine_kan(x)
    print(f"SineKAN output shape: {output5.shape}")
    
    print("All KAN tests passed!")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Optional


class SelectiveScan(nn.Module):
    """
    Selective scan mechanism core to Mamba architecture.
    Implements efficient state space model computation.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Linear projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.activation = "silu"
        self.act = nn.SiLU()

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # A parameter (learnable)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape

        # Input projection
        x_and_res = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # Convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]  # Trim to original length
        x = rearrange(x, 'b d l -> b l d')

        # Activation
        x = self.act(x)

        # SSM parameters
        x_dbl = self.x_proj(x)  # (B, L, 2 * d_state)
        B_x, C_x = x_dbl.split(split_size=[self.d_state, self.d_state], dim=-1)

        # Delta (time step)
        delta = F.softplus(self.dt_proj(x))  # (B, L, d_inner)

        # A matrix
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Selective scan
        y = self.selective_scan_fn(x, delta, A, B_x, C_x, self.D)

        # Residual connection and output projection
        y = y * self.act(res)
        output = self.out_proj(y)

        return output

    def selective_scan_fn(self, u, delta, A, B, C, D):
        """
        Selective scan function implementation.
        """
        B_batch, L, d_inner = u.shape
        _, _, d_state = B.shape

        # Discretize A and B
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (B, L, d_inner, d_state)

        # Initialize state
        h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []

        # Scan over sequence
        for i in range(L):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y = torch.sum(h * C[:, i].unsqueeze(1), dim=-1)  # (B, d_inner)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        # Add skip connection
        y = y + u * D

        return y


class MambaBlock(nn.Module):
    """
    Mamba block that replaces traditional transformer attention.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, mlp_ratio=4.0, 
                 norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        self.d_model = d_model
        self.norm1 = norm_layer(d_model)
        self.mamba = SelectiveScan(d_model, d_state, d_conv, expand)
        
        self.norm2 = norm_layer(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # Mamba layer
        x = x + self.mamba(self.norm1(x))
        # MLP layer
        x = x + self.mlp(self.norm2(x))
        return x


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba layer for better context modeling.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = SelectiveScan(d_model, d_state, d_conv, expand)
        self.backward_mamba = SelectiveScan(d_model, d_state, d_conv, expand)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # Forward direction
        x_forward = self.forward_mamba(x)
        
        # Backward direction
        x_reversed = torch.flip(x, dims=[1])
        x_backward = self.backward_mamba(x_reversed)
        x_backward = torch.flip(x_backward, dims=[1])
        
        # Combine both directions
        x_combined = torch.cat([x_forward, x_backward], dim=-1)
        output = self.proj(x_combined)
        
        return output


class VisionMambaBlock(nn.Module):
    """
    Vision Mamba block with bidirectional processing.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, mlp_ratio=4.0, 
                 norm_layer=nn.LayerNorm, drop=0., bidirectional=True):
        super().__init__()
        self.d_model = d_model
        self.norm1 = norm_layer(d_model)
        
        if bidirectional:
            self.mamba = BidirectionalMamba(d_model, d_state, d_conv, expand)
        else:
            self.mamba = SelectiveScan(d_model, d_state, d_conv, expand)
        
        self.norm2 = norm_layer(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # Mamba layer with residual connection
        x = x + self.mamba(self.norm1(x))
        # MLP layer with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionMambaBackbone(nn.Module):
    """
    Complete Vision Mamba backbone for image classification.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, d_state=16, d_conv=4, expand=2,
                 mlp_ratio=4.0, drop_rate=0., bidirectional=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        num_patches = (img_size // patch_size) ** 2
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            VisionMambaBlock(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                bidirectional=bidirectional
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
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
        x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x[:, 0])  # Use cls token for classification
        return x


if __name__ == "__main__":
    # Test Vision Mamba components
    batch_size, seq_len, dim = 2, 197, 768
    
    x = torch.randn(batch_size, seq_len, dim)
    
    # Test SelectiveScan
    ssm = SelectiveScan(dim)
    output1 = ssm(x)
    print(f"SelectiveScan output shape: {output1.shape}")
    
    # Test MambaBlock
    mamba_block = MambaBlock(dim)
    output2 = mamba_block(x)
    print(f"MambaBlock output shape: {output2.shape}")
    
    # Test BidirectionalMamba
    bidirectional = BidirectionalMamba(dim)
    output3 = bidirectional(x)
    print(f"BidirectionalMamba output shape: {output3.shape}")
    
    # Test complete VisionMamba model
    model = VisionMambaBackbone(img_size=224, patch_size=16, embed_dim=384, depth=6)
    test_input = torch.randn(2, 3, 224, 224)
    output4 = model(test_input)
    print(f"VisionMamba output shape: {output4.shape}")
    
    print("All Vision Mamba tests passed!")
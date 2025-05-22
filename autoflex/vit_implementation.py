import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# Import from our combined utilities file
from transformer_utils import (
    LayerNorm, 
    TransformerTrunk, 
    build_2d_sincos_posemb,
    set_seed
)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding with optional ResNet-style stem.
    
    Args:
        img_size: Image size.
        patch_size: Patch size.
        in_chans: Number of input channels.
        embed_dim: Embedding dimension.
        use_resnet_stem: Whether to use a ResNet-style stem.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, use_resnet_stem=True):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        if use_resnet_stem:
            # ResNet-style stem: conv-bn-relu-conv-bn-relu
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.ReLU(),
                nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.ReLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=patch_size[0] // 2, stride=patch_size[0] // 2)
            )
        else:
            # Original ViT: single convolution
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
            )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer with ResNet-style connections.
    
    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of input channels.
        num_classes: Number of classes for classification head.
        embed_dim: Embedding dimension.
        depth: Depth of transformer.
        head_dim: Dimension of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        use_resnet_stem: Whether to use ResNet-style stem for patching.
        global_pool: Whether to use global pooling for classification head.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        head_dim=64,
        mlp_ratio=4.0,
        use_bias=False,
        use_resnet_stem=True,
        global_pool='token',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.global_pool = global_pool
        
        # Set seeds for reproducibility
        set_seed()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim,
            use_resnet_stem=use_resnet_stem
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Add learnable sinusoidal position embedding 
        # Add +1 to account for the class token
        self.pos_embed = nn.Parameter(
            torch.cat([
                torch.zeros(1, 1, embed_dim),  # Class token position embedding
                build_2d_sincos_posemb(
                    h=self.patch_embed.grid_size[0],
                    w=self.patch_embed.grid_size[1],
                    embed_dim=embed_dim
                )
            ], dim=1),
            requires_grad=True  # Make the embeddings learnable
        )
        
        # Optional ResNet-style stem is handled in the PatchEmbed class
            
        # Transformer encoder
        self.norm_pre = LayerNorm(embed_dim, bias=use_bias)
        self.transformer = TransformerTrunk(
            dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            use_bias=use_bias
        )
        
        # Classification head
        self.norm = LayerNorm(embed_dim, bias=use_bias)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.init_weights()
        
    def init_weights(self):
        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize patch embedding
        nn.init.normal_(self.patch_embed.proj[-1].weight, std=0.02)
        
        # Initialize classification head
        if isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.bias)
            nn.init.xavier_uniform_(self.head.weight)
            
    def forward_features(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding (for both cls token and patches)
        x = x + self.pos_embed.expand(B, -1, -1)
        
        # Apply transformer directly - no need for additional blocks
        # since TransformerTrunk already contains the Block instances
        x = self.norm_pre(x)
        x = self.transformer(x)
        x = self.norm(x)
        
        return x
    
    def forward_head(self, x):
        if self.global_pool == 'token':
            x = x[:, 0]  # Use cls token
        elif self.global_pool == 'avg':
            x = x[:, 1:].mean(dim=1)  # Global avg pool without cls token
        
        x = self.head(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


# Example usage
def create_vit_model(
    img_size=224,
    patch_size=16, 
    in_chans=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    head_dim=64,
    mlp_ratio=4.0,
    use_resnet_stem=True
):
    """Create a ViT model with the specified parameters."""
    # Set seed for reproducibility
    set_seed()
    
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        head_dim=head_dim,
        mlp_ratio=mlp_ratio,
        use_resnet_stem=use_resnet_stem
    )
    
    return model


if __name__ == "__main__":
    # Set seed for entire script
    set_seed()
    
    # Create model
    model = create_vit_model()
    
    # Test with a random input
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f"Model output shape: {output.shape}")

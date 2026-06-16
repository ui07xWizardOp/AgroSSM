import torch
import torch.nn as nn
from typing import Optional

class MultiHeadLatentAttention(nn.Module):
    """
    Compresses coarse Key-Value pairs into a low-dimensional latent space,
    then fine-resolution queries attend to this compressed context.

    Used for: letting 10m satellite tokens query 31km weather context
    without spatially upsampling the weather data.
    """
    def __init__(self, embed_dim: int = 256, latent_dim: int = 32, num_heads: int = 8):
        super().__init__()
        self.compress = nn.Linear(embed_dim, latent_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, kdim=latent_dim, vdim=latent_dim, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, fine_tokens: torch.Tensor, coarse_tokens: torch.Tensor, coarse_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # fine_tokens: [B, N_fine, embed_dim]
        # coarse_tokens: [B, N_coarse, embed_dim]
        # coarse_mask: [B, N_coarse] (1=real, 0=pad)

        latent_kv = self.compress(coarse_tokens) # [B, N_coarse, latent_dim]

        key_padding_mask = None
        if coarse_mask is not None:
            key_padding_mask = (coarse_mask == 0.0)

        normed_fine = self.norm1(fine_tokens)

        attn_out, _ = self.attn(normed_fine, latent_kv, latent_kv, key_padding_mask=key_padding_mask, need_weights=False)

        x = fine_tokens + attn_out
        x = x + self.ffn(self.norm2(x))

        return x

import torch
import torch.nn as nn
from typing import Optional

class SlidingWindowAttentionBlock(nn.Module):
    """
    Local sliding-window multi-head attention.

    Each token attends only to tokens within a window of `window_size` positions.
    Every `global_every_n`-th instance uses full (global) attention instead.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, window_size: int = 5, is_global: bool = False):
        super().__init__()
        self.window_size = window_size
        self.is_global = is_global

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, N, embed_dim]
        # mask: [B, N] (1=real, 0=pad)
        B, N, _ = x.shape

        attn_mask = None
        if not self.is_global:
            # Create a diagonal band mask
            # shape: [N, N]
            idx = torch.arange(N, device=x.device)
            dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
            # True where attention is NOT allowed (PyTorch MHA mask convention)
            attn_mask = (dist > self.window_size // 2)

        key_padding_mask = None
        if mask is not None:
            # True where padding is (PyTorch MHA convention)
            key_padding_mask = (mask == 0.0)

        normed_x = self.norm1(x)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        return x

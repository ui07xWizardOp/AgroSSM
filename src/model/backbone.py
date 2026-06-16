import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from src.model.ssm_block import PhysicalSSMBlock
from src.model.swa_block import SlidingWindowAttentionBlock
from src.model.mla_block import MultiHeadLatentAttention

class HybridTemporalBackbone(nn.Module):
    """
    Interleaves SSM, SWA, and MLA blocks in the pattern:
        SSM -> SWA -> SSM -> MLA -> SSM -> SWA -> SSM -> MLA -> ...
    """
    def __init__(self, num_blocks: int = 12, embed_dim: int = 256, ssm_state_dim: int = 64,
                 swa_window_size: int = 5, mla_latent_dim: int = 32):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.block_types = []

        for i in range(num_blocks):
            pattern_idx = i % 4
            if pattern_idx == 0 or pattern_idx == 2:
                # SSM
                self.blocks.append(PhysicalSSMBlock(embed_dim, ssm_state_dim))
                self.block_types.append("SSM")
            elif pattern_idx == 1:
                # SWA
                # global_every_n logic can be implemented by passing is_global based on block index
                is_global = (i // 4) % 4 == 3 # arbitrary global_every_n logic based on config, let's say every 4th SWA
                self.blocks.append(SlidingWindowAttentionBlock(embed_dim, window_size=swa_window_size, is_global=is_global))
                self.block_types.append("SWA")
            elif pattern_idx == 3:
                # MLA
                self.blocks.append(MultiHeadLatentAttention(embed_dim, latent_dim=mla_latent_dim))
                self.block_types.append("MLA")

    def forward(self, fine_tokens: torch.Tensor, coarse_tokens: torch.Tensor,
                fine_mask: Optional[torch.Tensor] = None, coarse_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        last_substates = None

        x = fine_tokens

        for i, block in enumerate(self.blocks):
            b_type = self.block_types[i]

            if b_type == "SSM":
                x, substates = block(x)
                last_substates = substates
            elif b_type == "SWA":
                x = block(x, mask=fine_mask)
            elif b_type == "MLA":
                x = block(x, coarse_tokens, coarse_mask=coarse_mask)

        return x, last_substates

import torch
import torch.nn as nn

class MaskedReconstructionLoss(nn.Module):
    """
    Masked spatiotemporal reconstruction. Mask 75% of spatial tokens, 20% of full time steps. L1 loss on masked tokens only.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction='none')

    def forward(self, pred_tokens: torch.Tensor, target_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred_tokens, target_tokens: [B, N, embed_dim]
        # mask: [B, N] where 1 indicates masked tokens to compute loss on

        loss = self.loss_fn(pred_tokens, target_tokens) # [B, N, embed_dim]
        loss = loss.mean(dim=-1) # [B, N]

        # Only compute loss on masked tokens
        mask = mask.float()
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return masked_loss

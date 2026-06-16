import torch
import torch.nn as nn

class PhenologyTransitionLoss(nn.Module):
    """
    Phenology transition prediction. Given current state, predict days until next NDVI inflection point. Huber loss.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.HuberLoss(reduction='mean')

    def forward(self, phenology_pred: torch.Tensor, phenology_label: torch.Tensor) -> torch.Tensor:
        # phenology_pred: [B, T, 1]
        # phenology_label: [B, T]
        pred = phenology_pred.squeeze(-1)
        return self.loss_fn(pred, phenology_label)

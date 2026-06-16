import torch
import torch.nn as nn

class CropStressHead(nn.Module):
    """
    Spatial binary classification: P(stress) per spatial token per time step.
    Output: [B, T, 4, 4]
    """
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        # z_t: [B, T, N_spatial, embed_dim] where N_spatial = 16
        B, T, N, D = z_t.shape
        assert N == 16

        logits = self.mlp(z_t).squeeze(-1) # [B, T, 16]
        # Reshape to [B, T, 4, 4]
        return logits.view(B, T, 4, 4)

class PhenologyHead(nn.Module):
    """Regression: days until next phenological transition. Output: [B, T, 1]."""
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        # z_t could be spatially pooled first: [B, T, embed_dim]
        return self.mlp(z_t)

class YieldRiskHead(nn.Module):
    """Regression: yield anomaly %. Input: z_T (last time step). Output: [B, 1]."""
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_T: torch.Tensor) -> torch.Tensor:
        return self.mlp(z_T)

class DroughtTrajectoryHead(nn.Module):
    """Multi-step regression: stress at t+1, t+2, t+3. Input: z_t. Output: [B, T, 3]."""
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.mlp(z_t)

class ResidualWaterHead(nn.Module):
    """Regression with uncertainty: I_latent mean + log_variance. Input: z_t. Output: [B, T, 2]."""
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.mlp(z_t)

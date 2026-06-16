import torch
import torch.nn as nn

class DroughtTrajectoryLoss(nn.Module):
    """
    Multi-horizon drought-stress trajectory. Predict NDVI anomaly at t+1, t+2, t+3. MSE with exponential time discounting.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='none')
        # Exponential discounting: gamma^k where k=0,1,2
        gamma = 0.8
        self.register_buffer('discount', torch.tensor([gamma**0, gamma**1, gamma**2]))

    def forward(self, traj_pred: torch.Tensor, traj_target: torch.Tensor) -> torch.Tensor:
        # traj_pred: [B, T, 3]
        # traj_target: [B, T, 3]

        loss = self.loss_fn(traj_pred, traj_target) # [B, T, 3]

        # Apply discounting
        loss = loss * self.discount.unsqueeze(0).unsqueeze(0)

        return loss.mean()

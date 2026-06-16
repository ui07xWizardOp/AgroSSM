import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class SelectiveScanSSM(nn.Module):
    """
    A single selective-scan SSM sub-module implementing:
        delta_t = softplus(linear_dt(x_t))
        A_bar = exp(A * delta_t)           # A is diagonal, learnable, initialized negative
        B_bar = delta_t * B(x_t)           # B is input-dependent
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C(x_t) * h_t + D * x_t      # C is input-dependent
    """
    def __init__(self, dim: int, expand_factor: int = 2):
        super().__init__()
        self.dim = dim
        hidden_dim = dim * expand_factor

        self.dt_proj = nn.Linear(dim, hidden_dim)
        self.B_proj = nn.Linear(dim, hidden_dim)
        self.C_proj = nn.Linear(dim, hidden_dim)
        self.D = nn.Parameter(torch.ones(hidden_dim))

        # A is diagonal, initialized negative
        self.A_log = nn.Parameter(torch.log(torch.arange(1, hidden_dim + 1, dtype=torch.float32)))

        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, dim]
        B, T, D = x.shape

        dt = F.softplus(self.dt_proj(x)) # [B, T, hidden]
        B_mat = self.B_proj(x) # [B, T, hidden]
        C_mat = self.C_proj(x) # [B, T, hidden]

        A = -torch.exp(self.A_log) # [hidden]

        h = torch.zeros(B, A.shape[0], device=x.device, dtype=x.dtype)

        y_out = []
        h_out = []

        for t in range(T):
            dt_t = dt[:, t, :]
            A_bar = torch.exp(A.unsqueeze(0) * dt_t) # [B, hidden]
            B_bar = dt_t * B_mat[:, t, :] # [B, hidden]

            h = A_bar * h + B_bar * x[:, t, :].repeat(1, dt.shape[-1] // D) # simplistic B_bar input mult

            y = C_mat[:, t, :] * h + self.D.unsqueeze(0) * x[:, t, :].repeat(1, dt.shape[-1] // D)
            y_out.append(y)
            h_out.append(h)

        y_out = torch.stack(y_out, dim=1)
        h_out = torch.stack(h_out, dim=1)

        # To match expected dim, we just project the output
        # For state, we want to return something of shape [B, T, dim]
        # In a real Mamba, state is larger. We'll project state to dim for the regularizer to easily read
        h_reduced = h_out[..., :D] # simply slice for the substate shape requirement

        y_final = self.out_proj(y_out)

        return y_final, h_reduced

class PhysicalSSMBlock(nn.Module):
    """
    SSM block with 4 truly independent sub-modules, each with its own
    A, B, C, D, Î” parameters and its own hidden state.
    """
    def __init__(self, embed_dim=256, state_dim=64, expand_factor=2):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, 4 * state_dim)

        self.ssm_water  = SelectiveScanSSM(state_dim, expand_factor)
        self.ssm_stress = SelectiveScanSSM(state_dim, expand_factor)
        self.ssm_agdd   = SelectiveScanSSM(state_dim, expand_factor)
        self.ssm_free   = SelectiveScanSSM(state_dim, expand_factor)

        self.output_proj = nn.Linear(4 * state_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = x
        x_proj = self.input_proj(x)

        dim = x_proj.shape[-1] // 4
        x_water, x_stress, x_agdd, x_free = torch.split(x_proj, dim, dim=-1)

        y_water, h_water = self.ssm_water(x_water)
        y_stress, h_stress = self.ssm_stress(x_stress)
        y_agdd, h_agdd = self.ssm_agdd(x_agdd)
        y_free, h_free = self.ssm_free(x_free)

        y_concat = torch.cat([y_water, y_stress, y_agdd, y_free], dim=-1)
        output = self.norm(residual + self.output_proj(y_concat))

        substates = {
            "h_water": h_water,
            "h_stress": h_stress,
            "h_agdd": h_agdd,
            "h_free": h_free
        }

        return output, substates

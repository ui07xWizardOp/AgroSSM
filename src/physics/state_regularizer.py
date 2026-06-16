import torch
import torch.nn as nn
import torch.nn.functional as F

class WaterBalanceStateRegularizer(nn.Module):
    """
    Computes water-balance residual on h_water sub-states (H1/H2 unified formula).
    residual_t = decode_SM(h_water_t) - decode_SM(h_water_{t-1})
                 - (P_t + I_obs_t + I_latent_t - ET_t - R_t - D_t)
    """
    def __init__(self, decode_sm: nn.Module):
        super().__init__()
        self.decode_SM = decode_sm

    def forward(self, h_water, precipitation, et, i_obs, i_latent, soil_properties):
        # All inputs have shape [B, T] except soil [B, 7], h_water [B, T, state_dim]
        # Soil features: clay, sand, soc, phh2o, cec, bdod, ocd

        SM = self.decode_SM(h_water).squeeze(-1) # [B, T]

        delta_SM = SM[:, 1:] - SM[:, :-1] # [B, T-1]

        # We need inputs for t=1..T-1
        P = precipitation[:, 1:]
        ET = et[:, 1:]
        I_obs = i_obs[:, 1:]
        I_lat = i_latent[:, 1:]

        # Simplified R and D estimation based on soil
        clay = soil_properties[:, 0:1] # [B, 1]
        sand = soil_properties[:, 1:2]
        bdod = soil_properties[:, 5:6]

        # Dummy curve number logic based on clay/sand
        CN = 75.0 + 10.0 * (clay / 100.0) - 5.0 * (sand / 100.0)
        S = (25400.0 / CN) - 254.0

        # R = (P - 0.2*S)^2 / (P + 0.8*S) if P > 0.2*S else 0
        P_eff = torch.clamp(P - 0.2 * S, min=0.0)
        R = (P_eff ** 2) / (P + 0.8 * S + 1e-6)

        # Dummy K_sat logic based on bdod
        K_sat = 10.0 / (bdod + 0.1)
        # D = K_sat * exp(-alpha * SM_deficit) (let's simplify to a small factor of K_sat)
        D = K_sat * 0.1

        residual = delta_SM - (P + I_obs + I_lat - ET - R - D)
        loss = torch.mean(residual ** 2)
        return loss

class AGDDStateRegularizer(nn.Module):
    """
    Computes AGDD constraint on h_agdd sub-states.
    AGDD_t = AGDD_{t-1} + max(0, T_mean_t - T_base)
    """
    def __init__(self, decode_agdd: nn.Module):
        super().__init__()
        self.decode_AGDD = decode_agdd
        self.T_base = 10.0 # for maize

    def forward(self, h_agdd, temperature_mean, planting_doy, doy_sequence):
        # [B, T, state_dim], [B, T], [B], [B, T]
        AGDD_pred = self.decode_AGDD(h_agdd).squeeze(-1) # [B, T]

        # Calculate true AGDD
        GDD = torch.clamp(temperature_mean - self.T_base, min=0.0)
        # Only accumulate after planting DOY
        # doy_sequence: [B, T]
        mask = (doy_sequence >= planting_doy.unsqueeze(-1)).float()
        GDD = GDD * mask
        AGDD_true = torch.cumsum(GDD, dim=1) # [B, T]

        loss_fit = torch.mean((AGDD_pred - AGDD_true) ** 2)

        # Monotonicity penalty
        delta_AGDD = AGDD_pred[:, :-1] - AGDD_pred[:, 1:] # penalty if previous > current
        penalty = torch.mean(F.relu(delta_AGDD))

        return loss_fit + penalty

class StressAsymmetryRegularizer(nn.Module):
    """
    Computes asymmetry prior on h_stress sub-states.
    L_stress_asym = mean(relu(-delta_h_stress) * asymmetry_factor)
    """
    def __init__(self, decode_stress: nn.Module):
        super().__init__()
        self.decode_stress = decode_stress
        self.asymmetry_factor = 3.0

    def forward(self, h_stress):
        # [B, T, state_dim]
        stress_pred = self.decode_stress(h_stress).squeeze(-1) # [B, T]
        delta_stress = stress_pred[:, 1:] - stress_pred[:, :-1]

        loss = torch.mean(F.relu(-delta_stress) * self.asymmetry_factor)
        return loss

class SharedPhysicsDecoders(nn.Module):
    """Container for the shared decode functions across blocks/regularizers"""
    def __init__(self, state_dim: int = 64):
        super().__init__()
        self.decode_SM = nn.Sequential(
            nn.Linear(state_dim, 1),
            nn.Softplus()
        )
        self.decode_AGDD = nn.Sequential(
            nn.Linear(state_dim, 1),
            nn.Softplus()
        )
        self.decode_stress = nn.Linear(state_dim, 1)

        # Initialize positive weights for SM and AGDD
        nn.init.xavier_uniform_(self.decode_SM[0].weight)
        self.decode_SM[0].weight.data = torch.abs(self.decode_SM[0].weight.data)

        nn.init.xavier_uniform_(self.decode_AGDD[0].weight)
        self.decode_AGDD[0].weight.data = torch.abs(self.decode_AGDD[0].weight.data)

        nn.init.xavier_uniform_(self.decode_stress.weight)

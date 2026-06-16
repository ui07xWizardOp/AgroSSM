import torch
import torch.nn as nn
from typing import Dict, Any

from src.model.tokenizer import UnifiedTokenizer
from src.model.backbone import HybridTemporalBackbone
from src.model.moe import RegimeRoutedMoE
from src.model.heads import CropStressHead, PhenologyHead, YieldRiskHead, DroughtTrajectoryHead, ResidualWaterHead
from src.physics.state_regularizer import SharedPhysicsDecoders, WaterBalanceStateRegularizer, AGDDStateRegularizer, StressAsymmetryRegularizer

class AgroSSM(nn.Module):
    """
    AgroSSM: Compute-Efficient Process-Structured Agro-Ecosystem Model.
    """
    def __init__(self, embed_dim: int = 256, num_experts: int = 8, ssm_state_dim: int = 64):
        super().__init__()

        self.tokenizer = UnifiedTokenizer(embed_dim=embed_dim)

        self.backbone = HybridTemporalBackbone(embed_dim=embed_dim, ssm_state_dim=ssm_state_dim)

        self.moe = RegimeRoutedMoE(embed_dim=embed_dim, num_experts=num_experts)

        self.stress_head = CropStressHead(embed_dim=embed_dim)
        self.phenology_head = PhenologyHead(embed_dim=embed_dim)
        self.yield_head = YieldRiskHead(embed_dim=embed_dim)
        self.drought_traj_head = DroughtTrajectoryHead(embed_dim=embed_dim)
        self.residual_water_head = ResidualWaterHead(embed_dim=embed_dim)

        self.shared_decoders = SharedPhysicsDecoders(state_dim=ssm_state_dim)

        self.water_reg = WaterBalanceStateRegularizer(self.shared_decoders.decode_SM)
        self.agdd_reg = AGDDStateRegularizer(self.shared_decoders.decode_AGDD)
        self.stress_reg = StressAsymmetryRegularizer(self.shared_decoders.decode_stress)

    def forward(self, patch_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:

        # 1. Tokenizer
        tokens, token_masks = self.tokenizer(patch_dict)
        # tokens: [B, N_total, embed_dim]
        # We need to split into fine and coarse for the backbone
        # T*16 optical | T weather | 1 soil | 1 calendar
        B = tokens.shape[0]
        T = patch_dict["optical"].shape[1]

        fine_len = T * 16
        coarse_len = T + 2

        fine_tokens = tokens[:, :fine_len, :]
        fine_mask = token_masks[:, :fine_len]

        coarse_tokens = tokens[:, fine_len:, :]
        coarse_mask = token_masks[:, fine_len:]

        # 2. Backbone
        processed_fine, substates = self.backbone(fine_tokens, coarse_tokens, fine_mask=fine_mask, coarse_mask=coarse_mask)

        # 3. MoE
        moe_out, moe_lb_loss, expert_assignments = self.moe(processed_fine)

        # 4. Pooling temporal tokens
        # moe_out: [B, T*16, embed_dim] -> [B, T, 16, embed_dim]
        z_t_spatial = moe_out.view(B, T, 16, -1)
        z_t = z_t_spatial.mean(dim=2) # [B, T, embed_dim]

        # Last time step for yield
        z_T = z_t[:, -1, :] # [B, embed_dim]

        # 5. Heads
        stress_logits = self.stress_head(z_t_spatial) # [B, T, 4, 4]
        phenology_pred = self.phenology_head(z_t) # [B, T, 1]
        yield_pred = self.yield_head(z_T) # [B, 1]
        drought_traj = self.drought_traj_head(z_t) # [B, T, 3]
        residual_water = self.residual_water_head(z_t) # [B, T, 2]

        # 6. Physics Regularizers
        # Create dummy physical inputs based on weather/soil.
        # In a real impl, we extract the exact variables.
        # ERA5 order: t2m_min, t2m_max, t2m_mean, total_precipitation, u10, v10, d2m, ssrd, pet
        precipitation = patch_dict["weather"][:, :, 3]
        t2m_mean = patch_dict["weather"][:, :, 2]
        et = patch_dict["weather"][:, :, 8] # pet
        i_obs = (patch_dict["irrigation"].mean(dim=[-2, -1]) * patch_dict["irrigation_mask"].squeeze(-1)).unsqueeze(-1).repeat(1, T)
        i_latent = residual_water[..., 0] # mean prediction
        soil_properties = patch_dict["soil"]
        planting_doy = patch_dict["calendar"][:, 1]
        doy_sequence = patch_dict["doy_sequence"]

        # We must spatially pool the SSM sub-states to match the temporal sequence length (T)
        # substates are shape [B, T*16, state_dim]
        # We need [B, T, state_dim]
        pooled_substates = {}
        for k, v in substates.items():
            # v: [B, T*16, state_dim] -> [B, T, 16, state_dim] -> [B, T, state_dim]
            pooled_substates[k] = v.view(B, T, 16, -1).mean(dim=2)

        loss_water = self.water_reg(pooled_substates["h_water"], precipitation, et, i_obs, i_latent, soil_properties)
        loss_agdd = self.agdd_reg(pooled_substates["h_agdd"], t2m_mean, planting_doy, doy_sequence)
        loss_stress = self.stress_reg(pooled_substates["h_stress"])

        return {
            "stress_logits": stress_logits,
            "phenology_pred": phenology_pred,
            "yield_pred": yield_pred,
            "drought_traj": drought_traj,
            "residual_water": residual_water,
            "water_balance_loss": loss_water,
            "agdd_loss": loss_agdd,
            "stress_asymmetry_loss": loss_stress,
            "moe_load_balance_loss": moe_lb_loss,
            "substates": substates
        }

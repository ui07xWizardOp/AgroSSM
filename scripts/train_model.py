import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any

from src.model.agrossm import AgroSSM
from src.ssl.ssl_a_reconstruction import MaskedReconstructionLoss
from src.ssl.ssl_b_phenology import PhenologyTransitionLoss
from src.ssl.ssl_c_trajectory import DroughtTrajectoryLoss
from src.ssl.collapse_monitor import EmbeddingCollapseMonitor

def get_physics_lambda(epoch: int, lambda_max: float, warmup: int) -> float:
    return lambda_max * min(1.0, float(epoch) / float(warmup))

class AgroSSMLightning(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        embed_dim = config["model"]["embed_dim"]
        num_experts = config["model"]["moe"]["num_experts"]
        ssm_state_dim = config["model"]["backbone"]["ssm"]["state_dim"]

        self.model = AgroSSM(embed_dim, num_experts, ssm_state_dim)

        self.ssl_a = MaskedReconstructionLoss()
        self.ssl_b = PhenologyTransitionLoss()
        self.ssl_c = DroughtTrajectoryLoss()
        self.collapse_monitor = EmbeddingCollapseMonitor()

        # Supervised task losses
        # Using BCEWithLogits for stress because head outputs logits
        self.stress_loss = nn.BCEWithLogitsLoss()
        self.yield_loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)

        # Calculate individual losses

        # 1. Physics (State Regularizers)
        # Epoch can be obtained via self.current_epoch
        epoch = self.current_epoch
        phys_conf = self.config["training"]["physics"]
        warmup = phys_conf["warmup_epochs"]

        l_water = get_physics_lambda(epoch, phys_conf["water_balance_lambda_max"], warmup)
        l_agdd = get_physics_lambda(epoch, phys_conf["agdd_lambda_max"], warmup)
        l_stress_asym = get_physics_lambda(epoch, phys_conf["stress_asymmetry_lambda_max"], warmup)
        l_dag = get_physics_lambda(epoch, phys_conf["dag_lambda_max"], warmup)

        water_loss = outputs["water_balance_loss"] * l_water
        agdd_loss = outputs["agdd_loss"] * l_agdd
        stress_asym_loss = outputs["stress_asymmetry_loss"] * l_stress_asym

        # 2. SSL
        ssl_conf = self.config["training"]["ssl"]
        # Dummy inputs for SSL targets since full pipeline isn't wired up in batch loading
        # In practice, targets are generated from inputs/labels
        B, T = batch["optical"].shape[0], batch["optical"].shape[1]

        # Mocking SSL loss execution
        ssl_a_loss = torch.tensor(0.0, device=self.device)
        ssl_b_loss = torch.tensor(0.0, device=self.device)
        ssl_c_loss = torch.tensor(0.0, device=self.device)

        if "phenology_label" in batch:
            ssl_b_loss = self.ssl_b(outputs["phenology_pred"], batch["phenology_label"])

        if "drought_traj_target" in batch:
            ssl_c_loss = self.ssl_c(outputs["drought_traj"], batch["drought_traj_target"])

        ssl_loss = (ssl_conf["ssl_a_weight"] * ssl_a_loss +
                   ssl_conf["ssl_b_weight"] * ssl_b_loss +
                   ssl_conf["ssl_c_weight"] * ssl_c_loss)

        # 3. Supervised tasks
        sup_loss = torch.tensor(0.0, device=self.device)
        if "stress_label" in batch:
            sup_loss += self.stress_loss(outputs["stress_logits"], batch["stress_label"])

        if "yield_anomaly" in batch:
            sup_loss += self.yield_loss(outputs["yield_pred"].squeeze(-1), batch["yield_anomaly"].float())

        # 4. Auxiliary
        moe_lb_weight = self.config["training"]["moe_load_balance_weight"]
        moe_loss = outputs["moe_load_balance_loss"] * moe_lb_weight

        # Total
        dag_loss = torch.tensor(0.0, device=self.device) # Placeholder
        calibration_loss = torch.tensor(0.0, device=self.device) # Placeholder

        total_loss = (ssl_loss + water_loss + agdd_loss + stress_asym_loss +
                     dag_loss + moe_loss + sup_loss + calibration_loss)

        self.log("train_loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config["training"]["lr"]),
            weight_decay=self.config["training"]["weight_decay"]
        )
        return optimizer

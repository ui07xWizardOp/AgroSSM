import torch
import torch.nn as nn
from typing import Tuple

class RegimeRoutedMoE(nn.Module):
    """
    Top-1 routed Mixture-of-Experts with load-balance loss.

    Each expert is a small FFN: Linear(embed_dim, expert_hidden) -> GELU -> Linear(expert_hidden, embed_dim).
    The router is a Linear(embed_dim, num_experts) that scores each token against each expert.
    Top-1 selection: each token is processed by exactly one expert.
    """
    def __init__(self, embed_dim: int = 256, num_experts: int = 8, expert_hidden: int = 512):
        super().__init__()
        self.num_experts = num_experts

        self.router = nn.Linear(embed_dim, num_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, expert_hidden),
                nn.GELU(),
                nn.Linear(expert_hidden, embed_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, N, embed_dim]
        B, N, D = x.shape

        x_flat = x.view(-1, D)

        router_logits = self.router(x_flat) # [B*N, num_experts]
        router_probs = torch.softmax(router_logits, dim=-1) # [B*N, num_experts]

        # Top-1 routing
        max_probs, expert_assignments = torch.max(router_probs, dim=-1) # [B*N]

        output_flat = torch.zeros_like(x_flat)

        # Process each expert
        for i, expert in enumerate(self.experts):
            expert_mask = (expert_assignments == i)
            if expert_mask.any():
                expert_inputs = x_flat[expert_mask]
                expert_outputs = expert(expert_inputs)
                # Multiply by probability to allow gradients to flow into router
                output_flat[expert_mask] = expert_outputs * max_probs[expert_mask].unsqueeze(-1)

        output = output_flat.view(B, N, D)

        # Load balance loss
        # f_i = fraction of tokens routed to expert i
        # P_i = mean router probability for expert i across all tokens
        f = torch.bincount(expert_assignments, minlength=self.num_experts).float() / (B * N)
        P = router_probs.mean(dim=0)

        load_balance_loss = self.num_experts * torch.sum(f * P)

        expert_assignments = expert_assignments.view(B, N)

        return output, load_balance_loss, expert_assignments

import torch
import torch.nn as nn

class EmbeddingCollapseMonitor(nn.Module):
    """
    Compute eigenvalues of embedding covariance matrix. Alert if effective rank drops below threshold (INV-T3).
    """
    def __init__(self, threshold_ratio: float = 0.90):
        super().__init__()
        self.threshold_ratio = threshold_ratio

    def forward(self, embeddings: torch.Tensor) -> float:
        # embeddings: [B, N, embed_dim]
        # Flatten batch and sequence
        x = embeddings.reshape(-1, embeddings.size(-1)) # [B*N, embed_dim]

        # Center
        x_centered = x - x.mean(dim=0, keepdim=True)

        # Covariance matrix
        cov = torch.mm(x_centered.T, x_centered) / (x_centered.size(0) - 1)

        # Eigenvalues
        # Symmetric positive semi-definite matrix -> use eigh
        try:
            eigenvalues, _ = torch.linalg.eigh(cov)
        except:
            return 1.0 # fallback if svd fails

        # eigh returns eigenvalues in ascending order, so reverse
        eigenvalues = torch.flip(eigenvalues, dims=[0])

        # Filter negative eigenvalues caused by numerical instability
        eigenvalues = torch.clamp(eigenvalues, min=0.0)

        total_variance = eigenvalues.sum()
        if total_variance <= 1e-6:
            return 1.0 # collapsed

        # Top-1 variance explained
        top1_ratio = (eigenvalues[0] / total_variance).item()

        # Check against threshold
        if top1_ratio > self.threshold_ratio:
            # We can print or log this in practice. For now, just return the ratio.
            pass

        return top1_ratio

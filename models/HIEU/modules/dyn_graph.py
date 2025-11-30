import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraph(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        # learnable adjacency (symmetric non-negative via softplus)
        self.raw_A = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)
        self.mlp = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

    def adjacency(self):
        A = F.softplus(self.raw_A)
        A = (A + A.t()) / 2.0
        A = A - torch.diag_embed(torch.diag(A))  # zero diagonal
        return A

    def laplacian_smoothness(self, W: torch.Tensor, weight: float = 1e-3):
        # W: [num_nodes, d] per-node parameter vectors to be smoothed
        A = self.adjacency().detach()
        D = torch.diag(A.sum(dim=-1))
        L = D - A
        return weight * torch.trace(W.t() @ L @ W)

    def prior_loss(self, A_prior: torch.Tensor, weight: float = 1e-3):
        # A_prior: [N, N] prior adjacency (e.g., GC-based), compare with learned A
        A = self.adjacency()
        return weight * F.mse_loss(A, A_prior)

    @staticmethod
    def gc_features_from_adj(A: torch.Tensor):
        # A: [N, N] (j -> i as A[i, j])
        deg_in = A.sum(dim=1)
        deg_out = A.sum(dim=0)
        asym = deg_in - deg_out
        stats = torch.stack([deg_in, deg_out, asym], dim=-1)  # [N, 3]
        return stats

    def forward(self, features: torch.Tensor):
        # features: [B, num_nodes] graph-level context (e.g., returns or vol proxy)
        ctx = self.mlp(features)
        return ctx, self.adjacency()

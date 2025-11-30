import torch
import torch.nn as nn
from .configs import HIEUConfig
from .modules.revin_rlinear import RGRLCore, RevIN
from .modules.regime_encoder import RegimeEncoder
from .modules.dyn_graph import DynamicGraph
from .modules.freq_bank import FrequencyBank
from .modules.hyper_linear import HyperLinear
from .modules.probabilistic import QuantileHead

class HIEUModel(nn.Module):
    def __init__(self, config: HIEUConfig):
        super().__init__()
        self.cfg = config
        N = config.num_nodes
        # Core processes all nodes (channels=N)
        self.core = RGRLCore(config.seq_len, config.pred_len, channels=N)
        # Regime encoder takes N channels
        self.regime = RegimeEncoder(in_channels=N, num_regimes=config.num_regimes, latent_dim=config.regime_dim,
                                    temperature=config.regime_temp)
        # Dynamic graph over N nodes
        self.graph = DynamicGraph(num_nodes=N, hidden_dim=config.graph_hidden)
        # Frequency bank per node (channels=N)
        self.freq = FrequencyBank(channels=N, num_bands=config.num_bands, kernel=config.band_kernel)
        # HyperLinear is per-node; we will apply it per node in forward
        ctx_dim = config.regime_dim + config.graph_hidden + config.num_bands
        self.hyper = HyperLinear(config.seq_len, config.pred_len, rank=config.linear_rank, ctx_dim=ctx_dim)
        # Probabilistic head per node (apply per node and stack)
        self.qhead = QuantileHead(config.pred_len, config.quantiles)
        # optional prior
        self.A_prior = None

    def set_prior(self, A_prior: torch.Tensor):
        self.A_prior = A_prior

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        # x: [B, L, N]
        B, L, N = x.shape
        # Regime encoding
        z, logits, gate, recon = self.regime(x)
        # Graph context from last values per node
        features = x[:, -1, :]  # [B, N]
        g_ctx, A_learned = self.graph(features)
        # Frequency bank
        x_fused, bands, w = self.freq(x)
        if not self.cfg.use_freq_reweight:
            w = torch.ones_like(w) / w.numel()
        # Core baseline prediction
        y_core = self.core(x)  # [B, S, N]
        # Hyper ctx vector (same ctx for all nodes; can be extended with node-wise)
        ctx = torch.cat([z, g_ctx, w.unsqueeze(0).expand(B, -1)], dim=-1)
        # Apply HyperLinear per node
        y_hyper_nodes = []
        W_nodes = []
        for i in range(N):
            yi, Wi = self.hyper(x_fused[:, :, i:i+1], ctx)  # [B, S, 1]
            y_hyper_nodes.append(yi)
            W_nodes.append(Wi)
        y_hyper = torch.cat(y_hyper_nodes, dim=2)  # [B, S, N]
        y_point = y_core + y_hyper
        # Quantiles per node
        q_nodes = []
        for i in range(N):
            q_nodes.append(self.qhead(y_point[:, :, i:i+1]))  # [B, S, Q]
        q = torch.stack(q_nodes, dim=3)  # [B, S, Q, N]
        aux = None
        if return_aux:
            # Laplacian smoothness on average of W across batch and nodes (stack then mean)
            W_stack = torch.stack(W_nodes, dim=2).mean(dim=0).mean(dim=0)  # [S, L] approx -> not node-shaped; skip
            # For demonstration, smooth last-layer weights per node via an identity placeholder
            lap = None
            prior_term = None
            if self.cfg.use_gc_prior and (self.A_prior is not None):
                prior_term = self.graph.prior_loss(self.A_prior.to(A_learned.device), weight=self.cfg.graph_prior_weight)
            aux = {'logits': logits, 'gate': gate, 'recon': recon, 'A': A_learned,
                   'lap_smooth': lap if lap is not None else 0.0, 'prior_loss': prior_term}
            return y_point, q, aux
        return y_point

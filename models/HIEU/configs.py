class HIEUConfig:
    def __init__(self):
        # IO
        self.seq_len = 96
        self.pred_len = 96
        self.enc_in = 1
        self.batch_size = 32
        self.learning_rate = 5e-4
        self.weight_decay = 1e-4
        self.epochs = 12
        self.num_nodes = 1  # number of coins/nodes

        # Regime encoder
        self.num_regimes = 4
        self.regime_dim = 128
        self.regime_temp = 1.0  # gumbel-softmax temperature
        self.ssl_weight = 0.1
        self.use_gc_features = True

        # Graph
        self.graph_hidden = 128
        self.graph_prior_weight = 1e-2
        self.laplacian_weight = 1e-3
        self.use_gc_prior = True

        # Frequency bank
        self.num_bands = 5
        self.band_kernel = 15
        self.band_stride = 1
        self.use_freq_reweight = True

        # HyperLinear
        self.linear_rank = 16  # low-rank delta (LoRA-style)
        self.num_experts = 3  # per-band experts

        # Probabilistic
        self.quantiles = [0.1, 0.5, 0.9]
        self.pinball_weight = 0.2
        self.crps_weight = 0.2  # enable CRPS

        # TTA
        self.tta_steps = 2
        self.tta_lr = 1e-4
        self.tta_use_tent = True

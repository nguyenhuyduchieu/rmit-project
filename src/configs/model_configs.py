"""
Model configuration classes for different forecasting models
"""

class BaseConfig:
    """Base configuration class"""
    def __init__(self):
        # Common parameters
        self.seq_len = 96  # Input sequence length
        self.pred_len = 24  # Prediction length
        self.enc_in = 1  # Number of input features
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.epochs = 10

class LinearConfig(BaseConfig):
    """Configuration for Linear model"""
    def __init__(self):
        super().__init__()
        self.individual = True

class DLinearConfig(BaseConfig):
    """Configuration for DLinear model"""
    def __init__(self):
        super().__init__()
        self.individual = True
        self.kernel_size = 25

class NLinearConfig(BaseConfig):
    """Configuration for NLinear model"""
    def __init__(self):
        super().__init__()
        self.individual = True

class RLinearConfig(BaseConfig):
    """Configuration for RLinear model"""
    def __init__(self):
        super().__init__()
        self.individual = True
        self.alpha = 0.000001
        self.seed = 42
        self.max_train_N = None

class PatchTSTConfig(BaseConfig):
    """Configuration for PatchTST model"""
    def __init__(self):
        super().__init__()
        self.individual = True
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = 'end'
        self.e_layers = 3
        self.d_model = 128
        self.n_heads = 8
        self.d_ff = 256
        self.dropout = 0.1
        self.fc_dropout = 0.1
        self.head_dropout = 0.1
        self.revin = True
        self.affine = True
        self.subtract_last = False
        self.decomposition = False
        self.kernel_size = 25

class iTransformerConfig(BaseConfig):
    """Configuration for iTransformer model"""
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.e_layers = 3
        self.n_heads = 8
        self.d_ff = 256
        self.dropout = 0.1
        self.factor = 5
        self.activation = 'gelu'
        self.embed = 'timeF'
        self.freq = 'h'
        self.output_attention = False
        self.use_norm = True
        self.class_strategy = 'last'

class VanillaTransformerConfig(BaseConfig):
    """Configuration for Vanilla Transformer model"""
    def __init__(self):
        super().__init__()
        self.dec_in = 1
        self.c_out = 1
        self.task_name = 'long_term_forecast'
        self.d_model = 128
        self.e_layers = 3
        self.d_layers = 2
        self.n_heads = 8
        self.d_ff = 256
        self.dropout = 0.1
        self.factor = 5
        self.activation = 'gelu'
        self.embed = 'timeF'
        self.freq = 'h'

class RLGatedMoLEConfig(BaseConfig):
    """Configuration for RL-gated MoLE model"""
    def __init__(self):
        super().__init__()
        # Expert bank configuration
        self.num_experts = 8
        self.seq_len = 96
        self.pred_len = 96
        self.enc_in = 1
        
        # Router configuration
        self.state_dim = 20  # Number of state features
        self.router_hidden_dim = 256
        
        # Reward configuration
        self.reward_type = 'mda_rmse'  # 'mda_rmse' or 'pnl_cvar'
        self.reward_beta = 0.5
        
        # Training configuration
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 0.001
        
        # RL training configuration
        self.rl_epochs = 50
        self.rl_learning_rate = 0.0001
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate

class ProphetConfig(BaseConfig):
    """Configuration for Prophet model"""
    def __init__(self):
        super().__init__()
        self.yearly_seasonality = True
        self.weekly_seasonality = True
        self.daily_seasonality = False
        self.seasonality_mode = 'multiplicative'
        self.changepoint_prior_scale = 0.05
        self.seasonality_prior_scale = 10.0
        self.interval_width = 0.95
        self.uncertainty_samples = 1000

class AutoformerConfig(BaseConfig):
    """Configuration for Autoformer model"""
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.e_layers = 2
        self.d_layers = 1
        self.n_heads = 8
        self.d_ff = 256
        self.dropout = 0.1
        self.factor = 1
        self.activation = 'gelu'
        self.embed = 'timeF'
        self.freq = 'h'
        self.moving_avg = 25
        self.label_len = None  # Will be set to seq_len // 2 if None
        self.output_attention = False

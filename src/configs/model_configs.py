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

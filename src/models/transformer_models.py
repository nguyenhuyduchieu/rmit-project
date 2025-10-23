"""
Transformer models for time series forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt

# Import existing models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from existing implementations
try:
    from patchtst_model import PatchTST
    from itransformer_model import iTransformer
    from vanilla_transformer_model import VanillaTransformer
except ImportError:
    print("Warning: Could not import existing transformer models. Please ensure they are available.")
    
    # Fallback implementations
    class PatchTST(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.seq_len = config.seq_len
            self.pred_len = config.pred_len
            self.linear = nn.Linear(config.seq_len, config.pred_len)
        
        def forward(self, x):
            return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
    
    class iTransformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.seq_len = config.seq_len
            self.pred_len = config.pred_len
            self.linear = nn.Linear(config.seq_len, config.pred_len)
        
        def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
            return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
    
    class VanillaTransformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.seq_len = config.seq_len
            self.pred_len = config.pred_len
            self.linear = nn.Linear(config.seq_len, config.pred_len)
        
        def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
            return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)

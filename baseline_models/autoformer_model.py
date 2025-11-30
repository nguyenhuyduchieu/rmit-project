"""
Autoformer model wrapper for time series forecasting
Integrates Autoformer from Time-Series-Library into the benchmark framework
"""

import torch
import torch.nn as nn
import sys
import os

# Add Time-Series-Library to path
ts_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_1', 'Time-Series-Library')
sys.path.insert(0, ts_lib_path)

try:
    from models.Autoformer import Model as AutoformerModel
    from layers.Embed import DataEmbedding_wo_pos
    from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
    from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
    AUTOFORMER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Autoformer modules: {e}")
    AUTOFORMER_AVAILABLE = False
    AutoformerModel = None

class Autoformer(nn.Module):
    """
    Autoformer model wrapper for time series forecasting
    Paper: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
    https://openreview.net/pdf?id=I55UqU-M11y
    """
    
    def __init__(self, config):
        super(Autoformer, self).__init__()
        
        if not AUTOFORMER_AVAILABLE:
            raise ImportError("Autoformer modules are not available. Please ensure Time-Series-Library is properly set up.")
        
        # Create config object compatible with Autoformer
        class AutoformerConfig:
            def __init__(self, config):
                self.task_name = 'long_term_forecast'
                self.seq_len = config.seq_len
                # label_len is typically seq_len // 2 for long-term forecasting
                label_len_val = config.label_len if hasattr(config, 'label_len') and config.label_len is not None else config.seq_len // 2
                self.label_len = label_len_val
                self.pred_len = config.pred_len
                self.enc_in = config.enc_in
                self.dec_in = config.enc_in
                self.c_out = config.enc_in
                self.d_model = config.d_model if hasattr(config, 'd_model') else 128
                self.embed = config.embed if hasattr(config, 'embed') else 'timeF'
                self.freq = config.freq if hasattr(config, 'freq') else 'h'
                self.dropout = config.dropout if hasattr(config, 'dropout') else 0.1
                self.factor = config.factor if hasattr(config, 'factor') else 1
                self.n_heads = config.n_heads if hasattr(config, 'n_heads') else 8
                self.d_ff = config.d_ff if hasattr(config, 'd_ff') else 256
                self.e_layers = config.e_layers if hasattr(config, 'e_layers') else 2
                self.d_layers = config.d_layers if hasattr(config, 'd_layers') else 1
                self.moving_avg = config.moving_avg if hasattr(config, 'moving_avg') else 25
                self.activation = config.activation if hasattr(config, 'activation') else 'gelu'
                self.output_attention = False
        
        self.config_obj = AutoformerConfig(config)
        self.model = AutoformerModel(self.config_obj)
        
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.config = config  # Store original config for reference
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass
        
        Args:
            x_enc: encoder input [B, L, D] - may have multiple features, we use first column
            x_mark_enc: encoder time features (optional)
            x_dec: decoder input (optional) - if None, will be created from x_enc
            x_mark_dec: decoder time features (optional)
        """
        # Extract only first feature (target column) for Autoformer
        # Autoformer expects enc_in=1 (univariate forecasting)
        if x_enc.shape[2] > 1:
            x_enc = x_enc[:, :, 0:1]  # [B, L, 1]
        
        # Autoformer needs decoder input with shape [B, label_len + pred_len, D]
        label_len = self.config_obj.label_len
        
        if x_dec is None:
            # Create decoder input: last label_len values from x_enc + zeros for pred_len
            batch_size = x_enc.shape[0]
            # Use last label_len values from encoder
            dec_inp = torch.cat([
                x_enc[:, -label_len:, :],  # Last label_len values
                torch.zeros(batch_size, self.pred_len, x_enc.shape[2], device=x_enc.device)
            ], dim=1)
        else:
            # If x_dec is provided but has wrong shape, adjust it
            if x_dec.shape[1] == self.pred_len:
                # Only pred_len provided, prepend with last label_len from x_enc
                batch_size = x_enc.shape[0]
                dec_inp = torch.cat([
                    x_enc[:, -label_len:, :],
                    x_dec
                ], dim=1)
            else:
                dec_inp = x_dec
        
        # Create time mark features if not provided
        # Autoformer's temporal embedding expects 4 features: month, day, weekday, hour (for freq='h')
        batch_size, seq_len = x_enc.shape[0], x_enc.shape[1]
        
        if x_mark_enc is None or x_mark_enc.shape[2] == 0:
            # Create dummy time features: [month, day, weekday, hour]
            # For freq='h', we need 4 features
            x_mark_enc = torch.zeros(batch_size, seq_len, 4, device=x_enc.device)
        elif x_mark_enc.shape[2] != 4:
            # If shape is wrong, reshape to correct size
            if x_mark_enc.shape[2] < 4:
                # Pad with zeros
                padding = torch.zeros(batch_size, seq_len, 4 - x_mark_enc.shape[2], device=x_enc.device)
                x_mark_enc = torch.cat([x_mark_enc, padding], dim=2)
            else:
                # Take first 4 features
                x_mark_enc = x_mark_enc[:, :, :4]
        
        batch_size_dec, seq_len_dec = dec_inp.shape[0], dec_inp.shape[1]
        if x_mark_dec is None or x_mark_dec.shape[2] == 0:
            # Create dummy time features for decoder
            x_mark_dec = torch.zeros(batch_size_dec, seq_len_dec, 4, device=x_enc.device)
        elif x_mark_dec.shape[2] != 4:
            # If shape is wrong, reshape to correct size
            if x_mark_dec.shape[2] < 4:
                # Pad with zeros
                padding = torch.zeros(batch_size_dec, seq_len_dec, 4 - x_mark_dec.shape[2], device=x_enc.device)
                x_mark_dec = torch.cat([x_mark_dec, padding], dim=2)
            else:
                # Take first 4 features
                x_mark_dec = x_mark_dec[:, :, :4]
        
        # Call Autoformer model
        output = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec, mask)
        
        return output  # [B, pred_len, D]


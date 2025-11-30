# Baseline Models

This folder contains baseline model implementations for time series forecasting.

## Models

- **Linear Models** (`linear_models.py`): Linear, DLinear, NLinear, RLinear
- **Transformer Models** (`transformer_models.py`): PatchTST, iTransformer, VanillaTransformer
- **Individual Model Files**:
  - `autoformer_model.py` - Autoformer implementation
  - `prophet_model.py` - Prophet/Facebook Prophet model
  - `rlinear_model.py` - RLinear model
  - `itransformer_model.py` - iTransformer model
  - `patchtst_model.py` - PatchTST model
  - `vanilla_transformer_model.py` - Vanilla Transformer model
  - `crypto_ltsf_linear.py` - Crypto LTSF Linear implementation

## Usage

These models are used in the unified benchmark script:
```bash
python scripts/run_unified_benchmark.py
```

All models use the same prepared data from `data_prepare/` for fair comparison.


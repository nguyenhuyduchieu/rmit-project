# Baseline Models

This folder contains baseline model implementations for time series forecasting. These models serve as benchmarks for comparison with proposed models.

## Model Categories

### Linear Models (`linear_models.py`)
- **Linear**: Simple linear transformation
- **DLinear**: Decomposition + Linear (trend + seasonal)
- **NLinear**: Normalization + Linear
- **RLinear**: RevIN + Linear (reversible normalization)

### Transformer Models (`transformer_models.py`)
- **PatchTST**: Patch-based Time Series Transformer
- **iTransformer**: Inverted Transformer architecture
- **VanillaTransformer**: Standard Transformer for time series

### Individual Model Files
- `autoformer_model.py` - Autoformer (decomposition-based transformer)
- `prophet_model.py` - Facebook Prophet (statistical model)
- `rlinear_model.py` - RLinear standalone implementation
- `itransformer_model.py` - iTransformer standalone
- `patchtst_model.py` - PatchTST standalone
- `vanilla_transformer_model.py` - Vanilla Transformer standalone
- `crypto_ltsf_linear.py` - Crypto-specific LTSF Linear

## Usage

### Single-Asset Benchmark
```bash
# All baseline models included
python scripts/run_unified_benchmark.py
```

### Multi-Asset Benchmark
```bash
# All baseline models support multi-asset input
python scripts/run_multi_asset_benchmark.py
```

## Performance

### Single-Asset (BTCUSDT)
- **Best**: iTransformer (RMSE: 0.56, MAE: 0.41)
- **Best Linear**: PatchTST (RMSE: 21.20)

### Multi-Asset (5 cryptocurrencies)
- **Best**: SimpleMoLE (RMSE: 1.05, MAE: 0.58)
- All models show similar performance (RMSE: 1.05-1.06)

All models use the same prepared data for fair comparison.


# Scripts

This folder contains all execution scripts for running experiments and benchmarks.

## Main Benchmark Scripts

### 1. Single-Asset Benchmark
- **run_unified_benchmark.py** - Main benchmark script for single-asset forecasting
  - Uses prepared data from `data_prepare/`
  - All models use the same prepared data for fair comparison
  - Tests on BTCUSDT with technical indicators
  - **Usage**: `python scripts/run_unified_benchmark.py`

### 2. Multi-Asset Benchmark
- **run_multi_asset_benchmark.py** - Benchmark script for multi-asset forecasting
  - Tests on 5 cryptocurrencies: BTC, ETH, BNB, SOL, XRP
  - Uses log returns of Close prices
  - Automatically prepares multi-asset data
  - **Usage**: `python scripts/run_multi_asset_benchmark.py`

## Helper Scripts

- **test_hieu_multi_asset.py** - Test HIEU model specifically with multi-asset data
- **test_simple_mole.py** - SimpleMoLE model definition (used by benchmarks)

## Usage Examples

### Run Single-Asset Benchmark:
```bash
# 1. Prepare data first
python data_prepare/prepare_data.py --datasets BTCUSDT

# 2. Run benchmark
python scripts/run_unified_benchmark.py
```

### Run Multi-Asset Benchmark:
```bash
# No data preparation needed - script handles it automatically
python scripts/run_multi_asset_benchmark.py
```

### Generate Comparison Reports:
```bash
# After running benchmarks
python analysis/create_multi_asset_comparison.py
```

## Output

- **Results**: Saved to `analysis/` folder
- **Logs**: Saved to `logs/` folder
- **Metrics**: MAE, MSE, RMSE, MAPE, SMAPE, Trend Match


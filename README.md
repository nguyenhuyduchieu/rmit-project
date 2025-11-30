# RMIT Time Series Forecasting Project

## 🚀 Quick Start

### 1. Prepare Data (Bắt buộc trước khi chạy models)

```bash
# Prepare data cho tất cả datasets
python data_prepare/prepare_data.py

# Hoặc prepare cho specific datasets
python data_prepare/prepare_data.py --datasets BTCUSDT ETHUSDT

# Giới hạn số samples để test nhanh
python data_prepare/prepare_data.py --max_samples 5000
```

### 2. Chạy Benchmarks

```bash
# Single-Asset Benchmark (tất cả models dùng cùng prepared data)
python scripts/run_unified_benchmark.py

# Multi-Asset Benchmark (5 cryptocurrencies: BTC, ETH, BNB, SOL, XRP)
python scripts/run_multi_asset_benchmark.py
```

## Cấu trúc thư mục

Dự án đã được tổ chức lại thành các thư mục chính sau:

### 📁 `baseline_models/`
Chứa các model baseline để so sánh:
- `linear_models.py` - Linear, DLinear, NLinear, RLinear
- `transformer_models.py` - PatchTST, iTransformer, VanillaTransformer
- `autoformer_model.py` - Autoformer model
- `prophet_model.py` - Prophet model
- `rlinear_model.py` - RLinear model implementation
- `itransformer_model.py` - iTransformer model
- `patchtst_model.py` - PatchTST model
- `vanilla_transformer_model.py` - Vanilla Transformer model
- `crypto_ltsf_linear.py` - Crypto LTSF Linear implementation

### 📁 `models/`
Chứa các model chính (proposed models):
- `HIEU/` - HIEU model implementation
- `mole_rl.py` - RL-gated Mixture-of-RLinear Experts (MoLE)
- `mole_trainer.py` - MoLE trainer với offline RL
- `revin.py` - RevIN normalization module

### 📁 `scripts/`
Chứa tất cả các script để chạy experiments và tests:
- `run_unified_benchmark.py` - **Main single-asset benchmark** (tất cả models dùng cùng prepared data)
- `run_multi_asset_benchmark.py` - **Multi-asset benchmark** (5 cryptocurrencies)
- `test_hieu_multi_asset.py` - Test HIEU model với multi-asset data
- `test_simple_mole.py` - SimpleMoLE model definition

### 📁 `analysis/`
Chứa các file so sánh và phân tích kết quả cuối cùng:
- `MULTI_ASSET_COMPARISON.md` - Báo cáo so sánh multi-asset benchmark
- `multi_asset_comparison.csv` - Kết quả multi-asset benchmark
- `multi_asset_*.png` - Visualizations cho multi-asset results
- `HIEU_ARCHITECTURE_ANALYSIS.md` - Phân tích chi tiết về HIEU model
- `FEATURE_USAGE_ANALYSIS.md` - Phân tích cách models sử dụng features
- `create_multi_asset_comparison.py` - Script tạo multi-asset comparison report

### 📁 `src/`
Chứa các utilities và configs:
- `configs/` - Model configurations
- `data/` - Data preprocessing và dataset utilities
- `experiments/` - Experiment framework
- `utils/` - Utility functions

### 📁 `data/`
Chứa dữ liệu crypto raw (CSV files)

### 📁 `data_prepare/`
Chứa scripts để prepare data thống nhất cho tất cả models:
- `prepare_data.py` - Script để prepare và lưu data đã xử lý
- `load_prepared_data.py` - Script để load prepared data
- `{dataset_name}/` - Prepared data cho từng dataset (sau khi chạy prepare_data.py)

### 📁 `paper_1/`, `paper_2/`, `paper_3/`, `paper_4/`
Chứa code từ các papers tham khảo

## Cách sử dụng

### ⚠️ QUAN TRỌNG: Prepare Data trước

**Tất cả models bây giờ sử dụng cùng prepared data để đảm bảo fair comparison.**

1. **Prepare data** (chỉ cần chạy 1 lần):
```bash
python data_prepare/prepare_data.py --datasets BTCUSDT
```

2. **Chạy unified benchmark** (tất cả models dùng cùng data):
```bash
python scripts/run_unified_benchmark.py
```

### Xem kết quả so sánh:
```bash
# Multi-asset benchmark results
cat analysis/MULTI_ASSET_COMPARISON.md
cat analysis/multi_asset_comparison.csv

# Generate comparison report với visualizations
python analysis/create_multi_asset_comparison.py
```

## Data Preparation

### Format Data:
- **Input**: `[batch, seq_len, features]` - Tất cả features (~40+ technical indicators)
- **Target**: `[batch, pred_len, 1]` - Chỉ Close price (feature index 0)
- **Preprocessing**: Resample 15-min, add technical indicators, standardize
- **Split**: Train (<=2023), Valid (2024), Test (2025)

### Prepared Data Location:
Sau khi chạy `prepare_data.py`, data được lưu trong:
```
data_prepare/{dataset_name}/
├── train_data.npy
├── valid_data.npy
├── test_data.npy
├── scaler.pkl
├── metadata.pkl
└── feature_names.txt
```

## Benchmark Results

### Single-Asset Benchmark (BTCUSDT)
- **Best Model**: iTransformer (RMSE: 0.56, MAE: 0.41)
- **Best Linear**: PatchTST (RMSE: 21.20)
- Results: See `analysis/` folder

### Multi-Asset Benchmark (5 cryptocurrencies)
- **Best Model**: SimpleMoLE (RMSE: 1.05, MAE: 0.58)
- **HIEU Model**: RMSE: 1.05, MAE: 0.58 (xếp thứ 3)
- Results: See `analysis/MULTI_ASSET_COMPARISON.md`

## Important Notes

- **HIEU Model**: Designed for **multi-asset forecasting**, NOT single-asset
  - Single-asset: MAE=763.34 ❌ (very poor)
  - Multi-asset: MAE=0.58 ✅ (excellent)
- **Data Preparation**: Chạy `prepare_data.py` trước khi chạy single-asset benchmark
- **Multi-Asset Data**: Uses log returns of Close prices, automatically prepared by `run_multi_asset_benchmark.py`
- All results saved in `analysis/` folder
- Logs saved in `logs/` folder

## Documentation

- **HIEU Model**: See `models/HIEU/TECHNICAL_README.md` for comprehensive technical documentation
- **Architecture Analysis**: See `analysis/HIEU_ARCHITECTURE_ANALYSIS.md`
- **Feature Usage**: See `analysis/FEATURE_USAGE_ANALYSIS.md`

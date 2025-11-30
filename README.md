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

### 2. Chạy Unified Benchmark

```bash
# Chạy benchmark với prepared data (tất cả models dùng cùng data)
python scripts/run_unified_benchmark.py
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
- `run_*.py` - Các script chạy benchmark cho từng model
- `test_*.py` - Các script test cho từng model
- `run_full_benchmark_with_hieu.py` - Script chạy full benchmark
- `run_comprehensive_benchmark_with_mole.py` - Script benchmark với MoLE

### 📁 `analysis/`
Chứa các file so sánh và phân tích kết quả cuối cùng:
- `final_model_comparison.csv` - Kết quả tổng hợp tất cả models
- `FINAL_MODEL_COMPARISON.md` - Báo cáo so sánh chi tiết
- `*.png` - Các biểu đồ so sánh models
- `create_final_comparison.py` - Script tạo final comparison

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

### Chạy benchmark cho một model cụ thể (legacy):
```bash
python scripts/run_patchtst_benchmark.py
python scripts/run_itransformer_benchmark.py
# ... etc
```

### Xem kết quả so sánh:
```bash
# Xem báo cáo cuối cùng
cat analysis/FINAL_MODEL_COMPARISON.md

# Hoặc xem CSV
cat analysis/final_model_comparison.csv
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

## Lưu ý

- **Bắt buộc**: Chạy `prepare_data.py` trước khi chạy models
- Tất cả kết quả sẽ được lưu vào thư mục `analysis/`
- Đảm bảo đã cài đặt đầy đủ dependencies trước khi chạy
- Các import paths đã được cập nhật để phù hợp với cấu trúc mới
- **Unified benchmark** đảm bảo tất cả models dùng cùng data format

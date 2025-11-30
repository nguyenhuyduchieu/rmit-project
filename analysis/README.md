# Analysis Results

This folder contains comparison results, visualizations, and analysis reports for all models.

## Multi-Asset Benchmark Results

### Main Results
- **MULTI_ASSET_COMPARISON.md** - Detailed comparison report with rankings
- **multi_asset_comparison.csv** - Summary results (sorted by RMSE)
- **multi_asset_benchmark_results.csv** - Raw benchmark results

### Visualizations
1. **multi_asset_rmse_comparison.png** - RMSE comparison across all models
2. **multi_asset_mae_comparison.png** - MAE comparison across all models  
3. **multi_asset_radar_chart_top5.png** - Radar chart for top 5 models
4. **multi_asset_mae_vs_rmse_scatter.png** - Scatter plot of MAE vs RMSE

### Key Findings (Multi-Asset)
- **Best Model**: SimpleMoLE (RMSE: 1.05, MAE: 0.58)
- **2nd Best**: PatchTST (RMSE: 1.05, MAE: 0.58)
- **3rd Best**: HIEU (RMSE: 1.05, MAE: 0.58)
- All top models have very similar performance (RMSE: 1.05-1.06)

## Documentation

- **HIEU_ARCHITECTURE_ANALYSIS.md** - Detailed analysis of HIEU model architecture
- **FEATURE_USAGE_ANALYSIS.md** - Analysis of how models use input features
- **MULTI_ASSET_BENCHMARK_README.md** - Guide for multi-asset benchmark

## Scripts

- **create_multi_asset_comparison.py** - Generate multi-asset comparison report with visualizations
- **create_final_comparison.py** - Generate final comparison (legacy, for single-asset)

## ⚠️ Important: HIEU Model Results

Trong file `final_model_comparison.csv`, có **2 kết quả của HIEU**:

1. **HIEU (Single Asset)**: MAE=763.34, RMSE=889.05 - **KẾT QUẢ RẤT KÉM**
   - Đây là kết quả khi chạy HIEU với chỉ 1 asset (BTCUSDT)
   - **KHÔNG NÊN sử dụng kết quả này để so sánh** vì HIEU không được thiết kế cho single-asset

2. **HIEU (Multi-Asset)**: MAE=0.58, RMSE=1.05 - **KẾT QUẢ TỐT (xếp thứ 2)**
   - Đây là kết quả khi chạy HIEU với 5 assets (BTC, ETH, BNB, SOL, XRP)
   - Đây là cách sử dụng **ĐÚNG** của HIEU model

### Tại Sao HIEU Không Tốt Với Single Asset?

HIEU model được thiết kế với các module:
- **DynamicGraph**: Học mối quan hệ giữa các assets (vô dụng với N=1)
- **RegimeEncoder**: Phát hiện regime từ multi-asset patterns (thiếu thông tin với N=1)
- **HyperLinear**: Điều chỉnh predictions dựa trên context từ nhiều assets (context nghèo với N=1)

📖 **Xem chi tiết**: `HIEU_ARCHITECTURE_ANALYSIS.md`

All models were tested on the same prepared data (BTCUSDT) for fair comparison.


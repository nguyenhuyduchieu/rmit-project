# Analysis Results

This folder contains the final comparison results and visualizations for all models.

## Final Results

- **final_model_comparison.csv** - Combined results from all benchmarks
- **FINAL_MODEL_COMPARISON.md** - Detailed comparison report with rankings

## Visualizations

1. **rmse_comparison.png** - RMSE comparison across all models
2. **mae_comparison.png** - MAE comparison across all models  
3. **radar_chart_top5.png** - Radar chart for top 5 models
4. **mae_vs_rmse_scatter.png** - Scatter plot of MAE vs RMSE
5. **trend_match_comparison.png** - Trend match comparison

## Documentation

- **FEATURE_USAGE_ANALYSIS.md** - Analysis of how models use input features
- **create_final_comparison.py** - Script to generate final comparison

## Key Findings

- **Best Model (RMSE)**: iTransformer (0.56)
- **Best Model (MAE)**: iTransformer (0.41)
- **Best Linear Model**: PatchTST (RMSE: 21.20)
- **HIEU Model**: Works best with multi-asset data (MAE: 0.58)

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


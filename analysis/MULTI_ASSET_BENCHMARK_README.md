# Multi-Asset Benchmark Guide

## Tổng Quan

Script này chạy benchmark cho **tất cả các models** trên **multi-asset data** (5 cryptocurrencies: BTC, ETH, BNB, SOL, XRP).

## Tại Sao Cần Multi-Asset Benchmark?

1. **HIEU Model**: Được thiết kế đặc biệt cho multi-asset forecasting, cần benchmark để so sánh công bằng
2. **Cross-Asset Information**: Nhiều models có thể tận dụng thông tin từ các assets khác để cải thiện predictions
3. **Fair Comparison**: So sánh tất cả models trong cùng một setting (multi-asset)

## Data Format

- **Input**: Log returns của Close prices từ 5 assets
- **Shape**: `[batch, seq_len, num_assets]` = `[batch, 96, 5]`
- **Target**: Predict log returns cho tất cả 5 assets
- **Shape**: `[batch, pred_len, num_assets]` = `[batch, 96, 5]`
- **Standardization**: Per-asset standardization

## Models Được Test

1. **Linear Models**: Linear, DLinear, NLinear, RLinear
2. **Transformer Models**: iTransformer, PatchTST, VanillaTransformer
3. **MoLE**: SimpleMoLE
4. **HIEU**: HIEU (designed for multi-asset)

## Cách Chạy

### 1. Chạy Benchmark

```bash
cd /Users/hieuduc/Downloads/rmit
python scripts/run_multi_asset_benchmark.py
```

Kết quả sẽ được lưu vào:
- `logs/multi_asset_benchmark.log` - Log file
- `analysis/multi_asset_benchmark_results.csv` - Results CSV

### 2. Tạo Comparison Report

Sau khi benchmark chạy xong:

```bash
python analysis/create_multi_asset_comparison.py
```

Sẽ tạo:
- `analysis/multi_asset_comparison.csv` - Summary CSV
- `analysis/MULTI_ASSET_COMPARISON.md` - Detailed report
- `analysis/multi_asset_rmse_comparison.png` - RMSE visualization
- `analysis/multi_asset_mae_comparison.png` - MAE visualization
- `analysis/multi_asset_mae_vs_rmse_scatter.png` - Scatter plot
- `analysis/multi_asset_radar_chart_top5.png` - Radar chart for top 5

## Metrics

Tất cả metrics được tính **trung bình trên 5 assets**:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MSE**: Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **Trend Match**: Percentage of correct trend predictions

## So Sánh Với Single-Asset Benchmark

| Aspect | Single-Asset | Multi-Asset |
|--------|-------------|-------------|
| **Data** | 1 asset (BTCUSDT) với nhiều features | 5 assets (BTC, ETH, BNB, SOL, XRP) với log returns |
| **Input Shape** | `[batch, 96, 37]` (37 features) | `[batch, 96, 5]` (5 assets) |
| **Target** | Predict 1 asset | Predict 5 assets |
| **HIEU Performance** | Rất kém (MAE: 763) | Tốt (MAE: 0.58) |
| **Use Case** | Univariate/multivariate forecasting | Cross-asset forecasting |

## Notes

- Benchmark có thể mất thời gian (20-30 phút) tùy vào số lượng models và epochs
- Một số models có thể fail nếu không hỗ trợ multi-asset input - sẽ được skip và log lại
- HIEU model được expect sẽ có performance tốt nhất trong multi-asset setting

## Expected Results

Dựa trên kiến trúc:
- **HIEU**: Nên có performance tốt nhất (designed for multi-asset)
- **iTransformer**: Có thể tốt với multi-asset input
- **Linear Models**: Có thể hoạt động tốt với `individual=True`
- **PatchTST**: Có thể tốt với multi-asset input


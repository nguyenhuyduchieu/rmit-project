# Phân tích cách các Model sử dụng Features đầu vào

## Tổng quan

Dự án này có **2 loại data pipeline** khác nhau, dẫn đến các model sử dụng features khác nhau:

---

## 1. Data Pipeline 1: Multi-feature với Technical Indicators

### Models sử dụng pipeline này:
- **Baseline Linear Models**: Linear, DLinear, NLinear, RLinear
- **Transformer Models**: PatchTST, iTransformer, VanillaTransformer, Autoformer
- **Prophet Model**
- **MoLE (RL-gated)**

### Data Preprocessing (`src/data/preprocessing.py`):

#### Input Data:
- Raw CSV với OHLCV (Open, High, Low, Close, Volume)
- Resample về 15-minute timeframe

#### Feature Engineering:
Tạo **~40+ technical indicators**:
- **Price-based**: SMA (5, 10, 20, 50), EMA (5, 10, 20, 50)
- **Bollinger Bands**: BB_middle, BB_upper, BB_lower, BB_width, BB_position
- **RSI** (Relative Strength Index)
- **MACD**: MACD, MACD_signal, MACD_histogram
- **Stochastic**: Stoch_K, Stoch_D
- **Williams %R**
- **CCI** (Commodity Channel Index)
- **ATR** (Average True Range)
- **Volume indicators**: Volume_SMA, Volume_ratio
- **Price changes**: Price_change, Price_change_5, Price_change_10
- **Volatility**: Rolling volatility
- **Momentum**: Momentum_5, Momentum_10
- **ROC**: ROC_5, ROC_10

#### Data Format:
```python
# Shape: [samples, features] 
# features = ~40+ columns (OHLCV + technical indicators)
train_features = preprocessor.prepare_features(train_data)  # [T, F]
```

#### Dataset Creation (`src/data/dataset.py`):
```python
# TimeSeriesDataset
# Input: x [batch, seq_len, features] - TẤT CẢ features
# Target: y [batch, pred_len, 1] - CHỈ target_col_idx (thường là Close price, index=0)
```

#### Cách các Model xử lý:

**1. Linear Models (Linear, DLinear, NLinear, RLinear):**
- Nhận input: `[B, L, F]` với F = số features
- Có thể xử lý **multivariate** (F > 1) hoặc **univariate** (F = 1)
- Config `individual=True`: xử lý từng feature riêng biệt
- Config `individual=False`: xử lý chung tất cả features
- **Target**: Chỉ predict feature đầu tiên (Close price)

**2. Transformer Models (PatchTST, iTransformer, VanillaTransformer):**
- Nhận input: `[B, L, F]` 
- **PatchTST**: Có thể xử lý multivariate, nhưng thường dùng `enc_in=1` (chỉ Close)
- **iTransformer**: Tương tự, thường extract chỉ feature đầu tiên
- **Autoformer**: Extract chỉ feature đầu tiên: `x_enc = x_enc[:, :, 0:1]`
- **Target**: Predict Close price

**3. Prophet:**
- Nhận pandas DataFrame với datetime index
- Sử dụng tất cả features nhưng chủ yếu dựa vào Close price
- Có thể sử dụng regressors từ các features khác

**4. MoLE (RL-gated):**
- Nhận input: `[B, L, F]`
- **Expert Bank**: Sử dụng RLinear experts, mỗi expert có thể dùng features khác nhau
- **RL Router**: Extract state features từ data:
  - Volatility (1h, 4h, 1d)
  - Regime detection
  - Expert performance
  - Liquidity proxy
  - Trend features
- **Target**: Predict Close price (feature đầu tiên)

---

## 2. Data Pipeline 2: Multi-asset với Returns

### Models sử dụng pipeline này:
- **HIEU Model**

### Data Preprocessing (`models/HIEU/multi_asset_loader.py`):

#### Input Data:
- **Multiple crypto assets** (BTCUSDT, ETHUSDT, etc.)
- Chỉ sử dụng **Close price** của mỗi asset
- Resample về 15-minute timeframe

#### Feature Engineering:
- **Không có technical indicators**
- **Chỉ có**: Close prices của N assets
- Có thể convert sang **log returns** hoặc **percentage returns**
- Standardization: `(x - mean) / std` per asset

#### Data Format:
```python
# Shape: [samples, num_assets]
# Ví dụ: [T, 5] với 5 assets
aligned_df = load_align_close_series(data_dir, ['BTCUSDT', 'ETHUSDT', ...])
# Returns: [T, N] với N = số assets
```

#### Dataset Creation:
```python
# MultiAssetDataset
# Input: x [batch, seq_len, num_assets] - Close prices/returns của N assets
# Target: y [batch, pred_len, num_assets] - Predict TẤT CẢ assets
```

#### Cách HIEU Model xử lý:

**HIEU Model:**
- Nhận input: `[B, L, N]` với N = số assets (nodes)
- **Multivariate forecasting**: Predict tất cả N assets cùng lúc
- **Core components**:
  - `RGRLCore`: Baseline prediction cho tất cả nodes
  - `RegimeEncoder`: Encode regime từ tất cả nodes
  - `DynamicGraph`: Learn relationships giữa N nodes
  - `FrequencyBank`: Frequency analysis per node
  - `HyperLinear`: Per-node hypernetwork
  - `QuantileHead`: Probabilistic predictions per node
- **Output**: `[B, pred_len, N]` - Predictions cho tất cả assets

---

## So sánh chi tiết

### Feature Usage:

| Model | Input Shape | Features Used | Target |
|-------|------------|---------------|--------|
| **Linear Models** | `[B, L, F]` | Tất cả ~40+ features | Close (feature 0) |
| **Transformers** | `[B, L, F]` | Thường chỉ Close (F=1) | Close |
| **Prophet** | DataFrame | Tất cả features | Close |
| **MoLE** | `[B, L, F]` | Tất cả features + state features | Close |
| **HIEU** | `[B, L, N]` | Chỉ Close prices của N assets | Tất cả N assets |

### Data Preprocessing:

| Pipeline | Technical Indicators | Multi-asset | Returns | Standardization |
|----------|---------------------|-------------|---------|-----------------|
| **Pipeline 1** | ✅ Có (~40+) | ❌ Không | ❌ Không | ✅ StandardScaler |
| **Pipeline 2** | ❌ Không | ✅ Có | ✅ Có thể | ✅ Per-asset |

### Key Differences:

1. **Feature Richness**:
   - Pipeline 1: Rất nhiều features (OHLCV + 40+ indicators)
   - Pipeline 2: Chỉ Close prices của nhiều assets

2. **Target Prediction**:
   - Pipeline 1: Chỉ predict 1 target (Close price của 1 asset)
   - Pipeline 2: Predict nhiều targets (Close prices của N assets)

3. **Model Architecture**:
   - Pipeline 1: Univariate hoặc multivariate với nhiều features
   - Pipeline 2: Multivariate với ít features nhưng nhiều assets

---

## Vấn đề hiện tại

### ❌ Không thống nhất:
1. **HIEU model** sử dụng pipeline riêng (multi-asset, returns)
2. **Các baseline models** sử dụng pipeline khác (single-asset, nhiều features)
3. **Không fair comparison** vì:
   - Input features khác nhau
   - Data preprocessing khác nhau
   - Target format khác nhau (1 target vs N targets)

### ✅ Giải pháp đề xuất:

1. **Option 1: Unify HIEU với Pipeline 1**
   - Sử dụng technical indicators cho HIEU
   - Predict chỉ 1 asset (như các baseline)
   - So sánh fair hơn

2. **Option 2: Unify Baselines với Pipeline 2**
   - Chỉ dùng Close prices
   - Multi-asset forecasting
   - So sánh trên cùng task

3. **Option 3: Support cả 2 pipelines**
   - Mỗi model có thể chạy với cả 2 loại data
   - So sánh trên nhiều scenarios

---

## Code References

### Pipeline 1:
- `src/data/preprocessing.py` - CryptoDataPreprocessor
- `src/data/dataset.py` - TimeSeriesDataset
- Sử dụng trong: `scripts/run_full_benchmark_with_hieu.py`

### Pipeline 2:
- `models/HIEU/multi_asset_loader.py` - create_multiasset_loaders
- Sử dụng trong: `models/HIEU/` và `scripts/test_hieu.py`

---

## Kết luận

**Các model KHÔNG dùng cùng loại data:**
- Baselines: Multi-feature (40+ indicators), single-asset, univariate target
- HIEU: Single-feature (Close only), multi-asset, multivariate target

**Cần thống nhất data pipeline để có fair comparison!**


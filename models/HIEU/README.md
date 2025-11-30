# HIEU: Hypernetwork-Integrated Expert Unit (RGRL)

A lightweight, regime-adaptive, graph-aware, frequency-enhanced linear forecasting architecture for cryptocurrency time series.

## 📚 Documentation

- **[TECHNICAL_README.md](TECHNICAL_README.md)** - **Comprehensive technical deep dive** covering:
  - Mathematical foundations of all components
  - Detailed architecture explanations
  - Forward pass flow
  - Loss functions
  - Training process
  - Why multi-asset is essential

## 🏗️ Architecture Overview

HIEU combines several advanced techniques:
- **RevIN (Reversible Instance Normalization)** for robust normalization
- **Regime Detection** for market state identification
- **Dynamic Graph Learning** for cross-asset relationships
- **Frequency Analysis** for multi-scale pattern extraction
- **Hypernetworks** for adaptive linear transformations
- **Probabilistic Forecasting** for uncertainty quantification

## 🔧 Modules

- **RegimeEncoder**: regime discovery and routing (contrastive + reconstruction)
- **DynamicGraph**: learnable dynamic adjacency with GC-prior hooks and Laplacian regularization
- **FrequencyBank**: learnable FIR-like filter bank and band fusion
- **HyperLinear**: hypernetwork-conditioned linear experts (low-rank deltas)
- **ProbabilisticHeads**: quantile heads (pinball loss) and optional CRPS
- **TTA**: test-time adaptation (TENT) for distribution shifts

## ⚠️ Important: Multi-Asset Design

**HIEU is designed for MULTI-ASSET forecasting, NOT single-asset.**

- **Single Asset (N=1)**: MAE = 763.34, RMSE = 889.05 ❌
- **Multi-Asset (N=5)**: MAE = 0.58, RMSE = 1.05 ✅

See [TECHNICAL_README.md](TECHNICAL_README.md) for detailed explanation.

## 🚀 Usage

Entrypoint: `HIEU/model.py` exposes `HIEUModel` compatible with existing loaders: forward(x) -> [B, H, N].

```python
from models.HIEU.model import HIEUModel
from models.HIEU.configs import HIEUConfig

config = HIEUConfig()
config.num_nodes = 5  # Number of assets
config.seq_len = 96
config.pred_len = 96

model = HIEUModel(config)
y_point, q = model(x)  # x: [B, L, N]
```

## 📊 Scripts

- `test_hieu_multi_asset.py` - Test HIEU with multi-asset data (recommended)
- `run_multi_asset_benchmark.py` - Benchmark HIEU against other models

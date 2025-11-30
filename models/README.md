# Main Models

This folder contains the main/proposed models for time series forecasting.

## Models

### HIEU (Hypernetwork-Integrated Expert Unit)
- **Location**: `HIEU/`
- **Purpose**: Multi-asset cryptocurrency forecasting
- **Key Features**:
  - Dynamic Graph Learning for cross-asset relationships
  - Regime Detection for market state identification
  - Frequency Bank for multi-scale pattern extraction
  - Hypernetworks for adaptive transformations
  - Probabilistic forecasting with quantiles
- **Documentation**:
  - `HIEU/README.md` - Quick start guide
  - `HIEU/TECHNICAL_README.md` - **Comprehensive technical documentation** (mathematical foundations, architecture details)
- **Performance**:
  - Single-asset: MAE=763.34 ❌ (NOT recommended)
  - Multi-asset: MAE=0.58 ✅ (excellent, ranks 3rd)

### MoLE (RL-gated Mixture-of-RLinear Experts)
- **Files**: `mole_rl.py`, `mole_trainer.py`
- **Purpose**: Combines multiple RLinear experts with RL-based routing
- **Features**:
  - Multiple RLinear experts
  - RL-based expert selection
  - Offline RL training support
  - OPE (Off-Policy Evaluation)

### RevIN (Reversible Instance Normalization)
- **File**: `revin.py`
- **Purpose**: Normalization module used by various models
- **Features**: Reversible normalization for time series

## Usage

### HIEU Model (Multi-Asset - Recommended)
```bash
# Test HIEU with multi-asset data
python scripts/test_hieu_multi_asset.py

# Or run in multi-asset benchmark
python scripts/run_multi_asset_benchmark.py
```

### MoLE Model
```bash
# MoLE is included in unified benchmark
python scripts/run_unified_benchmark.py
```

## Important Notes

⚠️ **HIEU Model Design**:
- **Designed for MULTI-ASSET forecasting** (N > 1)
- **NOT suitable for single-asset** (N = 1)
- See `HIEU/TECHNICAL_README.md` for detailed explanation of why multi-asset is required


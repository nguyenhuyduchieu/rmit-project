# Main Models

This folder contains the main/proposed models for time series forecasting.

## Models

- **HIEU/** - HIEU model implementation (multi-asset forecasting)
  - Designed for multi-asset time series forecasting
  - Uses dynamic graph, regime encoding, frequency bank, and hypernetworks
  - See `HIEU/README.md` for details

- **mole_rl.py** - RL-gated Mixture-of-RLinear Experts (MoLE)
  - Combines multiple RLinear experts with RL-based routing
  - Supports offline RL training

- **mole_trainer.py** - MoLE trainer with offline RL
  - Implements offline RL training pipeline
  - Supports OPE (Off-Policy Evaluation)

- **revin.py** - RevIN (Reversible Instance Normalization)
  - Normalization module used by various models

## Usage

### HIEU Model (Multi-Asset)
```bash
python scripts/test_hieu_multi_asset.py
```

### MoLE Model
```bash
python scripts/test_mole_model.py
```

## Notes

- HIEU model works best with multi-asset data (multiple cryptocurrencies)
- For single asset, use baseline models instead


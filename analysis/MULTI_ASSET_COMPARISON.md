# Multi-Asset Model Comparison Report

Generated: 2025-11-30 19:42:52

## Overview

This report compares all models tested on **multi-asset data** (5 cryptocurrencies: BTC, ETH, BNB, SOL, XRP).

**Total Models Tested:** 7

## Summary Statistics

### Best Models by Metric

| Metric | Best Model | Value |
|--------|-----------|-------|
| **MAE** (Lower is Better) | SimpleMoLE | 0.579696 |
| **RMSE** (Lower is Better) | SimpleMoLE | 1.050197 |
| **MSE** (Lower is Better) | SimpleMoLE | 1.180850 |
| **MAPE** (Lower is Better) | SimpleMoLE | 122.336376 |
| **SMAPE** (Lower is Better) | NLinear | 162.219622 |
| **Trend_Match** (Higher is Better) | Linear | 0.00% |

## Detailed Results

### All Models (Sorted by RMSE)

| model | MAE | RMSE | MSE | MAPE | SMAPE | Trend_Match |
|:------|----:|-----:|----:|-----:|------:|------------:|
| SimpleMoLE | 0.579696 | 1.050197 | 1.180850 | 122.34 | 172.09 | 0.00% |
| PatchTST | 0.582599 | 1.052110 | 1.186924 | 131.18 | 168.16 | 0.00% |
| HIEU | 0.584168 | 1.054398 | 1.191687 | 137.46 | 165.72 | 0.00% |
| Linear | 0.583855 | 1.054680 | 1.192214 | 135.96 | 166.05 | 0.00% |
| DLinear | 0.584118 | 1.054964 | 1.192901 | 137.23 | 165.70 | 0.00% |
| RLinear | 0.587376 | 1.055934 | 1.195393 | 145.31 | 162.86 | 0.00% |
| NLinear | 0.588570 | 1.057017 | 1.197870 | 148.09 | 162.22 | 0.00% |

## Model Rankings

### Ranking by RMSE (Primary Metric)
1. **SimpleMoLE**: RMSE = 1.050197, MAE = 0.579696
2. **PatchTST**: RMSE = 1.052110, MAE = 0.582599
3. **HIEU**: RMSE = 1.054398, MAE = 0.584168
4. **Linear**: RMSE = 1.054680, MAE = 0.583855
5. **DLinear**: RMSE = 1.054964, MAE = 0.584118
6. **RLinear**: RMSE = 1.055934, MAE = 0.587376
7. **NLinear**: RMSE = 1.057017, MAE = 0.588570

### Ranking by MAE
1. **SimpleMoLE**: MAE = 0.579696, RMSE = 1.050197
2. **PatchTST**: MAE = 0.582599, RMSE = 1.052110
3. **Linear**: MAE = 0.583855, RMSE = 1.054680
4. **DLinear**: MAE = 0.584118, RMSE = 1.054964
5. **HIEU**: MAE = 0.584168, RMSE = 1.054398
6. **RLinear**: MAE = 0.587376, RMSE = 1.055934
7. **NLinear**: MAE = 0.588570, RMSE = 1.057017

## Notes

- All models were tested on the same multi-asset data (5 cryptocurrencies)
- Data: Log returns of Close prices, standardized per asset
- Metrics are averaged across all 5 assets
- Sequence Length: 96, Prediction Length: 96
- Data Split: Train (<=2023), Valid (2024), Test (2025)

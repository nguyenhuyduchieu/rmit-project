# Utilities

This folder contains utility functions used across the project.

## Files

- **metrics.py** - Metrics calculation functions
  - MAE, MSE, RMSE, MAPE, SMAPE, Trend Match

## Usage

```python
from utils.metrics import calculate_all_metrics
metrics = calculate_all_metrics(predictions, targets)
```


# Source Code

This folder contains core utilities, configurations, and framework code used throughout the project.

## Structure

### `configs/`
- **model_configs.py** - Configuration classes for all models
  - `LinearConfig`, `DLinearConfig`, `NLinearConfig`, `RLinearConfig`
  - `PatchTSTConfig`, `iTransformerConfig`, `VanillaTransformerConfig`
  - `RLGatedMoLEConfig`, `ProphetConfig`, `AutoformerConfig`

### `data/`
- **preprocessing.py** - Data preprocessing utilities
- **dataset.py** - Dataset classes and data loaders
  - `TimeSeriesDataset` - Single-asset dataset
  - `MultiVariateTimeSeriesDataset` - Multi-feature dataset

### `experiments/`
- **benchmark.py** - Benchmark experiment framework
  - Handles training, validation, and evaluation
  - Metrics calculation and result saving

### `utils/`
- **training.py** - Training utilities (`ModelTrainer` class)
- **metrics.py** - Metrics calculation functions
- Other utility functions

## Usage

These modules are imported by scripts and models throughout the project:

```python
from src.configs.model_configs import LinearConfig, PatchTSTConfig
from src.data.dataset import TimeSeriesDataset
from src.utils.training import ModelTrainer
from src.utils.metrics import calculate_all_metrics
```


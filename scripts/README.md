# Scripts

This folder contains all execution scripts for running experiments and tests.

## Main Scripts

### Unified Benchmark
- **run_unified_benchmark.py** - Main benchmark script using prepared data
  - All models use the same prepared data for fair comparison
  - Recommended script for comprehensive evaluation

### Individual Model Scripts
- **run_*_benchmark.py** - Benchmark scripts for specific models
- **test_*.py** - Test scripts for individual models

### Special Scripts
- **test_hieu_multi_asset.py** - Test HIEU model with multi-asset data
- **test_fixed_models.py** - Test fixed models (VanillaTransformer, SimpleMoLE, Prophet, etc.)

## Usage

### Run unified benchmark (recommended):
```bash
python scripts/run_unified_benchmark.py
```

### Test specific model:
```bash
python scripts/test_patchtst.py
python scripts/test_itransformer.py
# etc.
```

## Logs

All scripts generate logs in `logs/` directory for monitoring and debugging.


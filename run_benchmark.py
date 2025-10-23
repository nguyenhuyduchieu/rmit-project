"""
Main script to run comprehensive benchmark experiments
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.experiments.benchmark import BenchmarkExperiment
from src.configs.model_configs import (
    LinearConfig, DLinearConfig, NLinearConfig, RLinearConfig,
    PatchTSTConfig, iTransformerConfig, VanillaTransformerConfig
)

# Import model classes (you'll need to implement these)
# For now, we'll use placeholder imports
try:
    from src.models.linear_models import Linear, DLinear, NLinear, RLinear
    from src.models.transformer_models import PatchTST, iTransformer, VanillaTransformer
except ImportError:
    print("Model classes not found. Please implement them in src/models/")
    sys.exit(1)

def main():
    """Run comprehensive benchmark experiments"""
    
    # Configuration
    data_dir = '/Users/hieuduc/Downloads/rmit/data'
    results_dir = '/Users/hieuduc/Downloads/rmit/results'
    max_samples = 100000  # Limit data for faster training
    
    # Initialize benchmark experiment
    benchmark = BenchmarkExperiment(data_dir, results_dir)
    
    # Define models and their configurations
    models_configs = {
        'Linear': (Linear, LinearConfig()),
        'DLinear': (DLinear, DLinearConfig()),
        'NLinear': (NLinear, NLinearConfig()),
        'RLinear': (RLinear, RLinearConfig()),
        'PatchTST': (PatchTST, PatchTSTConfig()),
        'iTransformer': (iTransformer, iTransformerConfig()),
        'VanillaTransformer': (VanillaTransformer, VanillaTransformerConfig())
    }
    
    # Run benchmark
    print("Starting comprehensive benchmark experiments...")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Max samples per dataset: {max_samples}")
    print(f"Models to test: {list(models_configs.keys())}")
    
    results_df = benchmark.run_benchmark(
        models_configs=models_configs,
        max_samples=max_samples
    )
    
    if not results_df.empty:
        print(f"\nBenchmark completed successfully!")
        print(f"Total experiments: {len(results_df)}")
        print(f"Results saved to: {results_dir}")
    else:
        print("Benchmark failed - no successful experiments")

if __name__ == "__main__":
    main()

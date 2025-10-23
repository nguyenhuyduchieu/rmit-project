"""
Comprehensive benchmark script using the new framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.experiments.benchmark import BenchmarkExperiment
from src.configs.model_configs import (
    LinearConfig, DLinearConfig, NLinearConfig, RLinearConfig,
    PatchTSTConfig, iTransformerConfig, VanillaTransformerConfig
)
from src.models.linear_models import Linear, DLinear, NLinear, RLinearOLS
from src.models.transformer_models import PatchTST, iTransformer, VanillaTransformer

def main():
    """Run comprehensive benchmark experiments"""
    
    # Configuration
    data_dir = '/Users/hieuduc/Downloads/rmit/data'
    results_dir = '/Users/hieuduc/Downloads/rmit/results'
    max_samples = 50000  # Limit data for faster training
    
    # Initialize benchmark experiment
    benchmark = BenchmarkExperiment(data_dir, results_dir)
    
    # Define models and their configurations
    models_configs = {
        'Linear': (Linear, LinearConfig()),
        'DLinear': (DLinear, DLinearConfig()),
        'NLinear': (NLinear, NLinearConfig()),
        'RLinear': (RLinearOLS, RLinearConfig()),
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
        
        # Print top models by different metrics
        print(f"\nTop 3 models by MAE:")
        top_mae = results_df.groupby('model')['MAE'].mean().sort_values().head(3)
        for i, (model, mae) in enumerate(top_mae.items(), 1):
            print(f"  {i}. {model}: {mae:.6f}")
        
        print(f"\nTop 3 models by Trend Match:")
        top_trend = results_df.groupby('model')['Trend_Match'].mean().sort_values(ascending=False).head(3)
        for i, (model, trend) in enumerate(top_trend.items(), 1):
            print(f"  {i}. {model}: {trend:.2f}%")
    else:
        print("Benchmark failed - no successful experiments")

if __name__ == "__main__":
    main()

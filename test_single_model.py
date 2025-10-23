"""
Test single model with the new framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.experiments.benchmark import BenchmarkExperiment
from src.configs.model_configs import LinearConfig
from src.models.linear_models import Linear

def main():
    """Test single model"""
    
    # Configuration
    data_dir = '/Users/hieuduc/Downloads/rmit/data'
    results_dir = '/Users/hieuduc/Downloads/rmit/results'
    max_samples = 10000  # Small sample for testing
    
    # Initialize benchmark experiment
    benchmark = BenchmarkExperiment(data_dir, results_dir)
    
    # Test with just Linear model and one dataset
    models_configs = {
        'Linear': (Linear, LinearConfig())
    }
    
    # Test with just one dataset
    datasets = ['BTCUSDT']
    
    print("Testing Linear model on BTCUSDT...")
    
    results_df = benchmark.run_benchmark(
        models_configs=models_configs,
        datasets=datasets,
        max_samples=max_samples
    )
    
    if not results_df.empty:
        print(f"✓ Test successful!")
        print(f"Results: {len(results_df)} experiments")
        print(results_df)
    else:
        print("✗ Test failed")

if __name__ == "__main__":
    main()

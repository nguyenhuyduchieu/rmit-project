"""
Benchmark experiment framework for time series forecasting models
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from ..data.preprocessing import CryptoDataPreprocessor
from ..data.dataset import create_data_loaders
from ..utils.training import ModelTrainer
from ..utils.metrics import calculate_all_metrics

class BenchmarkExperiment:
    """Framework for running benchmark experiments on time series models"""
    
    def __init__(self, data_dir: str, results_dir: str = 'results'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.preprocessor = CryptoDataPreprocessor(data_dir)
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
    def run_single_experiment(self, model_class, config, dataset_name: str, 
                            max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Run a single experiment for one model on one dataset"""
        print(f"\n{'='*60}")
        print(f"Running {model_class.__name__} on {dataset_name}")
        print(f"{'='*60}")
        
        # Load and preprocess data
        file_path = os.path.join(self.data_dir, f"{dataset_name}.csv")
        train_data, valid_data, test_data = self.preprocessor.process_single_file(
            file_path, max_samples
        )
        
        # Check if we have enough data
        if len(train_data) < config.seq_len + config.pred_len:
            print(f"Not enough training data: {len(train_data)} samples")
            return None
        
        if len(valid_data) < config.seq_len + config.pred_len:
            print(f"Not enough validation data: {len(valid_data)} samples")
            return None
            
        if len(test_data) < config.seq_len + config.pred_len:
            print(f"Not enough test data: {len(test_data)} samples")
            return None
        
        # Normalize data
        scaler = StandardScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        valid_data_scaled = scaler.transform(valid_data)
        test_data_scaled = scaler.transform(test_data)
        
        # Create data loaders
        train_loader, valid_loader, test_loader = create_data_loaders(
            train_data_scaled, valid_data_scaled, test_data_scaled,
            config.seq_len, config.pred_len, config.batch_size
        )
        
        # Initialize model
        model = model_class(config)
        
        # Train model
        print(f"Training {model_class.__name__}...")
        trainer = ModelTrainer(model)
        training_results = trainer.train(
            train_loader, valid_loader, 
            epochs=config.epochs, 
            learning_rate=config.learning_rate
        )
        
        # Evaluate model
        print(f"Evaluating {model_class.__name__}...")
        preds, trues = trainer.evaluate(test_loader, scaler)
        
        # Calculate metrics
        metrics = calculate_all_metrics(preds, trues)
        
        return {
            'model': model_class.__name__,
            'dataset': dataset_name,
            'metrics': metrics,
            'training_results': training_results
        }
    
    def run_benchmark(self, models_configs: Dict[str, Dict], 
                     datasets: Optional[List[str]] = None,
                     max_samples: Optional[int] = None) -> pd.DataFrame:
        """Run benchmark experiments for multiple models and datasets"""
        
        if datasets is None:
            datasets = [f.replace('.csv', '') for f in self.preprocessor.get_all_crypto_files()]
        
        all_results = []
        
        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")
            
            for model_name, (model_class, config) in models_configs.items():
                try:
                    result = self.run_single_experiment(
                        model_class, config, dataset, max_samples
                    )
                    
                    if result is not None:
                        # Flatten metrics for easier analysis
                        row = {
                            'model': result['model'],
                            'dataset': result['dataset']
                        }
                        row.update(result['metrics'])
                        all_results.append(row)
                        
                        print(f"Results for {result['dataset']} - {result['model']}:")
                        for metric, value in result['metrics'].items():
                            print(f"  {metric}: {value:.6f}")
                    
                except Exception as e:
                    print(f"Error running {model_name} on {dataset}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Save results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(os.path.join(self.results_dir, 'benchmark_results.csv'), index=False)
            
            # Calculate and save summary
            summary_df = self._calculate_summary(results_df)
            summary_df.to_csv(os.path.join(self.results_dir, 'benchmark_summary.csv'), index=False)
            
            # Print summary
            self._print_summary(summary_df)
            
            return results_df
        else:
            print("No successful experiments completed.")
            return pd.DataFrame()
    
    def _calculate_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate summary statistics across all datasets"""
        summary_data = []
        
        for model in results_df['model'].unique():
            model_results = results_df[results_df['model'] == model]
            
            summary_row = {'model': model}
            for metric in ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'Trend_Match']:
                if metric in model_results.columns:
                    summary_row[metric] = model_results[metric].mean()
            
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def _print_summary(self, summary_df: pd.DataFrame):
        """Print summary results"""
        print(f"\n{'='*100}")
        print("BENCHMARK SUMMARY RESULTS")
        print(f"{'='*100}")
        
        print(f"\nAverage metrics across all datasets:")
        for _, row in summary_df.iterrows():
            print(f"\n{row['model']}:")
            for metric in ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'Trend_Match']:
                if metric in row:
                    print(f"  {metric}: {row[metric]:.6f}")
        
        # Model rankings
        print(f"\nModel Rankings:")
        for metric in ['MAE', 'RMSE', 'Trend_Match']:
            if metric in summary_df.columns:
                if metric == 'Trend_Match':
                    # Higher is better for Trend_Match
                    ranking = summary_df.sort_values(metric, ascending=False)
                else:
                    # Lower is better for MAE, RMSE
                    ranking = summary_df.sort_values(metric, ascending=True)
                
                print(f"\nBy {metric}:")
                for i, (_, row) in enumerate(ranking.iterrows(), 1):
                    print(f"  {i}. {row['model']}: {row[metric]:.6f}")
        
        print(f"\nDetailed results saved to: {self.results_dir}/benchmark_results.csv")
        print(f"Summary results saved to: {self.results_dir}/benchmark_summary.csv")

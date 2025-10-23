import pandas as pd
import numpy as np

def load_and_compare_results():
    """Load and compare results from all models"""
    
    # Load LTSF-Linear results
    try:
        ltsf_results = pd.read_csv('/Users/hieuduc/Downloads/rmit/ltsf_linear_results.csv')
        print("✓ Loaded LTSF-Linear results")
    except:
        print("✗ Could not load LTSF-Linear results")
        ltsf_results = None
    
    # Load RLinear results
    try:
        rlinear_results = pd.read_csv('/Users/hieuduc/Downloads/rmit/rlinear_results.csv')
        print("✓ Loaded RLinear results")
    except:
        print("✗ Could not load RLinear results")
        rlinear_results = None
    
    # Combine all results
    all_results = []
    
    if ltsf_results is not None:
        all_results.append(ltsf_results)
    
    if rlinear_results is not None:
        all_results.append(rlinear_results)
    
    if not all_results:
        print("No results found!")
        return
    
    # Concatenate all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Clean up trend_match column (replace NaN with 0)
    combined_results['trend_match'] = combined_results['trend_match'].fillna(0)
    
    # Print summary statistics
    print(f"\n{'='*100}")
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print(f"{'='*100}")
    
    # Group by model and calculate average metrics
    summary = combined_results.groupby('model').agg({
        'mae': 'mean',
        'mse': 'mean', 
        'rmse': 'mean',
        'trend_match': 'mean'
    }).round(6)
    
    print("\nAverage metrics across all datasets:")
    print(summary)
    
    # Save combined results
    combined_results.to_csv('/Users/hieuduc/Downloads/rmit/combined_benchmark_results.csv', index=False)
    summary.to_csv('/Users/hieuduc/Downloads/rmit/combined_benchmark_summary.csv')
    
    # Print detailed results for each dataset
    print(f"\nDetailed results by dataset:")
    print(combined_results.to_string(index=False))
    
    # Find best performing model for each metric
    print(f"\n{'='*80}")
    print("BEST PERFORMING MODELS BY METRIC")
    print(f"{'='*80}")
    
    for metric in ['mae', 'mse', 'rmse', 'trend_match']:
        if metric == 'trend_match':
            best_model = summary[metric].idxmax()
            best_value = summary.loc[best_model, metric]
            print(f"Best {metric.upper()}: {best_model} ({best_value:.6f})")
        else:
            best_model = summary[metric].idxmin()
            best_value = summary.loc[best_model, metric]
            print(f"Best {metric.upper()}: {best_model} ({best_value:.6f})")
    
    # Performance ranking
    print(f"\n{'='*80}")
    print("MODEL RANKINGS")
    print(f"{'='*80}")
    
    # Rank by MAE (lower is better)
    mae_ranking = summary.sort_values('mae')
    print("\nRanking by MAE (Mean Absolute Error):")
    for i, (model, row) in enumerate(mae_ranking.iterrows(), 1):
        print(f"{i}. {model}: {row['mae']:.6f}")
    
    # Rank by RMSE (lower is better)
    rmse_ranking = summary.sort_values('rmse')
    print("\nRanking by RMSE (Root Mean Square Error):")
    for i, (model, row) in enumerate(rmse_ranking.iterrows(), 1):
        print(f"{i}. {model}: {row['rmse']:.6f}")
    
    # Rank by Trend Match (higher is better)
    trend_ranking = summary.sort_values('trend_match', ascending=False)
    print("\nRanking by Trend Match (%):")
    for i, (model, row) in enumerate(trend_ranking.iterrows(), 1):
        print(f"{i}. {model}: {row['trend_match']:.2f}%")
    
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print("✓ Combined benchmark results saved to: combined_benchmark_results.csv")
    print("✓ Summary results saved to: combined_benchmark_summary.csv")
    print(f"✓ Total models tested: {len(summary)}")
    print(f"✓ Total datasets: {len(combined_results['dataset'].unique())}")
    print(f"✓ Total experiments: {len(combined_results)}")

if __name__ == "__main__":
    load_and_compare_results()

"""
Create final comparison of all models with visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_all_results():
    """Load and combine all benchmark results"""
    results = []
    
    # Load unified benchmark results
    unified_file = 'unified_benchmark_summary.csv'
    if os.path.exists(unified_file):
        df = pd.read_csv(unified_file)
        df['source'] = 'unified_benchmark'
        results.append(df)
    
    # Load fixed models results
    fixed_file = 'fixed_models_results.csv'
    if os.path.exists(fixed_file):
        df = pd.read_csv(fixed_file)
        df['source'] = 'fixed_models'
        results.append(df)
    
    # Load HIEU multi-asset results (average across assets)
    hieu_file = 'hieu_multi_asset_results.csv'
    if os.path.exists(hieu_file):
        df = pd.read_csv(hieu_file)
        # Calculate average across assets
        avg_row = {
            'model': 'HIEU (Multi-Asset)',
            'MAE': df['MAE'].mean(),
            'MSE': df['MSE'].mean(),
            'RMSE': df['RMSE'].mean(),
            'MAPE': df['MAPE'].mean(),
            'SMAPE': df['SMAPE'].mean(),
            'Trend_Match': df['Trend_Match'].mean(),
            'source': 'hieu_multi_asset'
        }
        results.append(pd.DataFrame([avg_row]))
    
    if not results:
        print("No results found!")
        return None
    
    # Combine all results
    combined = pd.concat(results, ignore_index=True)
    
    # Remove duplicates (keep first occurrence)
    combined = combined.drop_duplicates(subset=['model'], keep='first')
    
    # Sort by RMSE (lower is better)
    combined = combined.sort_values('RMSE', ascending=True).reset_index(drop=True)
    
    return combined


def create_visualizations(df, output_dir='.'):
    """Create visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bar plot: RMSE comparison
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('RMSE', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
    bars = plt.barh(df_sorted['model'], df_sorted['RMSE'], color=colors)
    plt.xlabel('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.title('Model Comparison: RMSE', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        plt.text(row['RMSE'] + max(df_sorted['RMSE']) * 0.01, i, 
                f"{row['RMSE']:.2f}", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar plot: MAE comparison
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('MAE', ascending=True)
    colors = plt.cm.plasma(np.linspace(0, 1, len(df_sorted)))
    bars = plt.barh(df_sorted['model'], df_sorted['MAE'], color=colors)
    plt.xlabel('MAE (Lower is Better)', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.title('Model Comparison: MAE', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        plt.text(row['MAE'] + max(df_sorted['MAE']) * 0.01, i, 
                f"{row['MAE']:.2f}", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Radar chart for top models
    top_models = df.nsmallest(5, 'RMSE')
    
    # Normalize metrics for radar chart (0-1 scale, lower is better for MAE/RMSE, higher for Trend_Match)
    metrics = ['MAE', 'RMSE', 'MAPE', 'Trend_Match']
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for idx, row in top_models.iterrows():
        values = []
        for metric in metrics:
            if metric == 'Trend_Match':
                # Higher is better, normalize to 0-1
                values.append(row[metric] / 100)
            else:
                # Lower is better, invert and normalize
                max_val = df[metric].max()
                min_val = df[metric].min()
                if max_val > min_val:
                    values.append(1 - (row[metric] - min_val) / (max_val - min_val))
                else:
                    values.append(1.0)
        
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Top 5 Models: Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart_top5.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Scatter plot: MAE vs RMSE
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['MAE'], df['RMSE'], s=200, alpha=0.6, c=range(len(df)), cmap='viridis')
    
    # Add labels
    for idx, row in df.iterrows():
        plt.annotate(row['model'], (row['MAE'], row['RMSE']), 
                    fontsize=8, alpha=0.7, ha='center')
    
    plt.xlabel('MAE', fontsize=12, fontweight='bold')
    plt.ylabel('RMSE', fontsize=12, fontweight='bold')
    plt.title('Model Performance: MAE vs RMSE', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Model Rank')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_vs_rmse_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Trend Match comparison
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('Trend_Match', ascending=False)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(df_sorted)))
    bars = plt.barh(df_sorted['model'], df_sorted['Trend_Match'], color=colors)
    plt.xlabel('Trend Match (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.title('Model Comparison: Trend Match (Higher is Better)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        plt.text(row['Trend_Match'] + 1, i, 
                f"{row['Trend_Match']:.2f}%", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_match_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created 5 visualization plots in {output_dir}/")


def create_summary_report(df, output_file='FINAL_MODEL_COMPARISON.md'):
    """Create final summary report"""
    
    report = f"""# Final Model Comparison Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report compares all models tested on BTCUSDT dataset using unified prepared data.

**Total Models Tested:** {len(df)}

## Summary Statistics

### Best Models by Metric

| Metric | Best Model | Value |
|--------|-----------|-------|
| **MAE** (Lower is Better) | {df.loc[df['MAE'].idxmin(), 'model']} | {df['MAE'].min():.6f} |
| **RMSE** (Lower is Better) | {df.loc[df['RMSE'].idxmin(), 'model']} | {df['RMSE'].min():.6f} |
| **MSE** (Lower is Better) | {df.loc[df['MSE'].idxmin(), 'model']} | {df['MSE'].min():.6f} |
| **MAPE** (Lower is Better) | {df.loc[df['MAPE'].idxmin(), 'model']} | {df['MAPE'].min():.6f} |
| **SMAPE** (Lower is Better) | {df.loc[df['SMAPE'].idxmin(), 'model']} | {df['SMAPE'].min():.6f} |
| **Trend Match** (Higher is Better) | {df.loc[df['Trend_Match'].idxmax(), 'model']} | {df['Trend_Match'].max():.2f}% |

## Detailed Results

### All Models (Sorted by RMSE)

"""
    
    # Add detailed table
    df_display = df[['model', 'MAE', 'RMSE', 'MSE', 'MAPE', 'SMAPE', 'Trend_Match']].copy()
    df_display = df_display.round(6)
    df_display['Trend_Match'] = df_display['Trend_Match'].apply(lambda x: f"{x:.2f}%")
    
    report += df_display.to_markdown(index=False)
    
    report += f"""

## Model Rankings

### Ranking by RMSE (Primary Metric)
"""
    
    for i, (idx, row) in enumerate(df.sort_values('RMSE').iterrows(), 1):
        report += f"{i}. **{row['model']}**: RMSE = {row['RMSE']:.6f}, MAE = {row['MAE']:.6f}\n"
    
    report += f"""

### Ranking by MAE
"""
    
    for i, (idx, row) in enumerate(df.sort_values('MAE').iterrows(), 1):
        report += f"{i}. **{row['model']}**: MAE = {row['MAE']:.6f}, RMSE = {row['RMSE']:.6f}\n"
    
    report += f"""

### Ranking by Trend Match
"""
    
    for i, (idx, row) in enumerate(df.sort_values('Trend_Match', ascending=False).iterrows(), 1):
        report += f"{i}. **{row['model']}**: Trend Match = {row['Trend_Match']:.2f}%\n"
    
    report += f"""

## Visualizations

The following plots are available:

1. **rmse_comparison.png** - RMSE comparison across all models
2. **mae_comparison.png** - MAE comparison across all models
3. **radar_chart_top5.png** - Radar chart for top 5 models
4. **mae_vs_rmse_scatter.png** - Scatter plot of MAE vs RMSE
5. **trend_match_comparison.png** - Trend match comparison

## Notes

- All models were tested on the same prepared data (BTCUSDT) for fair comparison
- HIEU model was tested with multi-asset data (5 assets) as it's designed for multi-asset forecasting
- Some models (Autoformer) were skipped due to missing dependencies
- Metrics are calculated on standardized data

## Data Source

- **Dataset**: BTCUSDT
- **Sequence Length**: 96
- **Prediction Length**: 96
- **Features**: 37 (Close price + 36 technical indicators)
- **Data Split**: Train (<=2023), Valid (2024), Test (2025)

"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✓ Created summary report: {output_file}")


def main():
    """Main function"""
    print("="*80)
    print("Creating Final Model Comparison")
    print("="*80)
    
    # Load all results
    print("\n1. Loading all benchmark results...")
    df = load_all_results()
    
    if df is None or len(df) == 0:
        print("No results found!")
        return
    
    print(f"   Loaded {len(df)} models")
    
    # Save combined results
    print("\n2. Saving combined results...")
    df.to_csv('final_model_comparison.csv', index=False)
    print(f"   Saved to: final_model_comparison.csv")
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    create_visualizations(df, output_dir='.')
    
    # Create summary report
    print("\n4. Creating summary report...")
    create_summary_report(df)
    
    print("\n" + "="*80)
    print("Final Comparison Created Successfully!")
    print("="*80)
    print("\nFiles created:")
    print("  - final_model_comparison.csv")
    print("  - FINAL_MODEL_COMPARISON.md")
    print("  - rmse_comparison.png")
    print("  - mae_comparison.png")
    print("  - radar_chart_top5.png")
    print("  - mae_vs_rmse_scatter.png")
    print("  - trend_match_comparison.png")


if __name__ == "__main__":
    main()


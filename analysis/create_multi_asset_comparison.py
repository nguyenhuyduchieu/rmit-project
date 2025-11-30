"""
Create comparison report for multi-asset benchmark results
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_multi_asset_results():
    """Load multi-asset benchmark results"""
    results_path = os.path.join(os.path.dirname(__file__), 'multi_asset_benchmark_results.csv')
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    return df


def create_multi_asset_comparison():
    """Create comprehensive comparison report for multi-asset results"""
    print("="*80)
    print("Creating Multi-Asset Comparison Report")
    print("="*80)
    
    # Load results
    df = load_multi_asset_results()
    if df is None or len(df) == 0:
        print("No results to compare")
        return
    
    print(f"\nLoaded {len(df)} model results")
    print(f"\nModels: {', '.join(df['model'].values)}")
    
    # Sort by RMSE
    df_sorted = df.sort_values('RMSE').reset_index(drop=True)
    
    # Create output directory
    output_dir = os.path.dirname(__file__)
    
    # 1. Create CSV summary
    summary_path = os.path.join(output_dir, 'multi_asset_comparison.csv')
    df_sorted.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary to: {summary_path}")
    
    # 2. Create markdown report
    report_path = os.path.join(output_dir, 'MULTI_ASSET_COMPARISON.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Multi-Asset Model Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overview\n\n")
        f.write(f"This report compares all models tested on **multi-asset data** (5 cryptocurrencies: BTC, ETH, BNB, SOL, XRP).\n\n")
        f.write(f"**Total Models Tested:** {len(df)}\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write("### Best Models by Metric\n\n")
        f.write("| Metric | Best Model | Value |\n")
        f.write("|--------|-----------|-------|\n")
        
        for metric in ['MAE', 'RMSE', 'MSE', 'MAPE', 'SMAPE', 'Trend_Match']:
            if metric in df.columns:
                if metric == 'Trend_Match':
                    best_idx = df[metric].idxmax()
                    best_value = df.loc[best_idx, metric]
                else:
                    best_idx = df[metric].idxmin()
                    best_value = df.loc[best_idx, metric]
                best_model = df.loc[best_idx, 'model']
                if metric == 'Trend_Match':
                    f.write(f"| **{metric}** (Higher is Better) | {best_model} | {best_value:.2f}% |\n")
                else:
                    f.write(f"| **{metric}** (Lower is Better) | {best_model} | {best_value:.6f} |\n")
        
        f.write("\n## Detailed Results\n\n")
        f.write("### All Models (Sorted by RMSE)\n\n")
        f.write("| model | MAE | RMSE | MSE | MAPE | SMAPE | Trend_Match |\n")
        f.write("|:------|----:|-----:|----:|-----:|------:|------------:|\n")
        
        for _, row in df_sorted.iterrows():
            f.write(f"| {row['model']} | {row.get('MAE', 'N/A'):.6f} | {row.get('RMSE', 'N/A'):.6f} | "
                   f"{row.get('MSE', 'N/A'):.6f} | {row.get('MAPE', 'N/A'):.2f} | "
                   f"{row.get('SMAPE', 'N/A'):.2f} | {row.get('Trend_Match', 'N/A'):.2f}% |\n")
        
        f.write("\n## Model Rankings\n\n")
        f.write("### Ranking by RMSE (Primary Metric)\n")
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            f.write(f"{i}. **{row['model']}**: RMSE = {row.get('RMSE', 'N/A'):.6f}, MAE = {row.get('MAE', 'N/A'):.6f}\n")
        
        f.write("\n### Ranking by MAE\n")
        df_mae = df.sort_values('MAE').reset_index(drop=True)
        for i, (_, row) in enumerate(df_mae.iterrows(), 1):
            f.write(f"{i}. **{row['model']}**: MAE = {row.get('MAE', 'N/A'):.6f}, RMSE = {row.get('RMSE', 'N/A'):.6f}\n")
        
        f.write("\n## Notes\n\n")
        f.write("- All models were tested on the same multi-asset data (5 cryptocurrencies)\n")
        f.write("- Data: Log returns of Close prices, standardized per asset\n")
        f.write("- Metrics are averaged across all 5 assets\n")
        f.write("- Sequence Length: 96, Prediction Length: 96\n")
        f.write("- Data Split: Train (<=2023), Valid (2024), Test (2025)\n")
    
    print(f"✓ Saved report to: {report_path}")
    
    # 3. Create visualizations
    print("\nCreating visualizations...")
    
    # RMSE Comparison
    plt.figure(figsize=(12, 6))
    plt.barh(df_sorted['model'], df_sorted['RMSE'], color='steelblue')
    plt.xlabel('RMSE', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('RMSE Comparison - Multi-Asset Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_asset_rmse_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: multi_asset_rmse_comparison.png")
    
    # MAE Comparison
    plt.figure(figsize=(12, 6))
    plt.barh(df_sorted['model'], df_sorted['MAE'], color='coral')
    plt.xlabel('MAE', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('MAE Comparison - Multi-Asset Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_asset_mae_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: multi_asset_mae_comparison.png")
    
    # MAE vs RMSE Scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(df['RMSE'], df['MAE'], s=200, alpha=0.6, c=range(len(df)), cmap='viridis')
    for i, row in df.iterrows():
        plt.annotate(row['model'], (row['RMSE'], row['MAE']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    plt.xlabel('RMSE', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('MAE vs RMSE - Multi-Asset Benchmark', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_asset_mae_vs_rmse_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: multi_asset_mae_vs_rmse_scatter.png")
    
    # Radar chart for top 5 models
    if len(df) >= 5:
        top5 = df_sorted.head(5)
        metrics = ['MAE', 'RMSE', 'MAPE', 'SMAPE', 'Trend_Match']
        available_metrics = [m for m in metrics if m in top5.columns]
        
        if len(available_metrics) >= 3:
            # Normalize metrics (lower is better for MAE, RMSE, MAPE, SMAPE; higher is better for Trend_Match)
            normalized = top5[available_metrics].copy()
            for metric in available_metrics:
                if metric == 'Trend_Match':
                    # Higher is better
                    normalized[metric] = (normalized[metric] - normalized[metric].min()) / (normalized[metric].max() - normalized[metric].min() + 1e-8)
                else:
                    # Lower is better - invert
                    normalized[metric] = 1 - (normalized[metric] - normalized[metric].min()) / (normalized[metric].max() - normalized[metric].min() + 1e-8)
            
            angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            for idx, row in top5.iterrows():
                values = [normalized.loc[idx, m] for m in available_metrics]
                values += values[:1]  # Complete the circle
                ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
                ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(available_metrics)
            ax.set_ylim(0, 1)
            ax.set_title('Top 5 Models - Multi-Asset Benchmark\n(Normalized Metrics)', 
                        size=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'multi_asset_radar_chart_top5.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: multi_asset_radar_chart_top5.png")
    
    print("\n" + "="*80)
    print("Multi-Asset Comparison Report Created Successfully!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  - {summary_path}")
    print(f"  - {report_path}")
    print(f"  - Visualizations in {output_dir}")


if __name__ == "__main__":
    create_multi_asset_comparison()


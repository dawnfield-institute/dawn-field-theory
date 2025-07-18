"""
Sweep Analysis Module for Symbolic Collapse Entropy Experiments

This module loads and interprets results from a parameter sweep, including:
- Aggregating t-test p-values and statistics across all runs
- Summarizing metrics (KL, JS, Wasserstein, QWCS)
- Visualizing p-values and metrics as heatmaps or tables
- Identifying parameter regions with significant differences

Usage:
    import sweep_analysis
    sweep_analysis.summarize_sweep('output/sweep_YYYYMMDD_HHMMSS')

"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from datetime import datetime

# --- Utility functions ---
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_sweep_dirs(sweep_root):
    return [d for d in glob.glob(os.path.join(sweep_root, 'd*_b*_e*')) if os.path.isdir(d)]

# --- Main analysis ---
def summarize_sweep(sweep_root, metric='p_value', depth=None):
    """
    Summarize t-test p-values (or other metrics) across the sweep.
    Args:
        sweep_root: Path to sweep directory
        metric: 'p_value', 't_stat', or any metric in metrics.json (default: 'p_value')
        depth: Specific tree depth to analyze (int), or None for all (default: None)
    """
    sweep_dirs = get_sweep_dirs(sweep_root)
    results = []
    for d in sweep_dirs:
        base = os.path.basename(d)
        parts = base.split('_')
        depth_val = int(parts[0][1:])
        breadth_val = int(parts[1][1:])
        ext_val = float(parts[2][1:])
        ttest_path = os.path.join(d, 'ttest_results.json')
        if not os.path.exists(ttest_path):
            continue
        ttest = load_json(ttest_path)
        for k, v in ttest.items():
            if depth is not None and int(k) != depth:
                continue
            results.append({
                'depth': depth_val,
                'breadth': breadth_val,
                'extinction_rate': ext_val,
                'tree_depth': int(k),
                'p_value': v.get('p_value'),
                't_stat': v.get('t_stat')
            })
    if not results:
        print('No results found.')
        return
    if not os.path.exists('output'):
        os.makedirs('output')
    if hasattr(summarize_sweep, 'output_dir'):
        output_dir = summarize_sweep.output_dir
    else:
        dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('output', f'sweep_analysis_{dt_str}')
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    tree_depths = [depth] if depth is not None else sorted(df['tree_depth'].unique())
    plot_files = []
    for td in tree_depths:
        sub = df[df['tree_depth'] == td]
        if sub.empty:
            continue
        agg = sub.groupby(['breadth', 'extinction_rate'])[metric].mean().unstack()
        pivot = agg
        plt.figure(figsize=(8,6))
        plt.title(f'{metric} at tree_depth={td}')
        plt.xlabel('Extinction Rate')
        plt.ylabel('Breadth')
        im = plt.imshow(pivot.values, aspect='auto', cmap='viridis',
                       extent=[min(pivot.columns), max(pivot.columns), min(pivot.index), max(pivot.index)])
        plt.colorbar(im, label=metric)
        plt.xticks(list(pivot.columns))
        plt.yticks(list(pivot.index))
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{metric}_tree_depth_{td}.png')
        plt.savefig(plot_path, dpi=300)
        plot_files.append(plot_path)
        plt.close()
    # Save summary table for selected metric
    summary_df = df.groupby(['tree_depth', 'breadth', 'extinction_rate'])[[metric]].min().reset_index()
    summary_path = os.path.join(output_dir, f'summary_{metric}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Plots saved to: {output_dir}\nSummary table saved to: {summary_path}")

# --- Example CLI ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Summarize symbolic collapse sweep results')
    parser.add_argument('sweep_root', type=str, help='Path to sweep directory')
    parser.add_argument('--metric', type=str, default='p_value', help="Metric to visualize (default: 'p_value')")
    parser.add_argument('--depth', type=int, default=None, help='Tree depth to analyze (default: all)')
    args = parser.parse_args()
    summarize_sweep(args.sweep_root, metric=args.metric, depth=args.depth)

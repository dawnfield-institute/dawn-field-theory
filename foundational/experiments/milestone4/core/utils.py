"""
Milestone 4: Shared utilities for experiment scripts.
"""

import json
import os
import time
import math
import numpy as np
from datetime import datetime, timezone


def save_results(results, experiment_name, results_dir=None):
    """Save experiment results to timestamped JSON file."""
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {filepath}")
    return filepath


def timer():
    """Simple context-manager timer."""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
    return Timer()


def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, ci=0.95, seed=42):
    """
    Compute bootstrap confidence interval for a statistic.
    
    Returns
    -------
    dict with 'estimate', 'ci_lower', 'ci_upper', 'std_error'
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[rng.integers(0, n, size=n)]
        estimates[i] = statistic(sample)
    
    alpha = (1 - ci) / 2
    return {
        'estimate': statistic(data),
        'ci_lower': float(np.percentile(estimates, 100 * alpha)),
        'ci_upper': float(np.percentile(estimates, 100 * (1 - alpha))),
        'std_error': float(np.std(estimates)),
    }


def monte_carlo_null(observed, generator_fn, n_trials=10000, seed=42):
    """
    Monte Carlo null test: what fraction of random trials match or exceed observed?
    
    Parameters
    ----------
    observed : float
        The observed statistic.
    generator_fn : callable
        Function(rng) → random statistic under null hypothesis.
    n_trials : int
        Number of Monte Carlo trials.
    seed : int
        Random seed.
    
    Returns
    -------
    dict with 'observed', 'null_mean', 'null_std', 'p_value', 'z_score'
    """
    rng = np.random.default_rng(seed)
    null_dist = np.array([generator_fn(rng) for _ in range(n_trials)])
    
    p_value = np.mean(null_dist >= observed)
    null_mean = np.mean(null_dist)
    null_std = np.std(null_dist)
    z_score = (observed - null_mean) / null_std if null_std > 0 else float('inf')
    
    return {
        'observed': float(observed),
        'null_mean': float(null_mean),
        'null_std': float(null_std),
        'p_value': float(p_value),
        'z_score': float(z_score),
        'n_trials': n_trials,
    }


def print_header(title, subtitle=None):
    """Print formatted experiment header."""
    print("\n" + "=" * 70)
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * 70)


def print_table(headers, rows, col_widths=None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2 
                      for i, h in enumerate(headers)]
    
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

"""
Milestone 3: Shared utilities for experiment scripts.

Common patterns: result saving, sieve, timing, statistical tests.
"""

import json
import os
import time
from datetime import datetime, timezone


def save_results(results, experiment_name, results_dir=None):
    """
    Save experiment results to a timestamped JSON file.

    Parameters
    ----------
    results : dict
        Experiment results (must be JSON-serializable).
    experiment_name : str
        Name like 'exp_01_fibonacci_memory'.
    results_dir : str or None
        Directory path. Defaults to '../results/' relative to this file.

    Returns
    -------
    str
        Path to the saved file.
    """
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

    print(f"Results saved: {filepath}")
    return filepath


def sieve_primes(limit):
    """
    Standard sieve of Eratosthenes.

    Parameters
    ----------
    limit : int
        Upper bound (inclusive).

    Returns
    -------
    list of int
        All primes up to limit.
    """
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]


def timer():
    """Simple context-manager timer."""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
    return Timer()


def experiment_header(name, description, paper=None, section=None):
    """
    Print a standard experiment header and return metadata dict.

    Parameters
    ----------
    name : str
        Experiment name (e.g., 'exp_01_fibonacci_memory').
    description : str
        One-line description.
    paper : str or None
        Target paper (e.g., 'Paper 1').
    section : str or None
        Target section (e.g., '§5.2').

    Returns
    -------
    dict
        Metadata for inclusion in results.
    """
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  {description}")
    if paper:
        print(f"  Target: {paper}" + (f" {section}" if section else ""))
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*70}\n")

    return {
        'experiment': name,
        'description': description,
        'paper': paper,
        'section': section,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'milestone': 'milestone3',
    }

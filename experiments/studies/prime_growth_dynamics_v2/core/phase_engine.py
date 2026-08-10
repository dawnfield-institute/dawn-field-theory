"""
Phase Emergence Engine: Core Library
=====================================

Reusable functions for exploring the three-phase emergence pipeline:
  Phase I:   Possibility Proliferation (bounded by MED)
  Phase II:  Symbolic Entropy Collapse (SEC)
  Phase III: Recursive Smoothing (PAC conservation)

Key constants:
  γ  = 0.5772156649... (Phase I→II interface cost)
  ln(φ) = 0.4812118250... (Phase II→III collapse efficiency)
  Ξ  = γ + ln(φ) = 1.0584... (total reconciliation threshold)
"""

import numpy as np
import math
import json
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from functools import lru_cache
from itertools import combinations_with_replacement


# =============================================================================
# Fundamental Constants
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2          # Golden ratio = 1.6180339887...
LN_PHI = math.log(PHI)                 # = 0.4812118250...
GAMMA = 0.5772156649015329             # Euler-Mascheroni constant
XI_ANALYTIC = GAMMA + LN_PHI           # = 1.0584274900...
XI_FORMULA = 1 + math.pi / 55          # = 1.0571198664...
XI_RULE110 = 1.0579                    # Measured from cellular automata

# Fibonacci sequence (first 30)
def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number (1-indexed: F1=1, F2=1, F3=2, ...)."""
    if n <= 0:
        return 0
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a

FIBS = [fibonacci(i) for i in range(1, 31)]  # F1..F30
F = {i: fibonacci(i) for i in range(1, 31)}  # F[n] dictionary

# Key Fibonacci numbers
F3, F4, F5, F6, F7, F9, F10, F12 = 2, 3, 5, 8, 13, 34, 55, 144


# =============================================================================
# Prime Sieve and Factorization
# =============================================================================

def sieve(limit: int) -> List[int]:
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]


def sieve_mask(limit: int) -> np.ndarray:
    """Return boolean array where True = prime."""
    mask = np.ones(limit + 1, dtype=bool)
    mask[0] = mask[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if mask[i]:
            mask[i*i::i] = False
    return mask


@lru_cache(maxsize=50000)
def factorize(n: int) -> Tuple[Tuple[int, int], ...]:
    """Prime factorization as tuple of (prime, exponent) pairs."""
    if n < 2:
        return ()
    factors = []
    d = 2
    temp = n
    while d * d <= temp:
        exp = 0
        while temp % d == 0:
            temp //= d
            exp += 1
        if exp > 0:
            factors.append((d, exp))
        d += 1
    if temp > 1:
        factors.append((temp, 1))
    return tuple(factors)


def omega(n: int) -> int:
    """Big omega: number of prime factors with multiplicity."""
    return sum(e for _, e in factorize(n))


# =============================================================================
# Phase Constant Algebra
# =============================================================================

def phase_formula_search(target: float, max_depth: int = 3,
                         tolerance: float = 0.02) -> List[Dict]:
    """
    Systematic search for expressions in {γ, ln(φ), Ξ, φ, π, F_n}
    that match a target value.

    Returns list of matches sorted by error.
    """
    # Building blocks
    atoms = {
        'γ': GAMMA,
        'ln(φ)': LN_PHI,
        'Ξ': XI_ANALYTIC,
        'φ': PHI,
        '1/φ': 1/PHI,
        'π': math.pi,
        '1': 1.0,
        '2': 2.0,
    }
    # Add Fibonacci numbers
    for i in range(3, 15):
        atoms[f'F{i}'] = float(fibonacci(i))

    matches = []

    # Level 1: single atoms and simple transforms
    for name, val in atoms.items():
        for op, fn, label in [
            ('id', lambda x: x, ''),
            ('inv', lambda x: 1/x if x != 0 else float('inf'), '1/'),
            ('sqrt', lambda x: math.sqrt(x) if x > 0 else float('inf'), '√'),
            ('log', lambda x: math.log(x) if x > 0 else float('inf'), 'ln('),
            ('exp', lambda x: math.exp(x) if abs(x) < 50 else float('inf'), 'e^'),
        ]:
            try:
                result = fn(val)
                if math.isfinite(result) and result != 0:
                    error = abs(result - target) / abs(target)
                    if error < tolerance:
                        expr = f"{label}{name}{')'if label in ['ln(','e^'] else ''}"
                        matches.append({
                            'expression': expr,
                            'value': result,
                            'error': error,
                            'error_pct': error * 100
                        })
            except (ValueError, OverflowError, ZeroDivisionError):
                pass

    # Level 2: binary operations
    ops = [
        ('+', lambda a, b: a + b),
        ('-', lambda a, b: a - b),
        ('*', lambda a, b: a * b),
        ('/', lambda a, b: a / b if b != 0 else float('inf')),
        ('^', lambda a, b: a ** b if abs(b) < 20 and a > 0 else float('inf')),
    ]

    atom_list = list(atoms.items())
    for i, (n1, v1) in enumerate(atom_list):
        for j, (n2, v2) in enumerate(atom_list):
            for op_sym, op_fn in ops:
                try:
                    result = op_fn(v1, v2)
                    if math.isfinite(result) and result != 0:
                        error = abs(result - target) / abs(target)
                        if error < tolerance:
                            expr = f"({n1} {op_sym} {n2})"
                            matches.append({
                                'expression': expr,
                                'value': result,
                                'error': error,
                                'error_pct': error * 100
                            })
                except (ValueError, OverflowError, ZeroDivisionError):
                    pass

    # Level 3: ternary a op1 b op2 c (if max_depth >= 3)
    if max_depth >= 3:
        key_atoms = [(n, v) for n, v in atom_list
                    if n in ['γ', 'ln(φ)', 'Ξ', 'φ', '1/φ', 'π',
                            'F3', 'F4', 'F5', 'F7', 'F10']]
        for n1, v1 in key_atoms:
            for n2, v2 in key_atoms:
                for n3, v3 in key_atoms:
                    for s1, f1 in ops[:4]:  # skip power for ternary
                        for s2, f2 in ops[:4]:
                            try:
                                result = f2(f1(v1, v2), v3)
                                if math.isfinite(result) and result != 0:
                                    error = abs(result - target) / abs(target)
                                    if error < tolerance:
                                        expr = f"(({n1} {s1} {n2}) {s2} {n3})"
                                        matches.append({
                                            'expression': expr,
                                            'value': result,
                                            'error': error,
                                            'error_pct': error * 100
                                        })
                            except:
                                pass

    # Deduplicate by rounding value
    seen = set()
    unique = []
    for m in sorted(matches, key=lambda x: x['error']):
        key = f"{m['expression']}_{m['value']:.8f}"
        if key not in seen:
            seen.add(key)
            unique.append(m)

    return unique[:50]  # Top 50


# =============================================================================
# Wave Interference (Phase III Smoothing)
# =============================================================================

def sieve_wave_interference(k: int, limit: int = 100000) -> Dict:
    """
    Compute smoothing wave interference at factor base size k.

    Uses the first k primes as smoothing waves and measures:
    - How much each wave smooths
    - Overlap/interference between waves
    - Residual roughness (remaining primes)
    """
    primes = sieve(limit)
    base_primes = primes[:k]

    # Track which numbers each wave smooths
    total = limit
    smoothed = np.zeros(limit + 1, dtype=bool)
    smoothed[0] = smoothed[1] = True

    wave_stats = []
    for p in base_primes:
        new_smoothed = 0
        for mult in range(p*p, limit + 1, p):
            if not smoothed[mult]:
                new_smoothed += 1
                smoothed[mult] = True
        # Also count the smaller multiples
        for mult in range(2*p, min(p*p, limit + 1), p):
            if not smoothed[mult]:
                new_smoothed += 1
                smoothed[mult] = True

        wave_stats.append({
            'prime': p,
            'new_smoothed': new_smoothed,
            'cumulative_smoothed': int(np.sum(smoothed)) - 2,  # exclude 0,1
            'fraction_smoothed': (int(np.sum(smoothed)) - 2) / (total - 1),
        })

    # Residual = primes beyond the base
    residual_primes = [p for p in primes if p > base_primes[-1]]
    is_prime_mask = sieve_mask(limit)
    actual_primes = int(np.sum(is_prime_mask))

    # Naive prediction (independent waves)
    naive_remaining = (limit - 1)
    for p in base_primes:
        naive_remaining *= (1 - 1/p)

    # Mertens product
    mertens = 1.0
    for p in base_primes:
        mertens *= (1 - 1/p)

    # Theoretical (with Mertens correction)
    theoretical_remaining = (limit - 1) * math.exp(-GAMMA) / math.log(base_primes[-1])

    return {
        'k': k,
        'base_primes': base_primes,
        'limit': limit,
        'actual_primes': actual_primes,
        'wave_stats': wave_stats,
        'naive_remaining': naive_remaining,
        'mertens_product': mertens,
        'theoretical_remaining': theoretical_remaining,
        'interference_ratio': naive_remaining / actual_primes if actual_primes > 0 else 0,
    }


def wave_destructive_interference(k: int, limit: int = 100000) -> float:
    """
    Measure destructive interference strength at factor base size k.

    Returns the ratio of actual-to-naive smoothing.
    Higher = more destructive interference (waves cancel more).
    """
    primes = sieve(limit)
    if k > len(primes):
        k = len(primes)
    base = primes[:k]

    # Naive (independent): fraction remaining = product(1 - 1/p)
    naive_frac = 1.0
    for p in base:
        naive_frac *= (1 - 1/p)

    # Actual: count primes > base[-1]
    actual_frac = sum(1 for p in primes if p > base[-1]) / limit

    # Interference = how much the naive overpredicts
    if actual_frac > 0:
        interference = naive_frac / actual_frac
    else:
        interference = float('inf')

    return interference


# =============================================================================
# PAC Tree Evolution (MED Bound Tests)
# =============================================================================

class PACNode:
    """A node in a PAC conservation tree."""
    def __init__(self, value: float, depth: int = 0):
        self.value = value
        self.depth = depth
        self.children: List['PACNode'] = []

    def split(self, n_children: int, noise: float = 0.01):
        """Split this node into n children, conserving total value."""
        if n_children < 2:
            return
        # Random partition that sums to self.value
        raw = np.random.dirichlet(np.ones(n_children)) * self.value
        # Add small noise to test stability
        raw += np.random.normal(0, noise * self.value / n_children, n_children)
        # Renormalize to conserve
        raw = raw * (self.value / raw.sum())
        self.children = [PACNode(v, self.depth + 1) for v in raw]


def evolve_pac_tree(initial_value: float, max_depth: int, max_children: int,
                    n_iterations: int = 100, noise: float = 0.01,
                    seed: int = 42) -> Dict:
    """
    Evolve a PAC tree with given depth and branching constraints.

    Returns stability metrics:
    - conservation_error: how well PAC is maintained
    - depth_reached: actual depth achieved
    - total_nodes: number of nodes
    - variance_ratio: variance of leaf values / mean (stability measure)
    - collapse_events: how many splits failed (negative values)
    """
    np.random.seed(seed)
    results = []

    for iteration in range(n_iterations):
        root = PACNode(initial_value)

        # Grow the tree
        frontier = [root]
        total_nodes = 1
        collapse_events = 0
        max_depth_reached = 0

        for d in range(max_depth):
            new_frontier = []
            for node in frontier:
                n_kids = min(max_children, np.random.randint(2, max_children + 1))
                node.split(n_kids, noise=noise)

                for child in node.children:
                    if child.value <= 0:
                        collapse_events += 1
                    else:
                        new_frontier.append(child)
                        total_nodes += 1

                max_depth_reached = max(max_depth_reached, d + 1)

            frontier = new_frontier
            if not frontier:
                break

        # Measure conservation
        leaf_values = [n.value for n in frontier] if frontier else [0]
        total_leaf = sum(leaf_values)
        conservation_error = abs(total_leaf - initial_value) / initial_value

        # Measure stability (low variance = stable)
        if len(leaf_values) > 1:
            variance_ratio = np.std(leaf_values) / np.mean(leaf_values) if np.mean(leaf_values) > 0 else float('inf')
        else:
            variance_ratio = 0

        results.append({
            'conservation_error': conservation_error,
            'depth_reached': max_depth_reached,
            'total_nodes': total_nodes,
            'n_leaves': len(leaf_values),
            'variance_ratio': variance_ratio,
            'collapse_events': collapse_events,
            'leaf_sum': total_leaf,
        })

    # Aggregate
    errors = [r['conservation_error'] for r in results]
    collapses = [r['collapse_events'] for r in results]
    variances = [r['variance_ratio'] for r in results]

    return {
        'max_depth': max_depth,
        'max_children': max_children,
        'n_iterations': n_iterations,
        'noise': noise,
        'mean_conservation_error': float(np.mean(errors)),
        'max_conservation_error': float(np.max(errors)),
        'mean_collapses': float(np.mean(collapses)),
        'total_collapses': int(sum(collapses)),
        'mean_variance_ratio': float(np.mean(variances)),
        'stability_score': float(1.0 - np.mean(collapses) / max(1, np.mean([r['total_nodes'] for r in results]))),
        'mean_depth_reached': float(np.mean([r['depth_reached'] for r in results])),
        'mean_leaves': float(np.mean([r['n_leaves'] for r in results])),
        'iterations': results,
    }


# =============================================================================
# Results I/O
# =============================================================================

def save_results(data: Dict, experiment_name: str,
                base_dir: str = None) -> str:
    """Save results as JSON with timestamp."""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(base_dir, filename)

    # Make everything serializable
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            result = convert(obj)
            if result is not obj:
                return result
            return super().default(obj)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    print(f"Results saved: {filepath}")
    return filepath

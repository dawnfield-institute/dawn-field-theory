"""
Milestone 4: Shared utilities for experiment scripts.
"""

import json
import os
import time
import math
import numpy as np
from scipy import stats
from datetime import datetime, timezone

# Landauer minimum energy threshold (kT·ln2, kT=1 in natural units).
# Mirrors constants.LANDAUER_MIN; kept here to avoid a circular import
# when energy_cascade / measure_exponent are called from utils itself.
_LANDAUER_MIN = math.log(2)


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


# ============================================================
# CASCADE ENGINE
# Shared by exp_03, exp_14, and any future turbulence experiments.
# Canonical version: exp_03_turbulence_mode_scaling.py (reference).
# ============================================================

def energy_cascade(injection_energy, n_scales, n_modes=8,
                   n_samples=15000, coupling_decay=0.3,
                   nonlinear_strength=0.3, coupling_matrix=None):
    """
    PAC energy cascade via eigenvalue-based partitioning.

    At each scale:
    1. Energy distributes across modes with coupling matrix C
    2. Eigenvalue analysis: organized fraction = lambda_max / sum(lambda)
    3. Organized energy stays at this scale (structure)
    4. Remaining energy transfers to next scale (cascade)

    The 0.98 transfer factor represents ~2% scale-to-scale dissipation,
    consistent with Landauer minimum floor: below kT*ln(2) no transfer occurs.

    Parameters
    ----------
    injection_energy : float
        Energy injected at the largest scale.
    n_scales : int
        Number of cascade scales (wavenumber octaves).
    n_modes : int
        Number of interacting modes per scale (the key control parameter).
    n_samples : int
        Monte Carlo samples for covariance estimation.
    coupling_decay : float
        Exponential decay rate of the mode-coupling matrix C[i,j].
    nonlinear_strength : float
        Strength of the nonlinear feedback term from the previous scale.
    coupling_matrix : ndarray (n_modes, n_modes), optional
        Injected base coupling matrix (e.g. a Dynkin graph-distance kernel,
        see ade_cascade/core/coupling.py). Default None reproduces the
        legacy kernel exp(-|i-j| * coupling_decay) exactly; the A-family
        path diagram under a graph-distance kernel equals that legacy kernel.

    Returns
    -------
    list of dicts, one per scale, with keys:
        k_index, wavenumber, P_input, org_fraction, E_organized,
        E_transfer, participation_ratio, alive
    """
    results = []
    P = injection_energy
    prev_dominant = None

    for k_idx in range(n_scales):
        if P < 1e-18:
            results.append({
                'k_index': k_idx, 'wavenumber': 2**(k_idx + 1),
                'P_input': 0, 'org_fraction': 0, 'alive': False
            })
            continue

        # Structured coupling matrix: C[i,j] = exp(-|i-j| * coupling_decay),
        # or an injected base matrix (must match n_modes)
        if coupling_matrix is not None:
            if coupling_matrix.shape != (n_modes, n_modes):
                raise ValueError(
                    f"coupling_matrix shape {coupling_matrix.shape} != "
                    f"({n_modes}, {n_modes})")
            C = coupling_matrix.copy()
        else:
            C = np.zeros((n_modes, n_modes))
            for i in range(n_modes):
                for j in range(n_modes):
                    C[i, j] = np.exp(-abs(i - j) * coupling_decay)

        # Nonlinear feedback from dominant eigenvector of previous scale
        if prev_dominant is not None:
            bias = np.outer(prev_dominant, prev_dominant)
            bias /= (np.max(np.abs(bias)) + 1e-15)
            C = C + bias * nonlinear_strength

        # Ensure C is symmetric and positive definite
        C = (C + C.T) / 2
        eigs_C = np.linalg.eigvalsh(C)
        psd_shift = 0.0
        if np.min(eigs_C) < 1e-10:
            psd_shift = abs(np.min(eigs_C)) + 1e-6
            C += np.eye(n_modes) * psd_shift

        # Energy distribution across modes
        means = P * np.exp(-np.arange(n_modes) * coupling_decay)
        means *= P / np.sum(means)

        try:
            sf = P / (np.trace(C) / n_modes) * 0.2
            samples = np.abs(np.random.multivariate_normal(
                means, C * sf, size=n_samples))
        except Exception:
            samples = np.random.exponential(
                P / n_modes, (n_samples, n_modes))

        # Eigenvalue analysis of sample covariance
        cov = np.cov(samples.T)
        eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 1e-30)
        total_var = np.sum(eigenvalues)
        org_frac = eigenvalues[-1] / total_var

        E_org = P * org_frac
        E_transfer = P * (1 - org_frac)

        # Landauer floor: transfer cannot fall below kT·ln(2)
        if E_transfer < _LANDAUER_MIN and P > _LANDAUER_MIN:
            E_transfer = _LANDAUER_MIN
            E_org = P - E_transfer
            org_frac = E_org / P

        _, eigvecs = np.linalg.eigh(cov)
        prev_dominant = eigvecs[:, -1]

        results.append({
            'k_index': k_idx,
            'wavenumber': 2**(k_idx + 1),
            'P_input': P,
            'org_fraction': org_frac,
            'E_organized': E_org,
            'E_transfer': E_transfer,
            'participation_ratio': (np.sum(eigenvalues)**2
                                    / np.sum(eigenvalues**2)),
            'psd_shift': psd_shift,
            'alive': True
        })

        P = E_transfer * 0.98  # ~2% scale-to-scale dissipation

    return results


def measure_exponent(results, trim=2):
    """
    Extract spectral exponent from cascade results via log-log regression.

    Fits E(k) ~ k^slope to the power spectrum, trimming `trim` points
    from each end to exclude injection and dissipation ranges.

    Parameters
    ----------
    results : list of dicts
        Output of energy_cascade().
    trim : int
        Number of scales to trim from each end of the inertial range.

    Returns
    -------
    (slope, r_squared, avg_org_fraction, std_error)
    Returns (None, None, None, None) if insufficient data.
    """
    alive = [r for r in results if r['alive'] and r['P_input'] > 1e-15]
    if len(alive) <= 2 * trim + 3:
        return None, None, None, None

    k_arr = np.array([r['wavenumber'] for r in alive])
    e_arr = np.array([r['P_input'] for r in alive])

    lk = np.log10(k_arr[trim:-trim])
    le = np.log10(e_arr[trim:-trim])

    if len(lk) < 4:
        return None, None, None, None

    slope, intercept, rval, pval, stderr = stats.linregress(lk, le)
    avg_org = np.mean([r['org_fraction'] for r in alive[trim:-trim]])

    return slope, rval**2, avg_org, stderr

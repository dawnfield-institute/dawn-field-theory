"""
ade_cascade core -- registered arm runner and statistics.

All values locked by journals/2026-07-17_ade-cascade-round1-preregistration.md:
canonical engine params, 100 seeds via seed_i = 42 + i*1000, percentile
bootstrap (10k resamples) 95% CI on the mean exponent. PSD-shift activation
recorded per arm (threat T1; contamination flag at max shift > 1e-3).
"""

import sys
import numpy as np
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_EXPERIMENTS / "milestone4" / "core"))

from utils import energy_cascade, measure_exponent   # noqa: E402

CANON = dict(n_scales=25, n_samples=15000, coupling_decay=0.1,
             nonlinear_strength=0.3)
N_SEEDS = 100
N_BOOT = 10000
T1_SHIFT_LIMIT = 1e-3


def run_arm(name, coupling, n_seeds=N_SEEDS):
    """Run one registered arm; returns per-seed exponents + T1 diagnostics."""
    exps, max_shift, shift_scales = [], 0.0, 0
    n_modes = coupling.shape[0]
    for i in range(n_seeds):
        np.random.seed(42 + i * 1000)
        res = energy_cascade(1.0, n_modes=n_modes,
                             coupling_matrix=coupling, **CANON)
        slope, r2, org, se = measure_exponent(res)
        if slope is not None:
            exps.append(slope)
        shifts = [r.get('psd_shift', 0.0) for r in res if r.get('alive')]
        if shifts:
            max_shift = max(max_shift, max(shifts))
            shift_scales += sum(1 for s in shifts if s > 0)
    exps = np.array(exps)
    return {
        "arm": name,
        "n_modes": int(n_modes),
        "n_valid": int(len(exps)),
        "mean": float(np.mean(exps)),
        "std": float(np.std(exps, ddof=1)),
        "ci95": bootstrap_ci(exps),
        "t1_max_psd_shift": float(max_shift),
        "t1_shift_activations": int(shift_scales),
        "t1_contaminated": bool(max_shift > T1_SHIFT_LIMIT),
        "per_seed": [float(x) for x in exps],
    }


def bootstrap_ci(x, n_boot=N_BOOT, seed=7):
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(x, size=len(x), replace=True))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def cis_overlap(a, b):
    return not (a["ci95"][1] < b["ci95"][0] or b["ci95"][1] < a["ci95"][0])

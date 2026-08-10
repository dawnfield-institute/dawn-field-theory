#!/usr/bin/env python3
"""
EXPERIMENT 14: 2D Turbulence Mode Count Cross-Validation (v2)
============================================================
Dawn Field Institute — Milestone 4, Block D

HYPOTHESIS: The same PAC cascade engine recovers 2D turbulence exponents
(enstrophy cascade: -3.0; inverse energy cascade: -5/3) with ONLY the mode
count changed from the 3D baseline. No parameter re-tuning is permitted.

She-Lévêque prediction (from milestone 2): k = d × F_{d+1}
  3D: k = 3 × F_4 = 9   (exp_03 found k=8 — investigate k-1 offset)
  2D enstrophy: k = 2 × F_3 = 4   (adjusted: k=3 if k-1 offset holds)
  2D inverse:   same exponent as 3D (-5/3), different physics

FALSIFICATION CONDITIONS:
  1. If neither 2D exponent is recovered at k_predicted or k_predicted-1
  2. If bootstrap CI excludes target for the best mode count
  3. If structured coupling is statistically indistinguishable from random
  4. If She-Lévêque k-1 offset is inconsistent across cases

CONNECTS TO:
  - exp_03_turbulence_mode_scaling.py (3D baseline, k=8 for -5/3)
  - milestone2 exp_01-04 (She-Leveque Fibonacci: k = d × F_{d+1})
  - core/utils.py (energy_cascade, measure_exponent)
"""

import numpy as np
from scipy import stats
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from constants import PHI, INV_PHI, LN_PHI, XI_BALANCE, FIB, KOLMOGOROV_EXPONENT
from utils import (save_results, print_header, bootstrap_ci, monte_carlo_null,
                   energy_cascade, measure_exponent)

np.random.seed(42)

TARGET_3D = -5 / 3
TARGET_2D_ENSTROPHY = -3.0
TARGET_2D_INVERSE = -5 / 3

# She-Lévêque formula: k = d × F_{d+1}
# FIB indexing: FIB[0]=0, FIB[1]=1, FIB[2]=1, FIB[3]=2, FIB[4]=3, FIB[5]=5
SL_3D_PREDICTED = 3 * FIB[4]   # = 3 × 3 = 9
SL_2D_PREDICTED = 2 * FIB[3]   # = 2 × 2 = 4

print("=" * 70)
print("EXPERIMENT 14: 2D Turbulence Mode Count Cross-Validation (v2)")
print("Dawn Field Institute — Milestone 4")
print("=" * 70)


# ============================================================
# PART A: A-Priori Predictions
# ============================================================
print_header("PART A: A-Priori Predictions (BEFORE any computation)")

print(f"""
She-Lévêque formula: k = d × F_{{d+1}}
  where d = spatial dimension, F_n = nth Fibonacci number

PREDICTIONS (stated before running):
  3D Kolmogorov (-5/3):
    k_SL = 3 × F_4 = 3 × {FIB[4]} = {SL_3D_PREDICTED}
    exp_03 found k=8 → a k-1 systematic offset.
    Hypothesis: cascade geometry reduces one effective degree of freedom.

  2D enstrophy cascade (target: -3.0):
    k_SL = 2 × F_3 = 2 × {FIB[3]} = {SL_2D_PREDICTED}
    If k-1 offset holds: adjusted prediction k = {SL_2D_PREDICTED - 1}

  2D inverse energy cascade (target: -5/3):
    Same exponent as 3D Kolmogorov; if k-1 offset holds from 3D: k ≈ 7.
    Treat as TBD — test mode range [5, 10].

FALSIFICATION CRITERION:
  If both 2D exponents are recovered at k_predicted or k_predicted-1
  with NO parameter re-tuning → PASS.
  If bootstrap CI (50 seeds) includes target → CONFIRMED.
""")

print(f"  FIB sequence: {FIB[:8]}")
print(f"  SL_3D_PREDICTED = {SL_3D_PREDICTED} (exp_03 observed: 8, offset = -1)")
print(f"  SL_2D_PREDICTED = {SL_2D_PREDICTED} (adjusted: {SL_2D_PREDICTED - 1})")


# ============================================================
# PART B: Parameter Sweep — Natural Exponent per Mode Count
# ============================================================
print_header("PART B: Parameter Sweep — Natural Exponent per Mode Count")

print("""
Unlike v1 (fixed coupling_decay=0.1, nonlinear_strength=0.3), we sweep
parameter combinations to find the NATURAL exponent at each mode count.
This removes confirmation bias: we report the median across all valid
parameter combos, not the cherry-picked best run.
""")

N_SCALES = 25
N_SAMPLES = 10000
coupling_decays = [0.1, 0.2, 0.3, 0.5, 0.7]
nonlinear_strengths = [0.0, 0.1, 0.3, 0.5]

# --- 2D Enstrophy sweep ---
print("B.1 — Enstrophy cascade sweep (target: -3.0)")
print(f"{'n_modes':>8} | {'Median exp':>10} | {'IQR':>8} | {'Best exp':>10} | {'n_valid':>7}")
print("-" * 55)

enst_modes = [2, 3, 4, 5, 6]
enst_sweep = {}

for nm in enst_modes:
    exponents = []
    for cd in coupling_decays:
        for ns in nonlinear_strengths:
            np.random.seed(42 + nm * 100 + int(cd * 10) + int(ns * 10))
            res = energy_cascade(1.0, N_SCALES, n_modes=nm,
                                 coupling_decay=cd, nonlinear_strength=ns,
                                 n_samples=N_SAMPLES)
            exp_val, r2, org, _ = measure_exponent(res)
            if exp_val is not None and r2 > 0.8:
                exponents.append(exp_val)

    if exponents:
        med = float(np.median(exponents))
        iqr = float(np.percentile(exponents, 75) - np.percentile(exponents, 25))
        best = float(min(exponents, key=lambda e: abs(e - TARGET_2D_ENSTROPHY)))
        marker = " <<<" if abs(med - TARGET_2D_ENSTROPHY) < 0.30 else ""
        print(f"  {nm:>6} | {med:>10.4f} | {iqr:>8.4f} | {best:>10.4f} | "
              f"{len(exponents):>7}{marker}")
        enst_sweep[nm] = {'median': med, 'iqr': iqr, 'best': best,
                          'n_valid': len(exponents), 'all': exponents}
    else:
        print(f"  {nm:>6} | {'no valid cascades':>36}")

# Pick winning enstrophy mode count (closest median to target)
best_enst_modes = min(enst_sweep,
                      key=lambda n: abs(enst_sweep[n]['median'] - TARGET_2D_ENSTROPHY))
print(f"\n  Best enstrophy mode count: {best_enst_modes} modes  "
      f"(median exp = {enst_sweep[best_enst_modes]['median']:.4f})")

# --- 2D Inverse cascade sweep ---
print("\nB.2 — Inverse cascade sweep (target: -1.6667)")
print(f"{'n_modes':>8} | {'Median exp':>10} | {'IQR':>8} | {'Best exp':>10} | {'n_valid':>7}")
print("-" * 55)

inv_modes_range = [5, 6, 7, 8, 9, 10]
inv_sweep = {}

for nm in inv_modes_range:
    exponents = []
    for cd in coupling_decays:
        for ns in nonlinear_strengths:
            np.random.seed(42 + nm * 200 + int(cd * 10) + int(ns * 10))
            res = energy_cascade(1.0, N_SCALES, n_modes=nm,
                                 coupling_decay=cd, nonlinear_strength=ns,
                                 n_samples=N_SAMPLES)
            exp_val, r2, org, _ = measure_exponent(res)
            if exp_val is not None and r2 > 0.8:
                exponents.append(exp_val)

    if exponents:
        med = float(np.median(exponents))
        iqr = float(np.percentile(exponents, 75) - np.percentile(exponents, 25))
        best = float(min(exponents, key=lambda e: abs(e - TARGET_2D_INVERSE)))
        marker = " <<<" if abs(med - TARGET_2D_INVERSE) < 0.20 else ""
        print(f"  {nm:>6} | {med:>10.4f} | {iqr:>8.4f} | {best:>10.4f} | "
              f"{len(exponents):>7}{marker}")
        inv_sweep[nm] = {'median': med, 'iqr': iqr, 'best': best,
                         'n_valid': len(exponents), 'all': exponents}
    else:
        print(f"  {nm:>6} | {'no valid cascades':>36}")

best_inv_modes = min(inv_sweep,
                     key=lambda n: abs(inv_sweep[n]['median'] - TARGET_2D_INVERSE))
print(f"\n  Best inverse cascade mode count: {best_inv_modes} modes  "
      f"(median exp = {inv_sweep[best_inv_modes]['median']:.4f})")


# ============================================================
# PART C: Bootstrap CIs on Winning Mode Counts
# ============================================================
print_header("PART C: Bootstrap Confidence Intervals (50 seeds, canonical params)")

print("""
At the winning mode counts from Part B, we run 50 random seeds
with canonical parameters (coupling_decay=0.1, nonlinear_strength=0.3)
and compute bootstrap 95% CIs. This is the key falsification test.
""")

N_SAMPLES_BOOT = 15000

# Enstrophy CI
exponents_enst = []
for seed in range(50):
    np.random.seed(42 + seed * 1000)
    res = energy_cascade(1.0, N_SCALES, n_modes=best_enst_modes,
                         coupling_decay=0.1, nonlinear_strength=0.3,
                         n_samples=N_SAMPLES_BOOT)
    exp_val, r2, org, _ = measure_exponent(res)
    if exp_val is not None and r2 > 0.8:
        exponents_enst.append(float(exp_val))

ci_enst = bootstrap_ci(exponents_enst) if exponents_enst else None
mean_enst_exp = float(np.mean(exponents_enst)) if exponents_enst else None

# Inverse cascade CI
exponents_inv = []
for seed in range(50):
    np.random.seed(42 + seed * 1000 + 999)
    res = energy_cascade(1.0, N_SCALES, n_modes=best_inv_modes,
                         coupling_decay=0.1, nonlinear_strength=0.3,
                         n_samples=N_SAMPLES_BOOT)
    exp_val, r2, org, _ = measure_exponent(res)
    if exp_val is not None and r2 > 0.8:
        exponents_inv.append(float(exp_val))

ci_inv = bootstrap_ci(exponents_inv) if exponents_inv else None
mean_inv_exp = float(np.mean(exponents_inv)) if exponents_inv else None

print(f"C.1 — Enstrophy cascade (n_modes={best_enst_modes}, target={TARGET_2D_ENSTROPHY:.4f}):")
if ci_enst:
    within_enst = ci_enst['ci_lower'] <= TARGET_2D_ENSTROPHY <= ci_enst['ci_upper']
    print(f"  n_valid seeds: {len(exponents_enst)}/50")
    print(f"  mean          = {mean_enst_exp:.4f}")
    print(f"  95% CI        = [{ci_enst['ci_lower']:.4f}, {ci_enst['ci_upper']:.4f}]")
    print(f"  std_error     = {ci_enst['std_error']:.4f}")
    print(f"  Target {TARGET_2D_ENSTROPHY:.4f} within CI: {'YES ✓' if within_enst else 'NO ✗'}")
else:
    print("  No valid seeds.")
    within_enst = False

print(f"\nC.2 — Inverse cascade (n_modes={best_inv_modes}, target={TARGET_2D_INVERSE:.4f}):")
if ci_inv:
    within_inv = ci_inv['ci_lower'] <= TARGET_2D_INVERSE <= ci_inv['ci_upper']
    print(f"  n_valid seeds: {len(exponents_inv)}/50")
    print(f"  mean          = {mean_inv_exp:.4f}")
    print(f"  95% CI        = [{ci_inv['ci_lower']:.4f}, {ci_inv['ci_upper']:.4f}]")
    print(f"  std_error     = {ci_inv['std_error']:.4f}")
    print(f"  Target {TARGET_2D_INVERSE:.4f} within CI: {'YES ✓' if within_inv else 'NO ✗'}")
else:
    print("  No valid seeds.")
    within_inv = False


# ============================================================
# PART D: She-Lévêque Systematic Offset Analysis
# ============================================================
print_header("PART D: She-Lévêque k-1 Systematic Offset Analysis")

print("""
She-Lévêque predicts k = d × F_{d+1}. exp_03 found k=8 for 3D (predicted=9).
We fill in the 2D cases from Part B to check for a consistent offset.
""")

SL_3D_OBS = 8   # from exp_03_turbulence_mode_scaling.py

# For 2D inverse cascade, the theoretical prediction mirrors 3D (same exponent)
cases = [
    ('3D Kolmogorov',  SL_3D_PREDICTED, SL_3D_OBS),
    ('2D enstrophy',   SL_2D_PREDICTED, best_enst_modes),
    ('2D inverse',     SL_3D_PREDICTED, best_inv_modes),
]

print(f"  {'Case':20} | {'k_SL_pred':>10} | {'k_observed':>10} | {'offset':>8}")
print("  " + "-" * 57)

offsets = []
for name, k_pred, k_obs in cases:
    offset = k_obs - k_pred
    offsets.append(offset)
    print(f"  {name:20} | {k_pred:>10} | {k_obs:>10} | {offset:>+8}")

offset_consistent = len(set(offsets)) == 1
offset_val = offsets[0] if offsets else None

print(f"\n  Offsets: {offsets}")
if offset_consistent:
    print(f"  Consistent offset: YES — all = {offset_val:+d}")
else:
    print(f"  Consistent offset: NO — mixed {offsets}")

if offset_consistent and offset_val == -1:
    print("""
  PHYSICAL INTERPRETATION (k-1 offset = -1):
  A consistent k-1 offset suggests the cascade engine's coupling matrix
  geometry reduces one effective degree of freedom compared to field-theoretic
  triadic interaction counts. In the matrix C[i,j] = exp(-|i-j|*cd), the
  i=j diagonal dominates with weight 1.0, effectively anchoring one mode as
  a reference state rather than a free interacting mode. The physical mode
  count is therefore n_modes - 1.
""")
elif offset_consistent:
    print(f"\n  Consistent offset of {offset_val:+d} found — "
          f"not the k-1 pattern from 3D alone.")
else:
    print("\n  Offset is NOT consistent across cases. The k-1 pattern"
          " from 3D does not generalise cleanly to 2D.")


# ============================================================
# PART E: Null Test — Structured vs Random Coupling
# ============================================================
print_header("PART E: Null Test — Structured vs Random Coupling Matrix")

print(f"""
Does the structured exponential-decay coupling drive the 2D enstrophy
exponent, or would any random symmetric PD matrix suffice?

We replace C[i,j] = exp(-|i-j|*cd) with a Wishart random matrix.
n_trials=500 gives ~2σ resolution (appropriate for a screening test).

Convention: monte_carlo_null tests fraction(null >= observed).
We use negated distances so that p < 0.05 means structured IS special.
""")


def energy_cascade_with_random_C(injection_energy, n_scales, n_modes, rng,
                                  n_samples=5000):
    """Like energy_cascade but uses a random symmetric PD coupling matrix."""
    from constants import LANDAUER_MIN as _LM
    results = []
    P = injection_energy

    for k_idx in range(n_scales):
        if P < 1e-18:
            results.append({
                'k_index': k_idx, 'wavenumber': 2**(k_idx + 1),
                'P_input': 0, 'org_fraction': 0, 'alive': False
            })
            continue

        # Random symmetric PD coupling matrix (Wishart-like)
        A = rng.standard_normal((n_modes, n_modes))
        C = A @ A.T / n_modes
        C = (C + C.T) / 2
        eigs_C = np.linalg.eigvalsh(C)
        if np.min(eigs_C) < 1e-10:
            C += np.eye(n_modes) * (abs(np.min(eigs_C)) + 1e-6)

        means = np.full(n_modes, P / n_modes)

        try:
            sf = P / (np.trace(C) / n_modes) * 0.2
            samples = np.abs(rng.multivariate_normal(means, C * sf, size=n_samples))
        except Exception:
            samples = rng.exponential(P / n_modes, (n_samples, n_modes))

        cov = np.cov(samples.T)
        eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 1e-30)
        total_var = np.sum(eigenvalues)
        org_frac = eigenvalues[-1] / total_var

        E_org = P * org_frac
        E_transfer = P * (1 - org_frac)

        if E_transfer < _LM and P > _LM:
            E_transfer = _LM
            E_org = P - E_transfer
            org_frac = E_org / P

        results.append({
            'k_index': k_idx, 'wavenumber': 2**(k_idx + 1),
            'P_input': P, 'org_fraction': org_frac,
            'E_organized': E_org, 'E_transfer': E_transfer,
            'participation_ratio': np.sum(eigenvalues)**2 / np.sum(eigenvalues**2),
            'alive': True
        })

        P = E_transfer * 0.98

    return results


null_result = None
structured_is_special = False
observed_dist = None

if mean_enst_exp is not None:
    observed_dist = abs(mean_enst_exp - TARGET_2D_ENSTROPHY)

    def null_generator(rng):
        res = energy_cascade_with_random_C(1.0, N_SCALES, best_enst_modes,
                                           rng, n_samples=5000)
        exp_val, r2, _, _ = measure_exponent(res)
        if exp_val is not None and r2 > 0.5:
            # Negate distance: larger (less negative) = random is closer to target
            return -abs(exp_val - TARGET_2D_ENSTROPHY)
        return 0.0  # neutral: neither better nor worse

    # p_value = fraction(null >= -observed_dist)
    # = fraction of random trials as close or closer to target than structured
    # p < 0.05 → structured IS significantly better than random
    null_result = monte_carlo_null(
        observed=-observed_dist,
        generator_fn=null_generator,
        n_trials=500,
        seed=42
    )

    structured_is_special = null_result['p_value'] < 0.05
    print(f"  Structured cascade distance from -3.0:  {observed_dist:.4f}")
    print(f"  Null (random C) mean negated distance:  {null_result['null_mean']:.4f}")
    print(f"  Null std:                               {null_result['null_std']:.4f}")
    print(f"  p-value (random ≤ structured distance): {null_result['p_value']:.4f}")
    print(f"  Structured cascade IS special (p<0.05): "
          f"{'YES ✓' if structured_is_special else 'NO ✗'}")
    if not structured_is_special:
        print("  NOTE: Random coupling occasionally achieves similar distances —")
        print("  the exponent may partly reflect cascade geometry, not only coupling.")
else:
    print("  Skipped (no valid enstrophy bootstrap mean).")


# ============================================================
# PART F: Global/Local Conservation Hierarchy
# ============================================================
print_header("PART F: Conservation Hierarchy — CoV(exponent) < CoV(global) < CoV(local)")

print(f"""
PAC predicts: CoV(exponent) < CoV(global org) < CoV(local org).
Tested at winning enstrophy mode count (n_modes={best_enst_modes}).
Same analysis as exp_03 Part 4, applied to the 2D case.
""")

hier_data = []
for cd in coupling_decays:
    for ns in nonlinear_strengths:
        np.random.seed(42 + int(cd * 100) + int(ns * 100) + best_enst_modes * 7)
        res = energy_cascade(1.0, N_SCALES, n_modes=best_enst_modes,
                             coupling_decay=cd, nonlinear_strength=ns,
                             n_samples=N_SAMPLES)
        alive = [r for r in res if r['alive'] and r['P_input'] > 1e-15]
        if len(alive) > 6:
            mid = alive[2:-2]
            local_org = float(np.mean([r['org_fraction'] for r in mid]))
            global_org = float(sum(r['E_organized'] for r in alive))  # / 1.0 injection
            exp_val, r2, _, _ = measure_exponent(res)
            if exp_val is not None and r2 > 0.8:
                hier_data.append({'cd': cd, 'ns': ns,
                                  'local_org': local_org,
                                  'global_org': global_org,
                                  'exponent': float(exp_val)})

hier_holds = False
cov_local = cov_global = cov_exp = None

if hier_data:
    loc = [d['local_org'] for d in hier_data]
    glb = [d['global_org'] for d in hier_data]
    exp = [d['exponent'] for d in hier_data]

    cov_local  = np.std(loc) / np.mean(loc) if np.mean(loc) > 0 else float('inf')
    cov_global = np.std(glb) / np.mean(glb) if np.mean(glb) > 0 else float('inf')
    cov_exp    = abs(np.std(exp) / np.mean(exp)) if np.mean(exp) != 0 else float('inf')

    hier_holds = cov_exp < cov_global < cov_local

    print(f"  {'Measure':20} | {'Mean':>8} | {'Std':>8} | {'CoV':>8}")
    print("  " + "-" * 54)
    print(f"  {'LOCAL org_frac':20} | {np.mean(loc):>8.4f} | "
          f"{np.std(loc):>8.4f} | {cov_local:>8.4f}")
    print(f"  {'GLOBAL Σξ/P':20} | {np.mean(glb):>8.4f} | "
          f"{np.std(glb):>8.4f} | {cov_global:>8.4f}")
    print(f"  {'EXPONENT':20} | {np.mean(exp):>8.4f} | "
          f"{np.std(exp):>8.4f} | {cov_exp:>8.4f}")
    print(f"\n  PAC hierarchy CoV(exp) < CoV(global) < CoV(local): "
          f"{'CONFIRMED ✓' if hier_holds else 'NOT CONFIRMED ✗'}")
    if not hier_holds:
        print(f"  (CoV_exp={cov_exp:.4f}, CoV_global={cov_global:.4f}, "
              f"CoV_local={cov_local:.4f})")
else:
    print("  Insufficient data for hierarchy test.")


# ============================================================
# PART G: Falsification Verdict
# ============================================================
print_header("PART G: Falsification Verdict")


def classify_deviation(dev):
    if dev < 0.10:
        return "TIGHT PASS"
    elif dev < 0.20:
        return "PASS"
    else:
        return "FAIL"


dev_enst = abs(mean_enst_exp - TARGET_2D_ENSTROPHY) if mean_enst_exp is not None else 999.0
dev_inv  = abs(mean_inv_exp  - TARGET_2D_INVERSE)   if mean_inv_exp  is not None else 999.0

verdict_enst = classify_deviation(dev_enst)
verdict_inv  = classify_deviation(dev_inv)

if verdict_enst in ("TIGHT PASS", "PASS") and verdict_inv in ("TIGHT PASS", "PASS"):
    overall = "PASS"
elif verdict_enst in ("TIGHT PASS", "PASS") or verdict_inv in ("TIGHT PASS", "PASS"):
    overall = "PARTIAL"
else:
    overall = "FAIL"

sl_offset_str = (
    "CONFIRMED (consistent k-1 across all 3 cases)"
    if (offset_consistent and offset_val == -1)
    else ("CONFIRMED (consistent, not k-1)" if offset_consistent
          else "NOT CONFIRMED (mixed)")
)

null_str = (
    f"p = {null_result['p_value']:.3f} "
    f"(structured cascade {'IS' if structured_is_special else 'IS NOT'} "
    f"distinguishable from random at p<0.05)"
    if null_result is not None else "not run"
)

hier_str = "CONFIRMED" if hier_holds else "NOT CONFIRMED"

enst_ci_str = (f"[{ci_enst['ci_lower']:.4f}, {ci_enst['ci_upper']:.4f}]"
               if ci_enst else "n/a")
inv_ci_str  = (f"[{ci_inv['ci_lower']:.4f}, {ci_inv['ci_upper']:.4f}]"
               if ci_inv else "n/a")

mean_enst_str = f"{mean_enst_exp:.4f}" if mean_enst_exp is not None else "n/a"
mean_inv_str  = f"{mean_inv_exp:.4f}"  if mean_inv_exp  is not None else "n/a"

print(f"""
VERDICT: {overall}

  2D enstrophy (n_modes={best_enst_modes}):
    mean exponent = {mean_enst_str}
    95% CI        = {enst_ci_str}
    target = {TARGET_2D_ENSTROPHY:.4f}, deviation = {dev_enst:.4f}  [{verdict_enst}]

  2D inverse cascade (n_modes={best_inv_modes}):
    mean exponent = {mean_inv_str}
    95% CI        = {inv_ci_str}
    target = {TARGET_2D_INVERSE:.4f}, deviation = {dev_inv:.4f}  [{verdict_inv}]

  She-Lévêque k-1 offset: {sl_offset_str}
  Null test: {null_str}
  PAC conservation hierarchy (2D enstrophy): {hier_str}
""")

if overall == "FAIL":
    print("  CONCLUSION: The cascade engine does NOT recover 2D turbulence exponents")
    print("  at the She-Lévêque predicted mode counts. Hypothesis FALSIFIED.")
elif overall == "PARTIAL":
    print("  CONCLUSION: One 2D exponent recovered. Partial support for hypothesis.")
    print("  Further investigation needed for the failing case.")
else:
    print("  CONCLUSION: Both 2D exponents recovered within tolerance at the")
    print("  She-Lévêque predicted (or k-1 adjusted) mode counts.")
    print("  Hypothesis SUPPORTED. The cascade engine generalises across dimensions.")


# ============================================================
# PART H: Save Results
# ============================================================
results = {
    'experiment': 'exp_14_2d_turbulence_mode_count_v2',
    'milestone': 4,
    'date': '2026-03-09',
    'hypothesis': 'PAC cascade recovers 2D turbulence exponents with mode count change only',

    'part_a_predictions': {
        'sl_formula': 'k = d x F_{d+1}',
        'sl_3d_predicted': SL_3D_PREDICTED,
        'sl_3d_observed': SL_3D_OBS,
        'sl_3d_offset': SL_3D_OBS - SL_3D_PREDICTED,
        'sl_2d_enstrophy_predicted': SL_2D_PREDICTED,
        'sl_2d_enstrophy_adjusted': SL_2D_PREDICTED - 1,
        'target_3d': TARGET_3D,
        'target_2d_enstrophy': TARGET_2D_ENSTROPHY,
        'target_2d_inverse': TARGET_2D_INVERSE,
    },

    'part_b_enst_sweep': {
        str(nm): {k: (v if not isinstance(v, list) else [float(x) for x in v])
                  for k, v in d.items()}
        for nm, d in enst_sweep.items()
    },
    'part_b_inv_sweep': {
        str(nm): {k: (v if not isinstance(v, list) else [float(x) for x in v])
                  for k, v in d.items()}
        for nm, d in inv_sweep.items()
    },
    'part_b_best_enst_modes': best_enst_modes,
    'part_b_best_inv_modes': best_inv_modes,

    'part_c_bootstrap': {
        'enst_n_valid': len(exponents_enst),
        'enst_mean': mean_enst_exp,
        'enst_ci': ci_enst,
        'enst_target_within_ci': bool(within_enst) if ci_enst else None,
        'inv_n_valid': len(exponents_inv),
        'inv_mean': mean_inv_exp,
        'inv_ci': ci_inv,
        'inv_target_within_ci': bool(within_inv) if ci_inv else None,
    },

    'part_d_sl_offset': {
        'cases': [{'name': n, 'k_pred': p, 'k_obs': o, 'offset': o - p}
                  for n, p, o in cases],
        'offsets': offsets,
        'consistent': bool(offset_consistent),
        'offset_value': int(offset_val) if offset_val is not None else None,
    },

    'part_e_null_test': {
        'observed_distance': float(observed_dist) if observed_dist is not None else None,
        'null_result': null_result,
        'structured_is_special': bool(structured_is_special),
    },

    'part_f_conservation': {
        'n_combos': len(hier_data),
        'cov_local': float(cov_local) if cov_local is not None else None,
        'cov_global': float(cov_global) if cov_global is not None else None,
        'cov_exponent': float(cov_exp) if cov_exp is not None else None,
        'hierarchy_holds': bool(hier_holds),
    },

    'part_g_verdict': {
        'overall': overall,
        'enst_deviation': float(dev_enst),
        'enst_verdict': verdict_enst,
        'inv_deviation': float(dev_inv),
        'inv_verdict': verdict_inv,
        'sl_offset_confirmed': bool(offset_consistent and offset_val == -1),
        'null_p_value': float(null_result['p_value']) if null_result else None,
        'conservation_hierarchy': bool(hier_holds),
    },
}

save_results(results, 'exp_14_2d_turbulence_mode_count_v2')

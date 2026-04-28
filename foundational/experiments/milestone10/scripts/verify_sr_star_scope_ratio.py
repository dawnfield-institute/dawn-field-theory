"""
Verify: sr* = gamma / ln(phi) = 1.199504...

The scope asymmetry prediction: the critical spectral radius where
h1 (mode-sequence entropy rate) crosses Xi is exactly the ratio of
PAC scope (gamma) to SEC scope (ln(phi)).

gamma = 0.5772156649... (Euler-Mascheroni, PAC's global/definitional cost)
ln(phi) = 0.4812118251... (SEC's local/generative cost)
gamma / ln(phi) = 1.19950432...

Xi = gamma + ln(phi) = 1.05842749...

The prediction: bisecting for sr* where h1 = Xi should yield
sr* = gamma/ln(phi) = 1.19950...

Test across multiple N values and seed counts.
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import (
    SelfApplicator,
    measure_entropy_rate,
    GAMMA_EM, LN_PHI, XI_BALANCE,
)

XI = GAMMA_EM + LN_PHI
SCOPE_RATIO = GAMMA_EM / LN_PHI  # 1.19950432...

print(f"Constants:")
print(f"  gamma     = {GAMMA_EM:.10f}")
print(f"  ln(phi)   = {LN_PHI:.10f}")
print(f"  Xi        = {XI:.10f}")
print(f"  gamma/ln(phi) = {SCOPE_RATIO:.10f}")
print()


def run_h1(N, sr, seed, n_steps=3000, burn_in=1000):
    """Run SelfApplicator at given sr, measure h1."""
    sa = SelfApplicator(rule_seed=seed, self_applies=True, symmetric=True, size=N)

    # Rescale W to target sr
    eigvals = np.linalg.eigvalsh(sa.W)
    current_sr = np.max(np.abs(eigvals))
    if current_sr > 1e-10:
        sa.W = sa.W * (sr / current_sr)
    sa._target_sr = sr

    # Burn in
    for _ in range(burn_in):
        sa.step()

    # Record trajectory
    traj = np.zeros((n_steps - burn_in, N))
    for t in range(n_steps - burn_in):
        sa.step()
        traj[t] = sa.state

    er = measure_entropy_rate(traj, n_modes=4, max_block=8)
    return er['h1']


def mean_h1(N, sr, n_seeds=20):
    """Average h1 across seeds."""
    vals = [run_h1(N, sr, seed) for seed in range(n_seeds)]
    return np.mean(vals), np.std(vals)


def bisect_sr_star(N, n_seeds=20, sr_low=0.8, sr_high=2.0, tol=0.005, max_iter=20):
    """Bisect to find sr* where h1 = Xi."""
    h_low, _ = mean_h1(N, sr_low, n_seeds)
    h_high, _ = mean_h1(N, sr_high, n_seeds)

    # Bracket check
    if not (h_low < XI < h_high):
        if h_low >= XI:
            sr_low = 0.5
            h_low, _ = mean_h1(N, sr_low, n_seeds)
        if h_high <= XI:
            sr_high = 3.0
            h_high, _ = mean_h1(N, sr_high, n_seeds)

    if not (h_low < XI < h_high):
        print(f"  WARNING: Cannot bracket Xi for N={N}")
        print(f"    h1({sr_low:.2f}) = {h_low:.4f}, h1({sr_high:.2f}) = {h_high:.4f}")
        return None

    for iteration in range(max_iter):
        sr_mid = (sr_low + sr_high) / 2
        h_mid, _ = mean_h1(N, sr_mid, n_seeds)

        if h_mid < XI:
            sr_low = sr_mid
            h_low = h_mid
        else:
            sr_high = sr_mid
            h_high = h_mid

        if sr_high - sr_low < tol:
            break

        if iteration % 5 == 0:
            print(f"    iter {iteration}: sr in [{sr_low:.4f}, {sr_high:.4f}], "
                  f"h1 in [{h_low:.4f}, {h_high:.4f}]")

    sr_star = (sr_low + sr_high) / 2
    h1_star, h1_std = mean_h1(N, sr_star, n_seeds)

    return {
        'sr_star': sr_star,
        'h1_star': h1_star,
        'h1_std': h1_std,
        'converged': sr_high - sr_low < tol,
    }


# ============================================================
# Main verification
# ============================================================
print("=" * 60)
print("VERIFICATION: sr* = gamma/ln(phi)?")
print("=" * 60)

results = {}
for N in [8, 16, 32]:
    print(f"\n--- N = {N} ---")
    result = bisect_sr_star(N, n_seeds=20, tol=0.003)

    if result is None:
        continue

    sr_star = result['sr_star']
    h1_star = result['h1_star']

    sr_error_pct = abs(sr_star - SCOPE_RATIO) / SCOPE_RATIO * 100
    h1_error_pct = abs(h1_star - XI) / XI * 100

    print(f"  sr*          = {sr_star:.6f}")
    print(f"  gamma/ln(phi)= {SCOPE_RATIO:.6f}")
    print(f"  sr* error    = {sr_error_pct:.2f}%")
    print(f"  h1(sr*)      = {h1_star:.6f} +/- {result['h1_std']:.4f}")
    print(f"  Xi           = {XI:.6f}")
    print(f"  h1 error     = {h1_error_pct:.2f}%")

    results[N] = {
        'sr_star': sr_star,
        'h1_star': h1_star,
        'sr_error_pct': sr_error_pct,
        'h1_error_pct': h1_error_pct,
    }


# ============================================================
# Also measure h1 at EXACTLY gamma/ln(phi) for each N
# ============================================================
print("\n" + "=" * 60)
print("DIRECT TEST: h1 at sr = gamma/ln(phi) = {:.6f}".format(SCOPE_RATIO))
print("=" * 60)

for N in [8, 16, 32]:
    h1_at_ratio, h1_std = mean_h1(N, SCOPE_RATIO, n_seeds=30)
    error_pct = abs(h1_at_ratio - XI) / XI * 100
    print(f"  N={N:3d}: h1 = {h1_at_ratio:.6f} +/- {h1_std:.4f}, "
          f"Xi = {XI:.6f}, error = {error_pct:.2f}%")


# ============================================================
# Scan h1 vs sr to see the crossing
# ============================================================
print("\n" + "=" * 60)
print("SCAN: h1 vs sr around the critical region (N=16)")
print("=" * 60)

sr_values = np.linspace(0.9, 1.5, 13)
print(f"  {'sr':>8s}  {'h1':>8s}  {'h1-Xi':>8s}")
print(f"  {'----':>8s}  {'----':>8s}  {'-----':>8s}")

for sr in sr_values:
    h1, _ = mean_h1(16, sr, n_seeds=15)
    marker = " <-- gamma/ln(phi)" if abs(sr - SCOPE_RATIO) < 0.03 else ""
    marker = marker or (" <-- Xi crossing" if abs(h1 - XI) < 0.05 else "")
    print(f"  {sr:8.4f}  {h1:8.4f}  {h1 - XI:+8.4f}{marker}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if results:
    sr_stars = [r['sr_star'] for r in results.values()]
    mean_sr = np.mean(sr_stars)
    print(f"  Mean sr* across N values: {mean_sr:.6f}")
    print(f"  gamma/ln(phi):            {SCOPE_RATIO:.6f}")
    print(f"  Difference:               {abs(mean_sr - SCOPE_RATIO):.6f} "
          f"({abs(mean_sr - SCOPE_RATIO)/SCOPE_RATIO*100:.2f}%)")
    print()
    print(f"  Prediction: sr* = gamma/ln(phi)")
    print(f"  Status: {'SUPPORTED' if abs(mean_sr - SCOPE_RATIO)/SCOPE_RATIO < 0.05 else 'NOT SUPPORTED'} "
          f"(threshold: 5%)")

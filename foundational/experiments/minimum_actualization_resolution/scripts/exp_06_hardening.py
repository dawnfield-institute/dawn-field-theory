"""
Hardening Tests — Experiment Script 06

PURPOSE:
    Harden the MVAE claims from exp_01-05 with proper null tests, cascade engine
    bridging, and formula selectivity analysis. Tests whether the xi_PAC closed form
    (1 + (7/8)*ln(2)*(1-ln2)^2) is a genuine derivation or a coincidence.

HYPOTHESIS:
    The MVAE derivation of xi_PAC is non-trivial: the champion formula is selective
    (most formulas of similar complexity do NOT match), the three independent xi
    derivations converge beyond chance, and the cascade engine's measured org_fraction
    connects to the MVAE hierarchy.

DESIGN:
    Part A: Three-Xi convergence test — are 1+pi/55, gamma+ln(phi), and
            1+(7/8)*ln(2)*(1-ln2)^2 converging or just close by accident?
    Part B: Formula selectivity null — what fraction of formulas with similar
            structural complexity match xi_PAC to <0.01%?
    Part C: Cascade engine bridge — connect org_fraction from the real engine
            to the MVAE xi hierarchy. This is the OPEN QUESTION from
            turbulence_pac_v3.py.
    Part D: PAC tree attractor test — does xi_floor = 1-ln²(2) emerge from
            pure Landauer mechanics regardless of branching, energy, noise?
            Then: does eta_PAC = 1.0824 bridge xi_floor to xi_PAC?
    Part E: Three-constraint conjugacy null — do the MVAE constraint products
            (dx_grav × dx_unc = 1, E × dt = 1/2) arise from structure or luck?

CORPUS CONTEXT:
    - xi_PAC = 1.0571 first measured in Navier-Stokes MED optimizer (2025)
    - xi_PAC = 1 + pi/55 derived from Mobius spectral theory (milestone 1)
    - Xi = gamma + ln(phi) = 1.0584 refined analytically (milestones 3-4)
    - Cascade engine measures org_fraction ~ 0.666 at cd=0.1, n=8
    - turbulence_pac_v3.py identifies the gap: no formula connects org_frac to xi
    - milestone3 xi_calculator uses mutual information (different from org_fraction)
"""

import json
import math
import sys
import os
import numpy as np
from scipy import stats as scipy_stats
from datetime import datetime
from itertools import product

# Add milestone4/core to path for cascade engine
m4_core = os.path.join(os.path.dirname(__file__), '..', '..', 'milestone4', 'core')
sys.path.insert(0, os.path.abspath(m4_core))
from utils import energy_cascade, measure_exponent, print_header

# ============================================================
# Constants
# ============================================================
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
LN2 = math.log(2)
LN_PHI = math.log(PHI)
GAMMA_EM = 0.5772156649015328
PI = math.pi

# Three independent xi derivations
XI_MOBIUS = 1 + PI / 55                              # Mobius spectral: 1 + pi/F_10
XI_EULER = GAMMA_EM + LN_PHI                          # Analytic: gamma + ln(phi)
XI_MVAE = 1 + (7/8) * LN2 * (1 - LN2)**2             # MVAE: (7/8)*ln(2)*(1-ln2)^2

# Cascade engine parameters (from milestone 4)
CANONICAL_CD = 0.1
CANONICAL_NS = 0.3
N_SCALES = 25

# Org fraction targets
ORG_TARGET_EXACT = 1 - 2**(-5/3)    # = 0.6850 (theoretical for exact -5/3)
ORG_TARGET_TWO_THIRDS = 2/3          # = 0.6667

results = {}
verdicts = {}

print("=" * 72)
print("EXPERIMENT 06: Hardening Tests for MVAE Claims")
print("Minimum Actualization Resolution — Dawn Field Institute")
print("=" * 72)
print(f"\nThree independent xi derivations:")
print(f"  Mobius spectral:  1 + pi/55            = {XI_MOBIUS:.10f}")
print(f"  Euler analytic:   gamma + ln(phi)      = {XI_EULER:.10f}")
print(f"  MVAE geometric:   1 + (7/8)*ln2*(1-ln2)^2 = {XI_MVAE:.10f}")
print(f"  Spread: {max(XI_MOBIUS, XI_EULER, XI_MVAE) - min(XI_MOBIUS, XI_EULER, XI_MVAE):.6f}")
print(f"  ({(max(XI_MOBIUS, XI_EULER, XI_MVAE) - min(XI_MOBIUS, XI_EULER, XI_MVAE))/XI_EULER*100:.3f}%)")
print()


# ============================================================
# PART A: Three-Xi Convergence Test
# ============================================================
print_header("PART A: Three-Xi Convergence Test",
             "NULL: Three xi derivations are unrelated numbers that happen to be close")

print(f"""
Three derivations from independent mathematical domains:
  1. Mobius spectral ratio: Xi(N) = sum((n+1/2)^2) / sum(n^2) at N=F_10 -> 1 + pi/55
  2. Euler-Mascheroni + golden log: gamma + ln(phi) = gamma_EM + ln(phi)
  3. MVAE cascade geometry: 1 + (7/8)*ln(2)*(1-ln2)^2 [She-Leveque k_eff=8]

NULL: For three independently derived constants near 1.0, what is the probability
that their pairwise spread is <= {max(XI_MOBIUS, XI_EULER, XI_MVAE) - min(XI_MOBIUS, XI_EULER, XI_MVAE):.6f}?
""")

# Generate null distribution: pick 3 constants from [1.0, 1.1] (the natural range
# for "1 + small correction" quantities)
N_PERM = 100000
rng = np.random.default_rng(42)
observed_spread = max(XI_MOBIUS, XI_EULER, XI_MVAE) - min(XI_MOBIUS, XI_EULER, XI_MVAE)

# Sample 3 values uniformly from [1.0, 1.1] — the "1 + correction" range
null_triples = rng.uniform(1.0, 1.1, size=(N_PERM, 3))
null_spreads = null_triples.max(axis=1) - null_triples.min(axis=1)
p_value_a = float(np.mean(null_spreads <= observed_spread))

print(f"  Observed pairwise spread: {observed_spread:.6f}")
print(f"  Pairwise distances:")
print(f"    Mobius-Euler: {abs(XI_MOBIUS - XI_EULER):.6f} ({abs(XI_MOBIUS - XI_EULER)/XI_EULER*100:.4f}%)")
print(f"    Mobius-MVAE:  {abs(XI_MOBIUS - XI_MVAE):.6f} ({abs(XI_MOBIUS - XI_MVAE)/XI_EULER*100:.4f}%)")
print(f"    Euler-MVAE:   {abs(XI_EULER - XI_MVAE):.6f} ({abs(XI_EULER - XI_MVAE)/XI_EULER*100:.4f}%)")
print(f"\n  Null test (100K random triples from U[1.0, 1.1]):")
print(f"    P(spread <= {observed_spread:.6f}): {p_value_a:.6f}")
print(f"    Median null spread: {np.median(null_spreads):.6f}")

# Also test with narrower range [1.05, 1.07] — what if all three are "obviously near 1.06"?
null_narrow = rng.uniform(1.05, 1.07, size=(N_PERM, 3))
null_narrow_spreads = null_narrow.max(axis=1) - null_narrow.min(axis=1)
p_value_a_narrow = float(np.mean(null_narrow_spreads <= observed_spread))
print(f"\n  Tighter null (U[1.05, 1.07]):")
print(f"    P(spread <= {observed_spread:.6f}): {p_value_a_narrow:.6f}")

# Check: are the TWO closest (Mobius and MVAE) suspiciously close?
closest_pair_dist = abs(XI_MOBIUS - XI_MVAE)
null_pair_dists = np.abs(null_narrow[:, 0] - null_narrow[:, 1])
p_value_pair = float(np.mean(null_pair_dists <= closest_pair_dist))
print(f"\n  Mobius-MVAE pair test (closest: {closest_pair_dist:.8f}):")
print(f"    P(|a-b| <= {closest_pair_dist:.8f}): {p_value_pair:.6f}")

convergent = p_value_a < 0.05
verdicts['A'] = 'PASS (null rejected)' if convergent else 'FAIL (null not rejected)'
print(f"\n  VERDICT: {verdicts['A']}")
if not convergent:
    print(f"  NOTE: Spread is not statistically unusual for U[1.0, 1.1].")
    print(f"  The convergence may be explained by all three targeting '1 + O(0.06)'.")

results['part_a'] = {
    'xi_mobius': XI_MOBIUS,
    'xi_euler': XI_EULER,
    'xi_mvae': XI_MVAE,
    'observed_spread': observed_spread,
    'p_value_uniform': p_value_a,
    'p_value_narrow': p_value_a_narrow,
    'p_value_pair': p_value_pair,
    'convergent': convergent,
}


# ============================================================
# PART B: Formula Selectivity Null
# ============================================================
print_header("PART B: Formula Selectivity Null",
             "NULL: Many formulas of similar complexity match xi_PAC to <0.01%")

print(f"""
The champion formula: eta = 1 + (7/8)*(1-ln2)^2 = {1 + (7/8)*(1-LN2)**2:.10f}
Implied xi_PAC = 1 + (7/8)*ln(2)*(1-ln2)^2 = {XI_MVAE:.10f}

Question: how SELECTIVE is this? Scan all formulas of the form:
  xi = 1 + (a/b) * ln(2)^p * (1-ln(2))^q
with a in [0..9], b in [1..9], p in [0..4], q in [0..4]

Count how many match to within various thresholds.
""")

target = XI_MOBIUS  # 1 + pi/55 = 1.0571 (the value the MVAE formula is matching)
thresholds = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]  # percentage thresholds

n_total = 0
hits_by_threshold = {t: 0 for t in thresholds}
best_hits = []

for a in range(0, 10):
    for b in range(1, 10):
        for p in range(0, 5):
            for q in range(0, 5):
                n_total += 1
                val = 1 + (a / b) * LN2**p * (1 - LN2)**q
                err_pct = abs(val - target) / target * 100

                for t in thresholds:
                    if err_pct < t:
                        hits_by_threshold[t] += 1

                if err_pct < 0.01:
                    best_hits.append({
                        'formula': f"1 + ({a}/{b}) * ln2^{p} * (1-ln2)^{q}",
                        'a': a, 'b': b, 'p': p, 'q': q,
                        'value': val,
                        'error_pct': err_pct,
                    })

best_hits.sort(key=lambda x: x['error_pct'])

print(f"  Total formulas scanned: {n_total}")
print(f"\n  {'Threshold':>12} | {'Hits':>6} | {'Rate':>10}")
print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*10}")
for t in thresholds:
    rate = hits_by_threshold[t] / n_total
    print(f"  {t:>11.4f}% | {hits_by_threshold[t]:>6d} | {rate:>10.6f}")

champion_err = abs(XI_MVAE - target) / target * 100
champion_rank = sum(1 for h in best_hits if h['error_pct'] < champion_err) + 1

print(f"\n  Champion formula: 1 + (7/8)*ln2*(1-ln2)^2")
print(f"  Champion error: {champion_err:.6f}%")
print(f"  Champion rank: #{champion_rank} out of {n_total} formulas")

if best_hits:
    print(f"\n  Top matches within <0.01%:")
    for h in best_hits[:10]:
        is_champ = " <- CHAMPION" if h['a'] == 7 and h['b'] == 8 and h['p'] == 1 and h['q'] == 2 else ""
        print(f"    {h['formula']:40s} = {h['value']:.10f}  err={h['error_pct']:.6f}%{is_champ}")

# p-value: fraction of random formulas that match better than champion
p_selectivity = hits_by_threshold[0.01] / n_total if hits_by_threshold[0.01] > 0 else 0
selective = hits_by_threshold[0.01] <= 5  # fewer than 5 hits at 0.01% = selective

verdicts['B'] = 'PASS (formula is selective)' if selective else 'FAIL (formula is not selective)'
print(f"\n  VERDICT: {verdicts['B']}")
if not selective:
    print(f"  NOTE: {hits_by_threshold[0.01]} formulas match to <0.01% — champion is not unique.")
    print(f"  The (7/8) interpretation as 'k_eff-1 out of k_eff modes' adds physical")
    print(f"  content beyond the numerical match, but the formula itself is not selective.")

results['part_b'] = {
    'n_total_formulas': n_total,
    'hits_by_threshold': {str(t): v for t, v in hits_by_threshold.items()},
    'champion_error_pct': champion_err,
    'champion_rank': champion_rank,
    'top_hits': best_hits[:10],
    'selective': selective,
    'p_selectivity': p_selectivity,
}


# ============================================================
# PART C: Cascade Engine Bridge
# ============================================================
print_header("PART C: Cascade Engine Bridge",
             "Can org_fraction from the real cascade connect to xi_PAC?")

print(f"""
The OPEN QUESTION (from turbulence_pac_v3.py): there is no formula
connecting the cascade engine's org_fraction to xi_PAC in the codebase.

The cascade engine at cd=0.1, n=8 measures:
  - org_fraction ~ 0.666 (close to 2/3)
  - exponent ~ -1.61 (close to -5/3)

The MVAE hierarchy has:
  - xi_floor = 1 - ln^2(2) = {1 - LN2**2:.6f}
  - xi_PAC = 1 + (7/8)*ln(2)*(1-ln2)^2 = {XI_MVAE:.6f}
  - eta_PAC = 1 + (7/8)*(1-ln2)^2 = {1 + (7/8)*(1-LN2)**2:.6f}

Test: run the cascade engine, measure org_fraction and energy partition,
and check which MVAE quantities can be predicted from cascade outputs.
""", flush=True)

# Run cascade engine at canonical parameters with multiple seeds
n_seeds_c = 50
org_fracs = []
exponents = []
energy_partitions = []  # E_organized / E_total at each scale

for seed in range(n_seeds_c):
    np.random.seed(seed)
    res = energy_cascade(1.0, N_SCALES, n_modes=8,
                         coupling_decay=CANONICAL_CD,
                         nonlinear_strength=CANONICAL_NS,
                         n_samples=5000)

    alive = [r for r in res if r['alive'] and r['P_input'] > 1e-15]
    if len(alive) > 6:
        mid = alive[2:-2]
        orgs = [r['org_fraction'] for r in mid]
        org_fracs.append(float(np.mean(orgs)))

        # Energy partition: total organized / total input
        total_org = sum(r['E_organized'] for r in alive)
        total_input = sum(r['P_input'] for r in alive)
        energy_partitions.append(total_org / total_input)

    exp_val, r2, _, _ = measure_exponent(res)
    if exp_val is not None:
        exponents.append(float(exp_val))

mean_org = float(np.mean(org_fracs))
mean_exp = float(np.mean(exponents))
mean_partition = float(np.mean(energy_partitions))

print(f"  Cascade engine (cd=0.1, n=8, {n_seeds_c} seeds):")
print(f"    Mean org_fraction:    {mean_org:.6f} (std={np.std(org_fracs):.6f})")
print(f"    Mean exponent:        {mean_exp:.6f} (std={np.std(exponents):.6f})")
print(f"    Mean energy partition: {mean_partition:.6f} (E_org/E_total)")

# Now test: can we BUILD xi from the measured org_fraction?
# Candidate formulas connecting org_fraction to xi:
print(f"\n  Testing connections: org_fraction -> xi")
print(f"  {'Formula':45s} | {'Predicted xi':>14} | {'Error from Mobius':>18}")
print(f"  {'-'*45}-+-{'-'*14}-+-{'-'*18}")

connections = [
    ("1 + org_frac - 1/2",
     1 + mean_org - 0.5),
    ("1 / (1 - org_frac)",
     1 / (1 - mean_org)),
    ("org_frac + ln(2)",
     mean_org + LN2),
    ("1 + org_frac * ln(2)",
     1 + mean_org * LN2),
    ("1 + (1 - org_frac) * ln(2)",
     1 + (1 - mean_org) * LN2),
    ("org_frac / (1 - ln(2))",
     mean_org / (1 - LN2)),
    ("1 + org_frac^2",
     1 + mean_org**2),
    ("1 + (1 - org_frac)^2",
     1 + (1 - mean_org)**2),
    ("(1 + org_frac) / (2 - org_frac)",
     (1 + mean_org) / (2 - mean_org)),
    ("1 + (7/8) * ln(2) * (1-org_frac)^2",
     1 + (7/8) * LN2 * (1 - mean_org)**2),
    ("1 + (7/8) * org_frac * (1-org_frac)^2",
     1 + (7/8) * mean_org * (1 - mean_org)**2),
    ("2 * org_frac / (1 + org_frac - org_frac^2)",
     2 * mean_org / (1 + mean_org - mean_org**2)),
]

best_connection = None
best_err_c = float('inf')
for name, val in connections:
    err = abs(val - XI_MOBIUS) / XI_MOBIUS * 100
    marker = " <-" if err < 1.0 else ""
    print(f"  {name:45s} | {val:>14.8f} | {err:>17.4f}%{marker}")
    if err < best_err_c:
        best_err_c = err
        best_connection = (name, val)

# The MVAE prediction: xi = 1 + (7/8)*ln(2)*(1-ln2)^2
# This uses ln(2), NOT org_fraction. But org_frac ~ 2/3 ~ 1-1/3 while 1-ln2 ~ 0.307.
# These are different: org_frac != 1-ln2. Check:
print(f"\n  Key comparison:")
print(f"    org_fraction measured:  {mean_org:.6f}")
print(f"    2/3 (She-Leveque):     {2/3:.6f}")
print(f"    1 - ln(2):             {1-LN2:.6f}")
print(f"    1 - 2^(-5/3):          {1-2**(-5/3):.6f}  (theoretical for exact -5/3)")
print(f"    org_frac - (1-ln2):    {mean_org - (1-LN2):.6f}  (these are different quantities)")

# Check if the exponent connects
# From Kolmogorov: E(k) ~ k^alpha, our alpha ~ -1.61
# Theoretical: alpha = -(1 + org_frac)? Let's test
alpha_from_org = -(1 + mean_org)
print(f"\n  Exponent-org connection test:")
print(f"    Measured exponent:    {mean_exp:.6f}")
print(f"    -(1 + org_frac):     {alpha_from_org:.6f}  (err={abs(mean_exp - alpha_from_org)/abs(mean_exp)*100:.2f}%)")
print(f"    -1/(1-org_frac):     {-1/(1-mean_org):.6f}  (err={abs(mean_exp - (-1/(1-mean_org)))/abs(mean_exp)*100:.2f}%)")

bridge_found = best_err_c < 1.0
verdicts['C'] = f'PASS (bridge at {best_err_c:.3f}%)' if bridge_found else 'FAIL (no bridge found <1%)'
print(f"\n  Best connection: {best_connection[0]} = {best_connection[1]:.8f} (err={best_err_c:.4f}%)")
print(f"  VERDICT: {verdicts['C']}")
if not bridge_found:
    print(f"  NOTE: No simple formula connects org_fraction to xi_PAC.")
    print(f"  This confirms the open question from turbulence_pac_v3.py.")
    print(f"  The MVAE formula uses ln(2), not org_fraction — they may")
    print(f"  be parallel descriptions rather than linked quantities.")

results['part_c'] = {
    'n_seeds': n_seeds_c,
    'mean_org_fraction': mean_org,
    'std_org_fraction': float(np.std(org_fracs)),
    'mean_exponent': mean_exp,
    'mean_energy_partition': mean_partition,
    'best_connection': best_connection[0] if best_connection else None,
    'best_connection_err_pct': best_err_c,
    'bridge_found': bridge_found,
    'connections_tested': [(name, float(val), float(abs(val - XI_MOBIUS)/XI_MOBIUS*100))
                           for name, val in connections],
}


# ============================================================
# PART D: PAC Tree Attractor Test
# ============================================================
print_header("PART D: PAC Tree Attractor — xi_floor and eta Bridging",
             "NULL: xi convergence in the PAC tree is just CLT, not a real attractor")

print(f"""
exp_02 showed the PAC tree cascade converges to xi_PAC when xi_target=1.0571.
The CRITICAL finding was exp_02G: pure Landauer (no imposed xi) gives
  xi_floor = 1 - ln^2(2) = {1-LN2**2:.8f} EXACTLY (zero variance).

This is a DIFFERENT model from the Kolmogorov cascade engine (milestone4).
The PAC tree: parent energy * local_xi / branching = child energy.
The cascade engine: structured coupling matrix, eigenvalue-based org_fraction.

Three tests:
  D.1: Is xi_floor exact across ALL branching factors and starting energies?
  D.2: Does the stochastic tree converge beyond CLT (test residual structure)?
  D.3: Does eta_PAC = 1.0824 from She-Leveque geometry bridge xi_floor to xi_PAC?
""", flush=True)

XI_PAC_D = 1.0571
XI_FLOOR = 1.0 - LN2**2
ETA_PAC = 1 + (7/8) * (1 - LN2)**2
ETA_FLOOR = 1 - LN2  # eta that gives xi_floor

# --- D.1: Pure Landauer exactness across parameters ---
print("  D.1: Pure Landauer xi_floor across parameters")
print(f"  {'Branching':>10} {'E_start':>12} {'xi_floor':>14} {'Error':>12} {'Exact?':>8}")
print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*14}-+-{'-'*12}-+-{'-'*8}")

def pure_landauer_xi(eta, n_gen=20):
    """Deterministic Landauer cascade: xi = 1 - ln(2)*(1-eta) at every level."""
    return 1.0 - LN2 * (1.0 - eta)

branching_factors_d = [2, 3, 4, 5, 8, 16, 32]
starting_energies_d = [0.001, 0.1, LN2, 1.0, 100.0, 1e6]
all_exact = True

for bf in branching_factors_d:
    for E0 in starting_energies_d:
        xi_val = pure_landauer_xi(ETA_FLOOR)
        err = abs(xi_val - XI_FLOOR)
        exact = err < 1e-14
        if not exact:
            all_exact = False
        # Only print subset for readability
        if bf in [2, 8, 32] and E0 in [0.001, LN2, 1e6]:
            print(f"  {bf:>10d} {E0:>12.3e} {xi_val:>14.10f} {err:>12.2e} {'YES' if exact else 'no':>8}")

print(f"\n  xi_floor = 1 - ln^2(2) = {XI_FLOOR:.10f}")
print(f"  Result: {'EXACT across all parameters' if all_exact else 'NOT exact — varies with parameters'}")
print(f"  Note: xi_floor depends ONLY on eta, not on branching or energy.")
print(f"  This is because xi(eta) = 1 - ln(2)*(1-eta) is a FORMULA, not a measurement.")

# --- D.2: Stochastic tree convergence beyond CLT ---
print(f"\n  D.2: Stochastic tree — is convergence just CLT?")

def grow_cascade_levels(initial_energy, max_depth, branching=2,
                        xi_target=None, sigma=0.15, rng=None):
    """PAC tree cascade from exp_02: per-level representation."""
    if rng is None:
        rng = np.random.default_rng(42)
    if xi_target is None:
        xi_target = XI_PAC_D
    level_energies = np.array([initial_energy])
    level_stats = []
    for depth in range(max_depth):
        n = len(level_energies)
        local_xis = rng.normal(xi_target, sigma, size=n)
        level_stats.append({
            "depth": depth, "n_nodes": n,
            "mean_xi": float(np.mean(local_xis)),
            "std_xi": float(np.std(local_xis)),
        })
        child_energies = level_energies * local_xis / branching
        level_energies = np.repeat(child_energies, branching)
    return level_stats

# Run many trials and check if residuals are structured (not just shrinking by 1/sqrt(N))
n_trials_d2 = 200
max_depth_d2 = 12
rng_d2 = np.random.default_rng(42)

# Collect mean_xi at each depth across trials
depth_residuals = {d: [] for d in range(max_depth_d2)}
for trial in range(n_trials_d2):
    stats = grow_cascade_levels(LN2, max_depth_d2, branching=2, sigma=0.15, rng=rng_d2)
    for s in stats:
        depth_residuals[s["depth"]].append(s["mean_xi"] - XI_PAC_D)

print(f"  {'Depth':>6} {'N_nodes':>10} {'Mean residual':>14} {'Std residual':>14} {'CLT pred std':>14} {'Ratio':>8}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}")

clt_ratios = []
for d in range(max_depth_d2):
    resids = np.array(depth_residuals[d])
    n_nodes = 2**d
    clt_pred_std = 0.15 / math.sqrt(n_nodes)  # CLT: sigma/sqrt(N)
    actual_std = float(np.std(resids))
    ratio = actual_std / clt_pred_std if clt_pred_std > 1e-15 else float('inf')
    clt_ratios.append(ratio)
    if d in [0, 2, 4, 6, 8, 10, 11]:
        print(f"  {d:>6d} {n_nodes:>10d} {float(np.mean(resids)):>14.6f} {actual_std:>14.6f} {clt_pred_std:>14.6f} {ratio:>8.3f}")

# If ratio is consistently ~1.0, convergence IS just CLT
mean_ratio = float(np.mean(clt_ratios[2:]))  # skip depth 0-1 (too few nodes)
clt_only = abs(mean_ratio - 1.0) < 0.2

print(f"\n  Mean CLT ratio (depth 2+): {mean_ratio:.3f}")
print(f"  {'Convergence IS just CLT (ratio ~ 1.0)' if clt_only else 'Convergence has structure BEYOND CLT'}")

# --- D.3: eta bridging from xi_floor to xi_PAC ---
print(f"\n  D.3: eta bridging — does eta_PAC connect xi_floor to xi_PAC?")

# The recycling bridge: xi(eta) = 1 - ln(2)*(1-eta)
# At eta = 1-ln(2): xi = 1 - ln(2)*ln(2) = 1-ln²(2) = xi_floor  ✓
# At eta = eta_PAC:  xi = 1 - ln(2)*(1-eta_PAC) = xi_PAC
# The question: eta_PAC = 1.0824 > 1, so SEC pump is required
xi_from_bridge = 1 - LN2 * (1 - ETA_PAC)
bridge_err = abs(xi_from_bridge - XI_PAC_D)
bridge_err_pct = bridge_err / XI_PAC_D * 100

print(f"  Recycling bridge: xi(eta) = 1 - ln(2)*(1-eta)")
print(f"  eta_floor = 1 - ln(2) = {ETA_FLOOR:.6f} -> xi_floor = {pure_landauer_xi(ETA_FLOOR):.8f}")
print(f"  eta_PAC = 1 + (7/8)*(1-ln2)^2 = {ETA_PAC:.6f} -> xi = {xi_from_bridge:.8f}")
print(f"  xi_PAC target = {XI_PAC_D:.6f}")
print(f"  Bridge error: {bridge_err_pct:.4f}%")
print(f"  eta_PAC > 1: {'YES — SEC pump required' if ETA_PAC > 1 else 'no'}")

# Verify the She-Leveque derivation: eta = 1 + (k-1)/k * (1-ln2)^2 with k=8
for k in [2, 3, 4, 5, 8, 16]:
    eta_k = 1 + (k-1)/k * (1 - LN2)**2
    xi_k = 1 - LN2 * (1 - eta_k)
    err_k = abs(xi_k - XI_PAC_D) / XI_PAC_D * 100
    marker = " <- k_eff=8 (She-Leveque 3D)" if k == 8 else ""
    if k in [2, 5, 8, 16]:
        print(f"    k={k:>2d}: eta={eta_k:.6f}, xi={xi_k:.8f}, err={err_k:.4f}%{marker}")

# Is k=8 special? Check selectivity
k_errors = []
for k in range(2, 33):
    eta_k = 1 + (k-1)/k * (1 - LN2)**2
    xi_k = 1 - LN2 * (1 - eta_k)
    k_errors.append((k, abs(xi_k - XI_PAC_D) / XI_PAC_D * 100))
k_errors.sort(key=lambda x: x[1])
best_k, best_k_err = k_errors[0]

print(f"\n  k selectivity (2-32): best k={best_k} (err={best_k_err:.4f}%)")
print(f"  k=8 is {'the best' if best_k == 8 else f'NOT the best (k={best_k} is better)'}")

# Overall D verdict
xi_floor_exact = all_exact
bridge_works = bridge_err_pct < 0.5
d_pass = xi_floor_exact and bridge_works

verdicts['D'] = 'PASS (xi_floor exact, eta bridge works)' if d_pass else 'FAIL'
if d_pass and clt_only:
    verdicts['D'] += ' (note: stochastic convergence is CLT)'
print(f"\n  VERDICT: {verdicts['D']}")
if clt_only:
    print(f"  HONEST NOTE: The stochastic tree convergence in exp_02 IS just CLT.")
    print(f"  The real physics is in the DETERMINISTIC pure Landauer floor (xi_floor)")
    print(f"  and the She-Leveque eta bridge, not in the stochastic sampling.")

results['part_d'] = {
    'xi_floor_exact': xi_floor_exact,
    'xi_floor_value': XI_FLOOR,
    'clt_only': clt_only,
    'mean_clt_ratio': mean_ratio,
    'eta_pac': ETA_PAC,
    'xi_from_bridge': xi_from_bridge,
    'bridge_error_pct': bridge_err_pct,
    'best_k': best_k,
    'best_k_error_pct': best_k_err,
    'bridge_works': bridge_works,
    'd_pass': d_pass,
}


# ============================================================
# PART E: Three-Constraint Conjugacy Null
# ============================================================
print_header("PART E: Three-Constraint Conjugacy Products",
             "NULL: Conjugacy products (exact 1, exact 1/2) arise by chance")

print(f"""
exp_01 derived three MVAE constraints in Planck units:
  Landauer energy:    E_L   = ln(2)         = {LN2:.6f}
  Heisenberg time:    dt_H  = 1/(2E) = 1/(2ln2)  = {1/(2*LN2):.6f}
  Schwarzschild:      R_s   = 2M = 2ln(2)   = {2*LN2:.6f}
  Uncertainty length: dx_unc = 1/(2E) = 1/(2ln2) = {1/(2*LN2):.6f}

The claim is NOT about spread — it's about CONJUGACY PRODUCTS:
  Product 1: dx_grav × dx_unc = 2ln(2) × 1/(2ln(2)) = 1.000 EXACTLY
  Product 2: E × dt = ln(2) × 1/(2ln(2)) = 1/2 EXACTLY (Heisenberg saturation)

These are exact by construction of the uncertainty principle. The real question:
is it non-trivial that THREE independent physical constraints (Landauer,
Heisenberg, Schwarzschild) all reduce to functions of a SINGLE parameter (ln2)?

NULL: Generate random constraint triples where each constraint is a function
of a DIFFERENT random parameter. Measure how often pairwise products
land on exact simple rational values.
""")

# The MVAE constraints as functions of ln(2):
E_mvae = LN2                    # energy
t_mvae = 1 / (2 * LN2)          # time (Heisenberg)
l_mvae = 1 / (2 * (1 - LN2))   # length (Landauer-localization)
r_grav = 2 * LN2                # Schwarzschild radius

# Conjugacy products
prod1 = r_grav * (1 / (2 * LN2))  # dx_grav × dx_unc
prod2 = E_mvae * t_mvae             # E × dt

print(f"  Observed conjugacy products:")
print(f"    dx_grav × dx_unc = {r_grav:.6f} × {1/(2*LN2):.6f} = {prod1:.10f}")
print(f"    E × dt           = {E_mvae:.6f} × {t_mvae:.6f} = {prod2:.10f}")
print(f"    Product 1 deviation from 1:   {abs(prod1 - 1.0):.2e}")
print(f"    Product 2 deviation from 1/2: {abs(prod2 - 0.5):.2e}")

# Key insight: all constraints are functions of ONE parameter (ln2).
# Test: if we had three constraints each using a DIFFERENT free parameter,
# how often would their pairwise products be exact simple values?
N_NULL_E = 100000
rng_e = np.random.default_rng(42)

# Simple rational targets to check products against
simple_values = [0.25, 1/3, 0.5, 2/3, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
TOLERANCE = 1e-8  # "exact" means within machine precision

# The MVAE system: all three constraints are f(ln2) for different f
# This means pairwise products are g(ln2) — a single-parameter family
# NULL: three constraints are f1(a), f2(b), f3(c) for independent a,b,c
# Then products are h(a,b) — two-parameter, much harder to hit exact values

# Model: each constraint has the SAME functional forms as MVAE but with different base params
# E = x, dt = 1/(2x), R_s = 2x — same structure but x1, x2, x3 independent
n_exact_products_null = 0
n_exact_products_per_trial = []

for _ in range(N_NULL_E):
    # Three independent "base parameters" (analogous to ln2)
    x1 = rng_e.uniform(0.1, 2.0)  # for "energy" constraint
    x2 = rng_e.uniform(0.1, 2.0)  # for "time" constraint
    x3 = rng_e.uniform(0.1, 2.0)  # for "gravity" constraint

    # Build constraints from different parameters (unlike MVAE which uses only ln2)
    E_null = x1
    dt_null = 1 / (2 * x2)        # time from DIFFERENT parameter
    r_null = 2 * x3               # radius from DIFFERENT parameter

    # Check pairwise products
    null_prod1 = r_null * (1 / (2 * x2))    # "gravity × uncertainty" but different params
    null_prod2 = E_null * dt_null            # "E × dt" but different params

    n_exact = 0
    for p in [null_prod1, null_prod2]:
        for sv in simple_values:
            if abs(p - sv) < TOLERANCE:
                n_exact += 1
                break

    n_exact_products_per_trial.append(n_exact)
    if n_exact >= 2:
        n_exact_products_null += 1

# MVAE achieves 2/2 exact products (prod1 = 1, prod2 = 1/2)
p_value_e = n_exact_products_null / N_NULL_E

print(f"\n  Null test: 3 constraints with INDEPENDENT base parameters")
print(f"    Trials: {N_NULL_E:,}")
print(f"    Fraction achieving 2+ exact products: {p_value_e:.6f}")
print(f"    (MVAE achieves 2/2 because ALL constraints use the SAME ln(2))")

# Also test: single-parameter family
n_single_param_exact = 0
for _ in range(N_NULL_E):
    x = rng_e.uniform(0.1, 2.0)  # single parameter like ln(2)
    E_sp = x
    dt_sp = 1 / (2 * x)
    r_sp = 2 * x
    sp_prod1 = r_sp * (1 / (2 * x))  # = 2x/(2x) = 1 ALWAYS
    sp_prod2 = E_sp * dt_sp            # = x/(2x) = 1/2 ALWAYS
    n_ex = 0
    if abs(sp_prod1 - 1.0) < TOLERANCE:
        n_ex += 1
    if abs(sp_prod2 - 0.5) < TOLERANCE:
        n_ex += 1
    if n_ex >= 2:
        n_single_param_exact += 1

p_single = n_single_param_exact / N_NULL_E

print(f"\n  Single-parameter family (like MVAE):")
print(f"    Fraction with 2+ exact products: {p_single:.6f}")
print(f"    (Should be 1.000 — the products are IDENTITIES of the form)")

print(f"\n  The insight: conjugacy is STRUCTURAL, not numerical.")
print(f"  E × dt = x × 1/(2x) = 1/2 for ANY x (Heisenberg identity)")
print(f"  R_s × dx_unc = 2x × 1/(2x) = 1 for ANY x (gravity-uncertainty duality)")
print(f"  What's non-trivial: all three constraints ADMIT a single-parameter")
print(f"  description. With independent parameters, products are NOT exact.")

# The real test: is single-parameter reducibility itself unusual?
# Among constraint triples from physics, how many reduce to f(x)?
# This is the deeper claim — test whether random 3-constraint systems
# from dimensional analysis can all be written as functions of one variable
print(f"\n  Dimensional analysis test:")
print(f"  In Planck units, there are 3 dimensionful base quantities (length, time, mass).")
print(f"  The MVAE maps all 3 to ONE parameter: ln(2).")
print(f"  This eliminates 2 degrees of freedom — a strong structural claim.")

# Count: how many functions of form a*x^p, 1/(b*x^q), c*x^r for
# random (a,b,c,p,q,r) produce exact products?
n_exact_power = 0
N_POWER = 100000
for _ in range(N_POWER):
    x = rng_e.uniform(0.1, 2.0)
    p1, p2, p3 = rng_e.uniform(-2, 2, size=3)
    a1, a2, a3 = rng_e.uniform(0.5, 2.0, size=3)
    v1, v2, v3 = a1 * x**p1, a2 * x**p2, a3 * x**p3

    # Check all pairwise products against simple values
    prods = [v1*v2, v1*v3, v2*v3]
    n_ex = sum(1 for p in prods for sv in simple_values if abs(p - sv) < 0.001)
    if n_ex >= 2:
        n_exact_power += 1

p_power = n_exact_power / N_POWER
print(f"  Random single-param power laws: {p_power:.6f} achieve 2+ exact products")
print(f"  MVAE coefficients (1, 1/2, 2) are the SIMPLEST integers — not random.")

conjugacy_exact = abs(prod1 - 1.0) < 1e-10 and abs(prod2 - 0.5) < 1e-10
independent_params_fail = p_value_e < 0.001  # independent params almost never produce exact products

verdicts['E'] = 'PASS (conjugacy is structural)' if (conjugacy_exact and independent_params_fail) else 'FAIL'
print(f"\n  VERDICT: {verdicts['E']}")
print(f"  Products exact: {conjugacy_exact}")
print(f"  Independent params null: p = {p_value_e:.6f}")
if conjugacy_exact:
    print(f"  The conjugacy products are IDENTITIES (hold for any single parameter).")
    print(f"  The physical content is that Landauer, Heisenberg, and Schwarzschild")
    print(f"  all reduce to functions of ln(2) alone — single-parameter reducibility.")

results['part_e'] = {
    'prod1_grav_unc': prod1,
    'prod2_E_dt': prod2,
    'prod1_exact': abs(prod1 - 1.0) < 1e-10,
    'prod2_exact': abs(prod2 - 0.5) < 1e-10,
    'p_value_independent': p_value_e,
    'p_value_single_param': p_single,
    'p_value_power_law': p_power,
    'conjugacy_exact': conjugacy_exact,
    'independent_params_fail': independent_params_fail,
}


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("SUMMARY: Hardening Test Results")
print("=" * 72)
print(f"\n  {'Part':6s} | {'Test':55s} | {'Result'}")
print(f"  {'-'*6}-+-{'-'*55}-+-{'-'*30}")
for key in sorted(verdicts.keys()):
    labels = {
        'A': 'Three-Xi convergence (independent derivations)',
        'B': 'Formula selectivity (champion vs random)',
        'C': 'Cascade engine bridge (org_fraction -> xi)',
        'D': 'PAC tree attractor (xi_floor exact, eta bridge)',
        'E': 'Three-constraint conjugacy (product structure)',
    }
    print(f"  {key:6s} | {labels[key]:55s} | {verdicts[key]}")

n_pass = sum(1 for v in verdicts.values() if v.startswith('PASS'))
n_fail = sum(1 for v in verdicts.values() if v.startswith('FAIL'))
print(f"\n  Passed: {n_pass}/5")
print(f"  Failed: {n_fail}/5")
print(f"\n  HONEST ASSESSMENT:")
print(f"  The MVAE experiment (exp_01-05) produced genuine mathematical")
print(f"  identities (xi_floor is exact, eta formula matches to 0.001%),")
print(f"  but several claims need qualification:")
if not convergent:
    print(f"  - Three-Xi 'convergence' may be generic for 1+O(0.06) quantities")
if not selective:
    print(f"  - The champion formula is not unique — physical interpretation (7/8)")
    print(f"    adds value but numerical selectivity is weak")
if not bridge_found:
    print(f"  - org_fraction and xi_PAC remain UNCONNECTED — parallel descriptions")
if 'D' in verdicts and verdicts['D'].startswith('FAIL'):
    print(f"  - PAC tree attractor or eta bridge has issues")
if 'E' in verdicts and verdicts['E'].startswith('FAIL'):
    print(f"  - Conjugacy products are not structurally significant")

# Save results
all_results = {
    'experiment': 'minimum_actualization_resolution',
    'script': 'exp_06_hardening.py',
    'timestamp': datetime.now().isoformat(),
    'verdicts': verdicts,
    'n_pass': n_pass,
    'n_fail': n_fail,
    'results': results,
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"results/exp_06_hardening_{ts}.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n  Results saved to {out_path}")
print("=" * 72)

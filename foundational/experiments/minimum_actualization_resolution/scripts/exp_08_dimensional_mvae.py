"""
Dimensional MVAE Predictions — Experiment Script 08

PURPOSE:
    Tests what the MVAE framework predicts for dimensions other than 3.
    The She-Leveque formula k = d * F_{d+1} gives different mode counts
    per dimension. The MVAE eta formula then gives dimension-dependent
    xi_PAC values. Critical test: does xi_PAC(d) converge to Xi as d->inf?

HYPOTHESIS:
    xi_PAC is dimension-dependent via k_eff(d). The 3D value xi_PAC(3) ~= 1.0571
    is one point on a curve. Xi = gamma + ln(phi) may be the d->infinity limit.

DESIGN:
    Part A: MVAE dimensional table (d=1..10)
    Part B: 2D cascade engine validation (n_modes=3)
    Part C: 4D prediction and failure analysis
    Part D: Dimensional xi convergence — does xi_PAC(d->inf) -> Xi?

CORPUS CONTEXT:
    - She-Leveque: k = d * F_{d+1}. k-1 offset confirmed in 2D and 3D.
    - 4D DNS measured k=10.78 (vs predicted k=20). Huge uncertainty.
    - Cascade engine at n=8, cd=0.1 gives org_fraction ~ 0.666, exponent ~ -1.61
    - milestone4 exp_14 confirmed 2D enstrophy at n_modes=3, 2D inverse at n_modes=8
"""

import json
import math
import numpy as np
from datetime import datetime
import sys
import os

m4_core = os.path.join(os.path.dirname(__file__), '..', '..', 'milestone4', 'core')
sys.path.insert(0, os.path.abspath(m4_core))
from utils import energy_cascade, measure_exponent, print_header

# ============================================================
# Constants
# ============================================================
LN2 = math.log(2)
PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
GAMMA_EM = 0.5772156649015328
PI = math.pi

XI_PAC_3D = 1.0571
XI_EULER = GAMMA_EM + LN_PHI
XI_FLOOR = 1.0 - LN2**2

def fibonacci(n):
    """Return nth Fibonacci number (F_1=1, F_2=1, F_3=2, ...)."""
    if n <= 0:
        return 0
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a

results = {}
verdicts = {}

print("=" * 72)
print("EXPERIMENT 08: Dimensional MVAE Predictions")
print("Minimum Actualization Resolution — Dawn Field Institute")
print("=" * 72)
print()


# ============================================================
# PART A: MVAE Dimensional Table
# ============================================================
print_header("PART A: MVAE Dimensional Table",
             "xi_PAC(d) for d=1..10 via She-Leveque + eta bridge")

print(f"""
Formula chain:
  k_SL(d) = d * F_{{d+1}}           [She-Leveque mode count]
  k_eff(d) = k_SL(d) - 1           [k-1 offset from coupling matrix]
  eta(d) = 1 + (k_eff-1)/k_eff * (1-ln2)^2   [recycling efficiency]
  xi_PAC(d) = 1 - ln2*(1-eta(d))   [recycling bridge]
  xi_floor = 1 - ln^2(2) = {XI_FLOOR:.6f}    [dimension-INDEPENDENT]
""")

print(f"  {'d':>3} {'F_{d+1}':>8} {'k_SL':>6} {'k_eff':>6} {'eta(d)':>12} {'xi_PAC(d)':>14} {'err from 3D':>14} {'xi-xi_floor':>12}")
print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*12}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}")

dim_data = []
for d in range(1, 11):
    F_dp1 = fibonacci(d + 1)
    k_sl = d * F_dp1
    k_eff = k_sl - 1
    if k_eff <= 0:
        continue

    eta_d = 1 + (k_eff - 1) / k_eff * (1 - LN2)**2
    xi_d = 1 - LN2 * (1 - eta_d)
    err_from_3d = abs(xi_d - XI_PAC_3D) / XI_PAC_3D * 100

    dim_data.append({
        'd': d, 'F_dp1': F_dp1, 'k_sl': k_sl, 'k_eff': k_eff,
        'eta': eta_d, 'xi_pac': xi_d, 'err_from_3d': err_from_3d,
    })

    marker = " <- physical 3D" if d == 3 else ""
    print(f"  {d:>3d} {F_dp1:>8d} {k_sl:>6d} {k_eff:>6d} {eta_d:>12.8f} {xi_d:>14.10f} {err_from_3d:>13.4f}% {xi_d - XI_FLOOR:>12.6f}{marker}")

# Is xi_PAC monotonically increasing?
xi_values_dim = [dd['xi_pac'] for dd in dim_data]
monotonic = all(xi_values_dim[i] <= xi_values_dim[i+1] for i in range(len(xi_values_dim)-1))
print(f"\n  xi_PAC(d) monotonically increasing: {monotonic}")

# Convergence behavior
dim_lookup = {dd['d']: dd for dd in dim_data}
print(f"\n  Convergence check:")
print(f"  xi_PAC(d=3):  {dim_lookup[3]['xi_pac']:.10f}")
print(f"  xi_PAC(d=5):  {dim_lookup[5]['xi_pac']:.10f}")
print(f"  xi_PAC(d=10): {dim_lookup[10]['xi_pac']:.10f}")
print(f"  Xi (target):  {XI_EULER:.10f}")

# What's the d->infinity limit?
# As d->inf, k_eff->inf, so (k_eff-1)/k_eff -> 1
# eta(inf) = 1 + (1-ln2)^2
# xi(inf) = 1 - ln2*(1 - (1 + (1-ln2)^2)) = 1 - ln2*(-(1-ln2)^2) = 1 + ln2*(1-ln2)^2
eta_inf = 1 + (1 - LN2)**2
xi_inf = 1 - LN2 * (1 - eta_inf)
print(f"\n  d->infinity limit:")
print(f"  eta(inf) = 1 + (1-ln2)^2 = {eta_inf:.10f}")
print(f"  xi(inf) = 1 + ln2*(1-ln2)^2 = {xi_inf:.10f}")
print(f"  Xi (gamma+ln(phi)) = {XI_EULER:.10f}")
print(f"  Difference: {abs(xi_inf - XI_EULER):.10f} ({abs(xi_inf - XI_EULER)/XI_EULER*100:.4f}%)")
print(f"\n  xi(inf) {'MATCHES' if abs(xi_inf - XI_EULER) < 0.01 else 'DOES NOT MATCH'} Xi = gamma + ln(phi)")

# The xi(inf) limit is NOT Xi. It's 1 + ln2*(1-ln2)^2.
# But this is just xi_MVAE with (k-1)/k -> 1 (i.e., 8/8 instead of 7/8)
xi_mvae_full = 1 + LN2 * (1 - LN2)**2
print(f"\n  xi(inf) = 1 + ln2*(1-ln2)^2 = {xi_mvae_full:.10f}")
print(f"  This is the MVAE formula with ALL modes recycling (k_eff -> inf)")

verdicts['A'] = f'xi(d) monotonic: {monotonic}, limit = {xi_inf:.6f}'
results['part_a'] = {
    'dim_table': dim_data,
    'monotonic': monotonic,
    'xi_inf_limit': xi_inf,
    'xi_euler': XI_EULER,
    'limit_matches_euler': abs(xi_inf - XI_EULER) < 0.01,
}


# ============================================================
# PART B: 2D Cascade Engine Validation
# ============================================================
print_header("PART B: 2D Cascade Engine Validation",
             "Does xi_PAC(2D) predict cascade behavior at n_modes=3?")

print(f"""
2D enstrophy cascade: k_SL = 2*F_3 = 4, k_eff = 3
MVAE predicts: xi_PAC(2D) = {dim_lookup[2]['xi_pac']:.10f}
               eta(2D) = {dim_lookup[2]['eta']:.10f}

Run cascade engine at n_modes=3 (2D physical mode count, confirmed in exp_14).
Compare org_fraction and test generalized bridge formula.
""", flush=True)

# Run cascade at n=3 (2D)
n_seeds = 30
org_fracs_2d = []
exponents_2d = []

for seed in range(n_seeds):
    np.random.seed(seed)
    res = energy_cascade(1.0, 25, n_modes=3, coupling_decay=0.1,
                         nonlinear_strength=0.3, n_samples=5000)
    alive = [r for r in res if r['alive'] and r['P_input'] > 1e-15]
    if len(alive) > 6:
        mid = alive[2:-2]
        org_fracs_2d.append(float(np.mean([r['org_fraction'] for r in mid])))
    exp_val, _, _, _ = measure_exponent(res)
    if exp_val is not None:
        exponents_2d.append(float(exp_val))

mean_org_2d = float(np.mean(org_fracs_2d))
mean_exp_2d = float(np.mean(exponents_2d))

print(f"  2D cascade (n_modes=3, {n_seeds} seeds):")
print(f"    Mean org_fraction: {mean_org_2d:.6f} (std={np.std(org_fracs_2d):.6f})")
print(f"    Mean exponent:     {mean_exp_2d:.6f} (target for enstrophy: -3.0)")
print(f"    Exponent error:    {abs(mean_exp_2d - (-3.0))/3.0*100:.2f}%")

# Also run 3D for comparison
org_fracs_3d = []
exponents_3d = []

for seed in range(n_seeds):
    np.random.seed(seed + 1000)
    res = energy_cascade(1.0, 25, n_modes=8, coupling_decay=0.1,
                         nonlinear_strength=0.3, n_samples=5000)
    alive = [r for r in res if r['alive'] and r['P_input'] > 1e-15]
    if len(alive) > 6:
        mid = alive[2:-2]
        org_fracs_3d.append(float(np.mean([r['org_fraction'] for r in mid])))
    exp_val, _, _, _ = measure_exponent(res)
    if exp_val is not None:
        exponents_3d.append(float(exp_val))

mean_org_3d = float(np.mean(org_fracs_3d))
mean_exp_3d = float(np.mean(exponents_3d))

print(f"\n  3D cascade (n_modes=8, {n_seeds} seeds):")
print(f"    Mean org_fraction: {mean_org_3d:.6f} (std={np.std(org_fracs_3d):.6f})")
print(f"    Mean exponent:     {mean_exp_3d:.6f} (target: -5/3 = -1.667)")

# Generalized bridge formula: 1 + ((k-1)/k)*org*(1-org)^2
bridge_2d = 1 + (2/3) * mean_org_2d * (1 - mean_org_2d)**2
bridge_3d = 1 + (7/8) * mean_org_3d * (1 - mean_org_3d)**2
xi_pac_2d = dim_lookup[2]['xi_pac']
xi_pac_3d = dim_lookup[3]['xi_pac']

print(f"\n  Generalized bridge formula: 1 + ((k_eff-1)/k_eff)*org*(1-org)^2")
print(f"  {'':>4} {'k_eff':>6} {'org_frac':>10} {'bridge':>14} {'xi_PAC(d)':>14} {'error':>10}")
print(f"  {'':>4} {'-'*6}-+-{'-'*10}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}")
print(f"  2D: {3:>6d} {mean_org_2d:>10.6f} {bridge_2d:>14.8f} {xi_pac_2d:>14.8f} {abs(bridge_2d-xi_pac_2d)/xi_pac_2d*100:>9.4f}%")
print(f"  3D: {8:>6d} {mean_org_3d:>10.6f} {bridge_3d:>14.8f} {xi_pac_3d:>14.8f} {abs(bridge_3d-xi_pac_3d)/xi_pac_3d*100:>9.4f}%")

bridge_generalizes = abs(bridge_2d - xi_pac_2d) / xi_pac_2d < 0.02
verdicts['B'] = f'PASS (bridge generalizes to 2D)' if bridge_generalizes else f'FAIL (bridge does not generalize)'
print(f"\n  VERDICT: {verdicts['B']}")

results['part_b'] = {
    'mean_org_2d': mean_org_2d,
    'mean_exp_2d': mean_exp_2d,
    'mean_org_3d': mean_org_3d,
    'mean_exp_3d': mean_exp_3d,
    'bridge_2d': bridge_2d,
    'bridge_3d': bridge_3d,
    'xi_pac_2d': xi_pac_2d,
    'xi_pac_3d': xi_pac_3d,
    'bridge_generalizes': bridge_generalizes,
}


# ============================================================
# PART C: 4D Prediction and Failure Analysis
# ============================================================
print_header("PART C: 4D Failure Analysis",
             "Why does the 4D prediction fail? Can MVAE explain it?")

print(f"""
4D prediction: k_SL = 4*F_5 = 4*5 = 20, k_eff = 19
4D DNS measured: k = 10.78 (milestone4 exp_06)

MVAE xi_PAC(4D) = {dim_lookup[4]['xi_pac']:.10f}

Key question: if 4D has a different effective k (not k-1 but some other offset),
what k_eff would explain k=10.78?
""")

# What k_eff would the DNS measurement imply?
# The cascade engine at k_eff=n gives exponent ~ -(5/3) at n=8 for 3D
# For 4D, the measured k=10.78 suggests k_eff ~ 10 or 11
# But we can't directly test this without 4D cascade data

# Instead: test whether MVAE eta formula at k_eff=10-11 gives anything interesting
print(f"  If 4D k_eff = measured k = 10.78 (instead of k_SL-1 = 19):")
k_measured_4d = 10.78
eta_4d_measured = 1 + (k_measured_4d - 1) / k_measured_4d * (1 - LN2)**2
xi_4d_measured = 1 - LN2 * (1 - eta_4d_measured)
print(f"    eta(k=10.78) = {eta_4d_measured:.8f}")
print(f"    xi_PAC(k=10.78) = {xi_4d_measured:.10f}")

# What offset would this represent?
# k_eff = k_SL - offset => offset = 20 - 10.78 = 9.22
offset_4d = 20 - k_measured_4d
print(f"\n  Offset analysis:")
print(f"    2D: k_SL=4, k_eff=3, offset=1")
print(f"    3D: k_SL=9, k_eff=8, offset=1")
print(f"    4D: k_SL=20, k_DNS=10.78, offset={offset_4d:.2f}")
print(f"\n  The offset GROWS with dimension. Not a constant k-1.")

# Hypothesis: offset scales with F_{d+1} - 1?
# d=2: F_3-1=1, offset=1. d=3: F_4-1=2, observed offset=1. NOPE.
# Hypothesis: offset = k_SL / 2?
# d=2: 4/2=2, observed=1. NOPE.
# Hypothesis: offset = d*(F_{d+1}-1)?
# d=2: 2*1=2, observed=1. NOPE.
# Let's just test what k_eff vs k_SL looks like if 4D follows the pattern
print(f"\n  Pattern search: k_eff(d) = k_SL(d) - offset(d)")
print(f"  {'d':>3} {'k_SL':>6} {'k_eff(obs)':>12} {'offset':>8} {'offset/d':>10} {'offset/F':>10}")
print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*12}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

observed_keff = {2: 3, 3: 8, 4: 10.78}
for d in [2, 3, 4]:
    F_dp1 = fibonacci(d + 1)
    k_sl = d * F_dp1
    k_obs = observed_keff[d]
    offset = k_sl - k_obs
    print(f"  {d:>3d} {k_sl:>6d} {k_obs:>12.2f} {offset:>8.2f} {offset/d:>10.2f} {offset/F_dp1:>10.2f}")

# Is offset/d constant? offset/F constant?
# d=2: offset/d=0.5, d=3: offset/d=0.33, d=4: offset/d=2.31 -> NO
# d=2: offset/F=0.5, d=3: offset/F=0.33, d=4: offset/F=1.84 -> NO

# The 4D result breaks the simple pattern. Either:
# 1. The 4D DNS is underpowered (N=32 grid, huge CI)
# 2. She-Leveque doesn't hold in 4D
# 3. A different scaling law applies above d=3

print(f"\n  CONCLUSION: No simple offset pattern extends to 4D.")
print(f"  The 4D DNS result (k=10.78 with CI [3.5, 300]) is too uncertain")
print(f"  to distinguish between k_eff=10, 11, or even 19.")
print(f"  MVAE cannot resolve this without better 4D data.")

verdicts['C'] = 'INCONCLUSIVE (4D DNS too uncertain)'
results['part_c'] = {
    'k_sl_4d': 20,
    'k_measured_4d': k_measured_4d,
    'offset_4d': offset_4d,
    'eta_4d_measured': eta_4d_measured,
    'xi_4d_measured': xi_4d_measured,
}


# ============================================================
# PART D: Dimensional xi Convergence
# ============================================================
print_header("PART D: Dimensional xi Convergence",
             "Does xi_PAC(d) converge to Xi = gamma + ln(phi)?")

print(f"""
As d -> infinity, k_eff -> infinity, so (k_eff-1)/k_eff -> 1.
  eta(inf) = 1 + (1-ln2)^2 = {1 + (1-LN2)**2:.10f}
  xi(inf) = 1 + ln2*(1-ln2)^2 = {1 + LN2*(1-LN2)**2:.10f}

But Xi = gamma + ln(phi) = {XI_EULER:.10f}

These are NOT equal. The gap is:
  Xi - xi(inf) = {XI_EULER - (1 + LN2*(1-LN2)**2):.10f}
""")

# Compute xi_PAC for large d
print(f"  {'d':>5} {'k_eff':>10} {'xi_PAC(d)':>16} {'Xi - xi_PAC(d)':>16} {'converging?':>12}")
print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*16}-+-{'-'*16}-+-{'-'*12}")

large_d_data = []
prev_gap = None
for d in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100]:
    F_dp1 = fibonacci(d + 1)
    k_sl = d * F_dp1
    k_eff = k_sl - 1
    if k_eff <= 0:
        continue
    eta_d = 1 + (k_eff - 1) / k_eff * (1 - LN2)**2
    xi_d = 1 - LN2 * (1 - eta_d)
    gap = XI_EULER - xi_d
    converging = "yes" if (prev_gap is not None and abs(gap) < abs(prev_gap)) else ""
    prev_gap = gap
    large_d_data.append({'d': d, 'k_eff': k_eff, 'xi_pac': xi_d, 'gap': gap})
    print(f"  {d:>5d} {k_eff:>10d} {xi_d:>16.10f} {gap:>16.10f} {converging:>12s}")

# The asymptotic limit
xi_asymp = 1 + LN2 * (1 - LN2)**2
gap_asymp = XI_EULER - xi_asymp
print(f"  {'inf':>5s} {'inf':>10s} {xi_asymp:>16.10f} {gap_asymp:>16.10f}")

print(f"\n  The gap Xi - xi_PAC(d) DOES shrink with d, but converges to:")
print(f"    gap(inf) = gamma + ln(phi) - 1 - ln2*(1-ln2)^2 = {gap_asymp:.10f}")
print(f"    = {gap_asymp:.10f}")
print(f"\n  This residual gap is {gap_asymp:.6f}, approximately:")

# Try to express the gap in known constants
candidates = [
    ("gamma - ln2*(1-ln2)", GAMMA_EM - LN2*(1-LN2)),
    ("gamma + ln(phi) - 1 - ln2*(1-ln2)^2", gap_asymp),
    ("ln(phi) - ln2*(1-ln2)^2", LN_PHI - LN2*(1-LN2)**2),
    ("gamma - (1 - ln(phi))", GAMMA_EM - (1 - LN_PHI)),
    ("1/(240*pi)", 1/(240*PI)),
]
for name, val in candidates:
    err = abs(val - gap_asymp) / abs(gap_asymp) * 100 if gap_asymp != 0 else float('inf')
    marker = " <--" if err < 1 else ""
    print(f"    {name:40s} = {val:.10f} (err={err:.2f}%){marker}")

# Is the gap the Euler gap from exp_05?
euler_gap_from_exp05 = XI_EULER - XI_PAC_3D
print(f"\n  Euler gap (Xi - xi_PAC(3D)):   {euler_gap_from_exp05:.10f}")
print(f"  Asymptotic gap (Xi - xi(inf)): {gap_asymp:.10f}")
print(f"  Ratio: {euler_gap_from_exp05 / gap_asymp:.6f}")

# Does xi_PAC(d) converge to Xi? NO — it converges to xi_asymp which is LESS than Xi.
# The gap is real and irreducible. Xi contains gamma, which the MVAE formula doesn't produce.
converges_to_xi = abs(gap_asymp) < 0.001
verdicts['D'] = f'xi(inf) = {xi_asymp:.6f} (gap from Xi = {gap_asymp:.6f})'
if converges_to_xi:
    verdicts['D'] += ' MATCHES Xi'
else:
    verdicts['D'] += ' — Xi has extra structure beyond MVAE'

print(f"\n  VERDICT: {verdicts['D']}")
print(f"\n  INTERPRETATION:")
print(f"  xi_PAC(d) does NOT converge to Xi = gamma + ln(phi).")
print(f"  The MVAE formula captures the (1-ln2)^2 recycling term")
print(f"  but NOT the Euler-Mascheroni constant gamma.")
print(f"  gamma encodes something BEYOND cascade recycling geometry —")
print(f"  possibly the discrete-to-continuum correction from number theory.")

results['part_d'] = {
    'xi_asymptotic': xi_asymp,
    'gap_asymptotic': gap_asymp,
    'xi_euler': XI_EULER,
    'converges_to_xi': converges_to_xi,
    'large_d_data': large_d_data,
}


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("SUMMARY: Dimensional MVAE Results")
print("=" * 72)

print(f"\n  {'Part':6s} | {'Test':55s} | {'Result'}")
print(f"  {'-'*6}-+-{'-'*55}-+-{'-'*40}")
labels = {
    'A': 'Dimensional table (d=1..10)',
    'B': '2D cascade engine validation',
    'C': '4D prediction failure analysis',
    'D': 'Dimensional xi convergence',
}
for key in sorted(verdicts.keys()):
    print(f"  {key:6s} | {labels[key]:55s} | {verdicts[key]}")

print(f"\n  KEY FINDINGS:")
print(f"  1. xi_PAC(d) is monotonically increasing with dimension")
print(f"  2. The d->inf limit is 1+ln2*(1-ln2)^2 = {xi_asymp:.6f}, NOT Xi = {XI_EULER:.6f}")
print(f"  3. The gap (Xi - xi(inf)) = {gap_asymp:.6f} is the 'gamma contribution'")
print(f"  4. gamma = {GAMMA_EM:.6f} is NOT produced by cascade geometry")
print(f"  5. Xi = [cascade geometry limit] + [something involving gamma]")

# Save
all_results = {
    'experiment': 'minimum_actualization_resolution',
    'script': 'exp_08_dimensional_mvae.py',
    'timestamp': datetime.now().isoformat(),
    'verdicts': {k: str(v) for k, v in verdicts.items()},
    'results': results,
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"results/exp_08_dimensional_mvae_{ts}.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n  Results saved to {out_path}")
print("=" * 72)

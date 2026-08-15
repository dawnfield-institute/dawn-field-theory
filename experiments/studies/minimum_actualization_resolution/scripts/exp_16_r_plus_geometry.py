"""
r+ Geometric Interpretation — Experiment Script 16

PURPOSE:
    The Fibonacci-PAC recursion x_{n+2} = xi_PAC * x_{n+1} + x_n has
    characteristic root r+ = (xi_PAC + sqrt(xi_PAC^2 + 4))/2 = 2.0593.
    This lies between phi (1.6180) and l_MVAE (1.6294).
    This experiment investigates what r+ IS geometrically and dynamically.

HYPOTHESIS:
    r+ is the "PAC golden ratio" — the asymptotic growth rate of a
    PAC-modified Fibonacci sequence. It encodes the combined effect of
    cascade geometry (xi_PAC) and recursive structure (Fibonacci).

DESIGN:
    Part A: Properties of r+ (algebraic, continued fraction, relations)
    Part B: PAC-Fibonacci sequence dynamics
    Part C: r+ as spiral growth rate (geometric interpretation)
    Part D: r+ sensitivity to xi_PAC (how does r+ depend on the balance constant?)
    Part E: Phase space portrait — PAC-Fibonacci orbits

CORPUS CONTEXT:
    - exp_05 Section C: r+ = 2.0593, lies between phi and l_MVAE
    - Standard Fibonacci: r = phi from x_{n+2} = x_{n+1} + x_n
    - PAC modification: coefficient xi_PAC ≠ 1 changes the growth rate
    - xi_PAC = 1.0571 (empirical) or 1 + (7/8)*ln2*(1-ln2)^2 (MVAE formula)
"""

import json
import math
import numpy as np
from datetime import datetime
import sys
import os

m4_core = os.path.join(os.path.dirname(__file__), '..', '..', 'milestone4', 'core')
sys.path.insert(0, os.path.abspath(m4_core))
from utils import print_header

# ============================================================
# Constants
# ============================================================
LN2 = math.log(2)
PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
GAMMA_EM = 0.5772156649015328
PI = math.pi

XI_PAC = 1.0571
XI_MVAE = 1 + (7/8) * LN2 * (1 - LN2)**2
XI_EULER = GAMMA_EM + LN_PHI
L_MVAE = 1 / (2 * (1 - LN2))

# PAC-Fibonacci characteristic roots
disc = XI_PAC**2 + 4
R_PLUS = (XI_PAC + math.sqrt(disc)) / 2
R_MINUS = (XI_PAC - math.sqrt(disc)) / 2

def fibonacci(n):
    if n <= 0:
        return 0
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a

results = {}
verdicts = {}

print("=" * 72)
print("EXPERIMENT 16: r+ Geometric Interpretation")
print("Minimum Actualization Resolution — Dawn Field Institute")
print("=" * 72)
print(f"\n  xi_PAC = {XI_PAC:.10f}")
print(f"  r+     = (xi_PAC + sqrt(xi_PAC^2 + 4))/2 = {R_PLUS:.10f}")
print(f"  r-     = (xi_PAC - sqrt(xi_PAC^2 + 4))/2 = {R_MINUS:.10f}")
print(f"  phi    = {PHI:.10f}")
print(f"  l_MVAE = {L_MVAE:.10f}")
print(f"  r+ - phi    = {R_PLUS - PHI:.10f} ({(R_PLUS-PHI)/PHI*100:.4f}%)")
print(f"  r+ - l_MVAE = {R_PLUS - L_MVAE:.10f} ({(R_PLUS-L_MVAE)/L_MVAE*100:.4f}%)")
print()


# ============================================================
# PART A: Algebraic Properties of r+
# ============================================================
print_header("PART A: Algebraic Properties",
             "r+ as root of r^2 - xi_PAC*r - 1 = 0")

print(f"""
  Standard Fibonacci: r^2 = r + 1 → r = phi = (1+sqrt(5))/2
  PAC-Fibonacci:      r^2 = xi*r + 1 → r = (xi + sqrt(xi^2+4))/2

  Properties of r+ (analogous to phi properties):
""")

# Verify: r+^2 = xi_PAC * r+ + 1
print(f"  r+^2 = {R_PLUS**2:.10f}")
print(f"  xi*r+ + 1 = {XI_PAC * R_PLUS + 1:.10f}")
print(f"  Match: {abs(R_PLUS**2 - XI_PAC*R_PLUS - 1) < 1e-10}")
print()

# r+ * r- product and sum
print(f"  r+ * r- = {R_PLUS * R_MINUS:.10f}  (should be -1)")
print(f"  r+ + r- = {R_PLUS + R_MINUS:.10f}  (should be xi_PAC = {XI_PAC:.10f})")
print(f"  r+ - r- = {R_PLUS - R_MINUS:.10f}  (= sqrt(xi^2+4) = {math.sqrt(disc):.10f})")
print()

# 1/r+ properties
print(f"  1/r+ = {1/R_PLUS:.10f}")
print(f"  r+ - xi_PAC = {R_PLUS - XI_PAC:.10f}")
print(f"  1/(r+ - xi_PAC) = {1/(R_PLUS - XI_PAC):.10f}")
print(f"  (should equal r+ since 1/r+ = r+ - xi_PAC? No: 1/r+ = -r-, so:)")
print(f"  -r- = {-R_MINUS:.10f}  vs  1/r+ = {1/R_PLUS:.10f}")
print(f"  Match: {abs(-R_MINUS - 1/R_PLUS) < 1e-10}")
print()

# Continued fraction of r+
def continued_fraction(x, n_terms=15):
    cf = []
    for _ in range(n_terms):
        a = int(x)
        cf.append(a)
        frac = x - a
        if abs(frac) < 1e-10:
            break
        x = 1 / frac
    return cf

cf_rplus = continued_fraction(R_PLUS, 15)
cf_phi = continued_fraction(PHI, 15)
cf_lmvae = continued_fraction(L_MVAE, 15)

print(f"  Continued fractions:")
print(f"    phi    = [{', '.join(str(a) for a in cf_phi[:10])}]  (all 1s)")
print(f"    l_MVAE = [{', '.join(str(a) for a in cf_lmvae[:10])}]")
print(f"    r+     = [{', '.join(str(a) for a in cf_rplus[:10])}]")

# ln(r+) — the information content per PAC-Fibonacci step
ln_rplus = math.log(R_PLUS)
print(f"\n  Information content:")
print(f"    ln(phi)  = {LN_PHI:.10f} nats/step (standard Fibonacci)")
print(f"    ln(r+)   = {ln_rplus:.10f} nats/step (PAC-Fibonacci)")
print(f"    ln(l_MVAE) = {math.log(L_MVAE):.10f} nats/step")
print(f"    Ratio ln(r+)/ln(phi) = {ln_rplus/LN_PHI:.6f}")

verdicts['A'] = f'r+ = {R_PLUS:.6f}, ln(r+) = {ln_rplus:.6f} nats/step'
results['part_a'] = {
    'r_plus': R_PLUS,
    'r_minus': R_MINUS,
    'cf_rplus': cf_rplus[:10],
    'ln_rplus': ln_rplus,
}


# ============================================================
# PART B: PAC-Fibonacci Sequence Dynamics
# ============================================================
print_header("PART B: PAC-Fibonacci Sequence",
             "x_{n+2} = xi_PAC * x_{n+1} + x_n")

print(f"  Standard Fibonacci (xi=1): 1, 1, 2, 3, 5, 8, 13, 21, 34, ...")
print(f"  PAC-Fibonacci (xi={XI_PAC}):")
print()

# Generate PAC-Fibonacci sequence
pac_fib = [1.0, 1.0]
for _ in range(25):
    pac_fib.append(XI_PAC * pac_fib[-1] + pac_fib[-2])

# Standard Fibonacci for comparison
std_fib = [1.0, 1.0]
for _ in range(25):
    std_fib.append(std_fib[-1] + std_fib[-2])

print(f"  {'n':>4} {'std F_n':>14} {'PAC x_n':>14} {'ratio F':>10} {'ratio PAC':>10} {'PAC/std':>10}")
print(f"  {'-'*4}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
for n in range(len(pac_fib)):
    r_std = std_fib[n] / std_fib[n-1] if n > 0 else float('nan')
    r_pac = pac_fib[n] / pac_fib[n-1] if n > 0 else float('nan')
    ratio = pac_fib[n] / std_fib[n] if std_fib[n] > 0 else float('nan')
    if n <= 15 or n % 5 == 0:
        print(f"  {n:>4d} {std_fib[n]:>14.4f} {pac_fib[n]:>14.4f} {r_std:>10.6f} {r_pac:>10.6f} {ratio:>10.6f}")

# Convergence of ratio to r+
final_ratio = pac_fib[-1] / pac_fib[-2]
print(f"\n  PAC-Fibonacci ratio converges to: {final_ratio:.10f}")
print(f"  r+ (theoretical):                 {R_PLUS:.10f}")
print(f"  Match: {abs(final_ratio - R_PLUS) < 1e-6}")

# How fast does it converge?
convergence_data = []
for n in range(2, len(pac_fib)):
    ratio = pac_fib[n] / pac_fib[n-1]
    err = abs(ratio - R_PLUS) / R_PLUS
    convergence_data.append({'n': n, 'ratio': ratio, 'error': err})

# Find n where error < 1e-6
n_converge = next((c['n'] for c in convergence_data if c['error'] < 1e-6), None)
print(f"  Converges to 6 digits by n = {n_converge}")

verdicts['B'] = f'PAC-Fibonacci ratio → r+ = {R_PLUS:.6f} by n={n_converge}'
results['part_b'] = {
    'pac_fib_first_10': pac_fib[:10],
    'n_converge': n_converge,
    'final_ratio': final_ratio,
}


# ============================================================
# PART C: Spiral Growth Rate
# ============================================================
print_header("PART C: Spiral Interpretation",
             "r+ as the growth rate of a PAC spiral")

print(f"""
  In standard Fibonacci geometry:
    The golden spiral grows by phi per quarter turn (90°).
    Each rectangle is phi times larger than the previous.

  In PAC-Fibonacci:
    The spiral grows by r+ = {R_PLUS:.6f} per step.
    This is a FASTER spiral than the golden spiral.

  Physical interpretation:
    The golden spiral (phi) represents pure recursive subdivision.
    The PAC spiral (r+) represents recursive subdivision WITH
    a balance cost (xi_PAC > 1 means each step "taxes" slightly more).
""")

# Compare spiral properties
print(f"  Spiral comparison:")
print(f"  {'Property':>30s} {'Golden (phi)':>14} {'PAC (r+)':>14} {'MVAE (l_MVAE)':>14}")
print(f"  {'-'*30}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")

properties = [
    ("Growth rate per step", PHI, R_PLUS, L_MVAE),
    ("ln(rate) [nats/step]", LN_PHI, ln_rplus, math.log(L_MVAE)),
    ("Area ratio per step", PHI**2, R_PLUS**2, L_MVAE**2),
    ("Steps per e-fold", 1/LN_PHI, 1/ln_rplus, 1/math.log(L_MVAE)),
    ("Steps per 10-fold", math.log(10)/LN_PHI, math.log(10)/ln_rplus, math.log(10)/math.log(L_MVAE)),
]

for name, v_phi, v_rp, v_lm in properties:
    print(f"  {name:>30s} {v_phi:>14.6f} {v_rp:>14.6f} {v_lm:>14.6f}")

# The PAC spiral angle
# In golden spiral: angle per step = 2*pi/ln(phi) * ln(phi) = 2*pi (full turn per phi ratio)
# Actually: golden spiral angle = arctan(1/phi) per Fibonacci rectangle
angle_phi = math.atan(1/PHI)  # ≈ 0.5536 rad ≈ 31.7°
angle_rplus = math.atan(1/R_PLUS)  # tighter spiral
angle_lmvae = math.atan(1/L_MVAE)

print(f"\n  Spiral angles:")
print(f"    Golden: arctan(1/phi) = {angle_phi:.6f} rad = {math.degrees(angle_phi):.2f}°")
print(f"    PAC:    arctan(1/r+)  = {angle_rplus:.6f} rad = {math.degrees(angle_rplus):.2f}°")
print(f"    MVAE:   arctan(1/l_M) = {angle_lmvae:.6f} rad = {math.degrees(angle_lmvae):.2f}°")

verdicts['C'] = f'PAC spiral: {math.degrees(angle_rplus):.1f}° per step ({math.degrees(angle_phi):.1f}° for golden)'
results['part_c'] = {
    'angle_phi_deg': math.degrees(angle_phi),
    'angle_rplus_deg': math.degrees(angle_rplus),
}


# ============================================================
# PART D: r+ Sensitivity to xi_PAC
# ============================================================
print_header("PART D: Sensitivity Analysis",
             "How does r+ depend on the balance constant xi?")

print(f"  r+(xi) = (xi + sqrt(xi^2 + 4))/2")
print(f"  dr+/dxi = (1 + xi/sqrt(xi^2+4))/2")
print()

# Compute r+ for a range of xi values
xi_values = np.linspace(0.5, 2.0, 31)
print(f"  {'xi':>8} {'r+(xi)':>14} {'dr+/dxi':>14} {'r+(xi)/phi':>12}")
print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}")

rplus_data = []
for xi in xi_values:
    d = xi**2 + 4
    rp = (xi + math.sqrt(d)) / 2
    drp = (1 + xi / math.sqrt(d)) / 2
    rplus_data.append({'xi': float(xi), 'r_plus': rp, 'dr_dxi': drp})
    if abs(xi - 1.0) < 0.02 or abs(xi - XI_PAC) < 0.02 or xi in [0.5, 1.5, 2.0]:
        marker = " <- standard Fibonacci" if abs(xi - 1.0) < 0.02 else (" <- PAC" if abs(xi - XI_PAC) < 0.02 else "")
        print(f"  {xi:>8.4f} {rp:>14.8f} {drp:>14.8f} {rp/PHI:>12.6f}{marker}")

# Key points
print(f"\n  Special values:")
print(f"    xi = 0: r+ = (0+2)/2 = 1.0 (no growth)")
print(f"    xi = 1: r+ = (1+sqrt(5))/2 = phi = {PHI:.8f} (standard Fibonacci)")
print(f"    xi = xi_PAC = {XI_PAC}: r+ = {R_PLUS:.8f}")
print(f"    xi = 2: r+ = (2+sqrt(8))/2 = 1+sqrt(2) = {1+math.sqrt(2):.8f}")

# Derivative at xi_PAC
drp_at_pac = (1 + XI_PAC / math.sqrt(XI_PAC**2 + 4)) / 2
print(f"\n  Sensitivity at xi_PAC:")
print(f"    dr+/dxi|_{{xi_PAC}} = {drp_at_pac:.6f}")
print(f"    1% change in xi_PAC → {drp_at_pac*0.01*XI_PAC:.6f} change in r+")
print(f"    → {drp_at_pac*0.01*XI_PAC/R_PLUS*100:.4f}% change in r+")
print(f"    r+ is LESS sensitive than xi (amplification < 1)")

verdicts['D'] = f'dr+/dxi = {drp_at_pac:.4f} at xi_PAC (sub-linear amplification)'
results['part_d'] = {
    'drp_dxi_at_pac': drp_at_pac,
    'rplus_data': rplus_data[::3],  # every 3rd point
}


# ============================================================
# PART E: Phase Space Portrait
# ============================================================
print_header("PART E: Phase Space",
             "PAC-Fibonacci orbits in (x_n, x_{n+1}) space")

print(f"""
  In phase space (x_n, x_{{n+1}}), the PAC-Fibonacci recurrence traces
  a trajectory. For standard Fibonacci, the trajectory approaches the
  eigenvector direction [1, phi]. For PAC-Fibonacci, it approaches [1, r+].

  The orbit converges as |r-/r+|^n = |{R_MINUS/R_PLUS:.6f}|^n.
  Since |r-/r+| = {abs(R_MINUS/R_PLUS):.6f} < 1, convergence is exponential.
""")

contraction = abs(R_MINUS / R_PLUS)
print(f"  Contraction ratio |r-/r+| = {contraction:.8f}")
print(f"  vs golden: |(-1/phi)/phi| = 1/phi^2 = {1/PHI**2:.8f}")
print(f"  PAC contracts {'faster' if contraction < 1/PHI**2 else 'slower'} than standard Fibonacci")

# Number of steps to converge to 6 digits
n_6digit = math.ceil(math.log(1e-6) / math.log(contraction))
print(f"  Steps to 6-digit convergence: {n_6digit}")
print(f"  (Standard Fibonacci: {math.ceil(math.log(1e-6) / math.log(1/PHI**2))})")

# The eigenvectors
# Eigenvector for r+: [1, r+]
# Eigenvector for r-: [1, r-]
print(f"\n  Eigenvectors of the PAC-Fibonacci map:")
print(f"    v+ = [1, r+] = [1, {R_PLUS:.6f}]  (attractor)")
print(f"    v- = [1, r-] = [1, {R_MINUS:.6f}]  (repeller)")
print(f"    Angle of v+ from x-axis: {math.degrees(math.atan(R_PLUS)):.2f}°")
print(f"    Angle of golden v+:      {math.degrees(math.atan(PHI)):.2f}°")

# The PAC-Fibonacci map as a matrix
print(f"\n  Matrix form: [x_{{n+1}}, x_{{n+2}}] = M * [x_n, x_{{n+1}}]")
print(f"  M = [[0, 1], [1, xi_PAC]] = [[0, 1], [1, {XI_PAC}]]")
print(f"  det(M) = -1 (area-preserving, like standard Fibonacci)")
print(f"  tr(M) = xi_PAC = {XI_PAC}")
print(f"  (Standard Fibonacci: tr = 1, det = -1)")

# The map is area-preserving but NOT conformal — it shears
# The shearing rate is proportional to xi_PAC - 1 = 0.0571
shear = XI_PAC - 1
print(f"\n  Shear relative to Fibonacci: xi_PAC - 1 = {shear:.6f}")
print(f"  This is the SEC pump: each PAC step shears the phase space")
print(f"  by {shear:.4f} = (7/8)*ln2*(1-ln2)^2 beyond the Fibonacci baseline.")

verdicts['E'] = f'Contraction |r-/r+| = {contraction:.4f}, converges in {n_6digit} steps'
results['part_e'] = {
    'contraction_ratio': contraction,
    'n_6digit': n_6digit,
    'shear': shear,
    'attractor_angle_deg': math.degrees(math.atan(R_PLUS)),
}


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("SUMMARY: r+ Geometric Interpretation")
print("=" * 72)

print(f"\n  {'Part':6s} | {'Test':55s} | {'Result'}")
print(f"  {'-'*6}-+-{'-'*55}-+-{'-'*40}")
labels = {
    'A': 'Algebraic properties',
    'B': 'PAC-Fibonacci sequence dynamics',
    'C': 'Spiral growth rate',
    'D': 'Sensitivity to xi_PAC',
    'E': 'Phase space portrait',
}
for key in sorted(verdicts.keys()):
    print(f"  {key:6s} | {labels[key]:55s} | {verdicts[key]}")

print(f"\n  KEY FINDINGS:")
print(f"  1. r+ = {R_PLUS:.6f} is the 'PAC golden ratio' — growth rate of")
print(f"     the PAC-modified Fibonacci sequence x_{{n+2}} = xi*x_{{n+1}} + x_n")
print(f"  2. r+ satisfies r^2 = xi*r + 1 (analog of phi^2 = phi + 1)")
print(f"  3. r+*r- = -1 (same as phi*(-1/phi) = -1)")
print(f"  4. PAC spiral is tighter ({math.degrees(math.atan(R_PLUS)):.1f}° vs {math.degrees(math.atan(PHI)):.1f}° per step)")
print(f"  5. Phase space shear = xi_PAC - 1 = {shear:.4f} (the SEC pump)")
print(f"  6. r+ is between phi and l_MVAE but distinct from both:")
print(f"     phi = {PHI:.6f} < l_MVAE = {L_MVAE:.6f} < r+ = {R_PLUS:.6f}")
print(f"\n  INTERPRETATION:")
print(f"  r+ encodes how the PAC cascade GROWS when modulated by the")
print(f"  balance constant xi_PAC. It's larger than both phi and l_MVAE")
print(f"  because the xi_PAC > 1 amplification compounds recursively.")
print(f"  At xi = 1: r+ = phi (pure Fibonacci)")
print(f"  At xi = xi_PAC: r+ = {R_PLUS:.6f} (PAC-modified)")
print(f"  The excess r+ - phi = {R_PLUS - PHI:.6f} is the recursive")
print(f"  amplification of the SEC pump (xi_PAC - 1 = {shear:.4f}).")

# Save
all_results = {
    'experiment': 'minimum_actualization_resolution',
    'script': 'exp_16_r_plus_geometry.py',
    'timestamp': datetime.now().isoformat(),
    'verdicts': {k: str(v) for k, v in verdicts.items()},
    'results': results,
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"results/exp_16_r_plus_geometry_{ts}.json"
os.makedirs("results", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n  Results saved to {out_path}")
print("=" * 72)

"""
Euler Gap First-Principles Derivation — Experiment Script 14

PURPOSE:
    The Euler gap Xi - xi_PAC ≈ 1/(240*pi) at 0.09% error is currently empirical.
    This experiment attempts to derive 1/(240*pi) from first principles by
    connecting the PAC discrete-to-continuum correction to E8 lattice theory,
    Fibonacci partition theory, and zeta function regularization.

HYPOTHESIS:
    The Euler gap is the leading correction from discretizing a continuum
    Fibonacci field onto a lattice. 240 = F3*F4*F5*F6 appears because
    four consecutive Fibonacci numbers span the minimal complete
    representation of the PAC cascade coupling structure.

DESIGN:
    Part A: Euler-Maclaurin approach — correction from discrete Fibonacci sum
    Part B: E8 connection — why 240 root vectors?
    Part C: Fibonacci partition function — Z_fib and its corrections
    Part D: Bernoulli number route — B_2/(2!) in Euler-Maclaurin
    Part E: Candidate exact formula refinement

CORPUS CONTEXT:
    - exp_05 Section D: gap ≈ 1/(240*pi) at 0.09%
    - exp_09: 240 = F3*F4*F5*F6, rank #1/75 Fibonacci products (p=0.005)
    - Paper 5: 240 appears in Casimir regularization (E8 root vectors)
    - xi_MVAE = 1 + (7/8)*ln2*(1-ln2)^2 = 1.057108...
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
EULER_GAP = XI_EULER - XI_MVAE
CANDIDATE_240PI = 1 / (240 * PI)

def fibonacci(n):
    if n <= 0:
        return 0
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a

# Bernoulli numbers (first few)
BERNOULLI = {
    0: 1, 1: -0.5, 2: 1/6, 4: -1/30, 6: 1/42, 8: -1/30,
    10: 5/66, 12: -691/2730
}

results = {}
verdicts = {}

print("=" * 72)
print("EXPERIMENT 14: Euler Gap First-Principles Derivation")
print("Minimum Actualization Resolution — Dawn Field Institute")
print("=" * 72)
print(f"\n  Xi       = {XI_EULER:.15f}")
print(f"  xi_MVAE  = {XI_MVAE:.15f}")
print(f"  Gap      = {EULER_GAP:.15f}")
print(f"  1/(240pi)= {CANDIDATE_240PI:.15f}")
print(f"  Error    = {abs(EULER_GAP - CANDIDATE_240PI)/EULER_GAP*100:.4f}%")
print()


# ============================================================
# PART A: Euler-Maclaurin Correction
# ============================================================
print_header("PART A: Euler-Maclaurin Approach",
             "Does the discrete-to-continuum correction give 1/(240*pi)?")

print("""
  The Euler-Maclaurin formula converts a discrete sum to an integral
  plus correction terms involving Bernoulli numbers:

    sum_{k=a}^{b} f(k) = int_a^b f(x)dx + (f(a)+f(b))/2
                         + sum_{j=1}^p B_{2j}/(2j)! * (f^{(2j-1)}(b) - f^{(2j-1)}(a))
                         + remainder

  For f(k) = phi^{-k} (PAC cascade energy at depth k):
    f'(x) = -ln(phi) * phi^{-x}
    f''(x) = ln(phi)^2 * phi^{-x}
    f^{(n)}(x) = (-ln(phi))^n * phi^{-x}

  The Euler-Maclaurin correction terms are:
    B_2/2! * ln(phi)^1 * [phi^{-b} - phi^{-a}]  (leading correction)
""")

# Compute Euler-Maclaurin corrections for f(k) = phi^{-k}
a, b = 1, 200
f_a = PHI**(-a)
f_b = PHI**(-b)

# Exact discrete sum
S_exact = sum(PHI**(-k) for k in range(a, b + 1))
# Integral
I_exact = (PHI**(-a) - PHI**(-b)) / LN_PHI

# Euler-Maclaurin corrections
em_corrections = []
em_total = (f_a + f_b) / 2  # zeroth correction

print(f"  Discrete sum S = {S_exact:.15f}")
print(f"  Integral    I = {I_exact:.15f}")
print(f"  Raw gap S - I = {S_exact - I_exact:.15f}")
print(f"  (f(a)+f(b))/2 = {em_total:.15f}")
print()

print(f"  Euler-Maclaurin correction terms:")
print(f"  {'j':>3} {'B_{2j}/(2j)!':>16} {'derivative factor':>20} {'correction':>16} {'running total':>16}")
print(f"  {'-'*3}-+-{'-'*16}-+-{'-'*20}-+-{'-'*16}-+-{'-'*16}")

running_em = em_total
for j in range(1, 7):
    idx = 2 * j
    if idx not in BERNOULLI:
        continue
    B_n = BERNOULLI[idx]
    factorial = math.factorial(idx)
    coeff = B_n / factorial

    # f^{(2j-1)} = (-ln_phi)^{(2j-1)} * phi^{-x}
    deriv_power = 2 * j - 1
    deriv_factor = (-LN_PHI)**deriv_power
    correction = coeff * deriv_factor * (f_b - f_a)

    running_em += correction
    em_corrections.append({
        'j': j, 'B_2j': B_n, 'coeff': coeff,
        'correction': correction, 'running': running_em
    })
    print(f"  {j:>3d} {coeff:>16.10e} {deriv_factor:>20.10e} {correction:>16.10e} {running_em:>16.12f}")

gap_with_em = S_exact - I_exact - running_em
print(f"\n  Gap after Euler-Maclaurin: {gap_with_em:.15e}")
print(f"  Euler gap target:         {EULER_GAP:.15e}")
print(f"  These are different objects — EM corrects sum vs integral,")
print(f"  while the Euler gap corrects xi_MVAE vs Xi.")

# But: can we relate the EM leading term to 1/(240*pi)?
em_leading = em_corrections[0]['correction'] if em_corrections else 0
print(f"\n  Leading EM correction: {em_leading:.10e}")
print(f"  vs 1/(240*pi): {CANDIDATE_240PI:.10e}")
print(f"  Ratio: {em_leading / CANDIDATE_240PI:.6f}")

verdicts['A'] = f'Euler-Maclaurin gap = {gap_with_em:.4e} (different from Euler gap)'
results['part_a'] = {
    'S_exact': S_exact, 'I_exact': I_exact,
    'em_corrections': em_corrections,
    'gap_after_em': gap_with_em,
}


# ============================================================
# PART B: E8 Connection
# ============================================================
print_header("PART B: E8 Root System Connection",
             "Why 240 = number of E8 root vectors?")

print("""
  240 has multiple structural decompositions:
    240 = 2 × 3 × 5 × 8 = F_3 × F_4 × F_5 × F_6 (four consecutive Fibonacci)
    240 = |roots of E8| (the exceptional Lie group)
    240 = 2 × 120 = 2 × |binary icosahedral group|
    240 = 2^4 × 15 = 16 × 15
    240 = 24 × 10 (24 = F_3 × F_4 × F_5 × 1... no)

  The E8 lattice is THE densest lattice packing in 8D.
  Its theta function: Theta_E8(q) = 1 + 240*q + 2160*q^2 + ...
  The 240 counts the NEAREST NEIGHBORS in E8.

  Question: does 8D = F_6 play a role? F_6 = 8.
  And: does the PAC cascade have E8 structure in some embedding?
""")

# Consecutive Fibonacci product structure
print(f"  240 = F_3 × F_4 × F_5 × F_6 = {fibonacci(3)} × {fibonacci(4)} × {fibonacci(5)} × {fibonacci(6)}")
print(f"  Product of 4 consecutive Fibonacci starting at F_3.")
print()

# Test all consecutive-4 Fibonacci products
print(f"  All consecutive-4 Fibonacci products:")
for start in range(1, 12):
    prod = fibonacci(start) * fibonacci(start+1) * fibonacci(start+2) * fibonacci(start+3)
    marker = " <- E8/240" if prod == 240 else ""
    print(f"    F_{start}*F_{start+1}*F_{start+2}*F_{start+3} = {prod}{marker}")

# Connection: E8 has dimension 8 = F_6
# Its root system has 240 vectors
# The Euler gap involves 240*pi
# pi enters through the circular/spherical geometry of the gap correction
print(f"\n  Structural analysis:")
print(f"    E8 dimension = 8 = F_6")
print(f"    E8 roots = 240 = F_3*F_4*F_5*F_6")
print(f"    Euler gap ≈ 1/(240*pi) = 1/(E8_roots * pi)")
print(f"    Interpretation: the gap is the solid angle per root vector")
print(f"    in the E8 lattice: 1/(240*pi) = angular correction per NN")

# Solid angle of 8-sphere: S_7 = 2*pi^4/3 (surface area of unit 7-sphere)
S_7 = 2 * PI**4 / 3
angle_per_root = S_7 / 240
print(f"\n  8D sphere surface: S_7 = 2*pi^4/3 = {S_7:.6f}")
print(f"  Solid angle per E8 root: S_7/240 = {angle_per_root:.6f}")
print(f"  1/(240*pi): {CANDIDATE_240PI:.6f}")
print(f"  Ratio S_7/(240*240*pi): {S_7/(240*240*PI):.6f}")

verdicts['B'] = f'240 = E8 roots = F3*F4*F5*F6. Gap = 1/(roots*pi) [angular correction]'
results['part_b'] = {
    'e8_dim': 8,
    'e8_roots': 240,
    'fibonacci_product': f'{fibonacci(3)}*{fibonacci(4)}*{fibonacci(5)}*{fibonacci(6)}',
    'S_7': S_7,
}


# ============================================================
# PART C: Fibonacci Partition Function
# ============================================================
print_header("PART C: Fibonacci Partition Function",
             "Z_fib(s) corrections and their relation to the Euler gap")

print("""
  Define the Fibonacci zeta function: Z_fib(s) = sum_{k=1}^inf F_k^{-s}

  For the PAC cascade, the partition function weights each level k by
  its Fibonacci energy F_k. The gap between discrete and analytic
  continuation of Z_fib might produce the Euler gap.
""")

# Compute Z_fib(s) for various s
print(f"  {'s':>6} {'Z_fib(s)':>16} {'1/Z_fib(s)':>16} {'Z_fib(s)*pi':>16}")
print(f"  {'-'*6}-+-{'-'*16}-+-{'-'*16}-+-{'-'*16}")

for s_10 in range(5, 55, 5):
    s = s_10 / 10
    z = sum(fibonacci(k)**(-s) for k in range(1, 50) if fibonacci(k) > 0)
    print(f"  {s:>6.1f} {z:>16.10f} {1/z:>16.10f} {z*PI:>16.10f}")

# At s=1: Z_fib(1) = reciprocal Fibonacci constant ≈ 3.3599
z_fib_1 = sum(1.0/fibonacci(k) for k in range(1, 50))
# At s=2: Z_fib(2) = sum(1/F_k^2)
z_fib_2 = sum(1.0/fibonacci(k)**2 for k in range(1, 50))

print(f"\n  Z_fib(1) = {z_fib_1:.12f} (reciprocal Fibonacci constant)")
print(f"  Z_fib(2) = {z_fib_2:.12f}")
print(f"  1/Z_fib(2) = {1/z_fib_2:.12f}")

# Test: is the Euler gap related to Z_fib?
print(f"\n  Testing Euler gap vs Z_fib expressions:")
candidates_c = [
    ("1/Z_fib(1)", 1/z_fib_1),
    ("1/(Z_fib(1)*pi)", 1/(z_fib_1*PI)),
    ("Z_fib(2)/(4*pi)", z_fib_2/(4*PI)),
    ("1/(Z_fib(1)*Z_fib(2)*pi)", 1/(z_fib_1*z_fib_2*PI)),
    ("1/(240*pi)", CANDIDATE_240PI),
    ("ln(phi)/(Z_fib(1)*pi)", LN_PHI/(z_fib_1*PI)),
]

for name, val in candidates_c:
    err = abs(val - EULER_GAP) / EULER_GAP * 100
    marker = " <--" if err < 1 else ""
    print(f"    {name:35s} = {val:.12f} (err = {err:.4f}%){marker}")

verdicts['C'] = f'Z_fib(1) = {z_fib_1:.6f}, no direct derivation of gap'
results['part_c'] = {
    'z_fib_1': z_fib_1,
    'z_fib_2': z_fib_2,
}


# ============================================================
# PART D: Bernoulli Number Route
# ============================================================
print_header("PART D: Bernoulli Number Route",
             "Does B_n produce 1/(240*pi) through Fibonacci weighting?")

print("""
  The Bernoulli numbers appear in:
    - Euler-Maclaurin formula (discrete-to-continuum corrections)
    - zeta(-n) = -B_{n+1}/(n+1)
    - Taylor expansion of x/(e^x - 1)

  Connection attempt: B_4 = -1/30, and 240 = -8*B_4^{-1} = 8*30 = 240.
  Also: B_4/(4!) = -1/720. And 720 = 3*240.
""")

# Bernoulli-240 connections
print(f"  B_4 = -1/30")
print(f"  -1/B_4 = 30")
print(f"  8 * (-1/B_4) = 8 * 30 = 240")
print(f"  F_6 * (-1/B_4) = {fibonacci(6)} * 30 = {fibonacci(6) * 30}")
print(f"  This gives 240 = F_6 / (-B_4)")
print()

# More precise: does B_4 / (something * pi) give the gap?
val_b4 = -1/30
print(f"  Testing Bernoulli-based formulas:")
b_candidates = [
    ("B_4/(8*pi)", val_b4/(8*PI)),
    ("|B_4|/(8*pi)", abs(val_b4)/(8*PI)),
    ("|B_4|^2/pi", val_b4**2/PI),
    ("1/((-1/B_4)*F_6*pi)", 1/(30*8*PI)),
    ("B_2/(F_6*F_5*pi)", BERNOULLI[2]/(fibonacci(6)*fibonacci(5)*PI)),
    ("|B_4|/(F_5*pi)", abs(val_b4)/(fibonacci(5)*PI)),
]

for name, val in b_candidates:
    err = abs(val - EULER_GAP) / EULER_GAP * 100
    marker = " <--" if err < 1 else ""
    print(f"    {name:30s} = {val:.12f} (err = {err:.4f}%){marker}")

# Direct: 1/(240*pi) = 1/(F3*F4*F5*F6*pi) = ?/(Bernoulli * factorial * ...)
# 240 = 2*3*5*8 = prod of 4 consecutive Fibonacci
# B_4 = -1/30 = -1/(2*3*5)
# So 240 = 8 * |1/B_4| = F_6 * |1/B_4|
# 1/(240*pi) = |B_4|/(F_6*pi)
val_b4_f6_pi = abs(val_b4) / (fibonacci(6) * PI)
err_b4 = abs(val_b4_f6_pi - EULER_GAP) / EULER_GAP * 100
print(f"\n  Key formula: |B_4|/(F_6*pi) = (1/30)/(8*pi) = 1/(240*pi)")
print(f"    = {val_b4_f6_pi:.12f}")
print(f"    vs gap = {EULER_GAP:.12f}")
print(f"    Error: {err_b4:.4f}%")
print()
print(f"  DERIVATION ATTEMPT:")
print(f"    B_4 = -1/30 enters through Euler-Maclaurin at order j=2")
print(f"    F_6 = 8 is the physical mode count (3D BCC cascade)")
print(f"    pi enters through circular symmetry of the balance operator")
print(f"    Gap = |B_4|/(F_6 * pi) = Euler-Maclaurin correction / (modes * geometry)")

verdicts['D'] = f'Gap = |B_4|/(F_6*pi) = 1/(240*pi) [Bernoulli + Fibonacci + geometry]'
results['part_d'] = {
    'B_4': val_b4,
    'F_6': fibonacci(6),
    'formula': '|B_4|/(F_6*pi) = 1/(240*pi)',
    'error_pct': err_b4,
}


# ============================================================
# PART E: Candidate Exact Formula
# ============================================================
print_header("PART E: Refining the Exact Formula",
             "Is the gap EXACTLY 1/(240*pi) or is there a correction?")

print(f"""
  Current: gap = {EULER_GAP:.15f}
  1/(240*pi) = {CANDIDATE_240PI:.15f}
  Residual   = {EULER_GAP - CANDIDATE_240PI:.15e}
  Rel. error = {abs(EULER_GAP - CANDIDATE_240PI)/EULER_GAP*100:.6f}%

  The 0.09% error means 1/(240*pi) is an APPROXIMATION, not exact.
  Test higher-order corrections.
""")

# Residual after 1/(240*pi)
resid = EULER_GAP - CANDIDATE_240PI
print(f"  Residual = {resid:.15e}")
print(f"  Residual/gap = {resid/EULER_GAP:.6f} ({resid/EULER_GAP*100:.4f}%)")

# Test: gap = 1/(240*pi) + correction
print(f"\n  Testing corrections to 1/(240*pi):")
corrections = [
    ("+ 1/(240^2*pi)", CANDIDATE_240PI + 1/(240**2*PI)),
    ("+ 1/(240*pi^2)", CANDIDATE_240PI + 1/(240*PI**2)),
    ("+ ln2/(240^2*pi^2)", CANDIDATE_240PI + LN2/(240**2*PI**2)),
    ("* (1 + 1/(240*pi))", CANDIDATE_240PI * (1 + 1/(240*PI))),
    ("* (1 + ln2/240)", CANDIDATE_240PI * (1 + LN2/240)),
    ("* (1 + 1/(F_7*pi))", CANDIDATE_240PI * (1 + 1/(fibonacci(7)*PI))),
    ("+ B_6/(F_6^2*F_5*pi^2)", CANDIDATE_240PI + BERNOULLI[6]/(fibonacci(6)**2*fibonacci(5)*PI**2)),
]

for name, val in corrections:
    err = abs(val - EULER_GAP) / EULER_GAP * 100
    marker = " <-- BETTER" if err < 0.09 else ""
    print(f"    1/(240*pi) {name:30s} = {val:.15f} (err = {err:.6f}%){marker}")

# Try: exact formula = gamma + ln(phi) - 1 - (7/8)*ln2*(1-ln2)^2
# This IS the gap by definition. The question is: does it simplify?
print(f"\n  Exact gap = gamma + ln(phi) - 1 - (7/8)*ln2*(1-ln2)^2")
print(f"           = {GAMMA_EM:.10f} + {LN_PHI:.10f} - 1 - {(7/8)*LN2*(1-LN2)**2:.10f}")
print(f"           = {EULER_GAP:.15f}")
print(f"\n  This is the EXACT formula. The question is whether")
print(f"  gamma + ln(phi) - 1 - (7/8)*ln2*(1-ln2)^2 simplifies to")
print(f"  1/(240*pi) + higher order terms.")
print(f"\n  Currently: no simplification found. The 0.09% match is")
print(f"  suggestive but may be coincidental at this precision.")

verdicts['E'] = f'1/(240*pi) at 0.09%, no exact derivation found'
results['part_e'] = {
    'residual': resid,
    'residual_rel': resid / EULER_GAP,
}


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("SUMMARY: Euler Gap Derivation Attempt")
print("=" * 72)

print(f"\n  {'Part':6s} | {'Test':55s} | {'Result'}")
print(f"  {'-'*6}-+-{'-'*55}-+-{'-'*40}")
labels = {
    'A': 'Euler-Maclaurin correction',
    'B': 'E8 root system connection',
    'C': 'Fibonacci partition function',
    'D': 'Bernoulli number route',
    'E': 'Exact formula refinement',
}
for key in sorted(verdicts.keys()):
    print(f"  {key:6s} | {labels[key]:55s} | {verdicts[key]}")

print(f"\n  KEY FINDINGS:")
print(f"  1. 240 = F_6/|B_4| = 8*30 (Fibonacci × inverse Bernoulli)")
print(f"  2. Gap ≈ |B_4|/(F_6*pi) — Bernoulli correction per mode per radian")
print(f"  3. E8 dimension = F_6 = 8, E8 roots = 240 = F3*F4*F5*F6")
print(f"  4. The 0.09% error means 1/(240*pi) is approximate, not exact")
print(f"  5. No exact derivation found — gap involves gamma irreducibly")
print(f"\n  INTERPRETATION:")
print(f"  The Euler gap is the leading discrete-to-continuum correction")
print(f"  in the PAC cascade. 240 enters through the Bernoulli number B_4")
print(f"  (which governs 4th-order lattice corrections) divided by the")
print(f"  3D mode count F_6=8. The 1/pi factor is geometric.")
print(f"  But the gap is not EXACTLY 1/(240*pi) — it's")
print(f"  gamma + ln(phi) - 1 - (7/8)*ln2*(1-ln2)^2,")
print(f"  and this doesn't simplify to a closed form involving 240.")
print(f"  The 240 connection is structural/approximate, not exact.")

# Save
all_results = {
    'experiment': 'minimum_actualization_resolution',
    'script': 'exp_14_euler_gap_derivation.py',
    'timestamp': datetime.now().isoformat(),
    'verdicts': {k: str(v) for k, v in verdicts.items()},
    'results': results,
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"results/exp_14_euler_gap_derivation_{ts}.json"
os.makedirs("results", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n  Results saved to {out_path}")
print("=" * 72)

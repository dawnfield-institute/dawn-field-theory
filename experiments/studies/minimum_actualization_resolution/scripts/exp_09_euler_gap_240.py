"""
Euler Gap — The 240 Connection — Experiment Script 09

PURPOSE:
    Tests whether the Euler gap (Xi - xi_PAC ~ 1/(240*pi)) shares the same
    240 as the Casimir regularization (240 = F3*F4*F5*F6 = E8 root vectors).
    Also probes whether gamma has a Fibonacci lattice origin.

HYPOTHESIS:
    The Euler gap is 1/(F3*F4*F5*F6*pi) exactly, connecting the MVAE
    discrete-to-continuum correction to E8 Fibonacci structure from Paper 5.

DESIGN:
    Part A: Precision verification — residual structure of the gap
    Part B: Fibonacci product scan — selectivity of 240
    Part C: Lattice refinement — does gap(n) converge from Fibonacci theory?
    Part D: gamma decomposition — Fibonacci harmonic series test

CORPUS CONTEXT:
    - exp_05 found Xi - xi_PAC ~ 1/(240*pi) at 0.09% error
    - Paper 5 (exp_06_casimir_mersenne.py): 240 = F3*F4*F5*F6 = E8 root vectors
    - gamma = 0.5772... appears in Xi but has NO first-principles derivation
    - Fibonacci harmonic series sum(1/F_k) converges to a known constant
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

def fibonacci(n):
    if n <= 0:
        return 0
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a

# Fibonacci sequence for reference
fibs = [fibonacci(n) for n in range(1, 25)]

results = {}
verdicts = {}

print("=" * 72)
print("EXPERIMENT 09: Euler Gap — The 240 Connection")
print("Minimum Actualization Resolution — Dawn Field Institute")
print("=" * 72)
print(f"\n  Xi = gamma + ln(phi) = {XI_EULER:.15f}")
print(f"  xi_PAC (nominal)     = {XI_PAC:.15f}")
print(f"  xi_MVAE (formula)    = {XI_MVAE:.15f}")
print(f"  Euler gap (Xi - xi_PAC):  {XI_EULER - XI_PAC:.15f}")
print(f"  Euler gap (Xi - xi_MVAE): {XI_EULER - XI_MVAE:.15f}")
print(f"  1/(240*pi):               {1/(240*PI):.15f}")
print()


# ============================================================
# PART A: Precision Verification
# ============================================================
print_header("PART A: Precision Verification",
             "What is the residual after subtracting 1/(240*pi)?")

# Use MVAE formula value (more precise than nominal 1.0571)
gap_nominal = XI_EULER - XI_PAC
gap_mvae = XI_EULER - XI_MVAE
candidate = 1 / (240 * PI)
fib_candidate = 1 / (fibonacci(3) * fibonacci(4) * fibonacci(5) * fibonacci(6) * PI)

print(f"  Two versions of xi_PAC:")
print(f"    Nominal (1.0571):     gap = {gap_nominal:.15f}")
print(f"    MVAE formula:         gap = {gap_mvae:.15f}")
print(f"    1/(240*pi):           val = {candidate:.15f}")
print(f"    1/(F3*F4*F5*F6*pi):  val = {fib_candidate:.15f}")
print(f"    (240 = F3*F4*F5*F6 = {fibonacci(3)}*{fibonacci(4)}*{fibonacci(5)}*{fibonacci(6)} = {fibonacci(3)*fibonacci(4)*fibonacci(5)*fibonacci(6)})")

# Residuals
resid_nominal = gap_nominal - candidate
resid_mvae = gap_mvae - candidate
pct_nominal = abs(resid_nominal) / gap_nominal * 100
pct_mvae = abs(resid_mvae) / gap_mvae * 100

print(f"\n  Residuals:")
print(f"    gap(nominal) - 1/(240*pi) = {resid_nominal:.15e} ({pct_nominal:.4f}%)")
print(f"    gap(MVAE)    - 1/(240*pi) = {resid_mvae:.15e} ({pct_mvae:.4f}%)")

# Is the residual itself a known constant?
print(f"\n  Residual analysis (using MVAE gap):")
print(f"  Residual = {resid_mvae:.15e}")
candidates_resid = [
    ("0", 0),
    ("1/(240^2*pi)", 1/(240**2 * PI)),
    ("1/(240*pi^2)", 1/(240 * PI**2)),
    ("ln2/(240*pi)", LN2 / (240 * PI)),
    ("(1-ln2)/(240*pi)", (1-LN2) / (240 * PI)),
    ("1/(F7*F8*pi)", 1 / (fibonacci(7) * fibonacci(8) * PI)),
    ("gamma/(240*pi)", GAMMA_EM / (240 * PI)),
]
for name, val in candidates_resid:
    if abs(val) > 1e-20:
        err = abs(val - resid_mvae) / abs(resid_mvae) * 100 if abs(resid_mvae) > 1e-20 else float('inf')
    else:
        err = float('inf') if abs(resid_mvae) > 1e-10 else 0
    marker = " <--" if err < 10 else ""
    print(f"    {name:25s} = {val:>15.10e} (err from residual: {err:.1f}%){marker}")

close_match = pct_mvae < 1.0  # within 1%
verdicts['A'] = f'gap ~ 1/(240*pi) at {pct_mvae:.2f}%'
print(f"\n  VERDICT: {verdicts['A']}")

results['part_a'] = {
    'gap_nominal': gap_nominal,
    'gap_mvae': gap_mvae,
    'candidate_240pi': candidate,
    'residual_mvae': resid_mvae,
    'pct_error_mvae': pct_mvae,
}


# ============================================================
# PART B: Fibonacci Product Scan
# ============================================================
print_header("PART B: Fibonacci Product Selectivity",
             "NULL: 240 is not the best Fibonacci product for the gap")

print(f"""
Test 1/(F_a * F_b * ... * pi) for consecutive Fibonacci products.
Is 240 = F3*F4*F5*F6 the BEST match for the Euler gap?
""")

# Test all consecutive Fibonacci products of length 2, 3, 4, 5
target_gap = gap_mvae
best_matches = []

for length in range(2, 7):
    for start in range(1, 16):
        if start + length > 24:
            break
        product = 1
        indices = list(range(start, start + length))
        for idx in indices:
            product *= fibonacci(idx)
        if product == 0:
            continue
        val = 1 / (product * PI)
        err = abs(val - target_gap) / target_gap * 100
        label = "*".join(f"F{i}" for i in indices)
        best_matches.append({
            'label': label,
            'product': product,
            'val': val,
            'error_pct': err,
            'length': length,
            'indices': indices,
        })

best_matches.sort(key=lambda x: x['error_pct'])

print(f"  Target gap: {target_gap:.15f}")
print(f"\n  {'Rank':>5} {'Formula':>25} {'Product':>10} {'1/(prod*pi)':>18} {'Error':>10}")
print(f"  {'-'*5}-+-{'-'*25}-+-{'-'*10}-+-{'-'*18}-+-{'-'*10}")
for i, m in enumerate(best_matches[:15]):
    marker = " <- THE 240" if m['product'] == 240 else ""
    print(f"  {i+1:>5d} {m['label']:>25s} {m['product']:>10d} {m['val']:>18.12f} {m['error_pct']:>9.4f}%{marker}")

# Where does 240 rank?
rank_240 = next((i+1 for i, m in enumerate(best_matches) if m['product'] == 240), None)
print(f"\n  240 = F3*F4*F5*F6 rank: #{rank_240} out of {len(best_matches)} Fibonacci products")

# NULL: random products of 4 integers in [2, 13]
N_NULL = 100000
rng = np.random.default_rng(42)
null_prods = np.prod(rng.integers(2, 14, size=(N_NULL, 4)), axis=1)
null_vals = 1 / (null_prods * PI)
null_errs = np.abs(null_vals - target_gap) / target_gap * 100
err_240 = abs(1/(240*PI) - target_gap) / target_gap * 100
p_random = float(np.mean(null_errs <= err_240))

print(f"\n  Random product null (4 integers from [2,13]):")
print(f"    P(random product closer than 240): {p_random:.6f}")
print(f"    240 error: {err_240:.4f}%")
print(f"    Median random error: {np.median(null_errs):.4f}%")

selective = rank_240 is not None and rank_240 <= 3
verdicts['B'] = f'PASS (240 rank #{rank_240})' if selective else f'FAIL (240 rank #{rank_240})'
print(f"\n  VERDICT: {verdicts['B']}")

results['part_b'] = {
    'rank_240': rank_240,
    'n_candidates': len(best_matches),
    'top_5': best_matches[:5],
    'p_random': p_random,
    'selective': selective,
}


# ============================================================
# PART C: Lattice Refinement
# ============================================================
print_header("PART C: Fibonacci Lattice Refinement",
             "Does gap(n) = ln(phi)/(F_n * F_{n+1}) converge to the Euler gap?")

print("""
exp_05 found the Euler gap shrinks with Fibonacci lattice refinement.
Candidate: gap(n) = ln(phi) / (F_n * F_{n+1})

Test: at what n does this formula best approximate the measured gap?
""")

print(f"  {'n':>4} {'F_n':>8} {'F_{n+1}':>8} {'F_n*F_{n+1}':>12} {'ln(phi)/(F_n*F_{n+1})':>22} {'Error from gap':>16}")
print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*22}-+-{'-'*16}")

lattice_data = []
best_lattice_n = None
best_lattice_err = float('inf')

for n in range(2, 16):
    fn = fibonacci(n)
    fn1 = fibonacci(n + 1)
    prod = fn * fn1
    gap_n = LN_PHI / prod
    err = abs(gap_n - target_gap) / target_gap * 100
    lattice_data.append({'n': n, 'fn': fn, 'fn1': fn1, 'gap_n': gap_n, 'error_pct': err})

    if err < best_lattice_err:
        best_lattice_err = err
        best_lattice_n = n

    print(f"  {n:>4d} {fn:>8d} {fn1:>8d} {prod:>12d} {gap_n:>22.12f} {err:>15.4f}%")

print(f"\n  Best lattice match: n={best_lattice_n} (error={best_lattice_err:.4f}%)")
print(f"  Compare with 1/(240*pi) error: {pct_mvae:.4f}%")

# Which formula is better?
lattice_better = best_lattice_err < pct_mvae
print(f"\n  Better formula: {'ln(phi)/(F_n*F_{n+1})' if lattice_better else '1/(240*pi)'}")

# Can we COMBINE them? E.g., gap = ln(phi) / (F_a * F_b * pi)?
print(f"\n  Combined formula test: gap = ln(phi) / (product * pi)")
combined = []
for length in range(2, 5):
    for start in range(2, 10):
        if start + length > 15:
            break
        prod = 1
        indices = list(range(start, start + length))
        for idx in indices:
            prod *= fibonacci(idx)
        val = LN_PHI / (prod * PI)
        err = abs(val - target_gap) / target_gap * 100
        combined.append({'indices': indices, 'product': prod, 'val': val, 'error': err})

combined.sort(key=lambda x: x['error'])
print(f"  {'Formula':>30s} {'Value':>18} {'Error':>10}")
print(f"  {'-'*30}-+-{'-'*18}-+-{'-'*10}")
for c in combined[:8]:
    label = "ln(phi)/(" + "*".join(f"F{i}" for i in c['indices']) + "*pi)"
    print(f"  {label:>30s} {c['val']:>18.12f} {c['error']:>9.4f}%")

verdicts['C'] = f'Best lattice: n={best_lattice_n} ({best_lattice_err:.2f}%)'
results['part_c'] = {
    'lattice_data': lattice_data,
    'best_n': best_lattice_n,
    'best_error': best_lattice_err,
    'lattice_better_than_240pi': lattice_better,
    'combined_top3': combined[:3],
}


# ============================================================
# PART D: gamma Decomposition — Fibonacci Harmonic Series
# ============================================================
print_header("PART D: gamma from Fibonacci Harmonic Series?",
             "Does sum(1/F_k) relate to gamma + ln(phi)?")

print(f"""
gamma = Xi - ln(phi) = {GAMMA_EM:.15f}
gamma = lim_{{n->inf}} (H_n - ln(n)) where H_n = sum_{{k=1}}^n 1/k

The reciprocal Fibonacci constant: sum_{{k=1}}^inf 1/F_k is known to converge
to approximately 3.3599 (irrational, not known to relate to gamma).

Test: does any Fibonacci harmonic series produce gamma or Xi?
""")

# Standard reciprocal Fibonacci constant
fib_harmonic = sum(1.0 / fibonacci(k) for k in range(1, 50))
print(f"  Reciprocal Fibonacci constant: sum(1/F_k) = {fib_harmonic:.10f}")
print(f"  gamma = {GAMMA_EM:.10f}")
print(f"  Xi = {XI_EULER:.10f}")
print(f"  Ratio fib_harmonic/Xi: {fib_harmonic/XI_EULER:.6f}")

# Standard harmonic series truncated at Fibonacci indices
print(f"\n  Harmonic number at Fibonacci indices:")
print(f"  {'n':>4} {'F_n':>8} {'H(F_n)':>14} {'H(F_n)-ln(F_n)':>16} {'diff from gamma':>16}")
print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*14}-+-{'-'*16}-+-{'-'*16}")

for n in range(3, 15):
    fn = fibonacci(n)
    # H(fn) = sum(1/k for k=1..fn)
    h_fn = sum(1.0/k for k in range(1, fn + 1))
    approx_gamma = h_fn - math.log(fn)
    diff = abs(approx_gamma - GAMMA_EM)
    print(f"  {n:>4d} {fn:>8d} {h_fn:>14.8f} {approx_gamma:>16.10f} {diff:>16.2e}")

# The H(F_n) - ln(F_n) converges to gamma (that's just the definition of gamma).
# Nothing special about Fibonacci here.

# More interesting: can we build gamma from Fibonacci products?
print(f"\n  gamma as Fibonacci expression:")
candidates_gamma = [
    ("1 - ln(phi)", 1 - LN_PHI),
    ("1/phi^2", 1/PHI**2),
    ("ln(phi)/ln(2)", LN_PHI / LN2),
    ("(F5-1)/(F7-1)", (fibonacci(5)-1)/(fibonacci(7)-1)),
    ("F4/(F5*phi)", fibonacci(4)/(fibonacci(5)*PHI)),
    ("1 - 1/phi - 1/phi^4", 1 - 1/PHI - 1/PHI**4),
    ("ln(phi) + 1/(2*pi)", LN_PHI + 1/(2*PI)),
    ("sum(1/F_k, k=3..7)/F5", sum(1/fibonacci(k) for k in range(3,8))/fibonacci(5)),
    ("2*ln(phi)/phi", 2*LN_PHI/PHI),
    ("pi/F5 - 1/phi", PI/fibonacci(5) - 1/PHI),
]

print(f"  {'Formula':>35s} {'Value':>14} {'Error from gamma':>18}")
print(f"  {'-'*35}-+-{'-'*14}-+-{'-'*18}")
best_gamma_match = None
best_gamma_err = float('inf')
for name, val in candidates_gamma:
    err = abs(val - GAMMA_EM) / GAMMA_EM * 100
    if err < best_gamma_err:
        best_gamma_err = err
        best_gamma_match = name
    marker = " <--" if err < 1 else ""
    print(f"  {name:>35s} {val:>14.10f} {err:>17.4f}%{marker}")

print(f"\n  Best match: {best_gamma_match} ({best_gamma_err:.4f}%)")

# Key test: is gamma derivable from phi/Fibonacci?
gamma_from_fib = best_gamma_err < 0.1  # within 0.1%
print(f"\n  gamma derivable from Fibonacci alone: {'POSSIBLY' if gamma_from_fib else 'NO'}")
print(f"  gamma appears to be an INDEPENDENT constant that enters Xi")
print(f"  through harmonic series / number-theoretic structure,")
print(f"  not through cascade geometry.")

verdicts['D'] = 'gamma NOT derived from Fibonacci' if not gamma_from_fib else 'gamma MATCHES Fibonacci expression'
results['part_d'] = {
    'fib_harmonic': fib_harmonic,
    'best_gamma_match': best_gamma_match,
    'best_gamma_error': best_gamma_err,
    'gamma_from_fibonacci': gamma_from_fib,
}


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("SUMMARY: Euler Gap 240 Connection")
print("=" * 72)

print(f"\n  {'Part':6s} | {'Test':55s} | {'Result'}")
print(f"  {'-'*6}-+-{'-'*55}-+-{'-'*40}")
labels = {
    'A': 'Precision (gap ~ 1/(240*pi))',
    'B': 'Fibonacci product selectivity',
    'C': 'Lattice refinement convergence',
    'D': 'gamma decomposition',
}
for key in sorted(verdicts.keys()):
    print(f"  {key:6s} | {labels[key]:55s} | {verdicts[key]}")

print(f"\n  KEY FINDINGS:")
print(f"  1. Euler gap = Xi - xi_MVAE = {gap_mvae:.10f}")
print(f"  2. 1/(240*pi) = {candidate:.10f} (error: {pct_mvae:.4f}%)")
print(f"  3. 240 = F3*F4*F5*F6 = 2*3*5*8 (Casimir E8 root vectors)")
print(f"  4. gamma is NOT derivable from Fibonacci/phi alone")
print(f"  5. Xi = [MVAE cascade limit] + [number-theoretic correction]")

# Save
all_results = {
    'experiment': 'minimum_actualization_resolution',
    'script': 'exp_09_euler_gap_240.py',
    'timestamp': datetime.now().isoformat(),
    'verdicts': {k: str(v) for k, v in verdicts.items()},
    'results': results,
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"results/exp_09_euler_gap_240_{ts}.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n  Results saved to {out_path}")
print("=" * 72)

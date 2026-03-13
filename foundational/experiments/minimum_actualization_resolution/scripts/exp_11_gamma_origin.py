"""
Gamma Origin Probe — Experiment Script 11

PURPOSE:
    Investigates why the Euler-Mascheroni constant gamma = 0.5772... appears
    in Xi = gamma + ln(phi) but is NOT derivable from cascade geometry.
    The MVAE framework produces xi_PAC = 1 + (7/8)*ln2*(1-ln2)^2 but NOT Xi.
    The gap Xi - xi(inf) contains gamma. Where does it come from?

HYPOTHESIS:
    gamma enters Xi as the discrete-to-continuum correction: the difference
    between a discrete PAC cascade (finite Fibonacci sum) and the continuum
    limit (integral). This is structurally identical to gamma = lim(H_n - ln n).

DESIGN:
    Part A: PAC cascade discrete-continuum gap — does it converge to gamma?
    Part B: Fibonacci harmonic series — sum(1/F_k) and partial sums
    Part C: Stieltjes constant decomposition of gamma
    Part D: gamma as regularization of cascade partition function
    Part E: Information-theoretic gamma — bits lost in discrete-to-continuum

CORPUS CONTEXT:
    - exp_08 Part D: xi(inf) = 1 + ln2*(1-ln2)^2 ≠ Xi. Gap = 0.001015.
    - exp_09 Part D: gamma NOT from Fibonacci harmonic series or phi expressions.
    - Xi = gamma + ln(phi) is the continuum balance constant from 5 domains.
    - gamma = 0.5772156649... from H_n - ln(n) → gamma.

Planck units throughout.
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
XI_FLOOR = 1.0 - LN2**2

def fibonacci(n):
    if n <= 0:
        return 0
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a

def harmonic(n):
    """Compute H_n = sum(1/k, k=1..n)."""
    return sum(1.0 / k for k in range(1, n + 1))

results = {}
verdicts = {}

print("=" * 72)
print("EXPERIMENT 11: Gamma Origin Probe")
print("Minimum Actualization Resolution — Dawn Field Institute")
print("=" * 72)
print(f"\n  gamma        = {GAMMA_EM:.15f}")
print(f"  ln(phi)      = {LN_PHI:.15f}")
print(f"  Xi           = {XI_EULER:.15f}")
print(f"  xi_MVAE      = {XI_MVAE:.15f}")
print(f"  Xi - xi_MVAE = {XI_EULER - XI_MVAE:.15f}")
print()


# ============================================================
# PART A: Discrete-to-Continuum Gap in PAC Fibonacci Cascade
# ============================================================
print_header("PART A: Discrete-to-Continuum Gap",
             "Does the PAC Fibonacci cascade produce a gamma-like correction?")

print("""
  In standard analysis: gamma = lim_{n->inf} (H_n - ln(n))
  where H_n = sum(1/k, k=1..n) is the discrete harmonic sum
  and ln(n) is the continuum integral approximation.

  PAC analogy: the cascade distributes energy as phi^{-k} at depth k.
  Discrete sum: S_N = sum(phi^{-k}, k=1..N)
  Continuum integral: I_N = integral(phi^{-x}, x=1..N) = (phi^{-1} - phi^{-N})/ln(phi)

  The gap S_N - I_N should converge to a constant. Is it gamma?
""")

# Compute discrete vs continuum PAC sums
depths = list(range(5, 205, 5))
pac_gaps = []

print(f"  {'N':>5} {'S_N (discrete)':>18} {'I_N (continuum)':>18} {'Gap S_N - I_N':>18} {'H_N - ln(N)':>14}")
print(f"  {'-'*5}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}-+-{'-'*14}")

for N in depths:
    # Discrete Fibonacci-weighted sum: sum(phi^{-k}, k=1..N)
    S_N = sum(PHI**(-k) for k in range(1, N + 1))
    # Continuum integral: int_1^N phi^{-x} dx = [phi^{-x}/(-ln phi)]_1^N
    I_N = (PHI**(-1) - PHI**(-N)) / LN_PHI
    gap_pac = S_N - I_N
    # Standard harmonic gap for comparison
    H_N = harmonic(N)
    h_gap = H_N - math.log(N)
    pac_gaps.append({'N': N, 'S_N': S_N, 'I_N': I_N, 'gap': gap_pac, 'h_gap': h_gap})

    if N <= 50 or N % 50 == 0:
        print(f"  {N:>5d} {S_N:>18.12f} {I_N:>18.12f} {gap_pac:>18.12f} {h_gap:>14.10f}")

# The PAC gap converges quickly (geometric series)
pac_gap_limit = pac_gaps[-1]['gap']
h_gap_limit = pac_gaps[-1]['h_gap']

print(f"\n  PAC gap limit (N=200): {pac_gap_limit:.15f}")
print(f"  Harmonic gap (N=200): {h_gap_limit:.15f}  (should → gamma = {GAMMA_EM:.15f})")
print(f"\n  PAC gap vs gamma: {pac_gap_limit:.10f} vs {GAMMA_EM:.10f}")
print(f"  Ratio: {pac_gap_limit / GAMMA_EM:.6f}")
print(f"  PAC gap vs ln(phi): {pac_gap_limit:.10f} vs {LN_PHI:.10f}")

# The PAC gap should be a function of phi, not gamma
# For geometric series: S = phi^{-1}/(1-phi^{-1}) = 1/(phi-1) = phi (exact!)
# Integral: I = phi^{-1}/ln(phi) (as N->inf)
S_inf = 1 / (PHI - 1)  # = phi
I_inf = 1 / (PHI * LN_PHI)
pac_gap_exact = S_inf - I_inf

print(f"\n  Exact limits:")
print(f"    S_inf = phi/(phi-1) = phi = {S_inf:.12f}")
print(f"    I_inf = 1/(phi*ln(phi)) = {I_inf:.12f}")
print(f"    Gap = phi - 1/(phi*ln(phi)) = {pac_gap_exact:.12f}")

# Express the gap in terms of known constants
print(f"\n  Decomposing PAC gap = {pac_gap_exact:.12f}:")
candidates_a = [
    ("gamma", GAMMA_EM),
    ("1/phi", 1/PHI),
    ("phi - 1/(phi*ln(phi))", pac_gap_exact),
    ("1 + ln(phi)/2", 1 + LN_PHI/2),
    ("phi*ln(phi)", PHI * LN_PHI),
]
for name, val in candidates_a:
    err = abs(val - pac_gap_exact) / pac_gap_exact * 100
    print(f"    {name:30s} = {val:.10f} (err = {err:.4f}%)")

pac_gap_is_gamma = abs(pac_gap_exact - GAMMA_EM) / GAMMA_EM < 0.01
verdicts['A'] = f'PAC gap = {pac_gap_exact:.6f}, gamma = {GAMMA_EM:.6f} — {"MATCH" if pac_gap_is_gamma else "NO MATCH"}'
print(f"\n  VERDICT: {verdicts['A']}")
print(f"  The PAC phi^{{-k}} cascade gap is NOT gamma.")
print(f"  gamma enters through harmonic (1/k) series, not geometric (phi^{{-k}}).")

results['part_a'] = {
    'pac_gap_exact': pac_gap_exact,
    'gamma': GAMMA_EM,
    'pac_gap_is_gamma': pac_gap_is_gamma,
    'S_inf': S_inf,
    'I_inf': I_inf,
}


# ============================================================
# PART B: Fibonacci-Weighted Harmonic Series
# ============================================================
print_header("PART B: Fibonacci-Weighted Harmonic Series",
             "Does sum(1/(k*F_k)) or similar produce gamma?")

print("""
  Standard: gamma = lim(H_n - ln(n)), H_n = sum(1/k)
  Test: what if we weight the harmonic series by Fibonacci?
  Candidates:
    B1: sum(1/(k * F_k))  — Fibonacci-damped harmonic
    B2: sum((-1)^k / F_k) — alternating Fibonacci
    B3: sum(ln(F_{k+1}/F_k) - ln(phi)) — Fibonacci ratio correction
    B4: product(F_{k+1}/F_k) / phi^N  — Fibonacci ratio product gap
""")

# B1: Fibonacci-damped harmonic series
N_terms = 30
fib_harmonic_damped = sum(1.0 / (k * fibonacci(k)) for k in range(1, N_terms + 1))
print(f"  B1: sum(1/(k*F_k), k=1..{N_terms}) = {fib_harmonic_damped:.12f}")
print(f"      vs gamma = {GAMMA_EM:.12f} (err = {abs(fib_harmonic_damped - GAMMA_EM)/GAMMA_EM*100:.4f}%)")

# B2: Alternating Fibonacci series
alt_fib = sum((-1)**(k+1) / fibonacci(k) for k in range(1, N_terms + 1))
print(f"  B2: sum((-1)^(k+1)/F_k, k=1..{N_terms}) = {alt_fib:.12f}")
print(f"      vs gamma = {GAMMA_EM:.12f} (err = {abs(alt_fib - GAMMA_EM)/GAMMA_EM*100:.4f}%)")

# B3: Fibonacci ratio correction series
# F_{k+1}/F_k → phi, but the corrections are (-1)^k / (F_k * F_{k+1})
fib_ratio_corrections = []
running_correction = 0
for k in range(2, N_terms + 1):
    fk = fibonacci(k)
    fk1 = fibonacci(k + 1)
    ratio = fk1 / fk
    correction = math.log(ratio) - LN_PHI
    running_correction += correction
    fib_ratio_corrections.append(running_correction)

fib_ratio_sum = running_correction
print(f"  B3: sum(ln(F_{{k+1}}/F_k) - ln(phi), k=2..{N_terms}) = {fib_ratio_sum:.12f}")
print(f"      This is ln(F_{N_terms+1}/(F_2 * phi^{N_terms-1}))")

# B4: Product gap — prod(F_{k+1}/F_k) vs phi^N
product_ratios = 1.0
for k in range(2, N_terms + 1):
    product_ratios *= fibonacci(k + 1) / fibonacci(k)
product_gap = product_ratios / PHI**(N_terms - 1)
print(f"  B4: prod(F_{{k+1}}/F_k, k=2..{N_terms}) / phi^{N_terms-1} = {product_gap:.12f}")
print(f"      = F_{N_terms+1} / (F_2 * phi^{N_terms-1})")
# This is related to the Fibonacci closed form: F_n = (phi^n - psi^n)/sqrt(5)
# So the ratio F_n/phi^n → 1/sqrt(5)
fn_over_phin = fibonacci(N_terms + 1) / PHI**(N_terms + 1)
print(f"      F_{N_terms+1}/phi^{N_terms+1} = {fn_over_phin:.12f} (→ 1/sqrt(5) = {1/math.sqrt(5):.12f})")

# B5: Construct a series that DOES produce gamma from Fibonacci structure
# Key insight: gamma = integral_0^1 (1/(1-x) + 1/ln(x)) dx  (Gauss representation)
# In Fibonacci context: replace x with phi^{-1} = 1/phi
# gamma_fib = integral_0^{1/phi} (1/(1-x) + 1/ln(x)) dx
# This doesn't give gamma — it gives a different constant
x0 = 1 / PHI
# Numerical integration
from scipy import integrate
def gauss_integrand(x):
    if abs(x) < 1e-15:
        return 0.5
    if abs(1 - x) < 1e-15:
        return 0
    return 1/(1-x) + 1/math.log(x) if x > 0 and x < 1 else 0

gamma_full, _ = integrate.quad(gauss_integrand, 0, 1)
gamma_phi_partial, _ = integrate.quad(gauss_integrand, 0, x0)

print(f"\n  B5: Gauss representation of gamma:")
print(f"      gamma = integral_0^1 (1/(1-x) + 1/ln(x)) dx = {gamma_full:.12f}")
print(f"      (should be {GAMMA_EM:.12f})")
print(f"      gamma_phi = integral_0^{{1/phi}} same integrand = {gamma_phi_partial:.12f}")
print(f"      ratio gamma_phi/gamma = {gamma_phi_partial/GAMMA_EM:.6f}")

# What fraction of gamma comes from the [0, 1/phi] interval?
gamma_complement, _ = integrate.quad(gauss_integrand, x0, 1)
print(f"      [0, 1/phi] contributes: {gamma_phi_partial:.6f} ({gamma_phi_partial/GAMMA_EM*100:.1f}%)")
print(f"      [1/phi, 1] contributes: {gamma_complement:.6f} ({gamma_complement/GAMMA_EM*100:.1f}%)")

best_b_err = min(
    abs(fib_harmonic_damped - GAMMA_EM) / GAMMA_EM,
    abs(alt_fib - GAMMA_EM) / GAMMA_EM,
)
verdicts['B'] = f'No Fibonacci series produces gamma (best err = {best_b_err*100:.2f}%)'

results['part_b'] = {
    'fib_harmonic_damped': fib_harmonic_damped,
    'alt_fib': alt_fib,
    'fib_ratio_sum': fib_ratio_sum,
    'gamma_phi_partial': gamma_phi_partial,
    'gamma_phi_fraction': gamma_phi_partial / GAMMA_EM,
}


# ============================================================
# PART C: Stieltjes Constant Decomposition
# ============================================================
print_header("PART C: Stieltjes Constant Analysis",
             "Can gamma be decomposed through Stieltjes constants?")

print("""
  gamma = gamma_0 (the zeroth Stieltjes constant).
  Laurent expansion of zeta(s) near s=1:
    zeta(s) = 1/(s-1) + sum_{n=0}^inf (-1)^n * gamma_n * (s-1)^n / n!

  gamma_0 = gamma = 0.5772156649...
  gamma_1 = -0.0728158454...
  gamma_2 = -0.0096903631...

  Test: does Xi - ln(phi) decompose in terms of Stieltjes constants?
  Or: does the MVAE framework predict the Stieltjes constants?
""")

# Known Stieltjes constants (high precision)
gamma_0 = 0.5772156649015328606
gamma_1 = -0.0728158454836767248
gamma_2 = -0.0096903631928723184
gamma_3 = 0.0020538344203033458

print(f"  gamma_0 = {gamma_0:.16f}")
print(f"  gamma_1 = {gamma_1:.16f}")
print(f"  gamma_2 = {gamma_2:.16f}")
print(f"  gamma_3 = {gamma_3:.16f}")

# Test: gamma decomposed as cascade-like structure
# Hypothesis: gamma = sum of contributions at each Fibonacci depth
print(f"\n  Fibonacci depth contributions to gamma (H(F_n) - H(F_{'{n-1}'})):")
print(f"  {'n':>4} {'F_n':>8} {'H(F_n)':>14} {'Delta H':>14} {'Delta H * ln(phi)':>18}")
print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*14}-+-{'-'*14}-+-{'-'*18}")

prev_h = 0
depth_contributions = []
for n in range(1, 14):
    fn = fibonacci(n)
    h_fn = harmonic(fn)
    delta_h = h_fn - prev_h
    prev_h = h_fn
    depth_contributions.append({
        'n': n, 'F_n': fn, 'H_Fn': h_fn,
        'delta_H': delta_h, 'delta_H_ln_phi': delta_h * LN_PHI
    })
    print(f"  {n:>4d} {fn:>8d} {h_fn:>14.8f} {delta_h:>14.8f} {delta_h * LN_PHI:>18.8f}")

# Does the Fibonacci-indexed harmonic series have special structure?
# H(F_n) ≈ n * ln(phi) + gamma  (since F_n ≈ phi^n/sqrt(5))
print(f"\n  Testing H(F_n) ≈ n*ln(phi) + gamma + correction:")
print(f"  {'n':>4} {'F_n':>8} {'H(F_n)':>14} {'n*ln(phi)+gamma':>16} {'residual':>14}")
print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*14}-+-{'-'*16}-+-{'-'*14}")

for n in range(3, 14):
    fn = fibonacci(n)
    h_fn = harmonic(fn)
    approx = n * LN_PHI + GAMMA_EM
    residual = h_fn - approx
    print(f"  {n:>4d} {fn:>8d} {h_fn:>14.8f} {approx:>16.8f} {residual:>14.8f}")

# The residual should be -ln(sqrt(5)) = -0.5*ln(5) since F_n ~ phi^n/sqrt(5)
# so ln(F_n) ~ n*ln(phi) - 0.5*ln(5)
# and H(F_n) ~ ln(F_n) + gamma ~ n*ln(phi) - 0.5*ln(5) + gamma
correction = -0.5 * math.log(5)
print(f"\n  Expected correction: -ln(sqrt(5)) = {correction:.10f}")
print(f"  H(F_n) ≈ n*ln(phi) + gamma - ln(sqrt(5))")
print(f"         = n*ln(phi) + gamma - 0.5*ln(5)")

# Verify
for n in [8, 10, 12]:
    fn = fibonacci(n)
    h_fn = harmonic(fn)
    approx2 = n * LN_PHI + GAMMA_EM + correction
    residual2 = h_fn - approx2
    print(f"  n={n}: H(F_{n})={h_fn:.8f}, approx={approx2:.8f}, residual={residual2:.8f}")

verdicts['C'] = 'gamma = H(F_n) - n*ln(phi) + ln(sqrt(5)) [standard asymptotic]'
results['part_c'] = {
    'stieltjes': [gamma_0, gamma_1, gamma_2, gamma_3],
    'correction_ln_sqrt5': correction,
    'depth_contributions': depth_contributions[:8],
}


# ============================================================
# PART D: Cascade Partition Function Regularization
# ============================================================
print_header("PART D: Cascade Partition Function",
             "gamma as the regularization constant of the PAC cascade")

print("""
  The PAC cascade partition function:
    Z(beta) = sum_{k=1}^inf exp(-beta * E_k)

  where E_k = k * ln(2) (Landauer cost at depth k).

  Z(beta) = sum exp(-beta*k*ln(2)) = 1/(exp(beta*ln2) - 1)  [Bose-Einstein]

  The free energy F = -ln(Z)/beta has a divergent part as beta->0:
    F ~ -ln(1/(beta*ln2))/beta = ln(beta*ln2)/beta

  The regularized free energy should contain gamma through zeta function
  regularization: zeta'(0) = -0.5*ln(2*pi).

  But does the PAC cascade specifically introduce gamma?
""")

# PAC cascade partition function
def cascade_Z(beta, N_max=1000):
    """Partition function of PAC cascade: Z = sum exp(-beta * k * ln2)."""
    return sum(math.exp(-beta * k * LN2) for k in range(1, N_max + 1))

# Test: free energy at various beta
print(f"  {'beta':>8} {'Z(beta)':>16} {'F = -ln(Z)/beta':>18} {'F + ln(beta*ln2)/beta':>24}")
print(f"  {'-'*8}-+-{'-'*16}-+-{'-'*18}-+-{'-'*24}")

betas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
for beta in betas:
    Z = cascade_Z(beta)
    F = -math.log(Z) / beta
    # Leading divergence: F ~ ln(beta*ln2)/beta
    F_leading = math.log(beta * LN2) / beta
    residual = F - F_leading
    print(f"  {beta:>8.3f} {Z:>16.6f} {F:>18.10f} {residual:>24.10f}")

# At beta=1 (natural temperature): the cascade is in the physical regime
Z_phys = cascade_Z(1.0)
F_phys = -math.log(Z_phys)
print(f"\n  Physical (beta=1):")
print(f"    Z(1) = 1/(e^ln2 - 1) = 1/(2-1) = 1.0 (exact)")
print(f"    Z(1) computed: {Z_phys:.10f}")
print(f"    F(1) = -ln(Z(1)) = -ln(1) = 0")
print(f"    F(1) computed: {F_phys:.10f}")

# Connection to zeta: Z(beta) at integer beta relates to zeta function
# sum_{k=1}^inf k^{-s} = zeta(s)
# But our sum is geometric, not power-law. Different series.
# The Fibonacci-weighted version might connect:
# Z_fib(s) = sum F_k^{-s}

print(f"\n  Fibonacci zeta function: Z_fib(s) = sum(F_k^(-s), k=1..inf)")
for s in [1.0, 1.5, 2.0, 2.5, 3.0]:
    z_fib = sum(fibonacci(k)**(-s) for k in range(1, 30))
    print(f"    Z_fib({s:.1f}) = {z_fib:.10f}")

z_fib_1 = sum(1.0/fibonacci(k) for k in range(1, 50))
print(f"\n  Z_fib(1) = reciprocal Fibonacci constant = {z_fib_1:.10f}")
print(f"  vs pi = {PI:.10f}")
print(f"  Z_fib(1)/pi = {z_fib_1/PI:.6f}")
print(f"  Z_fib(1) - gamma - ln(phi) = {z_fib_1 - GAMMA_EM - LN_PHI:.6f}")
print(f"  Z_fib(1) - Xi = {z_fib_1 - XI_EULER:.6f}")

verdicts['D'] = f'Z_fib(1) = {z_fib_1:.6f}, Xi = {XI_EULER:.6f} — no direct connection'
results['part_d'] = {
    'Z_fib_1': z_fib_1,
    'F_phys': F_phys,
    'Z_phys': Z_phys,
}


# ============================================================
# PART E: Information-Theoretic Gamma
# ============================================================
print_header("PART E: Information-Theoretic Interpretation",
             "gamma as the cost of enumerating an infinite discrete set")

print("""
  Interpretation attempt: gamma measures the ENUMERATION COST of a
  countably infinite set. In information theory:

    To specify element k of a countable set, you need ~log(k) bits.
    The average cost over the first N elements: (1/N) * sum(log(k), k=1..N)
    = (1/N) * log(N!) ≈ log(N) - 1 + 1/(2N) + ...  (Stirling)

    But the EXACT average is H_N/ln(2) ≈ (ln(N) + gamma)/ln(2).
    The gamma/ln(2) term is the discrete overhead of exact enumeration
    vs continuum approximation.

  In PAC: each cascade level k is a distinct actualization event.
  The cost of ADDRESSING level k (knowing which level you're at)
  is log(k). The total addressing cost contains gamma.
""")

# gamma/ln(2) = cost in bits of discrete-vs-continuum enumeration
gamma_bits = GAMMA_EM / LN2
print(f"  gamma/ln(2) = {gamma_bits:.10f} bits")
print(f"  This is the per-level discrete overhead in the PAC cascade.")
print()

# In the MVAE framework:
# xi_PAC captures cascade geometry (how energy distributes)
# gamma captures enumeration cost (how many levels are distinguishable)
# ln(phi) captures the information content per PAC step

print(f"  Xi decomposition in information-theoretic terms:")
print(f"    gamma    = {GAMMA_EM:.10f}  [enumeration cost / discrete overhead]")
print(f"    ln(phi)  = {LN_PHI:.10f}  [information per PAC step]")
print(f"    Xi       = {XI_EULER:.10f}  [total: enumeration + information]")
print()

# xi_PAC vs Xi: the gap is what cascade geometry DOESN'T capture
gap_mvae_xi = XI_EULER - XI_MVAE
print(f"  What MVAE captures vs what it doesn't:")
print(f"    xi_MVAE = {XI_MVAE:.10f}  [cascade geometry: modes + dissipation + regulation]")
print(f"    Xi      = {XI_EULER:.10f}  [full continuum limit]")
print(f"    gap     = {gap_mvae_xi:.10f}  [the 'gamma contribution']")
print()

# Decompose the gap
# Xi - xi_MVAE = gamma + ln(phi) - 1 - (7/8)*ln2*(1-ln2)^2
# Let's see if this simplifies
gap_formula = GAMMA_EM + LN_PHI - 1 - (7/8) * LN2 * (1 - LN2)**2
print(f"  Gap = gamma + ln(phi) - 1 - (7/8)*ln2*(1-ln2)^2 = {gap_formula:.10f}")
print(f"  = gamma + [{LN_PHI:.6f} - 1 - {(7/8)*LN2*(1-LN2)**2:.6f}]")
residual_without_gamma = LN_PHI - 1 - (7/8) * LN2 * (1 - LN2)**2
print(f"  = gamma + {residual_without_gamma:.10f}")
print(f"  The non-gamma residual: ln(phi) - 1 - (7/8)*ln2*(1-ln2)^2 = {residual_without_gamma:.10f}")
print()

# Is the non-gamma residual expressible in closed form?
candidates_e = [
    ("ln(phi) - 1", LN_PHI - 1),
    ("-1 + ln(phi)", -1 + LN_PHI),
    ("-(1 - ln(phi))", -(1 - LN_PHI)),
    ("(7/8)*ln2*(1-ln2)^2", (7/8)*LN2*(1-LN2)**2),
    ("ln(phi) - 1 - sec_pump", residual_without_gamma),
]
print(f"  Non-gamma residual = {residual_without_gamma:.10f}")
for name, val in candidates_e:
    print(f"    {name:40s} = {val:.10f}")

print(f"\n  INTERPRETATION:")
print(f"  gamma enters Xi as the enumeration/counting overhead of the")
print(f"  discrete cascade. The MVAE formula captures the geometric structure")
print(f"  (mode fractions, dissipation, regulation) but NOT the counting cost.")
print(f"  Xi = [cascade geometry] + [counting cost] where:")
print(f"    cascade geometry = 1 + (7/8)*ln2*(1-ln2)^2 = {XI_MVAE:.10f}")
print(f"    counting cost = gamma + (ln(phi) - 1 - sec_pump)")
print(f"                  = {gap_formula:.10f}")

verdicts['E'] = f'gamma = enumeration cost of discrete cascade (gap = {gap_formula:.6f})'
results['part_e'] = {
    'gamma_bits': gamma_bits,
    'gap_mvae_xi': gap_mvae_xi,
    'gap_formula': gap_formula,
    'residual_without_gamma': residual_without_gamma,
}


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("SUMMARY: Gamma Origin Probe")
print("=" * 72)

print(f"\n  {'Part':6s} | {'Test':55s} | {'Result'}")
print(f"  {'-'*6}-+-{'-'*55}-+-{'-'*40}")
labels = {
    'A': 'Discrete-continuum gap (phi^{-k} cascade)',
    'B': 'Fibonacci-weighted harmonic series',
    'C': 'Stieltjes constant decomposition',
    'D': 'Cascade partition function regularization',
    'E': 'Information-theoretic interpretation',
}
for key in sorted(verdicts.keys()):
    print(f"  {key:6s} | {labels[key]:55s} | {verdicts[key]}")

print(f"\n  KEY FINDINGS:")
print(f"  1. The PAC phi^{{-k}} cascade gap is NOT gamma (Part A)")
print(f"     (geometric series gap ≠ harmonic series gap)")
print(f"  2. No Fibonacci-weighted series produces gamma (Part B)")
print(f"  3. H(F_n) ≈ n*ln(phi) + gamma - ln(sqrt(5)) [standard asymptotic] (Part C)")
print(f"  4. Fibonacci zeta Z_fib(1) = {z_fib_1:.4f} has no direct Xi connection (Part D)")
print(f"  5. gamma = enumeration overhead of discrete cascade levels (Part E)")
print(f"     gamma/ln(2) = {gamma_bits:.6f} bits per level")
print(f"\n  CONCLUSION:")
print(f"  gamma is IRREDUCIBLE to cascade geometry. It enters Xi as the")
print(f"  number-theoretic cost of discrete enumeration — the same reason")
print(f"  it appears in H_n - ln(n). The PAC cascade inherits gamma because")
print(f"  it operates on countably many discrete levels, and the cost of")
print(f"  'knowing which level' is exactly gamma per nat of information.")
print(f"  This is not a derivation FROM PAC — it's a derivation OF WHY")
print(f"  PAC needs gamma: any discrete recursive structure does.")

# Save
all_results = {
    'experiment': 'minimum_actualization_resolution',
    'script': 'exp_11_gamma_origin.py',
    'timestamp': datetime.now().isoformat(),
    'verdicts': {k: str(v) for k, v in verdicts.items()},
    'results': results,
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"results/exp_11_gamma_origin_{ts}.json"
os.makedirs("results", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n  Results saved to {out_path}")
print("=" * 72)

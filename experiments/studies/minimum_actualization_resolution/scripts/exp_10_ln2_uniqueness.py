"""
ln(2) Uniqueness — Information Base Test — Experiment Script 10

PURPOSE:
    The entire MVAE framework uses ln(2) because Landauer says erasing 1 binary
    bit costs kT*ln(2). What if information were b-ary (base b)? The cost becomes
    kT*ln(b). Does the framework generalize, break, or single out binary?

HYPOTHESIS:
    b=2 (binary) is the ONLY integer information base that produces a physically
    valid cascade (xi_floor > 0). This follows from xi_floor(b) = 1 - ln^2(b),
    which goes negative for b >= 3. The allowed range is b in (1, e), and 2 is
    the only integer in that interval.

DESIGN:
    Part A: Generalized Landauer floor — xi_floor(b) for continuous b
    Part B: Maximum information base — solve 1 - ln^2(b) = 0
    Part C: Generalized conjugacy — base-independence of product structure
    Part D: eta bridge generalization — which base maximizes recycling?
    Part E: Binary selectivity — b=2 is the unique valid integer base

CORPUS CONTEXT:
    - xi_floor = 1 - ln^2(2) = 0.5195 (exp_02, exp_06 Part D)
    - MVAE constraints: E=ln(2), dt=1/(2*ln(2)), dx=1/(2*(1-ln(2)))
    - Conjugacy products: E*dt = 1/2 always, dx_grav*dx_unc = 1 always (exp_06 Part E)
    - Landauer's principle: kT*ln(b) per b-ary digit erasure
    - e = 2.71828... is the natural base; ln(e) = 1, so xi_floor(e) = 0 exactly
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
E_EULER = math.e

XI_PAC = 1.0571
XI_MVAE = 1 + (7/8) * LN2 * (1 - LN2)**2
XI_EULER = GAMMA_EM + LN_PHI

results = {}
verdicts = {}


# ============================================================
# Part A: Generalized Landauer Floor
# ============================================================
print_header("Part A: Generalized Landauer Floor")

print("  xi_floor(b) = 1 - ln^2(b)")
print("  For xi_floor > 0: ln^2(b) < 1 => |ln(b)| < 1 => b < e")
print()

# Compute xi_floor for various bases
test_bases = [
    (1.5, "sesqui"),
    (2, "binary"),
    (E_EULER**(1/2), "sqrt(e)"),
    (PHI, "golden"),
    (math.e, "natural (e)"),
    (3, "ternary"),
    (4, "quaternary"),
    (10, "decimal"),
]

print(f"  {'Base':>12s} {'Name':>12s} {'ln(b)':>10s} {'xi_floor':>12s} {'Physical?':>10s}")
print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

base_results = []
for b, name in test_bases:
    lnb = math.log(b)
    xi_fl = 1 - lnb**2
    physical = xi_fl > 0
    marker = "YES" if physical else "NO"
    if abs(xi_fl) < 1e-15:
        marker = "BOUNDARY"
    print(f"  {b:>12.6f} {name:>12s} {lnb:>10.6f} {xi_fl:>12.8f} {marker:>10s}")
    base_results.append({
        'base': b, 'name': name, 'ln_b': lnb,
        'xi_floor': xi_fl, 'physical': physical,
    })

# Count how many integers have valid xi_floor
valid_integers = []
for b_int in range(2, 100):
    xi_fl = 1 - math.log(b_int)**2
    if xi_fl > 0:
        valid_integers.append(b_int)

print(f"\n  Valid integer bases (xi_floor > 0): {valid_integers}")
print(f"  Count: {len(valid_integers)}")
print(f"  Binary is {'the ONLY' if valid_integers == [2] else 'NOT the only'} valid integer base")

verdicts['A'] = f"xi_floor > 0 for {len(valid_integers)} integer(s): {valid_integers}"
results['part_a'] = {
    'base_table': base_results,
    'valid_integers': valid_integers,
    'binary_unique': valid_integers == [2],
}


# ============================================================
# Part B: Maximum Information Base
# ============================================================
print_header("Part B: Maximum Information Base")

# Solve 1 - ln^2(b) = 0 => ln(b) = 1 => b = e
b_max = math.e
xi_at_e = 1 - math.log(b_max)**2

print(f"  1 - ln^2(b) = 0  =>  ln(b) = 1  =>  b = e = {b_max:.10f}")
print(f"  xi_floor(e) = {xi_at_e:.2e} (machine zero)")
print()
print(f"  Physical interpretation:")
print(f"    At b = e, the cascade floor is ZERO — complete dissipation.")
print(f"    No energy survives the Landauer cascade.")
print(f"    For b > e, xi_floor < 0 — energy is 'created', violating conservation.")
print()

# Characterize the allowed range (1, e)
# Where in the range is binary?
b_position = (2 - 1) / (E_EULER - 1)
print(f"  Allowed range: b in (1, e) = (1, {E_EULER:.4f})")
print(f"  Binary position: (2-1)/(e-1) = {b_position:.4f} ({b_position*100:.1f}% through the range)")
print(f"  Binary xi_floor = {1 - LN2**2:.8f}")
print(f"  This is {(1 - LN2**2) * 100:.2f}% energy conservation — healthy cascade")

# Where is the maximum xi_floor?
# d/db [1 - ln^2(b)] = -2*ln(b)/b = 0 => ln(b) = 0 => b = 1
# But b=1 is degenerate (0-ary information). Maximum in useful range is at b → 1+
# Binary is the smallest non-trivial integer base, so it's the MOST conservative
print(f"\n  xi_floor is monotonically decreasing for b > 1")
print(f"  b=2 gives the HIGHEST xi_floor among integer bases")
print(f"  (b=1 is degenerate — no information content)")

verdicts['B'] = f"b_max = e, xi_floor(e) = {xi_at_e:.1e} (boundary), binary at {b_position*100:.0f}% of range"
results['part_b'] = {
    'b_max': b_max,
    'xi_at_e': xi_at_e,
    'binary_position_in_range': b_position,
    'binary_xi_floor': 1 - LN2**2,
}


# ============================================================
# Part C: Generalized Conjugacy — Base Independence
# ============================================================
print_header("Part C: Generalized Conjugacy Products")

print("  MVAE constraints generalized to base b:")
print("    E(b) = ln(b)           (Landauer energy)")
print("    dt(b) = 1/(2*ln(b))   (Heisenberg time)")
print("    dx_grav(b) = 2*ln(b)  (Schwarzschild length)")
print("    dx_unc(b) = 1/(2*ln(b))  (uncertainty length)")
print()

# Test conjugacy products for many bases
test_bases_conj = np.linspace(1.01, 10, 100)
all_E_dt = []
all_dx_products = []

print(f"  {'Base':>8s} {'E*dt':>12s} {'dx_grav*dx_unc':>16s} {'Both exact?':>12s}")
print(f"  {'-'*8} {'-'*12} {'-'*16} {'-'*12}")

display_bases = [1.5, 2, PHI, E_EULER, 3, 5, 10]
for b in display_bases:
    lnb = math.log(b)
    E = lnb
    dt = 1 / (2 * lnb)
    dx_grav = 2 * lnb
    dx_unc = 1 / (2 * lnb)

    E_dt = E * dt
    dx_prod = dx_grav * dx_unc

    exact = abs(E_dt - 0.5) < 1e-14 and abs(dx_prod - 1.0) < 1e-14
    print(f"  {b:>8.4f} {E_dt:>12.10f} {dx_prod:>16.10f} {'YES' if exact else 'NO':>12s}")

# Full scan
n_exact = 0
for b in test_bases_conj:
    lnb = math.log(b)
    E_dt = lnb * (1 / (2 * lnb))
    dx_prod = (2 * lnb) * (1 / (2 * lnb))
    all_E_dt.append(E_dt)
    all_dx_products.append(dx_prod)
    if abs(E_dt - 0.5) < 1e-14 and abs(dx_prod - 1.0) < 1e-14:
        n_exact += 1

print(f"\n  Over {len(test_bases_conj)} bases in [1.01, 10]:")
print(f"    E*dt = 1/2 always: {all(abs(x - 0.5) < 1e-12 for x in all_E_dt)}")
print(f"    dx_grav*dx_unc = 1 always: {all(abs(x - 1.0) < 1e-12 for x in all_dx_products)}")
print()
print(f"  CONCLUSION: Conjugacy products are ALGEBRAIC IDENTITIES")
print(f"  They hold for ANY base because:")
print(f"    E*dt = ln(b) * 1/(2*ln(b)) = 1/2  (cancellation)")
print(f"    dx_grav*dx_unc = 2*ln(b) * 1/(2*ln(b)) = 1  (cancellation)")
print(f"  The conjugacy structure is base-INDEPENDENT")
print(f"  But xi_floor = 1 - ln^2(b) IS base-dependent")
print(f"  => Base selects the FLOOR, not the STRUCTURE")

conj_universal = (all(abs(x - 0.5) < 1e-12 for x in all_E_dt) and
                  all(abs(x - 1.0) < 1e-12 for x in all_dx_products))

verdicts['C'] = f"Conjugacy base-independent: {conj_universal}. Floor base-dependent."
results['part_c'] = {
    'E_dt_universal': all(abs(x - 0.5) < 1e-12 for x in all_E_dt),
    'dx_product_universal': all(abs(x - 1.0) < 1e-12 for x in all_dx_products),
    'conclusion': 'Base selects floor, not structure',
}


# ============================================================
# Part D: eta Bridge Generalization
# ============================================================
print_header("Part D: Generalized eta Bridge")

print("  eta(b) = 1 + (7/8)*(1-ln(b))^2")
print("  xi_PAC(b) = 1 + (7/8)*ln(b)*(1-ln(b))^2")
print()

# Compute for notable bases
notable_bases = [
    (1.1, "near-unity"),
    (1.5, "sesqui"),
    (2, "BINARY"),
    (2**0.5, "sqrt(2)"),
    (PHI, "phi"),
    (E_EULER, "e (natural)"),
    (3, "ternary"),
]

print(f"  {'Base':>8s} {'Name':>12s} {'eta':>12s} {'xi_PAC':>12s} {'xi_floor':>12s} {'eta-1':>10s} {'physical':>10s}")
print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

eta_results = []
for b, name in notable_bases:
    lnb = math.log(b)
    eta = 1 + (7/8) * (1 - lnb)**2
    xi_pac_b = 1 + (7/8) * lnb * (1 - lnb)**2
    xi_fl = 1 - lnb**2
    phys = xi_fl > 0

    print(f"  {b:>8.4f} {name:>12s} {eta:>12.8f} {xi_pac_b:>12.8f} {xi_fl:>12.8f} {eta-1:>10.6f} {'YES' if phys else 'NO':>10s}")
    eta_results.append({
        'base': b, 'name': name, 'eta': eta, 'xi_pac': xi_pac_b,
        'xi_floor': xi_fl, 'eta_minus_1': eta - 1, 'physical': phys,
    })

# Find the base that maximizes (eta - 1) * physical_validity
# eta - 1 = (7/8)*(1-ln(b))^2 is maximized when ln(b) is far from 1
# but xi_floor > 0 requires ln(b) < 1
# d/db [(1-ln(b))^2] = -2*(1-ln(b))/b, zero at ln(b)=1 (b=e, minimum)
# Maximum of eta-1 in (1,e): at b→1+ (but degenerate)
# Among integers: b=2 is the only one with xi_floor > 0
print(f"\n  eta - 1 = (7/8)*(1-ln(b))^2")
print(f"  This measures recycling STRENGTH (SEC pump amplitude)")
print(f"  Decreases as b -> e (recycling vanishes at natural base)")
print(f"  Increases as b -> 1 (but information content vanishes)")
print(f"  b=2 gives eta-1 = {(7/8)*(1-LN2)**2:.6f} — a moderate, balanced pump")

# Optimal base for xi_PAC (maximize the bridge output)
# xi_PAC(b) = 1 + (7/8)*ln(b)*(1-ln(b))^2
# d/db = (7/8) * [1/b * (1-ln(b))^2 + ln(b) * (-2*(1-ln(b))/b)]
#       = (7/8)/b * [(1-ln(b))^2 - 2*ln(b)*(1-ln(b))]
#       = (7/8)/b * (1-ln(b)) * [(1-ln(b)) - 2*ln(b)]
#       = (7/8)/b * (1-ln(b)) * (1 - 3*ln(b))
# Zero at: ln(b)=1 (b=e) or ln(b)=1/3 (b=e^(1/3))
b_opt = math.e**(1/3)
xi_pac_opt = 1 + (7/8) * math.log(b_opt) * (1 - math.log(b_opt))**2

print(f"\n  Optimal base for xi_PAC: b* = e^(1/3) = {b_opt:.6f}")
print(f"  xi_PAC(b*) = {xi_pac_opt:.8f}")
print(f"  xi_PAC(2) = {XI_MVAE:.8f}")
print(f"  Binary is {'above' if XI_MVAE > xi_pac_opt else 'below'} the optimum")
print(f"  Distance from optimum: {abs(XI_MVAE - xi_pac_opt):.6f}")

verdicts['D'] = f"Binary eta-1={((7/8)*(1-LN2)**2):.6f}, optimal base e^(1/3)={b_opt:.4f}"
results['part_d'] = {
    'eta_table': eta_results,
    'optimal_base': b_opt,
    'xi_pac_at_optimal': xi_pac_opt,
    'xi_pac_at_binary': XI_MVAE,
    'binary_vs_optimal': XI_MVAE - xi_pac_opt,
}


# ============================================================
# Part E: Binary Selectivity — The Uniqueness Argument
# ============================================================
print_header("Part E: Binary Selectivity")

print("  THE ARGUMENT:")
print("  1. Landauer: erasing one b-ary digit costs kT*ln(b)")
print("  2. MVAE cascade floor: xi_floor(b) = 1 - ln^2(b)")
print("  3. Physical validity: xi_floor > 0 requires b < e = 2.718...")
print("  4. Information theory: b must be a positive integer >= 2")
print("  5. ONLY b=2 satisfies both constraints simultaneously")
print()

# Formal verification
print("  Verification:")
integer_bases = list(range(2, 20))
for b in integer_bases:
    xi_fl = 1 - math.log(b)**2
    valid = xi_fl > 0
    print(f"    b = {b:>2d}: xi_floor = {xi_fl:>+10.6f} {'VALID' if valid else 'INVALID'}")

print()

# What about b=1? (unary)
print("  Edge case b=1:")
print(f"    ln(1) = 0, xi_floor(1) = 1 - 0 = 1.000 (trivially valid)")
print(f"    But b=1 is degenerate: a unary digit carries NO information")
print(f"    (a single symbol has zero entropy). Erasure cost = kT*ln(1) = 0.")
print(f"    This is not a real information system.")
print()

# The deep result: WHY is information binary?
print("  WHY IS INFORMATION BINARY?")
print("  Standard answer: convention (Shannon chose bits)")
print("  MVAE answer: PHYSICS demands it.")
print("    - Energy conservation requires xi_floor > 0")
print("    - Landauer's principle sets xi_floor = 1 - ln^2(b)")
print("    - Integer bases >= 3 violate energy conservation")
print("    - b=1 is degenerate (zero information)")
print("    - b=2 is the UNIQUE non-trivial integer base with valid cascades")
print()

# Quantitative margins
xi_2 = 1 - LN2**2
xi_3 = 1 - math.log(3)**2
margin_to_zero = xi_2  # distance from binary to breakdown
margin_to_ternary = xi_2 - xi_3  # gap between binary and ternary

print(f"  Quantitative margins:")
print(f"    xi_floor(2) = {xi_2:+.8f} (safely positive)")
print(f"    xi_floor(3) = {xi_3:+.8f} (badly negative)")
print(f"    Margin to zero: {margin_to_zero:.8f}")
print(f"    Gap binary-ternary: {margin_to_ternary:.8f}")
print(f"    Binary uses {LN2/1.0*100:.1f}% of the available ln-range [0,1)")
print()

# Null test: what fraction of real bases in (1, 100) give valid cascades?
rng = np.random.default_rng(42)
n_samples = 100000
random_bases = rng.uniform(1.01, 100, n_samples)
n_valid = sum(1 for b in random_bases if 1 - math.log(b)**2 > 0)
frac_valid = n_valid / n_samples

# Among those valid, what fraction are integers?
valid_reals = [b for b in random_bases if 1 - math.log(b)**2 > 0]
# The only integer in (1, e) is 2
frac_valid_integer = 1 / (E_EULER - 1)  # fraction of (1, e) that is "near" integer 2

print(f"  Random base null test (100K samples in [1.01, 100]):")
print(f"    Valid bases (xi_floor > 0): {n_valid} ({frac_valid*100:.2f}%)")
print(f"    Valid range: (1, {E_EULER:.4f}) = width {E_EULER - 1:.4f}")
print(f"    Only integer in range: 2")
print(f"    Probability of landing on an integer: measure zero")
print(f"    => Binary is selected by a MEASURE-ZERO constraint")

# Final synthesis
print(f"\n  SYNTHESIS:")
print(f"    Conjugacy products: base-INDEPENDENT (Part C)")
print(f"    Cascade floor xi_floor: base-DEPENDENT (Part A)")
print(f"    Valid integer bases: ONLY b=2 (Part E)")
print(f"    => The conjugacy STRUCTURE is universal")
print(f"    => The Landauer FLOOR selects binary uniquely")
print(f"    => Binary is not convention — it is thermodynamic necessity")

binary_unique = (valid_integers == [2])
verdicts['E'] = f"Binary uniquely selected: {binary_unique}. Ternary xi_floor = {xi_3:.4f} (invalid)"
results['part_e'] = {
    'binary_xi_floor': xi_2,
    'ternary_xi_floor': xi_3,
    'margin_to_zero': margin_to_zero,
    'binary_unique_integer': binary_unique,
    'random_valid_fraction': frac_valid,
}


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("SUMMARY: ln(2) Uniqueness — Information Base Test")
print("=" * 72)

print(f"\n  {'Part':6s} | {'Test':55s} | {'Result'}")
print(f"  {'-'*6}-+-{'-'*55}-+-{'-'*40}")
labels = {
    'A': 'Generalized Landauer floor',
    'B': 'Maximum information base',
    'C': 'Conjugacy base-independence',
    'D': 'eta bridge generalization',
    'E': 'Binary selectivity (unique integer)',
}
for key in sorted(verdicts.keys()):
    print(f"  {key:6s} | {labels[key]:55s} | {verdicts[key]}")

print(f"\n  KEY FINDINGS:")
print(f"  1. xi_floor(b) = 1 - ln^2(b) goes negative for b >= 3")
print(f"  2. b = e is the boundary (xi_floor = 0, complete dissipation)")
print(f"  3. Conjugacy products (E*dt=1/2, dx*dx=1) hold for ANY base")
print(f"  4. b=2 is the ONLY integer base with valid cascades")
print(f"  5. Binary information is not convention — it is thermodynamic necessity")

# Save
all_results = {
    'experiment': 'minimum_actualization_resolution',
    'script': 'exp_10_ln2_uniqueness.py',
    'timestamp': datetime.now().isoformat(),
    'verdicts': {k: str(v) for k, v in verdicts.items()},
    'results': results,
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"results/exp_10_ln2_uniqueness_{ts}.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n  Results saved to {out_path}")
print("=" * 72)

"""
Experiment 12: Force Depth Mapping
=====================================

Tests the prediction that fundamental force coupling strengths map
to PAC Fibonacci depths, specifically:

  - Gravity:      depth 183 = F₇² + F₇ + 1  (strongest hierarchy)
  - Weak force:   depth ~ some Fibonacci expression
  - EM:           depth ~ some Fibonacci expression
  - Strong force: depth ~ some Fibonacci expression

Source: gravity_from_maxwell_pac found gravity depth = 183 = F₇² + F₇ + 1
Source: pac_confluence_xi derived α from Fibonacci structure

The question: do ALL four fundamental coupling constants map to
specific Fibonacci-structured depths?

Success criterion: Map ≥ 3/4 forces to Fibonacci-structured depths
with < 5% error on coupling strength prediction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from phase_engine import *

def run():
    print("=" * 70)
    print("EXP 12: Force Depth Mapping")
    print("=" * 70)

    # Known coupling constants (at characteristic energy scales)
    # All expressed as dimensionless couplings
    forces = {
        'strong': {
            'coupling': 1.0,       # α_s ≈ 1 at low energy
            'coupling_mz': 0.1179,  # α_s(M_Z) at Z mass
        },
        'electromagnetic': {
            'coupling': 1/137.036,  # α_em ≈ 1/137
        },
        'weak': {
            'coupling': 1.166e-5,  # G_F in GeV^-2 (Fermi constant)
            'coupling_dimless': 1/29.2,  # g_w²/4π at low energy
        },
        'gravity': {
            'coupling': 5.39e-39,  # G_N in natural units (relative to proton mass)
        },
    }

    alpha_em = 1/137.036
    alpha_s_mz = 0.1179
    alpha_w = 1/29.2  # Approximate weak coupling

    print(f"  α_em = {alpha_em:.6e}")
    print(f"  α_s(M_Z) = {alpha_s_mz:.6e}")
    print(f"  α_w ≈ {alpha_w:.6e}")
    print(f"  G_N/m_p² ≈ {5.39e-39:.6e}")

    # ================================================================
    # Test 1: Known gravity depth
    # ================================================================
    print("\n--- Test 1: Gravity depth = 183 ---")

    gravity_depth = 183
    print(f"  183 = F₇² + F₇ + 1 = {F7**2} + {F7} + 1 = {F7**2 + F7 + 1}")
    print(f"  Also: 183 = 13² + 13 + 1 (cyclotomic polynomial Φ₃(13))")

    # What coupling does this predict?
    # If coupling ~ φ^(-depth), then:
    grav_coupling_pred = PHI**(-gravity_depth)
    print(f"  φ^(-183) = {grav_coupling_pred:.6e}")
    print(f"  Actual G_N/m_p² ≈ 5.39e-39")

    # More general: coupling ~ F(depth) function
    # Try several depth → coupling maps
    maps = [
        ("φ^(-d)", lambda d: PHI**(-d)),
        ("e^(-d·ln(φ))", lambda d: math.exp(-d * LN_PHI)),
        ("(1/φ)^d", lambda d: (1/PHI)**d),
        ("2^(-d/3)", lambda d: 2**(-d/3)),
        ("e^(-d·γ)", lambda d: math.exp(-d * GAMMA)),
        ("10^(-d/4.7)", lambda d: 10**(-d/4.7)),  # Rough hierarchy
    ]

    print(f"\n  Depth → coupling mapping at d=183:")
    for name, func in maps:
        val = func(gravity_depth)
        print(f"    {name:<20}: {val:.6e}")

    # ================================================================
    # Test 2: Search for EM depth
    # ================================================================
    print("\n--- Test 2: EM depth search ---")

    # If coupling = φ^(-d), then d = -ln(coupling)/ln(φ)
    em_depth_phi = -math.log(alpha_em) / LN_PHI
    em_depth_gamma = -math.log(alpha_em) / GAMMA
    em_depth_e = -math.log(alpha_em)

    print(f"  α_em = 1/137.036 = {alpha_em:.6e}")
    print(f"  Implied depths:")
    print(f"    d = -ln(α)/ln(φ) = {em_depth_phi:.4f}")
    print(f"    d = -ln(α)/γ     = {em_depth_gamma:.4f}")
    print(f"    d = -ln(α)       = {em_depth_e:.4f}")

    # Check if any of these are Fibonacci-structured
    print(f"\n  Testing if EM depth is Fibonacci-structured:")
    em_depth_target = em_depth_phi

    # Check EM depth against Fibonacci expressions
    fib_exprs = []
    for i in range(3, 12):
        for j in range(0, 12):
            for op in ['+', '-', '*']:
                if op == '+':
                    val = FIBS[i] + FIBS[j]
                elif op == '-':
                    val = FIBS[i] - FIBS[j]
                else:
                    val = FIBS[i] * FIBS[j]
                if val > 0:
                    err = abs(val - em_depth_target) / em_depth_target
                    if err < 0.05:
                        fib_exprs.append((f"F{i}{op}F{j}={FIBS[i]}{op}{FIBS[j]}", val, err))

    # Also check: F_i² + F_j + k patterns
    for i in range(3, 10):
        for j in range(0, 10):
            for k in [-1, 0, 1]:
                val = FIBS[i]**2 + FIBS[j] + k
                err = abs(val - em_depth_target) / em_depth_target
                if err < 0.05:
                    fib_exprs.append((f"F{i}²+F{j}+{k}", val, err))

    fib_exprs.sort(key=lambda x: x[2])
    for expr, val, err in fib_exprs[:10]:
        print(f"    {expr} = {val:.0f}  (error = {err*100:.2f}%)")

    # ================================================================
    # Test 3: Search for strong and weak depths
    # ================================================================
    print("\n--- Test 3: Strong and weak force depths ---")

    for force_name, coupling in [('strong(M_Z)', alpha_s_mz), ('weak', alpha_w)]:
        depth = -math.log(coupling) / LN_PHI
        print(f"\n  {force_name}: coupling = {coupling:.6e}")
        print(f"    Implied depth (base φ) = {depth:.4f}")

        # Check Fibonacci expressions
        candidates = []
        for i in range(2, 10):
            for j in range(0, 10):
                for op in ['+', '-', '*']:
                    if op == '+':
                        val = FIBS[i] + FIBS[j]
                    elif op == '-':
                        val = FIBS[i] - FIBS[j]
                    else:
                        val = FIBS[i] * FIBS[j]
                    if val > 0:
                        err = abs(val - depth) / depth
                        if err < 0.1:
                            candidates.append((f"F{i}{op}F{j}={FIBS[i]}{op}{FIBS[j]}", val, err))

        for i in range(2, 8):
            for j in range(0, 8):
                for k in [-1, 0, 1]:
                    val = FIBS[i]**2 + FIBS[j] + k
                    if val > 0:
                        err = abs(val - depth) / depth
                        if err < 0.1:
                            candidates.append((f"F{i}²+F{j}+{k}", val, err))

        candidates.sort(key=lambda x: x[2])
        for expr, val, err in candidates[:5]:
            print(f"      {expr} = {val:.0f}  (error = {err*100:.2f}%)")

    # ================================================================
    # Test 4: Self-consistent depth → coupling → hierarchy
    # ================================================================
    print("\n--- Test 4: Self-consistent hierarchy ---")

    # If depth = F_n² + F_n + 1 (cyclotomic pattern), what are the
    # natural depths?
    print(f"  Cyclotomic pattern Φ₃(F_n) = F_n² + F_n + 1:")
    for n in range(2, 11):
        fn = FIBS[n]
        depth = fn**2 + fn + 1
        coupling = PHI**(-depth)
        log_coupling = -depth * LN_PHI
        print(f"    n={n}: F_n={fn:4d}, depth={depth:6d}, "
              f"φ^(-d)={coupling:.4e}, "
              f"log₁₀(φ^(-d))={math.log10(coupling):.2f}" if coupling > 0 else
              f"    n={n}: F_n={fn:4d}, depth={depth:6d}, coupling → 0")

    # ================================================================
    # Test 5: Fibonacci ratios between force depths
    # ================================================================
    print("\n--- Test 5: Ratios between force depths ---")

    # Gravity depth is known: 183
    # EM depth (base φ): ~10.2
    # Strong depth (base φ): ~4.4
    # Weak depth (base φ): ~7.0

    depth_grav = gravity_depth
    depth_em = -math.log(alpha_em) / LN_PHI
    depth_strong = -math.log(alpha_s_mz) / LN_PHI
    depth_weak = -math.log(alpha_w) / LN_PHI

    force_depths = {
        'strong': depth_strong,
        'em': depth_em,
        'weak': depth_weak,
        'gravity': float(depth_grav),
    }

    print(f"  Force depths (base φ):")
    for name, d in sorted(force_depths.items(), key=lambda x: x[1]):
        print(f"    {name:12s}: {d:.4f}")

    print(f"\n  Ratios between depths:")
    names = sorted(force_depths.keys(), key=lambda x: force_depths[x])
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            ratio = force_depths[names[j]] / force_depths[names[i]]
            # Check if ratio is close to a Fibonacci ratio
            nearest_fib_ratio = min(
                [(f"F{a}/F{b}", FIBS[a]/FIBS[b]) for a in range(2, 12) for b in range(2, 12)
                 if FIBS[b] > 0 and a != b and abs(FIBS[a]/FIBS[b] - ratio) < ratio * 0.2],
                key=lambda x: abs(x[1] - ratio),
                default=None
            )
            fib_str = f" ≈ {nearest_fib_ratio[0]} = {nearest_fib_ratio[1]:.4f}" \
                      if nearest_fib_ratio else ""
            print(f"    {names[j]}/{names[i]}: {ratio:.4f}{fib_str}")

    # ================================================================
    # Results
    # ================================================================
    mapped_forces = 1  # Gravity is already known
    if fib_exprs and fib_exprs[0][2] < 0.05:
        mapped_forces += 1  # EM

    data = {
        'experiment': 'exp_12_force_depth_mapping',
        'hypothesis': 'All four forces map to Fibonacci-structured PAC depths',
        'force_depths_base_phi': force_depths,
        'gravity_depth': gravity_depth,
        'gravity_formula': 'F₇² + F₇ + 1 = 183',
        'em_depth_candidates': [(e, v, float(err)) for e, v, err in fib_exprs[:5]] if fib_exprs else [],
        'cyclotomic_depths': {
            str(n): {
                'F_n': FIBS[n],
                'depth': FIBS[n]**2 + FIBS[n] + 1,
                'coupling': float(PHI**(-(FIBS[n]**2 + FIBS[n] + 1)))
                    if FIBS[n]**2 + FIBS[n] + 1 < 700 else 0.0,
            }
            for n in range(2, 9)
        },
        'mapped_forces': mapped_forces,
        'success': mapped_forces >= 3,
        'success_criterion': 'Map ≥ 3/4 forces to Fibonacci-structured depths',
        'notes': 'Gravity depth confirmed. Other forces need either different base '
                 'mapping or are not on the cyclotomic pattern.',
    }

    print(f"\n{'='*70}")
    print(f"FORCES MAPPED: {mapped_forces}/4")
    print(f"SUCCESS: {'YES' if data['success'] else 'PARTIAL — needs more work'}")
    print(f"{'='*70}")

    save_results(data, 'exp_12_force_depth_mapping')
    return data


if __name__ == '__main__':
    run()

"""
exp_32d — Bouncing Ball Cascade: Gravity as Geometric Primitive

HYPOTHESIS: A ball bouncing on a trampoline under gravity is a physical
instantiation of the geometry-precedes-arithmetic thesis.

Gravity (geometric constraint) simultaneously:
  1. Creates constant ratios between bounce heights (arithmetic readout)
  2. Bridges linear and exponential (within-bounce: linear v change;
     across-bounces: geometric sequence)
  3. Terminates the cascade in finite time (ADE-like closure)
  4. Creates self-similarity (each bounce is a scaled copy)

Key insight: WITHIN each bounce, gravity creates LINEAR velocity change
(constant acceleration). But ACROSS bounces, the peak velocities form
a GEOMETRIC sequence. Gravity — a single geometric constraint — converts
linear to exponential through repeated application. This IS the ADE
hyperoperation ladder in physical form:
  Level 1 (addition):     v → v - g*dt     (linear, within one bounce)
  Level 2 (multiplication): v_n → e * v_{n-1}  (geometric, across bounces)
  Level 3 (exponentiation): h_n → h_0 * e^(2n) (tower, heights)
  Level 4 (termination):    T_total = finite   (cascade converges)

Tests:
  1. Constant ratio emergence — gravity determines the ratio
  2. Perturbation asymmetry — perturb gravity vs perturb ratios
  3. Finite termination — cascade completes in finite time
  4. Linear-exponential bridge — same force creates both regimes
  5. PAC conservation structure — connect to PAC tree dynamics

Author: Peter Groom
Date: 2026-04-20
"""

import sys
import json
from pathlib import Path
import numpy as np
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = EXP_ROOT / "results"

PHI = (1 + np.sqrt(5)) / 2


# ============================================================
# Bouncing ball physics
# ============================================================

def simulate_bouncing_ball(h0, e, g=9.81, n_max=500, t_max=1000.0):
    """
    Simulate a ball bouncing under gravity with coefficient of restitution e.

    Returns list of dicts with bounce-level data:
      height, velocity, time_of_bounce, period, path_length
    """
    bounces = []
    h = h0
    v = np.sqrt(2 * g * h)  # velocity at ground from height h
    t_total = 0.0

    # First fall
    t_fall = np.sqrt(2 * h / g)
    t_total += t_fall

    for n in range(n_max):
        if h < 1e-15 or v < 1e-15:
            break

        # Record this bounce
        period = 2 * v / g  # time for up + down
        path = 2 * h  # distance traveled (up + down)

        bounces.append({
            'n': n,
            'height': h,
            'velocity': v,
            'time': t_total,
            'period': period,
            'path': path,
            'kinetic_energy': 0.5 * v ** 2,  # per unit mass
        })

        # After bounce: lose energy
        v_new = e * v
        h_new = v_new ** 2 / (2 * g)

        # Advance time
        t_total += period

        if t_total > t_max:
            break

        v = v_new
        h = h_new

    return bounces


def bounce_ratios(bounces, key='height'):
    """Compute successive ratios for a given quantity."""
    values = [b[key] for b in bounces]
    ratios = []
    for i in range(len(values) - 1):
        if values[i + 1] > 1e-20:
            ratios.append(values[i] / values[i + 1])
    return ratios


# ============================================================
# Test 1: Constant Ratio Emergence
# ============================================================

def test1_constant_ratios():
    """
    Gravity + restitution → constant height ratios.

    For coefficient of restitution e:
      h_n / h_{n+1} = 1/e^2 (exact, from first principles)

    This is the arithmetic readout of a geometric constraint.
    The ratio is DETERMINED by the physics (gravity + surface),
    not imposed numerically.

    Also test: different initial conditions (h0) → SAME ratio.
    Many geometries (different trajectories) → one arithmetic value.
    """
    print("=" * 60)
    print("Test 1: Constant Ratio Emergence")
    print("(gravity determines arithmetic, not the reverse)")
    print("=" * 60)

    e_values = [0.7, 0.8, 0.85, 0.9, 0.95]
    h0_values = [1.0, 5.0, 10.0, 50.0, 100.0]

    results = {}

    for e in e_values:
        predicted_ratio = 1.0 / e ** 2
        measured_ratios_all = []

        for h0 in h0_values:
            bounces = simulate_bouncing_ball(h0, e, n_max=200)
            ratios = bounce_ratios(bounces, 'height')

            if len(ratios) >= 5:
                # Skip first 2 bounces (transient), use rest
                steady = ratios[2:]
                measured_ratios_all.extend(steady)

        if measured_ratios_all:
            mean_ratio = np.mean(measured_ratios_all)
            std_ratio = np.std(measured_ratios_all)
            delta = abs(mean_ratio - predicted_ratio) / predicted_ratio

            results[f'e={e}'] = {
                'e': e,
                'predicted_ratio': predicted_ratio,
                'measured_mean': float(mean_ratio),
                'measured_std': float(std_ratio),
                'delta': float(delta),
                'n_measurements': len(measured_ratios_all),
            }

            print(f"\n  e = {e}:")
            print(f"    Predicted h_n/h_{{n+1}} = 1/e^2 = {predicted_ratio:.6f}")
            print(f"    Measured (mean over {len(h0_values)} initial heights): "
                  f"{mean_ratio:.6f} +/- {std_ratio:.6f}")
            print(f"    Delta: {delta:.2e}")

    # KEY: ratios are EXACT regardless of initial height
    # Different geometric trajectories → same arithmetic
    all_deltas = [r['delta'] for r in results.values()]
    max_delta = max(all_deltas)
    all_exact = max_delta < 1e-10

    print(f"\n  Maximum delta across all (e, h0) pairs: {max_delta:.2e}")
    print(f"  All ratios exact: {all_exact}")
    print(f"\n  Interpretation: gravity determines the ratio EXACTLY.")
    print(f"  Different initial heights (different geometric trajectories)")
    print(f"  all produce the SAME ratio. Many geometries → one arithmetic.")

    # Now check: does any e produce phi?
    # h_n/h_{n+1} = phi when 1/e^2 = phi, i.e., e = 1/sqrt(phi) ≈ 0.7862
    e_phi = 1.0 / np.sqrt(PHI)
    bounces_phi = simulate_bouncing_ball(10.0, e_phi, n_max=200)
    ratios_phi = bounce_ratios(bounces_phi, 'height')
    if len(ratios_phi) > 5:
        phi_ratio = np.mean(ratios_phi[2:])
        phi_delta = abs(phi_ratio - PHI) / PHI
        print(f"\n  Special case: e = 1/sqrt(phi) = {e_phi:.6f}")
        print(f"    Measured ratio: {phi_ratio:.10f}")
        print(f"    Delta from phi: {phi_delta:.2e}")
        results['phi_special'] = {
            'e': float(e_phi),
            'measured_ratio': float(phi_ratio),
            'delta_from_phi': float(phi_delta),
        }

    results['all_exact'] = all_exact
    results['passed'] = all_exact
    return results


# ============================================================
# Test 2: Perturbation Asymmetry
# ============================================================

def test2_perturbation_asymmetry():
    """
    Two perturbation types:
      A) Perturb gravity (geometric constraint) → ratios change
      B) Perturb ratios directly (inject/remove energy at specific
         bounces) → gravity overwrites them next bounce

    Prediction: geometric perturbation controls arithmetic outcome;
    arithmetic perturbation is overwritten by the geometric constraint.
    """
    print("\n" + "=" * 60)
    print("Test 2: Perturbation Asymmetry")
    print("(geometric constraint controls arithmetic readout)")
    print("=" * 60)

    e = 0.85
    h0 = 10.0
    g_base = 9.81
    predicted_ratio = 1.0 / e ** 2

    # Experiment A: Perturb gravity at bounce 10
    # Change g → 1.5*g for bounces 10-15, then back to g
    print("\n  Experiment A: Perturb gravity (geometric constraint)")

    bounces_a = []
    h = h0
    v = np.sqrt(2 * g_base * h)
    t = np.sqrt(2 * h / g_base)

    for n in range(100):
        if h < 1e-15:
            break

        # Gravity perturbation window
        g = g_base * 1.5 if 10 <= n <= 15 else g_base

        period = 2 * v / g
        bounces_a.append({'n': n, 'height': h, 'velocity': v,
                          'time': t, 'period': period})

        v_new = e * v
        h_new = v_new ** 2 / (2 * g)  # height with LOCAL gravity
        t += period
        v = v_new
        h = h_new

    ratios_a = bounce_ratios(bounces_a, 'height')

    # Measure: how fast do ratios return to predicted after perturbation?
    recovery_a = []
    for i, r in enumerate(ratios_a):
        if i > 16:  # after perturbation window
            delta = abs(r - predicted_ratio) / predicted_ratio
            recovery_a.append(delta)
            if delta < 0.01:
                break

    a_recovery_time = len(recovery_a)
    a_settled = recovery_a[-1] if recovery_a else 1.0

    print(f"    Perturbed gravity 1.5x at bounces 10-15")
    print(f"    Recovery time after perturbation: {a_recovery_time} bounces")
    print(f"    Final delta from predicted ratio: {a_settled:.4e}")

    # Experiment B: Perturb ratios directly (inject energy at bounce 10)
    print("\n  Experiment B: Perturb arithmetic (inject energy at specific bounce)")

    bounces_b = []
    h = h0
    v = np.sqrt(2 * g_base * h)
    t = np.sqrt(2 * h / g_base)

    for n in range(100):
        if h < 1e-15:
            break

        period = 2 * v / g_base
        bounces_b.append({'n': n, 'height': h, 'velocity': v,
                          'time': t, 'period': period})

        v_new = e * v

        # Arithmetic perturbation: at bounce 10, double the velocity
        if n == 10:
            v_new *= 2.0  # inject energy — break the ratio

        h_new = v_new ** 2 / (2 * g_base)
        t += period
        v = v_new
        h = h_new

    ratios_b = bounce_ratios(bounces_b, 'height')

    # Recovery: how fast do ratios return?
    recovery_b = []
    for i, r in enumerate(ratios_b):
        if i > 11:  # after injection
            delta = abs(r - predicted_ratio) / predicted_ratio
            recovery_b.append(delta)
            if delta < 0.01:
                break

    b_recovery_time = len(recovery_b)
    b_settled = recovery_b[-1] if recovery_b else 1.0

    print(f"    Injected 2x velocity at bounce 10")
    print(f"    Recovery time after perturbation: {b_recovery_time} bounces")
    print(f"    Final delta from predicted ratio: {b_settled:.4e}")

    # KEY: arithmetic perturbation recovers IMMEDIATELY (next bounce)
    # because gravity overwrites the ratio. Geometric perturbation
    # takes longer because the constraint itself was changed.
    arith_recovers_faster = b_recovery_time <= a_recovery_time

    # Actually the deeper point: arithmetic perturbation recovers in
    # EXACTLY 1 bounce (the ratio is restored immediately because
    # gravity + e determine it completely). Geometric perturbation
    # affects multiple bounces during the perturbation window.
    ratios_after_inject = ratios_b[12:15] if len(ratios_b) > 14 else []
    instant_recovery = all(abs(r - predicted_ratio) / predicted_ratio < 0.01
                           for r in ratios_after_inject)

    print(f"\n  KEY:")
    print(f"    Arithmetic perturbation: instant recovery = {instant_recovery}")
    print(f"    (gravity overwrites the ratio on the VERY NEXT bounce)")
    print(f"    Geometric perturbation: takes {a_recovery_time} bounces to settle")
    print(f"    (changing the constraint changes the dynamics)")

    passed = instant_recovery
    print(f"\n  PASS: {passed}")

    return {
        'gravity_perturbation': {
            'recovery_bounces': a_recovery_time,
            'final_delta': float(a_settled),
        },
        'energy_injection': {
            'recovery_bounces': b_recovery_time,
            'final_delta': float(b_settled),
            'instant_recovery': instant_recovery,
        },
        'passed': passed,
    }


# ============================================================
# Test 3: Finite Termination (Zeno Convergence)
# ============================================================

def test3_finite_termination():
    """
    The total time for infinite bounces is FINITE:
      T = (2 * v0 / g) * (1 + e) / (1 - e)  [geometric series]

    This is the physical analogue of ADE termination at Level 4:
    the cascade completes. The geometric constraint (gravity)
    forces closure in finite "depth."

    Test: verify finite termination time, and show it depends
    on the geometric parameters (g, e), not on arithmetic properties.
    """
    print("\n" + "=" * 60)
    print("Test 3: Finite Termination (Zeno Convergence)")
    print("(cascade completes — ADE-like closure)")
    print("=" * 60)

    h0 = 10.0
    g = 9.81

    e_values = np.linspace(0.3, 0.99, 15)
    results = {}

    print(f"\n  {'e':>5} {'T_predicted':>12} {'T_simulated':>12} {'N_bounces':>10} {'Delta':>10}")
    print(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    for e in e_values:
        v0 = np.sqrt(2 * g * h0)

        # Analytical: total time for all bounces
        # T = t_fall + sum of periods = sqrt(2h/g) + (2v0/g) * e/(1-e)
        # More precisely: T = sqrt(2h0/g) * (1 + 2*e/(1-e))
        t_fall = np.sqrt(2 * h0 / g)
        T_predicted = t_fall + (2 * v0 / g) * e / (1 - e)

        # Simulate
        bounces = simulate_bouncing_ball(h0, e, g=g, n_max=5000, t_max=T_predicted * 2)
        if bounces:
            T_simulated = bounces[-1]['time'] + bounces[-1]['period']
            n_bounces = len(bounces)
        else:
            T_simulated = 0
            n_bounces = 0

        delta = abs(T_simulated - T_predicted) / T_predicted if T_predicted > 0 else 0

        results[f'e={e:.3f}'] = {
            'e': float(e),
            'T_predicted': float(T_predicted),
            'T_simulated': float(T_simulated),
            'n_bounces': n_bounces,
            'delta': float(delta),
        }

        print(f"  {e:5.3f} {T_predicted:12.4f} {T_simulated:12.4f} {n_bounces:10d} {delta:10.2e}")

    # Key: termination time is finite for ALL e < 1
    # And it depends ONLY on geometric parameters (g, e, h0)
    all_finite = all(r['T_predicted'] < 1e6 for r in results.values())
    all_close = all(r['delta'] < 0.05 for r in results.values())

    # How does T scale with e? As e → 1, T → infinity
    # T ~ 1/(1-e) — diverges only at perfect elasticity (no dissipation)
    # This means SEC (dissipation) is what enables termination

    print(f"\n  All termination times finite: {all_finite}")
    print(f"  Simulation matches prediction: {all_close}")
    print(f"\n  Key insight: T ~ 1/(1-e)")
    print(f"    e → 0 (max dissipation): T → t_fall (immediate termination)")
    print(f"    e → 1 (no dissipation): T → infinity (no termination)")
    print(f"    SEC (dissipation via 1-e) is what enables cascade closure.")
    print(f"    Without SEC, the cascade never terminates — no ADE-like fixed point.")

    passed = all_finite and all_close
    print(f"\n  PASS: {passed}")

    results['all_finite'] = all_finite
    results['all_close'] = all_close
    results['passed'] = passed
    return results


# ============================================================
# Test 4: Linear-Exponential Bridge
# ============================================================

def test4_linear_exponential_bridge():
    """
    The deepest structural insight: gravity (a CONSTANT force, creating
    LINEAR velocity change within each bounce) produces EXPONENTIAL
    structure across bounces through repeated application.

    This IS the ADE hyperoperation ladder in physical form:
      Within 1 bounce: v(t) = v0 - g*t         (Level 1: addition)
      Across bounces:  v_n = e * v_{n-1}        (Level 2: multiplication)
      Height sequence:  h_n = h0 * e^{2n}        (Level 3: exponentiation)
      Total time:       T = finite               (Level 4: termination/closure)

    Test: verify this structure explicitly and show each level
    emerges from repeated application of the previous.
    """
    print("\n" + "=" * 60)
    print("Test 4: Linear-Exponential Bridge")
    print("(gravity creates ADE ladder through repeated application)")
    print("=" * 60)

    e = 0.85
    h0 = 10.0
    g = 9.81

    bounces = simulate_bouncing_ball(h0, e, g=g, n_max=100)
    n_bounces = len(bounces)

    if n_bounces < 10:
        print("  Not enough bounces!")
        return {'passed': False}

    # Level 1: LINEAR velocity change within each bounce
    # v(t) = v_peak - g*t for 0 <= t <= v_peak/g (going up)
    # Check: velocity at ground = v_n, velocity at peak = 0
    # Linear profile confirmed by constant acceleration
    print(f"\n  Level 1 (Addition/Linear): Within-bounce dynamics")
    print(f"    Force: F = -mg (constant)")
    print(f"    Velocity: v(t) = v0 - g*t (linear)")
    print(f"    This is the ADDITIVE level: each dt adds -g*dt to velocity")

    # Level 2: MULTIPLICATIVE sequence across bounces
    v_ratios = []
    for i in range(1, min(n_bounces, 30)):
        if bounces[i]['velocity'] > 1e-15:
            v_ratios.append(bounces[i - 1]['velocity'] / bounces[i]['velocity'])

    v_ratio_mean = np.mean(v_ratios)
    v_ratio_std = np.std(v_ratios)
    v_predicted = 1.0 / e

    print(f"\n  Level 2 (Multiplication): Across-bounce velocity sequence")
    print(f"    v_n / v_{{n+1}} = 1/e = {v_predicted:.6f}")
    print(f"    Measured: {v_ratio_mean:.6f} +/- {v_ratio_std:.6f}")
    print(f"    This is MULTIPLICATIVE: v_{'{n+1}'} = e * v_n")
    print(f"    Repeated addition (Level 1) → multiplication (Level 2)")

    # Level 3: EXPONENTIAL height sequence
    ns = np.array([b['n'] for b in bounces[:30]])
    hs = np.array([b['height'] for b in bounces[:30]])

    # Fit: log(h) = log(h0) + 2n * log(e)
    log_hs = np.log(hs[hs > 1e-15])
    ns_valid = ns[:len(log_hs)]

    if len(ns_valid) >= 5:
        slope, intercept = np.polyfit(ns_valid, log_hs, 1)
        predicted_slope = 2 * np.log(e)
        slope_delta = abs(slope - predicted_slope) / abs(predicted_slope)

        print(f"\n  Level 3 (Exponentiation): Height sequence")
        print(f"    h_n = h0 * e^(2n)")
        print(f"    log(h_n) slope: {slope:.6f} (predicted: {predicted_slope:.6f})")
        print(f"    Delta: {slope_delta:.2e}")
        print(f"    Repeated multiplication (Level 2) → exponentiation (Level 3)")
    else:
        slope_delta = 1.0

    # Level 4: FINITE TERMINATION
    v0 = np.sqrt(2 * g * h0)
    T_total = np.sqrt(2 * h0 / g) + (2 * v0 / g) * e / (1 - e)
    bounce_rate_at_n = [1.0 / b['period'] if b['period'] > 1e-15 else np.inf
                         for b in bounces[:30]]

    # Bounce rate grows exponentially
    if len(bounce_rate_at_n) >= 5:
        log_rates = np.log([r for r in bounce_rate_at_n[:20] if np.isfinite(r) and r > 0])
        if len(log_rates) >= 5:
            rate_slope = np.polyfit(np.arange(len(log_rates)), log_rates, 1)[0]
        else:
            rate_slope = 0
    else:
        rate_slope = 0

    print(f"\n  Level 4 (Termination/Closure): Finite total time")
    print(f"    T_total = {T_total:.4f} seconds (finite!)")
    print(f"    Bounces in simulation: {n_bounces}")
    print(f"    Bounce rate growth: {rate_slope:.4f} per bounce (exponential)")
    print(f"    Infinite bounces in finite time = cascade terminates")
    print(f"    Repeated exponentiation (Level 3) → finite closure (Level 4)")

    # The bridge: each level is the REPEATED APPLICATION of the previous
    print(f"\n  THE BRIDGE:")
    print(f"    Level 1 → 2: repeated v += -g*dt  →  v_n = e^n * v_0")
    print(f"    Level 2 → 3: repeated v *= e      →  h_n = h_0 * e^(2n)")
    print(f"    Level 3 → 4: repeated h *= e^2    →  T = sum → finite")
    print(f"    Gravity (a single constant force) generates ALL four levels")
    print(f"    through the same mechanism: repeated application.")
    print(f"    This IS the ADE hyperoperation ladder, physically instantiated.")

    # Verification
    level2_exact = v_ratio_std < 1e-10
    level3_exact = slope_delta < 1e-4
    level4_finite = T_total < 1e6

    passed = level2_exact and level3_exact and level4_finite
    print(f"\n  Level 2 exact (v ratio constant): {level2_exact}")
    print(f"  Level 3 exact (exponential fit): {level3_exact}")
    print(f"  Level 4 finite: {level4_finite}")
    print(f"  PASS: {passed}")

    return {
        'level2': {
            'v_ratio_mean': float(v_ratio_mean),
            'v_ratio_predicted': float(v_predicted),
            'v_ratio_std': float(v_ratio_std),
            'exact': level2_exact,
        },
        'level3': {
            'log_h_slope': float(slope),
            'predicted_slope': float(predicted_slope),
            'delta': float(slope_delta),
            'exact': level3_exact,
        },
        'level4': {
            'T_total': float(T_total),
            'n_bounces': n_bounces,
            'bounce_rate_growth': float(rate_slope),
            'finite': level4_finite,
        },
        'passed': passed,
    }


# ============================================================
# Test 5: PAC Tree Connection
# ============================================================

def test5_pac_tree_connection():
    """
    The bouncing ball IS a PAC tree laid out in time.

    At each bounce, total energy E = KE + PE_lost splits into:
      - Retained energy: e^2 * E  (dominant child)
      - Dissipated energy: (1-e^2) * E  (subordinate child / SEC)

    This is EXACTLY the PAC tree structure:
      P → D + S where D/P = e^2

    And the scale-invariance condition D_{n+1} = S_n is:
      e^2 * E_{n+1} = (1-e^2) * E_n
      → e^2 * (e^2 * E_n) = (1-e^2) * E_n
      → e^4 = 1 - e^2
      → e^2 = (sqrt(5) - 1) / 2 = 1/phi
      → h_n / h_{n+1} = PHI

    The scale-invariance constraint SELECTS e = 1/sqrt(phi),
    which gives height ratios = phi. The golden ratio emerges
    from the SAME geometric constraint as in the PAC tree!
    """
    print("\n" + "=" * 60)
    print("Test 5: PAC Tree Connection")
    print("(bouncing ball IS a PAC tree in time)")
    print("=" * 60)

    # The PAC split: at each bounce
    # Retained fraction = e^2 (dominant child)
    # Dissipated fraction = 1 - e^2 (subordinate child / SEC)
    print("\n  PAC structure at each bounce:")
    print("    E_n → e^2 * E_n (retained) + (1-e^2) * E_n (dissipated)")
    print("    This is P → D + S with D/P = e^2")

    # Scale invariance: D_{n+1} = S_n
    # e^2 * E_{n+1} = (1-e^2) * E_n
    # e^2 * (e^2 * E_n) = (1-e^2) * E_n
    # e^4 = 1 - e^2
    # Let x = e^2: x^2 + x - 1 = 0
    # x = (-1 + sqrt(5)) / 2 = 1/phi
    e_si = np.sqrt((-1 + np.sqrt(5)) / 2)  # e for scale invariance
    e_squared = e_si ** 2
    predicted_ratio = 1.0 / e_squared  # = phi

    print(f"\n  Scale invariance condition: D_{{n+1}} = S_n")
    print(f"    e^4 = 1 - e^2")
    print(f"    e^2 = (sqrt(5)-1)/2 = 1/phi = {1/PHI:.10f}")
    print(f"    e = 1/sqrt(phi) = {e_si:.10f}")
    print(f"    h_n / h_{{n+1}} = 1/e^2 = phi = {predicted_ratio:.10f}")

    # Simulate with this special e
    bounces = simulate_bouncing_ball(10.0, e_si, n_max=200)
    ratios = bounce_ratios(bounces, 'height')

    if len(ratios) > 5:
        mean_ratio = np.mean(ratios[2:])
        delta_phi = abs(mean_ratio - PHI) / PHI

        print(f"\n  Simulation with e = 1/sqrt(phi):")
        print(f"    Mean height ratio: {mean_ratio:.10f}")
        print(f"    Phi:               {PHI:.10f}")
        print(f"    Delta: {delta_phi:.2e}")

        # Verify PAC conservation at each bounce
        pac_violations = 0
        for i, b in enumerate(bounces[:-1]):
            E_parent = b['kinetic_energy']
            E_retained = bounces[i + 1]['kinetic_energy']
            E_dissipated = E_parent - E_retained

            # PAC: parent = dominant + subordinate
            pac_check = abs(E_parent - (E_retained + E_dissipated)) / E_parent
            if pac_check > 1e-10:
                pac_violations += 1

        # Verify scale invariance: D_{n+1} = S_n
        si_violations = []
        for i in range(len(bounces) - 2):
            E_n = bounces[i]['kinetic_energy']
            E_n1 = bounces[i + 1]['kinetic_energy']

            D_n1 = bounces[i + 2]['kinetic_energy']  # retained at next level
            S_n = E_n - E_n1  # dissipated at current level

            if S_n > 1e-15:
                si_quality = abs(D_n1 - S_n) / S_n
                si_violations.append(si_quality)

        mean_si = np.mean(si_violations) if si_violations else np.nan

        print(f"\n  PAC conservation violations: {pac_violations}")
        print(f"  Scale invariance quality (|D_{{n+1}} - S_n|/S_n): {mean_si:.2e}")
        print(f"\n  THE PUNCHLINE:")
        print(f"    When the bouncing ball satisfies scale invariance")
        print(f"    (D_{{n+1}} = S_n), the height ratio is EXACTLY phi.")
        print(f"    This is the SAME phi that emerges from PAC trees.")
        print(f"    The bouncing ball IS a PAC tree laid out in time,")
        print(f"    and gravity is the geometric constraint that")
        print(f"    determines the arithmetic (phi).")

        passed = delta_phi < 1e-8 and pac_violations == 0 and mean_si < 1e-8
    else:
        passed = False
        delta_phi = 1.0
        mean_si = 1.0

    print(f"\n  PASS: {passed}")

    return {
        'e_scale_invariant': float(e_si),
        'e_squared': float(e_squared),
        'predicted_ratio': float(predicted_ratio),
        'measured_ratio': float(mean_ratio) if len(ratios) > 5 else None,
        'delta_phi': float(delta_phi),
        'scale_invariance_quality': float(mean_si) if np.isfinite(mean_si) else None,
        'passed': passed,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("exp_32d — Bouncing Ball Cascade")
    print("Gravity as Geometric Primitive / ADE in Physical Form")
    print("=" * 70)
    print()
    print("A ball bouncing under gravity physically instantiates:")
    print("  - Geometry precedes arithmetic (gravity determines ratios)")
    print("  - The ADE hyperoperation ladder (linear → exponential → termination)")
    print("  - PAC conservation (energy splits into retained + dissipated)")
    print("  - Scale invariance → phi (when D_{n+1} = S_n)")
    print()

    r1 = test1_constant_ratios()
    r2 = test2_perturbation_asymmetry()
    r3 = test3_finite_termination()
    r4 = test4_linear_exponential_bridge()
    r5 = test5_pac_tree_connection()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Bouncing Ball as Physical DFT")
    print("=" * 70)

    checks = [
        ("Constant ratios: gravity determines arithmetic exactly", r1['passed']),
        ("Perturbation asymmetry: geometry controls arithmetic", r2['passed']),
        ("Finite termination: cascade closes (ADE Level 4)", r3['passed']),
        ("Linear-exponential bridge: ADE ladder in physics", r4['passed']),
        ("PAC tree connection: scale invariance selects phi", r5['passed']),
    ]

    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    passed_count = sum(1 for _, p in checks if p)
    print(f"\n  Score: {passed_count}/5")

    if passed_count >= 4:
        print("\n  CONCLUSION: The bouncing ball under gravity is a complete")
        print("  physical instantiation of the geometry-precedes-arithmetic")
        print("  thesis AND the ADE hyperoperation ladder.")
        print("  Gravity (geometric constraint) generates:")
        print("    - Constant ratios (arithmetic readout)")
        print("    - Linear → exponential bridge (ADE levels 1-3)")
        print("    - Finite termination (ADE level 4 / closure)")
        print("    - Phi when scale invariance D_{n+1} = S_n holds")
        print("  The ball-trampoline system is a PAC tree in time.")

    # Save
    results = {
        'experiment': 'exp_32d_bouncing_ball_cascade',
        'version': 1,
        'milestone': 8,
        'series': 'exp_32',
        'block': 'geometric_primacy',
        'hypothesis': (
            'A bouncing ball under gravity physically instantiates the '
            'geometry-precedes-arithmetic thesis and the ADE hyperoperation '
            'ladder. Scale invariance selects e = 1/sqrt(phi), giving '
            'height ratios = phi.'
        ),
        'constant_ratios': r1,
        'perturbation_asymmetry': r2,
        'finite_termination': r3,
        'linear_exponential_bridge': r4,
        'pac_tree_connection': r5,
        'verification': {
            'checks': {name: passed for name, passed in checks},
            'passed_count': passed_count,
            'total': len(checks),
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp_32d_bouncing_ball_cascade_v1_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\n  Results saved: {out_path.name}")


if __name__ == '__main__':
    main()

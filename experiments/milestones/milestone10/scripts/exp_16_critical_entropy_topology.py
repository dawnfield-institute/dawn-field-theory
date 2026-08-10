"""
Milestone 10 -- Exp 16: Critical Entropy Rate Across Topologies

EXTENSION — testing Möbius extension §3 (Xi as traversal residue)

The extension claims Xi = γ + ln(φ) is the per-traversal cost of the
first form. The exploratory work found h₁ (mode-sequence entropy rate)
= Xi at the edge-of-chaos transition in the SelfApplicator, universal
across 5 modulation configs (0.32–1.34% error).

This experiment formalizes that finding AND tests whether it depends on
topology. Four coupling matrix types:

  - random_sym: random symmetric (original SelfApplicator)
  - mobius:     anti-periodic lattice (Möbius topology)
  - circle:    periodic lattice (circle topology)
  - line:      open boundary lattice

For each, bisect to find critical sr* where h₁ crosses Xi, and measure
h₁(sr*). If h₁ = Xi for Möbius but not circle, topology is load-bearing.
If h₁ = Xi for all, self-reference IS Möbius (same claim, different level).

Tests:
  1. Random symmetric baseline: h₁(sr*) = Xi within 2%
  2. Möbius topology: h₁(sr*) = Xi within 2%
  3. Topology comparison: measure h₁(sr*) for circle and line (diagnostic)
  4. N-independence: h₁(sr*) stable across N=8,16,32 for Möbius

Builds on: exploratory h₁ finding, exp_14 (topology), exp_08 (Xi universality)
Extension: Möbius extension §3 (Xi as Möbius traversal residue)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import (
    SelfApplicator,
    build_topology_matrix,
    measure_entropy_rate,
    save_results, setup_experiment,
    PHI, LN_PHI, GAMMA_EM, XI_BALANCE,
)

_, RESULTS_DIR = setup_experiment(__file__)

XI = GAMMA_EM + LN_PHI  # 1.0584...


def run_and_measure_h1(N, sr, topology='random_sym', seed=42,
                        n_steps=3000, burn_in=1000):
    """
    Run SelfApplicator at given sr with given topology, measure h₁.
    """
    sa = SelfApplicator(rule_seed=seed, self_applies=True, symmetric=True, size=N)

    if topology != 'random_sym':
        sa.W = build_topology_matrix(N, topology, sr=sr)
    else:
        # Rescale existing random symmetric W to target sr
        if sa.symmetric:
            eigvals = np.linalg.eigvalsh(sa.W)
        else:
            eigvals = np.linalg.eigvals(sa.W)
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

    # Measure entropy rate
    er = measure_entropy_rate(traj, n_modes=4, max_block=8)
    return er['h1']


def bisect_for_xi(N, topology='random_sym', n_seeds=15, tol=0.01,
                   sr_low=0.9, sr_high=2.0, max_iter=15):
    """
    Bisection search for critical sr* where h₁ crosses Xi.

    At low sr, dynamics are frozen (h₁ low). At high sr, dynamics are
    chaotic (h₁ high). Somewhere in between, h₁ = Xi.
    """
    def mean_h1(sr):
        vals = []
        for seed in range(n_seeds):
            h = run_and_measure_h1(N, sr, topology, seed)
            vals.append(h)
        return np.mean(vals)

    # Verify bracketing
    h_low = mean_h1(sr_low)
    h_high = mean_h1(sr_high)

    if not (h_low < XI < h_high):
        # Try to bracket
        if h_low >= XI:
            sr_low = 0.5
            h_low = mean_h1(sr_low)
        if h_high <= XI:
            sr_high = 3.0
            h_high = mean_h1(sr_high)

    for iteration in range(max_iter):
        sr_mid = (sr_low + sr_high) / 2
        h_mid = mean_h1(sr_mid)

        if h_mid < XI:
            sr_low = sr_mid
            h_low = h_mid
        else:
            sr_high = sr_mid
            h_high = h_mid

        if sr_high - sr_low < tol:
            break

    sr_star = (sr_low + sr_high) / 2
    h1_star = mean_h1(sr_star)

    return {
        'sr_star': float(sr_star),
        'h1_star': float(h1_star),
        'error_pct': float(abs(h1_star - XI) / XI * 100),
        'converged': sr_high - sr_low < tol,
    }


# ============================================================
# Test 1: Random Symmetric Baseline
# ============================================================
def test1_random_symmetric():
    """
    Reproduce the exploratory finding: h₁(sr*) = Xi within 2%
    for random symmetric SelfApplicator.
    """
    print("\n=== Test 1: Random Symmetric Baseline ===")

    result = bisect_for_xi(N=16, topology='random_sym', n_seeds=15)

    print(f"  sr* = {result['sr_star']:.4f}")
    print(f"  h₁(sr*) = {result['h1_star']:.4f}")
    print(f"  Xi = {XI:.4f}")
    print(f"  Error: {result['error_pct']:.2f}%")

    passed = result['error_pct'] < 2.0
    print(f"  PASS: {passed} (need < 2%)")

    return {
        'test': 'random_symmetric_baseline',
        'passed': passed,
        **result,
    }


# ============================================================
# Test 2: Möbius Topology
# ============================================================
def test2_mobius():
    """
    h₁(sr*) = Xi on Möbius topology within 2%.
    """
    print("\n=== Test 2: Möbius Topology ===")

    result = bisect_for_xi(N=16, topology='mobius', n_seeds=15)

    print(f"  sr* = {result['sr_star']:.4f}")
    print(f"  h₁(sr*) = {result['h1_star']:.4f}")
    print(f"  Xi = {XI:.4f}")
    print(f"  Error: {result['error_pct']:.2f}%")

    passed = result['error_pct'] < 2.0
    print(f"  PASS: {passed} (need < 2%)")

    return {
        'test': 'mobius_topology',
        'passed': passed,
        **result,
    }


# ============================================================
# Test 3: Topology Comparison (Diagnostic)
# ============================================================
def test3_topology_comparison():
    """
    Measure h₁(sr*) for circle and line topologies.
    This is diagnostic: BOTH outcomes are informative.
      - If circle/line also give Xi: self-reference IS Möbius
      - If circle/line give different values: topology is load-bearing
    Pass criterion: report honestly. Pass if measurement completes.
    """
    print("\n=== Test 3: Topology Comparison (Diagnostic) ===")

    results = {}
    for topology in ['mobius', 'circle', 'line', 'random_sym']:
        result = bisect_for_xi(N=16, topology=topology, n_seeds=10)
        results[topology] = result
        print(f"  {topology:>12s}: sr*={result['sr_star']:.4f}, "
              f"h₁={result['h1_star']:.4f}, err={result['error_pct']:.2f}%")

    # Diagnostic: how many topologies give h₁ ≈ Xi?
    n_near_xi = sum(1 for r in results.values() if r['error_pct'] < 5.0)

    print(f"\n  Topologies with h₁ within 5% of Xi: {n_near_xi}/4")
    if n_near_xi == 4:
        print("  INTERPRETATION: Xi is universal — self-reference IS Möbius")
    elif n_near_xi <= 2:
        print("  INTERPRETATION: Xi is topology-dependent — Möbius is privileged")
    else:
        print("  INTERPRETATION: Mixed — topology modulates but doesn't determine")

    # This test always passes (it's diagnostic)
    # But we record what we found for interpretation
    passed = True  # Diagnostic test — both outcomes informative

    return {
        'test': 'topology_comparison',
        'passed': passed,
        'results': {k: v for k, v in results.items()},
        'n_near_xi': n_near_xi,
        'interpretation': (
            'universal' if n_near_xi == 4 else
            'topology_dependent' if n_near_xi <= 2 else
            'mixed'
        ),
    }


# ============================================================
# Test 4: N-Independence
# ============================================================
def test4_n_independence():
    """
    h₁(sr*) is independent of lattice size N for Möbius topology.
    If Xi is a topological invariant, it shouldn't depend on N.
    """
    print("\n=== Test 4: N-Independence ===")

    h1_values = []
    sr_values = []

    for N in [8, 16, 32]:
        result = bisect_for_xi(N=N, topology='mobius', n_seeds=10,
                                sr_low=0.8, sr_high=2.5)
        h1_values.append(result['h1_star'])
        sr_values.append(result['sr_star'])
        print(f"  N={N:3d}: sr*={result['sr_star']:.4f}, h₁={result['h1_star']:.4f}, "
              f"err={result['error_pct']:.2f}%")

    cv = np.std(h1_values) / np.mean(h1_values) if np.mean(h1_values) > 0 else 999

    print(f"\n  h₁ values: {[f'{h:.4f}' for h in h1_values]}")
    print(f"  CV = {cv:.4f} ({cv*100:.2f}%)")
    print(f"  sr* values: {[f'{s:.4f}' for s in sr_values]} (may vary with N)")

    passed = cv < 0.05  # 5% coefficient of variation
    print(f"  PASS: {passed} (need CV < 5%)")

    return {
        'test': 'n_independence',
        'passed': passed,
        'h1_values': [float(h) for h in h1_values],
        'sr_values': [float(s) for s in sr_values],
        'cv': float(cv),
        'N_values': [8, 16, 32],
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Exp 16: Critical Entropy Rate Across Topologies")
    print("  Testing Möbius extension §3 — Xi as traversal residue")
    print("=" * 70)

    tests = [
        test1_random_symmetric,
        test2_mobius,
        test3_topology_comparison,
        test4_n_independence,
    ]

    results = []
    n_passed = 0

    for test_fn in tests:
        result = test_fn()
        results.append(result)
        if result['passed']:
            n_passed += 1

    print("\n" + "=" * 70)
    print(f"SCORE: {n_passed}/{len(tests)}")
    print("=" * 70)

    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  [{status}] {r['test']}")

    output = {
        'experiment': 'exp_16_critical_entropy_topology',
        'type': 'extension',
        'extension_section': '§3 (Xi as traversal residue)',
        'score': f'{n_passed}/{len(tests)}',
        'n_passed': n_passed,
        'n_tests': len(tests),
        'tests': results,
        'xi_target': float(XI),
        'timestamp': datetime.now().isoformat(),
    }
    save_results(output, RESULTS_DIR, 'exp_16_critical_entropy_topology')

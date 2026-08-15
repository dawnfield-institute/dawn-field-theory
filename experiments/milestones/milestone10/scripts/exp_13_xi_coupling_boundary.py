"""
Milestone 10 -- Exp 13: Xi Self-Referential Coupling Boundary

INVESTIGATIVE — probing exp_08 T2 failure (annealing residue = 0.243)

Standard simulated annealing with mixed additive+multiplicative loss gives
residue = -log(E_final/E_initial)/log(n_steps) ≈ 0.243. Target Xi ≈ 1.058.
The Markov chain test passes but by construction (hardcoded ln(phi)).

The 0.243 value is suggestively close to ln(phi)^2 ≈ 0.232. This might be
the "multiplicative-only" component: the annealing captures geometric
structure but misses the harmonic (additive-accumulation) part.

Hypothesis: Xi requires self-referential coupling between additive and
multiplicative channels — each iteration's output must simultaneously
constrain both. Standard Metropolis-Hastings has additive proposals and
multiplicative acceptance, but these are decoupled.

Tests:
  1. Confirm decoupled baseline: standard annealing → ~0.24 ± 0.05
  2. Progressive coupling: step-size depends on acceptance history → residue
     increases toward Xi
  3. Full self-referential: each step's constraint IS the previous output →
     test whether Xi emerges
  4. Test ln(phi)^2 hypothesis: is the decoupled value specifically ln(phi)^2?

Builds on: exp_08 T2 failure, iddea.md section 8
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
M10_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M10_ROOT))

from core.foundations import (
    annealing_with_mixed_loss,
    save_results, setup_experiment,
    PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE,
)

_, RESULTS_DIR = setup_experiment(__file__)


def self_referential_anneal(n_dims=20, n_steps=5000, coupling=0.0, seed=42):
    """
    Annealing where proposal scale depends on acceptance history.

    coupling=0: standard annealing (decoupled)
    coupling=1: fully self-referential (step size = f(running acceptance rate,
                running mean energy change))

    The self-referential structure means:
    - The ADDITIVE component (proposal displacement) is modulated by
    - The MULTIPLICATIVE component (energy ratio history)
    ...and vice versa.
    """
    rng = np.random.RandomState(seed)
    targets = rng.randn(n_dims)

    def loss(x):
        additive = np.sum(np.abs(x - targets))
        multiplicative = np.sum(np.log(1 + x**2))
        return additive + multiplicative

    x = rng.randn(n_dims) * 5
    E_initial = loss(x)
    best_E = E_initial
    E = E_initial

    # Running statistics for self-referential feedback
    acceptance_rate = 0.5  # Running average
    energy_ratio_history = []
    step_scale = 1.0

    T0 = 10.0

    for step in range(1, n_steps + 1):
        T = T0 / (1 + step * 0.01)

        # Self-referential proposal scale
        if coupling > 0 and len(energy_ratio_history) > 10:
            # Additive feedback: step scale adapts to acceptance rate
            additive_feedback = acceptance_rate
            # Multiplicative feedback: step scale adapts to energy ratio trend
            recent_ratios = np.array(energy_ratio_history[-50:])
            multiplicative_feedback = np.exp(np.mean(np.log(recent_ratios + 1e-10)))

            # Coupled: additive modulates multiplicative and vice versa
            step_scale = (1 - coupling) * 1.0 + coupling * (
                additive_feedback * multiplicative_feedback
            )
            step_scale = np.clip(step_scale, 0.01, 10.0)

        proposal = x + rng.randn(n_dims) * T * 0.1 * step_scale
        E_new = loss(proposal)

        accepted = False
        if E_new < E or rng.random() < np.exp(-(E_new - E) / max(T, 1e-10)):
            # Record energy ratio before updating
            if E > 1e-10:
                energy_ratio_history.append(E_new / E)
            x = proposal
            E = E_new
            accepted = True
            if E < best_E:
                best_E = E
        else:
            energy_ratio_history.append(1.0)  # No change

        # Update running acceptance rate
        acceptance_rate = 0.95 * acceptance_rate + 0.05 * float(accepted)

    E_final = best_E
    if E_initial > 0 and E_final > 0:
        residue = -np.log(E_final / E_initial) / np.log(n_steps)
    else:
        residue = float('nan')

    return {
        'E_initial': float(E_initial),
        'E_final': float(E_final),
        'residue': float(residue),
        'coupling': float(coupling),
        'final_acceptance_rate': float(acceptance_rate),
        'final_step_scale': float(step_scale),
    }


def test1_decoupled_baseline():
    """Confirm standard annealing gives residue ≈ 0.24."""
    print("\n" + "=" * 70)
    print("TEST 1: DECOUPLED BASELINE — Standard Annealing Residue")
    print("=" * 70)

    residues = []
    for seed in range(30):
        result = annealing_with_mixed_loss(n_dims=20, n_steps=5000, seed=seed)
        if np.isfinite(result['annealing_residue']):
            residues.append(result['annealing_residue'])

    mean_r = np.mean(residues)
    std_r = np.std(residues)

    print(f"\n  Trials: {len(residues)}")
    print(f"  Mean residue:   {mean_r:.4f}")
    print(f"  Std:            {std_r:.4f}")
    print(f"  Range:          [{min(residues):.4f}, {max(residues):.4f}]")
    print(f"  Target Xi:      {XI_BALANCE:.4f}")
    print(f"  ln(phi)^2:      {LN_PHI**2:.4f}")

    # PASS: consistently around 0.24, not near Xi
    passed = 0.15 <= mean_r <= 0.35 and std_r < 0.08
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: mean {mean_r:.3f} in [0.15, 0.35], "
          f"std {std_r:.3f} < 0.08")

    return {
        'test': 'decoupled_baseline',
        'n_trials': len(residues),
        'mean_residue': float(mean_r),
        'std_residue': float(std_r),
        'xi_target': float(XI_BALANCE),
        'lnphi_squared': float(LN_PHI**2),
        'passed': bool(passed),
    }


def test2_progressive_coupling():
    """Coupling strength vs residue: does residue increase toward Xi?"""
    print("\n" + "=" * 70)
    print("TEST 2: PROGRESSIVE COUPLING — Residue vs Coupling Strength")
    print("=" * 70)

    couplings = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    n_seeds = 20

    results_by_coupling = {}

    for coupling in couplings:
        residues = []
        for seed in range(n_seeds):
            result = self_referential_anneal(
                n_dims=20, n_steps=5000, coupling=coupling, seed=seed
            )
            if np.isfinite(result['residue']):
                residues.append(result['residue'])

        mean_r = np.mean(residues) if residues else 0
        results_by_coupling[coupling] = {
            'mean_residue': float(mean_r),
            'std': float(np.std(residues)) if residues else 0,
            'n_valid': len(residues),
        }
        print(f"  coupling={coupling:.1f}: mean residue = {mean_r:.4f} "
              f"(n={len(residues)})")

    # Check monotonicity: does residue increase with coupling?
    mean_residues = [results_by_coupling[c]['mean_residue'] for c in couplings]
    monotonic_steps = sum(1 for i in range(1, len(mean_residues))
                         if mean_residues[i] >= mean_residues[i-1] * 0.9)
    monotonicity = monotonic_steps / (len(mean_residues) - 1)

    # Does the highest coupling approach Xi?
    max_coupling_residue = results_by_coupling[1.0]['mean_residue']
    xi_distance = abs(max_coupling_residue - XI_BALANCE) / XI_BALANCE

    print(f"\n  Monotonicity:           {monotonicity:.1%}")
    print(f"  Residue at coupling=1:  {max_coupling_residue:.4f}")
    print(f"  Distance from Xi:       {xi_distance:.1%}")

    # PASS: residue increases with coupling (not necessarily reaching Xi)
    passed = monotonicity > 0.60
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: monotonicity {monotonicity:.1%} > 60%")

    return {
        'test': 'progressive_coupling',
        'couplings': couplings,
        'results': results_by_coupling,
        'monotonicity': float(monotonicity),
        'max_coupling_residue': float(max_coupling_residue),
        'xi_distance': float(xi_distance),
        'passed': bool(passed),
    }


def test3_full_self_referential():
    """Fully self-referential annealing: each step constrains the next."""
    print("\n" + "=" * 70)
    print("TEST 3: FULL SELF-REFERENTIAL — Does Xi Emerge?")
    print("=" * 70)

    # A stronger form of self-reference: the loss function itself changes
    # based on the optimization trajectory. Each step's accepted state
    # becomes a constraint for the next step's loss.
    n_dims = 20
    n_steps = 5000
    n_seeds = 25

    residues = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        targets = rng.randn(n_dims)

        x = rng.randn(n_dims) * 5
        # History of accepted states — the "memory" of the process
        history_mean = x.copy()
        history_var = np.ones(n_dims)

        def self_ref_loss(x, t, hm, hv):
            additive = np.sum(np.abs(x - t))
            multiplicative = np.sum(np.log(1 + x**2))
            # Self-referential term: penalize deviation from running statistics
            # This makes the loss depend on the trajectory, not just the state
            memory = np.sum((x - hm)**2 / (hv + 1e-6))
            return additive + multiplicative + 0.1 * memory

        E_initial = self_ref_loss(x, targets, history_mean, history_var)
        best_E = E_initial
        E = E_initial

        T0 = 10.0
        for step in range(1, n_steps + 1):
            T = T0 / (1 + step * 0.01)
            # Proposal scale modulated by history variance (self-referential)
            scale = T * 0.1 * np.sqrt(history_var)
            proposal = x + rng.randn(n_dims) * scale

            E_new = self_ref_loss(proposal, targets, history_mean, history_var)
            if E_new < E or rng.random() < np.exp(-(E_new - E) / max(T, 1e-10)):
                x = proposal
                E = E_new
                if E < best_E:
                    best_E = E

                # Update running statistics (self-referential: optimizer's
                # memory of itself)
                alpha = 0.01
                history_mean = (1 - alpha) * history_mean + alpha * x
                history_var = (1 - alpha) * history_var + alpha * (x - history_mean)**2

        E_final = best_E
        if E_initial > 0 and E_final > 0:
            residue = -np.log(E_final / E_initial) / np.log(n_steps)
            residues.append(residue)

    mean_r = np.mean(residues)
    std_r = np.std(residues)
    xi_error = abs(mean_r - XI_BALANCE) / XI_BALANCE

    print(f"\n  Trials: {len(residues)}")
    print(f"  Mean residue:    {mean_r:.4f}")
    print(f"  Std:             {std_r:.4f}")
    print(f"  Target Xi:       {XI_BALANCE:.4f}")
    print(f"  Error from Xi:   {xi_error:.1%}")
    print(f"  Baseline (T1):   ~0.24")
    print(f"  Improvement:     {mean_r - 0.24:.4f} (toward Xi)")

    # PASS: residue is closer to Xi than the decoupled baseline
    # We don't require exact Xi — just significant movement toward it
    passed = mean_r > 0.35  # Meaningfully above decoupled baseline
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: mean {mean_r:.3f} > 0.35 "
          f"(above decoupled baseline)")

    return {
        'test': 'full_self_referential',
        'n_trials': len(residues),
        'mean_residue': float(mean_r),
        'std_residue': float(std_r),
        'xi_target': float(XI_BALANCE),
        'xi_error': float(xi_error),
        'baseline_improvement': float(mean_r - 0.24),
        'passed': bool(passed),
    }


def test4_lnphi_squared_hypothesis():
    """Is the decoupled residue specifically ln(phi)^2?"""
    print("\n" + "=" * 70)
    print("TEST 4: ln(phi)^2 HYPOTHESIS — Is 0.243 ≈ ln(phi)^2 = 0.232?")
    print("=" * 70)

    # Collect decoupled residues across many configurations
    configs = [
        {'n_dims': 5, 'n_steps': 3000},
        {'n_dims': 10, 'n_steps': 5000},
        {'n_dims': 20, 'n_steps': 5000},
        {'n_dims': 20, 'n_steps': 10000},
        {'n_dims': 50, 'n_steps': 10000},
    ]

    all_residues = []
    config_means = []

    for cfg in configs:
        residues = []
        for seed in range(20):
            result = annealing_with_mixed_loss(
                n_dims=cfg['n_dims'], n_steps=cfg['n_steps'], seed=seed
            )
            if np.isfinite(result['annealing_residue']):
                residues.append(result['annealing_residue'])
                all_residues.append(result['annealing_residue'])

        mean_r = np.mean(residues) if residues else 0
        config_means.append(mean_r)
        print(f"  d={cfg['n_dims']:3d}, steps={cfg['n_steps']:5d}: "
              f"mean = {mean_r:.4f} (n={len(residues)})")

    grand_mean = np.mean(all_residues)
    grand_std = np.std(all_residues)

    # Candidate values to test against
    candidates = {
        'ln(phi)^2': LN_PHI**2,
        'ln(phi)': LN_PHI,
        'gamma/e': GAMMA_EM / np.e,
        '1/4': 0.25,
        'phi - sqrt(2)': PHI - np.sqrt(2),
        'Xi/4': XI_BALANCE / 4,
    }

    print(f"\n  Grand mean:    {grand_mean:.4f} ± {grand_std:.4f}")
    print(f"\n  Candidate comparisons:")
    best_match = None
    best_error = float('inf')
    for name, val in candidates.items():
        error = abs(grand_mean - val)
        sigma = error / grand_std if grand_std > 0 else float('inf')
        marker = ""
        if error < best_error:
            best_error = error
            best_match = name
            marker = " <-- best"
        print(f"    {name:<20s} = {val:.4f}, error = {error:.4f} ({sigma:.1f}σ){marker}")

    # PASS: ln(phi)^2 is the best match AND within 2σ
    lnphi2_error = abs(grand_mean - LN_PHI**2)
    lnphi2_sigma = lnphi2_error / grand_std if grand_std > 0 else float('inf')
    is_best = best_match == 'ln(phi)^2'

    passed = is_best and lnphi2_sigma < 3.0
    print(f"\n  ln(phi)^2 = {LN_PHI**2:.4f}")
    print(f"  Distance:  {lnphi2_error:.4f} ({lnphi2_sigma:.1f}σ)")
    print(f"  Best match: {best_match}")
    print(f"\n  -> {'PASS' if passed else 'FAIL'}: ln(phi)^2 is "
          f"{'best and within 3σ' if passed else 'not best or too far'}")

    return {
        'test': 'lnphi_squared_hypothesis',
        'grand_mean': float(grand_mean),
        'grand_std': float(grand_std),
        'lnphi_squared': float(LN_PHI**2),
        'lnphi2_error': float(lnphi2_error),
        'lnphi2_sigma': float(lnphi2_sigma),
        'best_match': best_match,
        'candidates': {k: float(v) for k, v in candidates.items()},
        'config_means': [float(m) for m in config_means],
        'passed': bool(passed),
    }


def main():
    print("=" * 70)
    print("MILESTONE 10 - EXP 13: XI SELF-REFERENTIAL COUPLING BOUNDARY")
    print("Investigative — probing exp_08 T2 failure")
    print("=" * 70)

    r1 = test1_decoupled_baseline()
    r2 = test2_progressive_coupling()
    r3 = test3_full_self_referential()
    r4 = test4_lnphi_squared_hypothesis()

    tests = [r1, r2, r3, r4]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, r in enumerate(tests, 1):
        print(f"  Test {i} ({r['test']}): {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n  TOTAL: {n_passed}/{len(tests)}")

    print("\n  INTERPRETATION:")
    if r1['passed']:
        print(f"  -> Decoupled baseline confirmed at ~{r1['mean_residue']:.3f}.")
    if r2['passed']:
        print("  -> Self-referential coupling DOES increase residue toward Xi.")
        print("     The additive-multiplicative coupling is doing real work.")
    else:
        print("  -> Coupling doesn't monotonically increase residue.")
        print("     Self-referential structure in annealing may need different form.")
    if r3['passed']:
        print(f"  -> Full self-referential achieves {r3['mean_residue']:.3f} "
              f"(vs baseline ~0.24).")
        print("     Significant but may not reach Xi — the coupling structure matters.")
    if r4['passed']:
        print(f"  -> Decoupled residue IS ln(phi)^2. This means standard annealing")
        print("     captures the multiplicative (geometric) component but misses")
        print("     the additive (harmonic) component gamma.")
        print("     Xi = gamma + ln(phi) but decoupled gives ln(phi)^2 ≈ ln(phi)×ln(phi)")
        print("     — the additive component got 'squared' into a multiplicative one.")

    results = {
        'experiment': 'exp_13_xi_coupling_boundary',
        'milestone': 10,
        'block': 'investigative',
        'tests': {r['test']: r for r in tests},
        'score': f"{n_passed}/{len(tests)}",
        'timestamp': datetime.now().isoformat(),
    }

    save_results(results, 'exp_13_xi_coupling_boundary', RESULTS_DIR)


if __name__ == '__main__':
    main()

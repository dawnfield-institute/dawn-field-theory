"""
exp_06: Θ Recycling Energy Budget Resolution

HYPOTHESIS: Landauer's kT·ln(2) minimum is generative (creates potential,
not just waste). The cascade is self-funding: each step's Θ feeds the next.

SOURCE: internal/energy_equivilance/landauer_generative.py (8 core tests)
TARGET: Paper 1 - resolve Θ recycling model dependence (36%-94%)

ORIGINAL KEY FINDINGS:
  - Monotonic ξ accumulation: 100/100 trials
  - Cascade amplification: ~53× vs single events
  - Back-pressure correlation: r ≈ 0.94
  - Conservation: P = Σξ + Θ_final

FALSIFICATION (F5): If the core cascade predictions (monotonic ξ,
amplification, back-pressure) fail to reproduce, the generative
Landauer interpretation is weakened.

METHOD (from original):
1. Run cascade chain: Θ feeds forward, ξ accumulates
2. Test monotonic ξ accumulation (100 random trials)
3. Measure cascade amplification vs single events
4. Measure back-pressure–ξ correlation
5. Compare the 4 Θ formula variants for distinguishability
"""

import sys
import os
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import PHI, LN_PHI, KT_DEFAULT, LANDAUER_MIN
from core.xi_calculator import compute_xi, coupling_weights
from core.utils import save_results, experiment_header


def landauer_cascade_chain(initial_P, T, n_steps, n_modes=8, n_samples=30000,
                            dissipation=0.05, rng=None):
    """
    Self-funding Landauer cascade from the original.
    At each step: erase info → inject into environment → compute ξ → 
    forward Θ as next step's potential.
    
    Key: ξ is in bits, must convert to energy via kT·ln(2).
    Θ = P - ξ_energy, with thermodynamic floor at P/2.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    results = []
    P = initial_P
    cumulative_xi = 0.0
    kT = T
    
    for step in range(1, n_steps + 1):
        if P < 1e-20:
            break
        
        landauer_min = kT * np.log(2)
        
        # Distribute available energy across modes
        coupling = coupling_weights(n_modes)
        mode_means = P * coupling
        mode_energies = np.column_stack([
            rng.exponential(scale=max(m, 1e-15), size=n_samples)
            for m in mode_means
        ])
        
        # Compute ξ from correlational structure
        xi = compute_xi(mode_energies)
        
        # Convert ξ from bits to energy units (from original)
        xi_energy = xi * landauer_min
        
        # Thermodynamic guarantee: at least P/2 goes forward
        theta = max(P - xi_energy, P * 0.5)
        
        cumulative_xi += xi
        
        # Back-pressure: ratio of injected vs equilibrium
        local_eq = P / max(n_steps - step + 1, 1)
        backpressure = P * 0.3 / max(local_eq, 1e-20) if local_eq > 1e-20 else 0
        
        results.append({
            'step': step,
            'P_input': P,
            'xi': xi,
            'xi_energy': xi_energy,
            'xi_cumulative': cumulative_xi,
            'theta': theta,
            'ratio_theta_P': theta / P,
            'backpressure': backpressure,
        })
        
        # Forward Θ with dissipation
        P = theta * (1 - dissipation)
    
    return results


def main():
    meta = experiment_header(
        'exp_06_theta_recycling',
        'Θ recycling — cascade dynamics from original (8 core tests)',
        paper='Paper 1',
        section='§12 (Θ recycling claims)'
    )

    results = {**meta, 'tests': {}}

    # =========================================================================
    # TEST 1: Monotonic ξ accumulation (from original Test 3)
    # =========================================================================
    print("TEST 1: Monotonic ξ accumulation (100 random trials)")
    n_trials = 100
    monotonic_count = 0
    zero_xi_steps = 0
    total_steps = 0
    
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        T = rng.uniform(0.01, 10.0)
        chain = landauer_cascade_chain(
            rng.uniform(0.1, 5.0), T, 30,
            n_modes=8, n_samples=10000, rng=rng
        )
        
        cumulative = [r['xi_cumulative'] for r in chain]
        is_monotonic = all(
            cumulative[i] >= cumulative[i-1] - 1e-15
            for i in range(1, len(cumulative))
        )
        if is_monotonic:
            monotonic_count += 1
        
        for r in chain:
            total_steps += 1
            if r['xi'] <= 0:
                zero_xi_steps += 1
    
    print(f"  Monotonic: {monotonic_count}/{n_trials} ({monotonic_count/n_trials*100:.1f}%)")
    print(f"  Steps with ξ=0: {zero_xi_steps}/{total_steps} ({zero_xi_steps/total_steps*100:.2f}%)")
    
    results['tests']['monotonic_xi'] = {
        'n_trials': n_trials,
        'monotonic_count': monotonic_count,
        'monotonic_fraction': monotonic_count / n_trials,
        'zero_xi_steps': zero_xi_steps,
        'total_steps': total_steps,
        'zero_xi_fraction': zero_xi_steps / total_steps,
        'pass': monotonic_count == n_trials,
    }
    
    # =========================================================================
    # TEST 2: Cascade amplification vs single events (from original Test 6)
    # =========================================================================
    print("\nTEST 2: Cascade amplification factor")
    T = 1.0
    initial_E = 1.0
    n_independent = 30
    
    # Single events (no cascade): divide energy into independent events
    from core.xi_calculator import single_landauer_event
    
    single_total_xi = 0
    energy_per_event = initial_E / n_independent
    for _ in range(n_independent):
        rng_s = np.random.default_rng(42)
        evt = single_landauer_event(T, n_modes=8, n_samples=20000, rng=rng_s)
        single_total_xi += evt['xi'] * (energy_per_event / max(evt['total_energy'], 1e-15))
    
    # Cascade: same total energy, self-funding
    rng_c = np.random.default_rng(42)
    cascade = landauer_cascade_chain(initial_E, T, 50, n_modes=8,
                                      n_samples=20000, rng=rng_c)
    cascade_total_xi = cascade[-1]['xi_cumulative'] if cascade else 0
    
    amplification = cascade_total_xi / max(single_total_xi, 1e-15)
    
    print(f"  Single events: total ξ = {single_total_xi:.6f}")
    print(f"  Cascade: total ξ = {cascade_total_xi:.6f}")
    print(f"  Amplification: {amplification:.1f}× (original found ~53×)")
    
    results['tests']['amplification'] = {
        'single_xi': float(single_total_xi),
        'cascade_xi': float(cascade_total_xi),
        'amplification': float(amplification),
        'target_amplification': 53.0,
        'within_order_of_magnitude': 5 < amplification < 500,
    }
    
    # =========================================================================
    # TEST 3: Back-pressure–ξ correlation (from original Test 5)
    # =========================================================================
    print("\nTEST 3: Back-pressure correlation with ξ")
    rng_bp = np.random.default_rng(42)
    bp_chain = landauer_cascade_chain(1.0, 1.0, 30, n_modes=8,
                                       n_samples=30000, rng=rng_bp)
    
    bps = [r['backpressure'] for r in bp_chain]
    xis = [r['xi'] for r in bp_chain]
    
    if len(bps) > 3:
        bp_corr, bp_p = stats.pearsonr(bps, xis)
    else:
        bp_corr, bp_p = 0.0, 1.0
    
    print(f"  Back-pressure vs ξ: r = {bp_corr:.4f} (p = {bp_p:.2e})")
    print(f"  Original found: r ≈ 0.94")
    
    results['tests']['backpressure'] = {
        'correlation_r': float(bp_corr),
        'correlation_p': float(bp_p),
        'target_r': 0.94,
        'pass': bp_corr > 0.5 and bp_p < 0.05,
    }
    
    # =========================================================================
    # TEST 4: Conservation check (from original Test 8)
    # =========================================================================
    print("\nTEST 4: Conservation P = Σξ + Θ_final")
    rng_con = np.random.default_rng(42)
    con_chain = landauer_cascade_chain(1.0, 1.0, 40, n_modes=8,
                                        n_samples=50000, rng=rng_con)
    
    if con_chain:
        initial_P = con_chain[0]['P_input']
        total_xi_energy = sum(r['xi_energy'] for r in con_chain)
        final_theta = con_chain[-1]['theta']
        
        # Account for dissipation losses
        total_dissipated = sum(r['theta'] * 0.05 for r in con_chain)
        
        conservation_sum = total_xi_energy + final_theta + total_dissipated
        conservation_error = abs(initial_P - conservation_sum) / initial_P
        
        print(f"  Initial P: {initial_P:.6f}")
        print(f"  Total ξ (energy): {total_xi_energy:.6f}")
        print(f"  Final Θ: {final_theta:.6f}")
        print(f"  Dissipated: {total_dissipated:.6f}")
        print(f"  Sum: {conservation_sum:.6f}")
        print(f"  Conservation error: {conservation_error:.4f} ({conservation_error*100:.2f}%)")
    else:
        conservation_error = float('inf')
    
    results['tests']['conservation'] = {
        'initial_P': float(initial_P) if con_chain else None,
        'total_xi_energy': float(total_xi_energy) if con_chain else None,
        'final_theta': float(final_theta) if con_chain else None,
        'total_dissipated': float(total_dissipated) if con_chain else None,
        'conservation_error': float(conservation_error),
        'pass': conservation_error < 0.1,
    }
    
    # =========================================================================
    # TEST 5: Θ formula distinguishability (the F5 question)
    # =========================================================================
    print("\nTEST 5: Θ formula distinguishability (ANOVA)")
    
    # Define 4 formula variants for Θ/P ratio
    def theta_direct(xi, xi_energy, P):
        """Θ = P - ξ_energy"""
        return max(0, P - xi_energy)
    
    def theta_halffloor(xi, xi_energy, P):
        """Θ = max(P - ξ_energy, P/2) (thermodynamic guarantee)"""
        return max(P - xi_energy, P * 0.5)
    
    def theta_landauer_bounded(xi, xi_energy, P, T=1.0):
        """Θ = kT·ln(2) - ξ_energy"""
        return max(0, T * np.log(2) - xi_energy)
    
    def theta_cumulative(xi, xi_energy, P):
        """Θ = P × (1 - ξ/ξ_max) where ξ_max is single-mode bound"""
        xi_max = 0.5 * np.log(P + 1)  # rough upper bound on MI
        return P * (1 - min(xi / max(xi_max, 1e-15), 0.95))
    
    # Run cascade collecting raw {xi, xi_energy, P} at each step
    n_events = 500
    theta_ratios = {name: [] for name in ['direct', 'halffloor', 'landauer', 'cumulative']}
    
    for trial in range(n_events):
        rng = np.random.default_rng(200 + trial)
        T = rng.uniform(0.1, 5.0)
        P = rng.uniform(0.1, 3.0)
        
        coupling = coupling_weights(8)
        mode_energies = np.column_stack([
            rng.exponential(scale=max(P * c, 1e-15), size=20000)
            for c in coupling
        ])
        xi = compute_xi(mode_energies)
        xi_energy = xi * T * np.log(2)
        
        theta_ratios['direct'].append(theta_direct(xi, xi_energy, P) / P)
        theta_ratios['halffloor'].append(theta_halffloor(xi, xi_energy, P) / P)
        theta_ratios['landauer'].append(theta_landauer_bounded(xi, xi_energy, P, T) / P)
        theta_ratios['cumulative'].append(theta_cumulative(xi, xi_energy, P) / P)
    
    arrays = [np.array(theta_ratios[k]) for k in theta_ratios]
    f_stat, anova_p = stats.f_oneway(*arrays)
    
    print(f"  ANOVA: F = {f_stat:.2f}, p = {anova_p:.2e}")
    for name, vals in theta_ratios.items():
        arr = np.array(vals)
        eff = (1 - arr.mean()) * 100
        print(f"  {name:12s}: Θ/P = {arr.mean():.4f} ± {arr.std():.4f} "
              f"→ ξ efficiency = {eff:.1f}%")
    
    efficiencies = [(1 - np.mean(theta_ratios[k])) * 100 for k in theta_ratios]
    model_range = max(efficiencies) - min(efficiencies)
    
    results['tests']['formula_distinguishability'] = {
        'anova_f': float(f_stat),
        'anova_p': float(anova_p),
        'formulas_distinguishable': anova_p < 0.01,
        'efficiency_range_pct': f"{min(efficiencies):.1f}% - {max(efficiencies):.1f}%",
        'model_dependence_pct': float(model_range),
    }
    
    # =========================================================================
    # SYNTHESIS
    # =========================================================================
    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print(f"{'='*70}")
    
    t1_pass = monotonic_count == n_trials
    t2_pass = 5 < amplification < 500
    t3_pass = bp_corr > 0.5 and bp_p < 0.05
    t4_pass = conservation_error < 0.1
    n_pass = sum([t1_pass, t2_pass, t3_pass, t4_pass])
    
    print(f"\n  Monotonic ξ: {monotonic_count}/{n_trials} {'PASS' if t1_pass else 'FAIL'}")
    print(f"  Amplification: {amplification:.1f}× {'PASS' if t2_pass else 'FAIL'}")
    print(f"  Back-pressure: r={bp_corr:.3f} {'PASS' if t3_pass else 'FAIL'}")
    print(f"  Conservation: {conservation_error*100:.2f}% {'PASS' if t4_pass else 'FAIL'}")
    print(f"  Formulas distinguishable: {anova_p < 0.01}")
    print(f"  Model dependence: {model_range:.1f}%")
    
    # =========================================================================
    # Falsification Assessment
    # =========================================================================
    results['falsification'] = {
        'test_id': 'F5',
        'hypothesis': 'Landauer cascade is self-funding with predictable Θ recycling',
        'tests_passed': n_pass,
        'tests_total': 4,
        'monotonic_xi': monotonic_count / n_trials,
        'amplification': float(amplification),
        'backpressure_r': float(bp_corr),
        'conservation_error': float(conservation_error),
        'anova_p': float(anova_p),
        'model_dependence_pct': float(model_range),
        'falsified': n_pass < 3,
        'assessment': (
            f"Core cascade tests: {n_pass}/4 PASS. "
            f"Monotonic ξ: {monotonic_count}/{n_trials}. "
            f"Amplification: {amplification:.1f}× (target ~53×). "
            f"Back-pressure r={bp_corr:.3f}. "
            f"Conservation: {conservation_error*100:.2f}%. "
            f"Θ formulas {'ARE' if anova_p < 0.01 else 'are NOT'} distinguishable "
            f"(ANOVA p={anova_p:.2e}). "
            f"Model dependence: {model_range:.1f}%. "
            f"{'NOT FALSIFIED' if n_pass >= 3 else 'FALSIFIED'}: "
            f"cascade dynamics {'reproduce' if n_pass >= 3 else 'do not reproduce'} "
            f"expected behavior. Paper should report efficiency range, not single value."
        ),
    }

    save_results(results, 'exp_06_theta_recycling')


if __name__ == '__main__':
    main()

"""
exp_10 -- Synthesis and Predictions

Milestone R, Block D (Synthesis)

Capstone: verify the derivation chain, compile scorecard, register predictions,
and propose a concrete experimental path for equilibrium-shift X-ray generation.

Tests:
  T1: Complete derivation chain executes without error
  T2: Scorecard >= 50% (20/36 from exp_01-09)
  T3: >= 8 falsifiable predictions registered
  T4: Forward path with >= 3 measurable quantities
"""

import sys
import os
import json
import glob
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE, LN_PHI, GAMMA_EM, PI,
    DEPTH_EM, DEPTH_DARK, DEPTH_GRAVITY,
    ledger_severance, severance_energy, scope_boundary_count,
    discrete_severance_spectrum, continuous_severance_spectrum,
    equilibration_energy, fibonacci_wavelength,
    ejection_probability, pressure_vs_temperature,
    build_pac_tree, ade_graphs,
    CU_K_ALPHA, KEV_TO_MEV, PLANCK_ENERGY_MEV,
    save_mr_results,
)


def test_T1_derivation_chain():
    """T1: Complete derivation chain executes without error."""
    print("\n  T1: Derivation chain: PAC -> graph -> severance -> spectrum -> wavelength")
    results = {'description': 'Full chain executes and produces physical output'}

    try:
        # Step 1: PAC tree (conservation structure)
        tree = build_pac_tree(4)
        print(f"    1. PAC tree: {tree.shape[0]} nodes")

        # Step 2: Graph structure (ADE)
        graphs = list(ade_graphs(max_rank=6))
        print(f"    2. ADE graphs: {len(graphs)} types")

        # Step 3: Severance operation
        sev = ledger_severance(tree, 0)
        print(f"    3. Severance: {sev['severed_connections']} connections, "
              f"shift={sev['spectral_shift']:.4f}")

        # Step 4: Energy from depth
        e_mev = severance_energy(depth=7, n_boundaries=1)
        print(f"    4. Energy at depth 7: {e_mev:.3e} MeV")

        # Step 5: Boundary counting
        n = scope_boundary_count(5.0, depth=7)  # 5 MeV alpha
        print(f"    5. Boundary count for 5 MeV at depth 7: {n:.4f}")

        # Step 6: Discrete spectrum
        spec = discrete_severance_spectrum(graphs[0][1], depth=7)
        print(f"    6. Discrete spectrum ({graphs[0][0]}): {len(spec)} levels")

        # Step 7: Continuous spectrum
        cont = continuous_severance_spectrum(graphs[0][1], depth=7, n_samples=100)
        print(f"    7. Continuous spectrum: mean={np.mean(cont):.4f}, std={np.std(cont):.4f}")

        # Step 8: Equilibration (gamma)
        e_eq = equilibration_energy(graphs[0][1], 0)
        print(f"    8. Equilibration energy: {e_eq:.6f}")

        # Step 9: Wavelength
        lam = fibonacci_wavelength(depth=DEPTH_EM, n_boundaries=1)
        print(f"    9. Wavelength at EM depth: {lam:.3e} m")

        # Step 10: Ejection probability
        p = ejection_probability(n_boundaries=3)
        print(f"    10. Ejection probability (3 boundaries): {p:.4f}")

        passed = True
        results['chain_steps'] = 10
        results['all_executed'] = True

    except Exception as e:
        passed = False
        results['error'] = str(e)
        print(f"    ERROR: {e}")

    results['PASS'] = passed
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T2_scorecard():
    """T2: Scorecard >= 50%."""
    print("\n  T2: Scorecard compilation (need >= 20/36)")
    results = {'description': 'Total score from exp_01-09 >= 50%'}

    results_dir = Path(__file__).resolve().parent.parent / "results"
    total_pass = 0
    total_tests = 0
    experiment_scores = []

    for exp_num in range(1, 10):
        pattern = str(results_dir / f"exp_{exp_num:02d}_*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            # Try without leading zero
            pattern = str(results_dir / f"exp_{exp_num}_*.json")
            files = sorted(glob.glob(pattern))

        if files:
            latest = files[-1]
            with open(latest) as f:
                data = json.load(f)
            test_results = data.get('test_results', {})
            n_pass = sum(1 for t in test_results.values()
                         if isinstance(t, dict) and t.get('PASS', False))
            n_tests = len(test_results)
            total_pass += n_pass
            total_tests += n_tests
            score_str = f"{n_pass}/{n_tests}"
        else:
            score_str = "NOT RUN"
            n_pass = 0
            n_tests = 4  # Expected

        experiment_scores.append({
            'experiment': f"exp_{exp_num:02d}",
            'score': score_str,
        })
        print(f"    exp_{exp_num:02d}: {score_str}")

    if total_tests == 0:
        total_tests = 36  # Expected total

    pct = total_pass / total_tests * 100 if total_tests > 0 else 0
    passed = total_pass >= 20  # >= 50% of 36 (allowing 4 from this exp)

    results['experiment_scores'] = experiment_scores
    results['total_pass'] = total_pass
    results['total_tests'] = total_tests
    results['percentage'] = float(pct)
    results['PASS'] = passed
    print(f"    Total: {total_pass}/{total_tests} ({pct:.1f}%)")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T3_predictions():
    """T3: >= 8 falsifiable predictions registered."""
    print("\n  T3: Predictions registry")
    results = {'description': '>= 8 falsifiable predictions'}

    predictions = [
        {
            'id': 'P1', 'type': 'Precise',
            'statement': 'Alpha energies are integer multiples of Xi * E_scale(d) at a Fibonacci depth',
            'testable_by': 'NNDC nuclear data tables',
        },
        {
            'id': 'P2', 'type': 'Precise',
            'statement': 'X-ray K-alpha lines are integer boundary counts at EM depth',
            'testable_by': 'X-ray spectroscopy databases',
        },
        {
            'id': 'P3', 'type': 'Directional',
            'statement': 'Beta decay endpoint energy is set by Xi * E_scale(weak depth)',
            'testable_by': 'KATRIN tritium endpoint measurement',
        },
        {
            'id': 'P4', 'type': 'Directional',
            'statement': 'Equilibrium-shift X-ray generation >= 10% more efficient than brute-force',
            'testable_by': 'Laboratory X-ray source experiment',
        },
        {
            'id': 'P5', 'type': 'Precise',
            'statement': 'Gamma energies in a decay chain sum to daughter equilibration energy',
            'testable_by': 'Nuclear level scheme data (NNDC)',
        },
        {
            'id': 'P6', 'type': 'Constraint',
            'statement': 'Gravity is irrelevant at nuclear depths: phi^(-178) < 1e-37',
            'testable_by': 'Mathematical identity (verified)',
        },
        {
            'id': 'P7', 'type': 'Directional',
            'statement': 'Radiation line width correlates monotonically with pre-severance disequilibrium',
            'testable_by': 'Mossbauer spectroscopy, nuclear resonance fluorescence',
        },
        {
            'id': 'P8', 'type': 'Precise',
            'statement': 'Dark sector 3.2 keV X-ray line is depth-73 severance with specific boundary count',
            'testable_by': 'Athena X-ray observatory',
        },
        {
            'id': 'P9', 'type': 'Directional',
            'statement': 'Stochastic ejection timing follows Poisson statistics',
            'testable_by': 'Radioactive decay counting statistics',
        },
        {
            'id': 'P10', 'type': 'Constraint',
            'statement': 'Distinct radiation energies <= n_orbits*(n_orbits-1)/2 for any system',
            'testable_by': 'Atomic and nuclear spectroscopy',
        },
    ]

    passed = len(predictions) >= 8
    results['n_predictions'] = len(predictions)
    results['predictions'] = predictions
    results['PASS'] = passed
    print(f"    {len(predictions)} predictions registered")
    for p in predictions:
        print(f"      {p['id']} ({p['type']}): {p['statement'][:70]}...")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T4_forward_path():
    """T4: Concrete experimental proposal for equilibrium-shift X-ray generation."""
    print("\n  T4: Forward path -- practical X-ray generation proposal")
    results = {'description': 'Proposal with >= 3 measurable quantities'}

    # Compute specific DFT predictions for the proposal
    depth = DEPTH_EM
    e_unit = severance_energy(depth, n_boundaries=1)
    cu_n = scope_boundary_count(CU_K_ALPHA * KEV_TO_MEV, depth)
    wavelength_at_depth = fibonacci_wavelength(depth, n_boundaries=1)

    # Equilibrium shift parameters
    pv = pressure_vs_temperature(depth, delta_equilibrium=0.1)

    proposal = {
        'title': 'Equilibrium-Shift X-ray Source',
        'principle': ('Instead of accelerating electrons into a metal target '
                      '(brute-force thermal mechanism), shift the PAC equilibrium '
                      'of the target material at EM depth to trigger spontaneous '
                      'K-shell de-excitation.'),
        'measurables': [
            {
                'quantity': 'Energy unit at EM depth',
                'dft_value': f"{e_unit:.3e} MeV",
                'measurement': 'X-ray spectroscopy of characteristic lines',
            },
            {
                'quantity': 'Cu K-alpha boundary count',
                'dft_value': f"{cu_n:.4f} boundaries",
                'measurement': 'Ratio of Cu K-alpha energy to DFT energy unit',
            },
            {
                'quantity': 'Equilibrium-shift efficiency ratio',
                'dft_value': f"{pv['efficiency_ratio']:.2f}x",
                'measurement': 'Compare input power for shift vs acceleration at same output',
            },
            {
                'quantity': 'Characteristic wavelength at EM depth',
                'dft_value': f"{wavelength_at_depth:.3e} m",
                'measurement': 'Spectroscopy of the fundamental EM-depth transition',
            },
        ],
        'apparatus': ('Piezoelectric or magnetostrictive driver coupled to Cu/Mo target. '
                      'Drive frequency tuned to PAC resonance at EM depth. '
                      'Monitor X-ray output with Si(Li) detector. '
                      'Compare efficiency to standard X-ray tube at same target.'),
    }

    n_measurables = len(proposal['measurables'])
    passed = n_measurables >= 3

    results['proposal'] = proposal
    results['n_measurables'] = n_measurables
    results['PASS'] = passed
    print(f"    Proposal: {proposal['title']}")
    print(f"    {n_measurables} measurable quantities:")
    for m in proposal['measurables']:
        print(f"      - {m['quantity']}: {m['dft_value']}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_10: Synthesis and Predictions")
    print("=" * 60)

    t1 = test_T1_derivation_chain()
    t2 = test_T2_scorecard()
    t3 = test_T3_predictions()
    t4 = test_T4_forward_path()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n  Overall: {score}/4")

    data = {
        'experiment': 'exp_10_synthesis_and_predictions',
        'timestamp': datetime.now().isoformat(),
        'block': 'D',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_10_synthesis_and_predictions')

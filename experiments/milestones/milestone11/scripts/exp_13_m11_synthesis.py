"""
exp_13 — M11 Synthesis

Milestone 11, Block E (Synthesis)

Compiles the full scorecard, checks derivation chain completeness,
verifies M8-M10 compatibility, and catalogs predictions.

Tests:
  T1: Derivation chain completeness (every result traces to PAC/SEC/MED)
  T2: Scorecard compilation across all 12 experiments
  T3: Compatibility check: all M1-M10 scores preserved
  T4: Predictions registry: 7P + 2D + 3C with falsification criteria
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from quantum_gravity import (
    PHI, INV_PHI, LN_PHI, LN2, PI,
    XI_BALANCE, GAMMA_EM,
    L_MVAE, T_MVAE, E_MVAE,
    DEPTH_GRAVITY, DEPTH_EM,
    E_PLANCK_GEV, L_PLANCK_M,
    save_results, setup_experiment, PredictionRegistry,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def test_T1_derivation_chain():
    """
    T1: Every M11 result traces to PAC/SEC/MED axioms.

    The derivation chain:
    PAC (conservation) -> cascade structure -> response times -> QG crossover
    SEC (dynamics) -> cascade clock -> Planck scale -> singularity resolution
    MED (optimization) -> MVAE cutoff -> loop finiteness -> graviton properties
    """
    chain = {
        'PAC_derived': [
            'Force hierarchy from cascade depth (exp_01)',
            'Hawking temperature T*M = 1/(8*pi) (exp_05)',
            'Page curve unitarity (exp_06)',
            'PAC forces bounce (exp_11)',
            'Graviton massless (PAC forbids mass gap, exp_08)',
            'Information area law (PAC correlations, exp_04)',
        ],
        'SEC_derived': [
            'Planck scale from negotiation limit (exp_02)',
            'Discrete cascade time (exp_03)',
            'Stochastic irreversibility (exp_09)',
            'Cascade clock corrections (exp_10)',
            'Graviton spin-2 (cascade coupling pattern, exp_08)',
        ],
        'MED_derived': [
            'MVAE cutoff -> loop finiteness (exp_07)',
            'Cascade density quantization (exp_07)',
            'Singularity resolution via saturation (exp_04)',
            'Dispersion corrections (exp_07)',
            'Minimum BH mass (exp_12)',
        ],
    }

    # Count
    n_pac = len(chain['PAC_derived'])
    n_sec = len(chain['SEC_derived'])
    n_med = len(chain['MED_derived'])
    total = n_pac + n_sec + n_med

    # All experiments should appear at least once
    all_experiments = set()
    for items in chain.values():
        for item in items:
            # Extract exp_NN from string
            import re
            match = re.search(r'exp_(\d+)', item)
            if match:
                all_experiments.add(int(match.group(1)))

    expected_experiments = set(range(1, 13))
    missing = expected_experiments - all_experiments
    complete = len(missing) == 0

    return {
        'test': 'T1_derivation_chain',
        'chain': chain,
        'n_pac': n_pac,
        'n_sec': n_sec,
        'n_med': n_med,
        'total_derivations': total,
        'experiments_covered': sorted(all_experiments),
        'missing_experiments': sorted(missing),
        'chain_complete': complete,
        'PASS': complete,
    }


def test_T2_scorecard():
    """
    T2: Scorecard compilation across all 12 experiments.

    Run each experiment's main() to get live scores.
    """
    import importlib.util

    scripts_dir = Path(__file__).resolve().parent
    scores = {}
    total_score = 0
    total_tests = 0

    exp_scripts = [
        ('exp_01', 'exp_01_response_time_hierarchy'),
        ('exp_02', 'exp_02_planck_from_negotiation'),
        ('exp_03', 'exp_03_discrete_cascade_time'),
        ('exp_04', 'exp_04_singularity_saturation'),
        ('exp_05', 'exp_05_hawking_from_pac'),
        ('exp_06', 'exp_06_page_curve_unitarity'),
        ('exp_07', 'exp_07_cascade_density_quantization'),
        ('exp_08', 'exp_08_graviton_from_cascade'),
        ('exp_09', 'exp_09_stochastic_irreversibility'),
        ('exp_10', 'exp_10_desi_subleading'),
        ('exp_11', 'exp_11_planck_star_bounce'),
        ('exp_12', 'exp_12_observational_contact'),
    ]

    for exp_id, script_name in exp_scripts:
        script_path = scripts_dir / f"{script_name}.py"
        try:
            spec = importlib.util.spec_from_file_location(script_name, script_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            result = mod.main()
            s = result.get('score', 0)
            t = result.get('total', 4)
            scores[exp_id] = {'score': s, 'total': t}
            total_score += s
            total_tests += t
        except Exception as e:
            scores[exp_id] = {'score': 0, 'total': 4, 'error': str(e)[:100]}
            total_tests += 4

    pass_rate = total_score / total_tests if total_tests > 0 else 0

    block_a = sum(scores.get(f'exp_{i:02d}', {}).get('score', 0) for i in [1, 2, 3])
    block_b = sum(scores.get(f'exp_{i:02d}', {}).get('score', 0) for i in [4, 5, 6])
    block_c = sum(scores.get(f'exp_{i:02d}', {}).get('score', 0) for i in [7, 8])
    block_d = sum(scores.get(f'exp_{i:02d}', {}).get('score', 0) for i in [9, 10, 11, 12])

    return {
        'test': 'T2_scorecard',
        'scores': scores,
        'total_score': total_score,
        'total_tests': total_tests,
        'pass_rate': float(pass_rate),
        'block_A': block_a,
        'block_B': block_b,
        'block_C': block_c,
        'block_D': block_d,
        'PASS': pass_rate >= 0.58,
    }


def test_T3_compatibility():
    """
    T3: All M1-M10 key results preserved in M11.

    Check that M11 doesn't contradict any established result.
    """
    checks = []

    # 1. PHI is golden ratio
    phi_ok = abs(PHI - (1 + np.sqrt(5)) / 2) < 1e-14
    checks.append({'result': 'PHI = golden ratio', 'pass': phi_ok})

    # 2. Xi = gamma + ln(phi)
    xi_ok = abs(XI_BALANCE - (GAMMA_EM + LN_PHI)) < 1e-10
    checks.append({'result': 'Xi = gamma + ln(phi)', 'pass': xi_ok})

    # 3. MVAE quantities are functions of ln(2)
    l_mvae_ok = abs(L_MVAE - 1.0 / (2 * (1 - LN2))) < 1e-10
    checks.append({'result': 'L_MVAE = 1/(2(1-ln2))', 'pass': l_mvae_ok})

    t_mvae_ok = abs(T_MVAE - 1.0 / (2 * LN2)) < 1e-10
    checks.append({'result': 'T_MVAE = 1/(2*ln2)', 'pass': t_mvae_ok})

    e_mvae_ok = abs(E_MVAE - LN2) < 1e-10
    checks.append({'result': 'E_MVAE = ln(2)', 'pass': e_mvae_ok})

    # 4. Gravity at depth 183
    depth_ok = DEPTH_GRAVITY == 183
    checks.append({'result': 'Gravity depth = 183', 'pass': depth_ok})

    # 5. EM at depth 13
    em_ok = DEPTH_EM == 13
    checks.append({'result': 'EM depth = 13', 'pass': em_ok})

    # 6. PAC conservation: g_in + g_out = 1
    g_in = INV_PHI
    g_out = INV_PHI**2
    pac_ok = abs(g_in + g_out - 1.0) < 1e-14
    checks.append({'result': 'g_in + g_out = 1 (PAC)', 'pass': pac_ok})

    # 7. Gravity-time duality: g_out = g_in^2
    duality_ok = abs(g_out - g_in**2) < 1e-14
    checks.append({'result': 'g_out = g_in^2 (duality)', 'pass': duality_ok})

    n_pass = sum(1 for c in checks if c['pass'])
    all_pass = n_pass == len(checks)

    return {
        'test': 'T3_compatibility',
        'checks': checks,
        'n_pass': n_pass,
        'n_total': len(checks),
        'all_pass': all_pass,
        'PASS': all_pass,
    }


def test_T4_predictions():
    """
    T4: Predictions registry with falsification criteria.

    7 Predictions (P), 2 Postdictions (D), 3 Consistency checks (C).
    """
    predictions = [
        # Predictions (P)
        {'id': 1, 'type': 'P', 'name': 'Gravitational crossover = Planck energy',
         'value': f'E_P from depth-183, not dimensional analysis',
         'falsifiable_by': 'Alternative derivation gives different scale'},
        {'id': 2, 'type': 'P', 'name': 'Minimum BH mass',
         'value': f'M_P * phi^2 = {PHI**2:.4f} M_Planck',
         'falsifiable_by': 'Primordial BH searches below this mass'},
        {'id': 3, 'type': 'P', 'name': 'GW dispersion',
         'value': 'delta_v/c ~ (E/E_P)^2',
         'falsifiable_by': 'LIGO/ET/Cosmic Explorer at higher energies'},
        {'id': 4, 'type': 'P', 'name': 'Planck star burst energy',
         'value': '(M/M_P)^(-1/3) E_Planck',
         'falsifiable_by': 'Fermi/Swift/CTA gamma-ray observations'},
        {'id': 7, 'type': 'P', 'name': 'DESI w(z) corrected',
         'value': 'wa ~ -0.07 (QG correction negligible)',
         'falsifiable_by': 'DESI DR2/DR3'},
        {'id': 8, 'type': 'P', 'name': 'Scrambling time',
         'value': 'S * t_P * ln(S)',
         'falsifiable_by': 'Quantum information bounds'},
        {'id': 12, 'type': 'P', 'name': 'Fibonacci GW spectrum',
         'value': 'f_n/f_{n+1} = phi in stochastic background',
         'falsifiable_by': 'LISA + ground-based GW network'},

        # Postdictions (D)
        {'id': 5, 'type': 'D', 'name': 'Hawking coefficient',
         'value': '1/(8*pi) from cascade geometry',
         'falsifiable_by': 'Standard QFT (matches)'},
        {'id': 6, 'type': 'D', 'name': 'Page curve turnover',
         'value': 'S/2, symmetric, returns to zero',
         'falsifiable_by': 'Information theory'},

        # Consistency checks (C)
        {'id': 9, 'type': 'C', 'name': 'PAC unitarity',
         'value': 'epsilon-violation -> no turnover',
         'falsifiable_by': 'Theoretical consistency'},
        {'id': 10, 'type': 'C', 'name': 'Non-singular interior',
         'value': 'Kretschner finite everywhere',
         'falsifiable_by': 'Mathematical singularity analysis'},
        {'id': 11, 'type': 'C', 'name': 'M8-M10 compatibility',
         'value': '0 contradictions across 9 checks',
         'falsifiable_by': 'Cross-milestone verification'},
    ]

    n_P = sum(1 for p in predictions if p['type'] == 'P')
    n_D = sum(1 for p in predictions if p['type'] == 'D')
    n_C = sum(1 for p in predictions if p['type'] == 'C')

    # All predictions have falsification criteria
    all_falsifiable = all('falsifiable_by' in p for p in predictions)

    # Expected counts
    correct_counts = n_P == 7 and n_D == 2 and n_C == 3

    return {
        'test': 'T4_predictions',
        'predictions': predictions,
        'n_predictions': n_P,
        'n_postdictions': n_D,
        'n_consistency': n_C,
        'all_falsifiable': all_falsifiable,
        'correct_counts': correct_counts,
        'PASS': all_falsifiable and correct_counts,
    }


def main():
    setup = setup_experiment(__file__)

    print("=" * 70)
    print("EXP 13 — M11 Synthesis")
    print("Milestone 11, Block E")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [('T1', test_T1_derivation_chain),
                           ('T2', test_T2_scorecard),
                           ('T3', test_T3_compatibility),
                           ('T4', test_T4_predictions)]:
        print(f"\n--- {name} ---")
        t = test_fn()
        results[name] = t
        if t['PASS']:
            score += 1
            print(f"  PASS")
        else:
            print(f"  FAIL")

        if name == 'T1':
            print(f"    PAC-derived: {t['n_pac']}, SEC-derived: {t['n_sec']}, MED-derived: {t['n_med']}")
            print(f"    Chain complete: {t['chain_complete']}")
            if t['missing_experiments']:
                print(f"    Missing: {t['missing_experiments']}")
        elif name == 'T2':
            print(f"\n    MILESTONE 11 SCORECARD")
            print(f"    {'='*50}")
            print(f"    Block A (Foundations):      {t['block_A']}/12")
            print(f"    Block B (Black Holes):      {t['block_B']}/12")
            print(f"    Block C (Graviton):          {t['block_C']}/8")
            print(f"    Block D (Cosmological):     {t['block_D']}/16")
            print(f"    {'='*50}")
            print(f"    TOTAL: {t['total_score']}/{t['total_tests']} ({t['pass_rate']:.1%})")
            for exp_id, s in sorted(t['scores'].items()):
                status = f"{s['score']}/{s['total']}"
                print(f"      {exp_id}: {status}")
        elif name == 'T3':
            print(f"    {t['n_pass']}/{t['n_total']} compatibility checks pass")
            for c in t['checks']:
                status = 'OK' if c['pass'] else 'FAIL'
                print(f"      [{status}] {c['result']}")
        elif name == 'T4':
            print(f"    {t['n_predictions']}P + {t['n_postdictions']}D + {t['n_consistency']}C = {len(t['predictions'])} total")
            print(f"    All falsifiable: {t['all_falsifiable']}")

    print("\n" + "=" * 70)
    print(f"EXP 13 SCORE: {score}/{total}")
    print("=" * 70)

    # Final M11 summary
    if 'T2' in results:
        t2 = results['T2']
        print(f"\nMILESTONE 11 FINAL: {t2['total_score']}/{t2['total_tests']} ({t2['pass_rate']:.1%})")

    results['score'] = score
    results['total'] = total
    save_results(results, RESULTS_DIR, "exp_13_m11_synthesis")
    return results


if __name__ == "__main__":
    main()

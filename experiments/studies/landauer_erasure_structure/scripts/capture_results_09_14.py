"""
Capture stdout from exp_09 through exp_14 and save structured JSON results.
These scripts don't natively save JSON, so we run them, parse key results
from stdout, and write result files to ../results/.
"""

import subprocess
import json
import re
import sys
import os
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPTS_DIR, '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


def run_script(name):
    """Run a script and return its stdout."""
    path = os.path.join(SCRIPTS_DIR, name)
    print(f"\n{'='*60}")
    print(f"Running {name}...")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True,
        encoding='utf-8', errors='replace',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        timeout=300
    )
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        print(f"  stderr: {result.stderr[:500]}")
        return None
    stdout = result.stdout or ''
    print(f"  OK ({len(stdout)} chars)")
    return stdout


def extract_numbers(text, pattern):
    """Extract first number after a regex pattern."""
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, IndexError):
            return None
    return None


def parse_exp09(output):
    """Parse exp_09 conservative RBF binding results."""
    results = {
        'experiment': 'exp_09_conservative_rbf',
        'title': 'Conservative RBF Binding',
        'timestamp': datetime.now().isoformat(),
    }

    # Q1: RBF vs unbound
    xi_vu = extract_numbers(output, r'RBF vs Unbound.*?[ξx]\s*=\s*([\d.]+)')
    p_vu = extract_numbers(output, r'RBF vs Unbound.*?p\s*=\s*([\d.eE+-]+)')
    if xi_vu is None:
        xi_vu = extract_numbers(output, r'Q1.*?excess.*?([\d.]+)')
    if p_vu is None:
        p_vu = extract_numbers(output, r'Q1.*?p\s*[=:]\s*([\d.eE+-]+)')

    # Q2: Nonlinear vs linear
    xi_vl = extract_numbers(output, r'[Nn]onlinear.*?[ξx]\s*=\s*([\d.]+)')
    p_vl = extract_numbers(output, r'[Nn]onlinear.*?p\s*=\s*([\d.eE+-]+)')
    if xi_vl is None:
        xi_vl = extract_numbers(output, r'Q2.*?excess.*?([\d.]+)')
    if p_vl is None:
        p_vl = extract_numbers(output, r'Q2.*?p\s*[=:]\s*([\d.eE+-]+)')

    # Q4: Balance operator
    xi_measured = extract_numbers(output, r'[Bb]alance.*?[Ξξ]\s*[=:≈]\s*([\d.]+)')

    # Q5: Conservation
    cons = extract_numbers(output, r'S_final/S_initial\s*[=:]\s*([\d.]+)')

    results['main_results'] = {
        'rbf_vs_unbound': {'xi_excess': xi_vu, 'p_value': p_vu},
        'nonlinear_vs_linear': {'xi_excess': xi_vl, 'p_value': p_vl},
        'balance_operator': xi_measured,
        'conservation_ratio': cons,
    }

    # Also store raw output for full traceability
    results['raw_output'] = output
    return results


def parse_exp10(output):
    """Parse exp_10 thermodynamic cascade results."""
    results = {
        'experiment': 'exp_10_thermodynamic_cascade',
        'title': 'Multi-generation Thermodynamic Cascade',
        'timestamp': datetime.now().isoformat(),
    }

    single_xi = extract_numbers(output, r'[Ss]ingle.*?[ξx]\s*[=:]\s*([\d.]+)')
    cascade_xi = extract_numbers(output, r'[Cc]ascade.*?[ξx]\s*[=:]\s*([\d.]+)')
    ratio = extract_numbers(output, r'(\d+)[×x]\s*(?:more|amplification|cascade)')
    if ratio is None:
        ratio = extract_numbers(output, r'[Rr]atio.*?(\d+)')
    p_val = extract_numbers(output, r'p\s*[=:]\s*([\d.eE+-]+)')
    lifespan = extract_numbers(output, r'[Ll]ifespan.*?([\d.]+)\s*gen')
    if lifespan is None:
        lifespan = extract_numbers(output, r'([\d.]+)\s*gen')

    results['main_results'] = {
        'single_xi': single_xi,
        'cascade_xi': cascade_xi,
        'amplification_ratio': ratio,
        'p_value': p_val,
        'mean_lifespan_generations': lifespan,
    }
    results['raw_output'] = output
    return results


def parse_exp11(output):
    """Parse exp_11 time computation results."""
    results = {
        'experiment': 'exp_11_time_computation',
        'title': 'Time as Computational Density',
        'timestamp': datetime.now().isoformat(),
    }

    dense_xi = extract_numbers(output, r'[Dd]ense.*?[ξx].*?([\d.]+)')
    sparse_xi = extract_numbers(output, r'[Ss]parse.*?[ξx].*?([\d.]+)')
    ratio = extract_numbers(output, r'(\d+)[×x]')
    p_val = extract_numbers(output, r'p\s*[=:]\s*([\d.eE+-]+)')

    results['main_results'] = {
        'dense_xi_per_tick': dense_xi,
        'sparse_xi_per_tick': sparse_xi,
        'ratio': ratio,
        'p_value': p_val,
    }
    results['raw_output'] = output
    return results


def parse_exp12(output):
    """Parse exp_12 causal lag test results."""
    results = {
        'experiment': 'exp_12_causal_lag_test',
        'title': 'Causal Lag Hypothesis Test',
        'timestamp': datetime.now().isoformat(),
    }

    # Look for lag deviations
    lag0_dev = extract_numbers(output, r'[Ll]ag\s*=?\s*0.*?dev.*?([\d.]+)%')
    lag1_dev = extract_numbers(output, r'[Ll]ag\s*=?\s*1.*?dev.*?([\d.]+)%')
    lag2_dev = extract_numbers(output, r'[Ll]ag\s*=?\s*2.*?dev.*?([\d.]+)%')
    eff_ratio = extract_numbers(output, r'[Ee]ffective.*?ratio.*?([\d.]+)')
    ratio_error = extract_numbers(output, r'ratio.*?error.*?([\d.]+)%')

    # Tests passed
    tests = re.findall(r'(PASS|FAIL)', output, re.IGNORECASE)

    results['main_results'] = {
        'lag_deviations_pct': {'lag_0': lag0_dev, 'lag_1': lag1_dev, 'lag_2': lag2_dev},
        'effective_ratio': eff_ratio,
        'ratio_error_pct': ratio_error,
        'tests_passed': sum(1 for t in tests if t.upper() == 'PASS'),
        'tests_total': len(tests) if tests else None,
    }
    results['raw_output'] = output
    return results


def parse_exp13(output):
    """Parse exp_13 causal falsification results."""
    results = {
        'experiment': 'exp_13_causal_falsification',
        'title': 'Causal Lag Falsification Suite',
        'timestamp': datetime.now().isoformat(),
    }

    tests_passed = extract_numbers(output, r'(\d+)/6')
    best_lag = extract_numbers(output, r'[Oo]ptimal.*?lag.*?([\d.]+)')
    lag1_wins = extract_numbers(output, r'lag.*?1.*?wins.*?(\d+)')

    # Collect all ✓/✗ or PASS/FAIL
    checks = re.findall(r'[✓✗]|PASS|FAIL', output)

    results['main_results'] = {
        'tests_passed': int(tests_passed) if tests_passed else None,
        'tests_total': 6,
        'optimal_lag': best_lag,
        'lag1_win_count': int(lag1_wins) if lag1_wins else None,
        'check_marks': len([c for c in checks if c in ('✓', 'PASS')]),
    }
    results['raw_output'] = output
    return results


def parse_exp14(output):
    """Parse exp_14 PAC conservation results."""
    results = {
        'experiment': 'exp_14_pac_conservation',
        'title': 'PAC Ratio vs Magnitude Conservation',
        'timestamp': datetime.now().isoformat(),
    }

    ratio = extract_numbers(output, r'A/\(A\+[ξx]\)\s*[=:]\s*([\d.]+)')
    if ratio is None:
        ratio = extract_numbers(output, r'ratio.*?(0\.4\d+)')
    slope = extract_numbers(output, r'[Ss]lope.*?([\d.]+)')
    r_sq = extract_numbers(output, r'R[²2]\s*[=:]\s*([\d.]+)')
    shuffle_change = extract_numbers(output, r'[Ss]huffle.*?(\d+\.?\d*)%')
    cv_I = extract_numbers(output, r'cv.*?I.*?([\d.]+)')

    results['main_results'] = {
        'A_over_total': ratio,
        'ln_phi': 0.4812,
        'regression_slope': slope,
        'R_squared': r_sq,
        'shuffle_change_pct': shuffle_change,
        'cv_I_total': cv_I,
    }
    results['raw_output'] = output
    return results


def save_result(results, exp_name):
    """Save results JSON, stripping raw_output to a separate file."""
    # Save full version (with raw output)
    fname = f"{exp_name}_{TIMESTAMP}.json"
    path = os.path.join(RESULTS_DIR, fname)

    # For the main JSON, truncate raw_output to first 2000 chars to keep manageable
    save_data = dict(results)
    if 'raw_output' in save_data:
        raw = save_data.pop('raw_output')
        save_data['raw_output_chars'] = len(raw)
        save_data['raw_output_first_2000'] = raw[:2000]

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"  Saved: {fname}")
    return path


EXPERIMENTS = [
    ('exp_09_conservative_rbf.py', parse_exp09, 'exp_09_conservative_rbf'),
    ('exp_10_thermodynamic_cascade.py', parse_exp10, 'exp_10_thermodynamic_cascade'),
    ('exp_11_time_computation.py', parse_exp11, 'exp_11_time_computation'),
    ('exp_12_causal_lag_test.py', parse_exp12, 'exp_12_causal_lag_test'),
    ('exp_13_causal_falsification.py', parse_exp13, 'exp_13_causal_falsification'),
    ('exp_14_pac_conservation.py', parse_exp14, 'exp_14_pac_conservation'),
]


if __name__ == '__main__':
    print(f"Capturing results for exp_09–14 at {TIMESTAMP}")
    print(f"Results dir: {RESULTS_DIR}")

    succeeded = 0
    failed = 0

    for script_name, parser, result_name in EXPERIMENTS:
        output = run_script(script_name)
        if output is None:
            failed += 1
            continue

        results = parser(output)
        save_result(results, result_name)
        succeeded += 1

    print(f"\n{'='*60}")
    print(f"Done: {succeeded} succeeded, {failed} failed")
    print(f"Results in: {os.path.abspath(RESULTS_DIR)}")

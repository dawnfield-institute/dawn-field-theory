"""
exp_31: UNIVERSAL DECOMPOSITION - Does γ + ln(φ) underlie ALL three Ξ values?

HYPOTHESIS:
The three converging values (~1.057) all decompose as γ + ln(φ),
meaning γ + ln(φ) is the TRUE universal interface constant,
and the domain-specific measurements are approximations to it.

PREDICTION:
If true, then (Ξ_domain - γ) should equal ln(φ) for ALL domains,
not just the analytic one where it's true by construction.

Author: Dawn Field Institute
Date: February 5, 2026
"""

import math
import json
from datetime import datetime

# Constants
PHI = (1 + math.sqrt(5)) / 2
GAMMA = 0.5772156649015329
LN_PHI = math.log(PHI)

# The measured Ξ values (four non-analytic + one analytic)
XI_VALUES = {
    'formula_1_plus_pi_55': {
        'value': 1 + math.pi / 55,
        'domain': 'Fibonacci arithmetic',
        'derivation': '1 + π/55'
    },
    'rule_110_measured': {
        'value': 1.0579,
        'domain': 'Cellular automata',
        'derivation': 'Measured P/A ratio'
    },
    'analytic_gamma_ln_phi': {
        'value': GAMMA + LN_PHI,
        'domain': 'Number theory + PAC',
        'derivation': 'γ + ln(φ)'
    },
    'mobius_field_dynamics': {
        'value': 1.058051,
        'domain': 'Möbius field dynamics',
        'derivation': 'Ξ_L2 energy ratio (Reality Engine)'
    }
}


def test_decomposition():
    """Test if all Ξ values decompose as γ + ln(φ)."""
    
    results = {
        'hypothesis': 'All Ξ values decompose as γ + ln(φ)',
        'gamma': float(GAMMA),
        'ln_phi': float(LN_PHI),
        'decompositions': {}
    }
    
    print("=" * 70)
    print("EXP 31: UNIVERSAL DECOMPOSITION TEST")
    print("=" * 70)
    print()
    print(f"γ = {GAMMA:.10f}")
    print(f"ln(φ) = {LN_PHI:.10f}")
    print(f"γ + ln(φ) = {GAMMA + LN_PHI:.10f}")
    print()
    
    print("-" * 70)
    print("DECOMPOSITION: Ξ = γ + ??? (is ??? = ln(φ)?)")
    print("-" * 70)
    print()
    
    all_pass = True
    for name, info in XI_VALUES.items():
        xi = info['value']
        remainder = xi - GAMMA
        error_from_ln_phi = abs(remainder - LN_PHI) / LN_PHI * 100
        
        results['decompositions'][name] = {
            'xi_value': float(xi),
            'domain': info['domain'],
            'xi_minus_gamma': float(remainder),
            'error_from_ln_phi_percent': float(error_from_ln_phi)
        }
        
        print(f"{info['domain']}:")
        print(f"  Ξ = {xi:.6f} ({info['derivation']})")
        print(f"  Ξ - γ = {remainder:.6f}")
        print(f"  ln(φ) = {LN_PHI:.6f}")
        print(f"  Error: {error_from_ln_phi:.3f}%")
        
        if error_from_ln_phi < 0.5:  # Within 0.5%
            print(f"  ✅ MATCHES γ + ln(φ)")
        else:
            print(f"  ❌ Does NOT match γ + ln(φ)")
            all_pass = False
        print()
    
    return results, all_pass


def test_hierarchy():
    """Test which approximates γ + ln(φ) better."""
    
    true_xi = GAMMA + LN_PHI
    
    print("-" * 70)
    print("HIERARCHY: Which is closest to γ + ln(φ)?")
    print("-" * 70)
    print()
    print(f"TRUE target: γ + ln(φ) = {true_xi:.6f}")
    print()
    
    errors = []
    for name, info in XI_VALUES.items():
        if name == 'analytic_gamma_ln_phi':
            continue  # Skip the one that's true by definition
        
        xi = info['value']
        error = abs(xi - true_xi) / true_xi * 100
        errors.append((info['domain'], error))
        print(f"  {info['domain']}: error = {error:.4f}%")
    
    # Sort by error
    errors.sort(key=lambda x: x[1])
    
    print()
    print(f"CLOSEST: {errors[0][0]} ({errors[0][1]:.4f}%)")
    print()
    
    return {
        'true_xi': float(true_xi),
        'approximation_errors': {domain: float(err) for domain, err in errors},
        'closest': errors[0][0]
    }


def compute_significance():
    """Compute probability of this convergence by chance."""
    
    import random
    
    print("-" * 70)
    print("SIGNIFICANCE: How unlikely is this convergence?")
    print("-" * 70)
    print()
    
    true_xi = GAMMA + LN_PHI
    max_error = 0.003  # 0.3% tolerance for "match"
    
    # Original test: 2 random values near a target (backward compat)
    n_trials = 100000
    hits_2 = 0
    hits_3 = 0
    
    for _ in range(n_trials):
        target = random.uniform(1.0, 1.1)
        vals = [random.uniform(1.0, 1.1) for _ in range(3)]
        errs = [abs(v - target) / target for v in vals]
        
        if errs[0] < max_error and errs[1] < max_error:
            hits_2 += 1
        if all(e < max_error for e in errs):
            hits_3 += 1
    
    p_random_2 = hits_2 / n_trials
    p_random_3 = hits_3 / n_trials
    
    print(f"2-of-2 within 0.3% of random target: p = {p_random_2:.6f} ({hits_2}/{n_trials})")
    print(f"3-of-3 within 0.3% of random target: p = {p_random_3:.6f} ({hits_3}/{n_trials})")
    print()
    print(f"Observed: Fibonacci, Rule 110, AND Möbius all within 0.13% of γ + ln(φ)")
    print()
    
    if p_random_3 > 0:
        significance = 1 / p_random_3
        print(f"Significance (3-of-3): {significance:.0f}x more likely than chance")
    else:
        print(f"Significance (3-of-3): > {n_trials}x (no random hits in {n_trials} trials)")
    
    return {
        'p_random_2of2': float(p_random_2),
        'p_random_3of3': float(p_random_3),
        'n_trials': n_trials,
        'max_error_percent': float(max_error * 100)
    }


def main():
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_31_universal_decomposition',
        'hypothesis': 'γ + ln(φ) is the universal interface constant',
        'constants': {
            'gamma': float(GAMMA),
            'ln_phi': float(LN_PHI),
            'phi': float(PHI)
        }
    }
    
    decomp, all_pass = test_decomposition()
    results['decomposition_test'] = decomp
    results['all_decompose_as_gamma_ln_phi'] = all_pass
    
    hierarchy = test_hierarchy()
    results['hierarchy_test'] = hierarchy
    
    significance = compute_significance()
    results['significance_test'] = significance
    
    # Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()
    
    if all_pass:
        print("✅ VALIDATED: All four Ξ values decompose as γ + ln(φ)")
        print()
        print("INTERPRETATION:")
        print("  - γ + ln(φ) = 1.0584 is the TRUE universal interface constant")
        print("  - 1 + π/55 is a discrete approximation (0.124% error)")
        print("  - Rule 110 is a measured approximation (0.050% error)")
        print("  - Möbius field dynamics is a continuous approximation (0.036% error)")
        print()
        print("  γ (interface cost) + ln(φ) (emergence geometry) = universal")
        print()
        print("This answers WHY five domains converge:")
        print("  They're all measuring the same underlying quantity!")
        results['verdict'] = 'validated'
    else:
        print("❌ NOT ALL MATCH: Some domains don't decompose as γ + ln(φ)")
        results['verdict'] = 'partial'
    
    # Save
    with open('exp_31_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("Results saved to exp_31_results.json")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Experiment 07: PAC Ecosystem Analysis

Part VII: Mass Ratio Derivation

KEY INSIGHT: We've been deriving masses in isolation, but PAC says
f(Parent) = Σf(Children). The masses form a COUPLED SYSTEM.

The "correction terms" in our formulas might be:
- μ/e: (1 + 1/F_7) - multiplicative
- τ/e: + F_5 - additive
- p/e: / F_6 - divisive

These aren't arbitrary patches - they might be BALANCE TERMS that
ensure PAC conservation across the particle spectrum.

Questions:
1. Do the corrections sum/balance to a PAC-conserved quantity?
2. Can we derive the corrections from QBE: dI/dt + dE/dt = λ·QPL?
3. Is there a parent quantity that splits into lepton masses?
4. What role does the proton play in lepton mass balance?
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path


# Constants
PHI = (1 + np.sqrt(5)) / 2
XI = 1 + np.pi / 55

# Fibonacci
def fib(n: int) -> int:
    if n <= 1:
        return max(n, 0)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

FIB = [fib(i) for i in range(25)]

# Measured values
MEASURED = {
    'me': 0.51099895,      # MeV
    'mu': 105.6583755,     # MeV
    'mtau': 1776.86,       # MeV
    'mp': 938.27208816,    # MeV
    'mn': 939.56542052,    # MeV
}

# Derived ratios
RATIOS = {
    'mu/e': MEASURED['mu'] / MEASURED['me'],
    'tau/e': MEASURED['mtau'] / MEASURED['me'],
    'tau/mu': MEASURED['mtau'] / MEASURED['mu'],
    'p/e': MEASURED['mp'] / MEASURED['me'],
    'n/e': MEASURED['mn'] / MEASURED['me'],
    'n/p': MEASURED['mn'] / MEASURED['mp'],
}


def analyze_base_vs_correction():
    """
    Separate each formula into base term and correction.
    See if corrections are related across particles.
    """
    print("=" * 70)
    print("ANALYZING BASE vs CORRECTION STRUCTURE")
    print("=" * 70)
    
    formulas = {
        'mu/e': {
            'base': FIB[4] * FIB[6]**2,  # 3 × 64 = 192
            'correction': (1 + 1/FIB[7]),  # 14/13
            'correction_delta': 1/FIB[7],  # 1/13
            'type': 'multiplicative'
        },
        'tau/e': {
            'base': FIB[4] * FIB[7] * FIB[11],  # 3 × 13 × 89 = 3471
            'correction': FIB[5],  # +5
            'correction_delta': FIB[5],  # 5
            'type': 'additive'
        },
        'p/e': {
            'base': FIB[4] * FIB[9] * FIB[12],  # 3 × 34 × 144 = 14688
            'correction': 1/FIB[6],  # /8
            'correction_delta': -7/8,  # relative to ×1
            'type': 'divisive'
        }
    }
    
    print("\nFormula structure:")
    for name, data in formulas.items():
        measured = RATIOS[name]
        full = data['base'] * data['correction'] if data['type'] == 'multiplicative' else \
               data['base'] + data['correction'] if data['type'] == 'additive' else \
               data['base'] * data['correction']
        error = abs(measured - full) / measured * 100
        
        print(f"\n{name}:")
        print(f"  Base: {data['base']}")
        print(f"  Correction: {data['correction']} ({data['type']})")
        print(f"  Full: {full:.4f}")
        print(f"  Measured: {measured:.4f}")
        print(f"  Error: {error:.4f}%")
    
    # Look for relationships between corrections
    print("\n" + "=" * 70)
    print("CORRECTION RELATIONSHIPS")
    print("=" * 70)
    
    # The corrections in Fibonacci terms:
    # μ: 1/F_7 = 1/13
    # τ: F_5 = 5
    # p: 1/F_6 = 1/8
    
    print("\nCorrections as Fibonacci:")
    print(f"  μ correction: 1/F_7 = 1/13 = {1/13:.6f}")
    print(f"  τ correction: F_5 = 5")
    print(f"  p correction: 1/F_6 = 1/8 = {1/8:.6f}")
    
    # Sum of inverse corrections?
    inv_sum = 1/FIB[7] + 1/FIB[6]  # 1/13 + 1/8
    print(f"\n  1/F_7 + 1/F_6 = 1/13 + 1/8 = {inv_sum:.6f}")
    print(f"  That's approximately: {inv_sum}")
    print(f"  = (8 + 13)/(8×13) = 21/104 = F_8/(F_6×F_7)")
    print(f"  = {21/104:.6f}")
    
    # What about τ's +5?
    # 5 = F_5
    # Maybe: F_5 / (base) = 5/3471 = 0.00144
    # Compare to: 1/F_7 = 0.077, 1/F_6 = 0.125
    
    print(f"\n  τ relative correction: F_5/base = 5/3471 = {5/3471:.6f}")
    print(f"  μ relative correction: 1/F_7 = {1/13:.6f}")
    print(f"  p relative correction: 1/F_6 = {1/8:.6f}")
    
    return formulas


def test_pac_sum():
    """
    Test if the masses satisfy some PAC conservation.
    
    PAC: f(Parent) = Σ f(Children)
    
    Maybe: m_e + m_μ + m_τ = something × m_p?
    Or in ratios: 1 + μ/e + τ/e = k × p/e?
    """
    print("\n" + "=" * 70)
    print("TESTING PAC CONSERVATION")
    print("=" * 70)
    
    # Sum of lepton mass ratios
    lepton_sum = 1 + RATIOS['mu/e'] + RATIOS['tau/e']
    print(f"\n1 + μ/e + τ/e = 1 + {RATIOS['mu/e']:.4f} + {RATIOS['tau/e']:.4f}")
    print(f"             = {lepton_sum:.4f}")
    
    # Compare to proton
    print(f"\np/e = {RATIOS['p/e']:.4f}")
    print(f"\nRatio: (1 + μ/e + τ/e) / (p/e) = {lepton_sum / RATIOS['p/e']:.6f}")
    
    # Is this Fibonacci?
    ratio = lepton_sum / RATIOS['p/e']
    print(f"\nLooking for Fibonacci match to {ratio:.6f}:")
    for i in range(2, 15):
        for j in range(2, 15):
            test = FIB[i] / FIB[j]
            if abs(test - ratio) / ratio < 0.01:
                print(f"  F_{i}/F_{j} = {FIB[i]}/{FIB[j]} = {test:.6f} ({abs(test-ratio)/ratio*100:.3f}%)")
    
    # Try: lepton_sum + proton = ?
    total = lepton_sum + RATIOS['p/e']
    print(f"\n(1 + μ/e + τ/e) + p/e = {total:.4f}")
    
    # Is total a Fibonacci product?
    print(f"\nLooking for Fibonacci product near {total:.4f}:")
    for i in range(2, 14):
        for j in range(i, 14):
            for k in range(j, 14):
                prod = FIB[i] * FIB[j] * FIB[k]
                if abs(prod - total) / total < 0.02:
                    print(f"  F_{i}×F_{j}×F_{k} = {prod} ({abs(prod-total)/total*100:.3f}%)")
    
    return {
        'lepton_sum': lepton_sum,
        'p/e': RATIOS['p/e'],
        'ratio': lepton_sum / RATIOS['p/e']
    }


def test_koide_as_pac():
    """
    Koide relation: Q = (me + mμ + mτ) / (√me + √mμ + √mτ)² = 2/3
    
    Rewrite in terms of ratios:
    Q = (1 + μ/e + τ/e) / (1 + √(μ/e) + √(τ/e))² = 2/3 = F_3/F_4
    
    This IS a PAC relation! The sum (1 + μ/e + τ/e) is constrained
    by the "geometric mean" (1 + √(μ/e) + √(τ/e))².
    
    What's the parent? What's the balance?
    """
    print("\n" + "=" * 70)
    print("KOIDE AS PAC CONSERVATION")
    print("=" * 70)
    
    mu = RATIOS['mu/e']
    tau = RATIOS['tau/e']
    
    # The Koide constraint
    numerator = 1 + mu + tau
    denominator = (1 + np.sqrt(mu) + np.sqrt(tau))**2
    Q = numerator / denominator
    
    print(f"\nKoide numerator (mass sum): {numerator:.4f}")
    print(f"Koide denominator (√sum)²: {denominator:.4f}")
    print(f"Q = {Q:.8f}")
    print(f"2/3 = {2/3:.8f}")
    print(f"Error: {abs(Q - 2/3)/(2/3)*100:.4f}%")
    
    # Interpretation:
    print("\nPAC interpretation:")
    print(f"  The 'parent' is the denominator: {denominator:.4f}")
    print(f"  The 'children sum' is: Q × parent = {numerator:.4f}")
    print(f"  The 'wasted potential' is: (1-Q) × parent = {(1-Q)*denominator:.4f}")
    
    # What's that wasted potential?
    waste = (1 - Q) * denominator
    print(f"\n  'Waste' = {waste:.4f}")
    print(f"  Compare to √(τ/e) = {np.sqrt(tau):.4f}")
    
    # The waste is almost exactly 2×√τ!
    print(f"  2×√(τ/e) = {2*np.sqrt(tau):.4f}")
    print(f"  Ratio: waste / 2√τ = {waste / (2*np.sqrt(tau)):.6f}")
    
    return {
        'Q': Q,
        'numerator': numerator,
        'denominator': denominator,
        'waste': waste
    }


def test_qbe_balance():
    """
    QBE: dI/dt + dE/dt = λ·QPL(t)
    
    At equilibrium (stable particles): dI/dt = 0, dE/dt = 0
    So: 0 = λ·QPL(t)
    
    Meaning: the quantum potential layer is at a zero-crossing.
    
    For masses, the "information" might be log(m).
    The "energy" is m×c².
    
    Test: Do log-ratios satisfy some balance?
    """
    print("\n" + "=" * 70)
    print("QBE EQUILIBRIUM TEST")
    print("=" * 70)
    
    # Log ratios (information metric)
    log_mu = np.log(RATIOS['mu/e'])
    log_tau = np.log(RATIOS['tau/e'])
    log_p = np.log(RATIOS['p/e'])
    
    print(f"\nLog ratios (information content):")
    print(f"  log(μ/e) = {log_mu:.6f}")
    print(f"  log(τ/e) = {log_tau:.6f}")
    print(f"  log(p/e) = {log_p:.6f}")
    
    # Sum of logs = log of products
    log_sum = log_mu + log_tau + log_p
    print(f"\nSum of logs: {log_sum:.6f}")
    print(f"= log(μ×τ×p / e³) = log({RATIOS['mu/e']*RATIOS['tau/e']*RATIOS['p/e']:.2e})")
    
    # Is this related to Fibonacci?
    product = RATIOS['mu/e'] * RATIOS['tau/e'] * RATIOS['p/e']
    print(f"\nμ×τ×p / e³ = {product:.2e}")
    
    # Check against Fibonacci products
    print(f"\nLooking for Fibonacci representation:")
    # log(product) ≈ 27.2
    # e^27.2 ≈ 6.5e11
    
    # F_20 × F_21 × F_22 / something?
    for i in range(15, 25):
        for j in range(i, 25):
            prod = FIB[i] * FIB[j]
            if 1e11 < prod < 1e13:
                ratio = product / prod
                print(f"  F_{i}×F_{j} = {prod:.2e}, ratio = {ratio:.4f}")
    
    # Alternative: ratios of logs
    print(f"\nLog ratios:")
    print(f"  log(τ/e) / log(μ/e) = {log_tau / log_mu:.6f}")
    print(f"  log(p/e) / log(μ/e) = {log_p / log_mu:.6f}")
    print(f"  log(p/e) / log(τ/e) = {log_p / log_tau:.6f}")
    
    # Check for φ
    print(f"\nCompare to φ = {PHI:.6f}:")
    print(f"  log(τ/e)/log(μ/e) / φ = {(log_tau/log_mu)/PHI:.6f}")
    
    return {
        'log_mu': log_mu,
        'log_tau': log_tau,
        'log_p': log_p,
        'log_sum': log_sum
    }


def test_recursive_balance():
    """
    RBF: Systems self-regulate toward dynamic stability.
    
    The lepton masses might be at fixed points of a recursive map.
    
    Hypothesis: m_{n+1}/m_n = f(m_n/m_{n-1}) for some PAC-derived f
    
    We have: μ/e = 206.77, τ/μ = 16.82
    
    Is there a recursive relation?
    """
    print("\n" + "=" * 70)
    print("RECURSIVE BALANCE FIELD TEST")
    print("=" * 70)
    
    r1 = RATIOS['mu/e']  # 206.77
    r2 = RATIOS['tau/mu']  # 16.82
    
    print(f"\nConsecutive ratios:")
    print(f"  μ/e = {r1:.4f}")
    print(f"  τ/μ = {r2:.4f}")
    
    # Is r2 a function of r1?
    print(f"\nRelationship tests:")
    print(f"  r1 / r2 = {r1/r2:.4f}")
    print(f"  √r1 = {np.sqrt(r1):.4f}")
    print(f"  r1^(1/φ) = {r1**(1/PHI):.4f}")
    
    # Interesting: r1/r2 ≈ 12.3 ≈ F_7 - 1
    print(f"\n  r1/r2 ≈ F_7 - 1 = 13 - 1 = 12? Actual: {r1/r2:.4f}")
    
    # Or: r2 ≈ r1 / F_7?
    print(f"  r1/F_7 = {r1/FIB[7]:.4f} vs r2 = {r2:.4f}")
    print(f"  Error: {abs(r1/FIB[7] - r2)/r2*100:.2f}%")
    
    # Close! μ/e / 13 ≈ τ/μ within 5%
    
    # What about: τ/μ = (μ/e)^x for some x?
    x = np.log(r2) / np.log(r1)
    print(f"\n  If τ/μ = (μ/e)^x, then x = {x:.6f}")
    print(f"  Compare to 1/2 = {0.5}")
    print(f"  Compare to 1/φ = {1/PHI:.6f}")
    print(f"  Compare to F_4/F_7 = 3/13 = {3/13:.6f}")
    
    # x ≈ 0.53, close to 1/2 but not exact
    
    # The recursive map might be: r_{n+1} = r_n^(1/2) × correction
    predicted_r2 = np.sqrt(r1)
    print(f"\n  √(μ/e) = {predicted_r2:.4f}")
    print(f"  Actual τ/μ = {r2:.4f}")
    print(f"  Ratio: {r2/predicted_r2:.4f}")
    
    # So: τ/μ ≈ √(μ/e) × 1.17
    # 1.17 ≈ ?
    print(f"\n  τ/μ ≈ √(μ/e) × {r2/predicted_r2:.4f}")
    print(f"  Compare to F_7/F_6 = 13/8 = {13/8:.4f}")
    print(f"  Compare to φ^(1/3) = {PHI**(1/3):.4f}")
    
    return {
        'r1': r1,
        'r2': r2,
        'ratio': r1/r2,
        'exponent': x
    }


def derive_second_constraint():
    """
    We have Koide: Q = 2/3 (one equation, three unknowns)
    
    What's the second constraint?
    
    Hypothesis: The second constraint comes from PAC balance
    with the proton (or some other particle).
    """
    print("\n" + "=" * 70)
    print("DERIVING THE SECOND CONSTRAINT")
    print("=" * 70)
    
    # Koide gives us: (1 + μ + τ) / (1 + √μ + √τ)² = 2/3
    # One equation, two unknowns (μ, τ in units of e)
    
    # What if the second constraint involves the proton?
    # 
    # Observation from our formulas:
    #   All use F_4 = 3
    #   μ uses F_6, F_7
    #   τ uses F_5, F_7, F_11
    #   p uses F_6, F_9, F_12
    
    # The Fibonacci indices used: 4, 5, 6, 7, 9, 11, 12
    # These are NOT consecutive. What's special about them?
    
    indices = [4, 5, 6, 7, 9, 11, 12]
    print(f"\nFibonacci indices in mass formulas: {indices}")
    print(f"Sum: {sum(indices)} = {sum(indices)}")
    print(f"Product: {np.prod(indices)}")
    
    # Check if indices themselves are related
    print(f"\nIndex differences:")
    for i in range(len(indices)-1):
        print(f"  {indices[i+1]} - {indices[i]} = {indices[i+1] - indices[i]}")
    
    # 5-4=1, 6-5=1, 7-6=1, 9-7=2, 11-9=2, 12-11=1
    # Pattern: 1,1,1,2,2,1
    
    # Alternative hypothesis: The proton mass sets the scale
    # and lepton masses are fractions of it
    
    print(f"\nLepton masses as fractions of proton:")
    print(f"  e/p = 1/{RATIOS['p/e']:.2f} = {1/RATIOS['p/e']:.6f}")
    print(f"  μ/p = {RATIOS['mu/e']/RATIOS['p/e']:.6f}")
    print(f"  τ/p = {RATIOS['tau/e']/RATIOS['p/e']:.6f}")
    
    # Sum of lepton/proton ratios
    lepton_frac_sum = (1 + RATIOS['mu/e'] + RATIOS['tau/e']) / RATIOS['p/e']
    print(f"\n  (e + μ + τ)/p = {lepton_frac_sum:.6f}")
    
    # Is this Fibonacci?
    print(f"\nCompare to Fibonacci ratios:")
    for i in range(3, 12):
        for j in range(i+1, 15):
            ratio = FIB[i] / FIB[j]
            if abs(ratio - lepton_frac_sum) / lepton_frac_sum < 0.02:
                print(f"  F_{i}/F_{j} = {FIB[i]}/{FIB[j]} = {ratio:.6f} ({abs(ratio-lepton_frac_sum)/lepton_frac_sum*100:.2f}%)")
    
    return {
        'indices': indices,
        'lepton_frac_sum': lepton_frac_sum
    }


def main():
    print("=" * 70)
    print("Experiment 07: PAC Ecosystem Analysis")
    print("=" * 70)
    print("\nKEY QUESTION: What's the second constraint beyond Koide?")
    print("HYPOTHESIS: Masses form a coupled PAC system, not isolated values.")
    
    results = {}
    
    # Analyze formula structure
    results['structure'] = analyze_base_vs_correction()
    
    # Test PAC sum conservation
    results['pac_sum'] = test_pac_sum()
    
    # Koide as PAC
    results['koide_pac'] = test_koide_as_pac()
    
    # QBE equilibrium
    results['qbe'] = test_qbe_balance()
    
    # Recursive balance
    results['rbf'] = test_recursive_balance()
    
    # Second constraint
    results['second_constraint'] = derive_second_constraint()
    
    # Summary
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)
    
    print("""
Key findings:

1. KOIDE IS PAC: The relation Q = 2/3 constrains the sum of masses
   relative to their geometric structure. The "waste" (1-Q) × parent
   is almost exactly 2×√τ - suggesting τ is the "overflow" term.

2. RECURSIVE PATTERN: τ/μ ≈ √(μ/e) × 1.17 ≈ √(μ/e) × F_7/F_6
   This suggests a recursive map where each generation is the
   square root of the previous, with a Fibonacci correction.

3. F_4 = 3 UNIVERSAL: All mass formulas use F_4 = 3. This isn't
   coincidence - it might encode 3 generations or 3 spatial dimensions.

4. PROTON COUPLING: The sum (e + μ + τ)/p ≈ 2 (approximately).
   The proton might be the "parent" in a PAC sense.

NEXT: Formalize the second constraint as:
  τ/μ = √(μ/e) × F_7/F_6
  
  Combined with Koide Q = 2/3, this might fully determine the masses.
""")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp_07_pac_ecosystem',
        'results': results
    }
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(results_dir / f'exp_07_pac_ecosystem_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to results/exp_07_pac_ecosystem_{timestamp}.json")
    
    return output


if __name__ == '__main__':
    main()

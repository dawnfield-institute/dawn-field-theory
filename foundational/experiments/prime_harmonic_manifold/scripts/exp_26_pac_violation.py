#!/usr/bin/env python3
"""
exp_26_pac_violation.py - PAC Necessity Proof

HYPOTHESIS: PAC is not just observed in structure - it's REQUIRED for structure.
            Breaking PAC should cause structural collapse.

PAC: Ψ(k) = Ψ(k+1) + Ψ(k+2)
     Solution: Ψ = φ^(-k) where φ = (1+√5)/2

TESTS:
1. PAC-compliant recursion → expect stable structure
2. Violate coefficient ratio → expect instability/collapse
3. Non-φ base → expect divergence or decay to uniformity
4. Random coefficients → expect chaos/no structure

METRICS:
- Structure depth: How many stable levels form
- Convergence: Does the system reach equilibrium?
- Information content: Entropy of final state
- Boundary count: Number of distinct phase boundaries

If PAC is necessary, violations should show:
- Reduced structure depth
- Failed convergence (blow-up or collapse)
- Either maximum entropy (uniformity) or zero entropy (collapse)
- No stable boundaries
"""

import numpy as np
from scipy import stats
from datetime import datetime
import json
import os

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

def pac_recursion(depth: int, base: float, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """
    Generate recursion sequence with Ψ(k) = a*Ψ(k+1) + b*Ψ(k+2)
    
    PAC-compliant: a=1, b=1, base=φ → stable
    Violation: any deviation → should show instability
    """
    # Start from bottom (large k) and work up
    sequence = np.zeros(depth)
    
    # Initialize with base^(-k) for bottom two levels
    sequence[-1] = base ** (-(depth-1))
    sequence[-2] = base ** (-(depth-2))
    
    # Propagate upward: Ψ(k) = a*Ψ(k+1) + b*Ψ(k+2)
    for k in range(depth - 3, -1, -1):
        sequence[k] = a * sequence[k+1] + b * sequence[k+2]
    
    return sequence


def measure_structure(sequence: np.ndarray) -> dict:
    """Measure structural properties of a recursion sequence."""
    
    # Normalize to avoid numerical issues
    seq_norm = sequence / (np.max(np.abs(sequence)) + 1e-15)
    
    # 1. Convergence check: does ratio stabilize?
    if len(sequence) > 2:
        ratios = sequence[:-1] / (sequence[1:] + 1e-15)
        ratio_variance = np.var(ratios[np.isfinite(ratios)])
        converged = ratio_variance < 0.01
        final_ratio = np.median(ratios[np.isfinite(ratios)]) if np.any(np.isfinite(ratios)) else np.nan
    else:
        converged = False
        ratio_variance = np.inf
        final_ratio = np.nan
    
    # 2. Stability: no blow-up or collapse
    max_val = np.max(np.abs(sequence))
    min_val = np.min(np.abs(sequence[sequence != 0])) if np.any(sequence != 0) else 0
    stable = np.isfinite(max_val) and max_val < 1e10 and min_val > 1e-15
    
    # 3. Information content (entropy of normalized sequence)
    seq_pos = np.abs(seq_norm) + 1e-15
    seq_prob = seq_pos / np.sum(seq_pos)
    entropy = -np.sum(seq_prob * np.log2(seq_prob + 1e-15))
    max_entropy = np.log2(len(sequence))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # 4. Structure depth: levels with meaningful distinction
    if stable:
        # Count how many levels have distinct values (not collapsed to same value)
        unique_rounded = len(np.unique(np.round(seq_norm, decimals=3)))
        structure_depth = unique_rounded
    else:
        structure_depth = 0
    
    # 5. Boundary count: phase transitions in the sequence
    if len(sequence) > 1:
        log_seq = np.log(np.abs(sequence) + 1e-15)
        diff = np.diff(log_seq)
        # Boundaries where growth rate changes significantly
        boundaries = np.sum(np.abs(np.diff(diff)) > 0.5) if len(diff) > 1 else 0
    else:
        boundaries = 0
    
    return {
        'converged': converged,
        'ratio_variance': float(ratio_variance),
        'final_ratio': float(final_ratio),
        'stable': stable,
        'max_value': float(max_val),
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'structure_depth': int(structure_depth),
        'boundaries': int(boundaries)
    }


def run_violation_test(name: str, depth: int, base: float, a: float, b: float) -> dict:
    """Run a single PAC violation test."""
    sequence = pac_recursion(depth, base, a, b)
    metrics = measure_structure(sequence)
    
    # Calculate deviation from PAC
    pac_base = PHI
    pac_a, pac_b = 1.0, 1.0
    base_deviation = abs(base - pac_base)
    coeff_deviation = abs(a - pac_a) + abs(b - pac_b)
    
    return {
        'name': name,
        'parameters': {
            'depth': depth,
            'base': float(base),
            'a': float(a),
            'b': float(b)
        },
        'pac_deviation': {
            'base_from_phi': float(base_deviation),
            'coeff_from_pac': float(coeff_deviation),
            'total': float(base_deviation + coeff_deviation)
        },
        'metrics': metrics,
        'sequence_sample': [float(x) for x in sequence[:10]]  # First 10 values
    }


def main():
    print("=" * 70)
    print("EXP 26: PAC VIOLATION - NECESSITY PROOF")
    print("=" * 70)
    print()
    print("HYPOTHESIS: Breaking PAC breaks structure")
    print("If true: PAC is necessary, not just observed")
    print()
    
    depth = 50  # Deep enough to see asymptotic behavior
    
    # Define test cases
    tests = [
        # Control: PAC-compliant
        ("PAC-compliant (φ, 1:1)", PHI, 1.0, 1.0),
        
        # Base violations (wrong ratio)
        ("Base = 1.5 (not φ)", 1.5, 1.0, 1.0),
        ("Base = 2.0 (doubling)", 2.0, 1.0, 1.0),
        ("Base = √2 ≈ 1.414", np.sqrt(2), 1.0, 1.0),
        ("Base = e ≈ 2.718", np.e, 1.0, 1.0),
        ("Base = π ≈ 3.14", np.pi, 1.0, 1.0),
        ("Base = 1.0 (no growth)", 1.0, 1.0, 1.0),
        
        # Coefficient violations (wrong recursion)
        ("Coeff 1:0.5 (half second)", PHI, 1.0, 0.5),
        ("Coeff 1:2 (double second)", PHI, 1.0, 2.0),
        ("Coeff 0.5:1 (half first)", PHI, 0.5, 1.0),
        ("Coeff 2:1 (double first)", PHI, 2.0, 1.0),
        ("Coeff 1:0 (no second term)", PHI, 1.0, 0.0),
        
        # Combined violations
        ("Base=2, Coeff 1:0.5", 2.0, 1.0, 0.5),
        ("Base=1.5, Coeff 2:1", 1.5, 2.0, 1.0),
        
        # Random coefficients
        ("Random coefficients", PHI, np.random.uniform(0.5, 2.0), np.random.uniform(0.5, 2.0)),
        
        # Near-PAC (small violations)
        ("Base = φ+0.01", PHI + 0.01, 1.0, 1.0),
        ("Base = φ-0.01", PHI - 0.01, 1.0, 1.0),
        ("Coeff 1.01:0.99", PHI, 1.01, 0.99),
    ]
    
    results = []
    
    print("-" * 70)
    print(f"{'Test Name':<30} {'Converged':<10} {'Stable':<8} {'Depth':<8} {'Entropy':<10}")
    print("-" * 70)
    
    for name, base, a, b in tests:
        result = run_violation_test(name, depth, base, a, b)
        results.append(result)
        
        m = result['metrics']
        print(f"{name:<30} {str(m['converged']):<10} {str(m['stable']):<8} {m['structure_depth']:<8} {m['normalized_entropy']:.4f}")
    
    print("-" * 70)
    print()
    
    # Analyze: correlation between PAC deviation and structure
    deviations = [r['pac_deviation']['total'] for r in results]
    depths = [r['metrics']['structure_depth'] for r in results]
    converged = [1 if r['metrics']['converged'] else 0 for r in results]
    stable = [1 if r['metrics']['stable'] else 0 for r in results]
    
    # Statistical tests
    depth_corr, depth_p = stats.spearmanr(deviations, depths)
    conv_corr, conv_p = stats.spearmanr(deviations, converged)
    
    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    print()
    print(f"Correlation: PAC deviation vs structure depth: r = {depth_corr:.4f}, p = {depth_p:.4f}")
    print(f"Correlation: PAC deviation vs convergence: r = {conv_corr:.4f}, p = {conv_p:.4f}")
    print()
    
    # Compare PAC-compliant vs all violations
    pac_result = results[0]  # First test is PAC-compliant
    violations = results[1:]
    
    pac_depth = pac_result['metrics']['structure_depth']
    pac_converged = pac_result['metrics']['converged']
    pac_stable = pac_result['metrics']['stable']
    
    violation_depths = [r['metrics']['structure_depth'] for r in violations]
    violation_converged = [r['metrics']['converged'] for r in violations]
    violation_stable = [r['metrics']['stable'] for r in violations]
    
    print("=" * 70)
    print("PAC-COMPLIANT vs VIOLATIONS")
    print("=" * 70)
    print()
    print(f"PAC-compliant structure depth: {pac_depth}")
    print(f"Mean violation structure depth: {np.mean(violation_depths):.2f}")
    print(f"Max violation structure depth: {np.max(violation_depths)}")
    print()
    print(f"PAC-compliant converged: {pac_converged}")
    print(f"Violations converged: {sum(violation_converged)}/{len(violation_converged)}")
    print()
    print(f"PAC-compliant stable: {pac_stable}")
    print(f"Violations stable: {sum(violation_stable)}/{len(violation_stable)}")
    print()
    
    # Key finding: does violating PAC reduce structure?
    depth_reduction = pac_depth - np.mean(violation_depths)
    
    print("=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    print()
    
    if depth_corr < -0.3 and depth_p < 0.05:
        print("✅ CONFIRMED: Greater PAC deviation → Less structure")
        print(f"   Correlation: r = {depth_corr:.4f}, p = {depth_p:.4f}")
        print(f"   Average depth reduction: {depth_reduction:.2f} levels")
        necessity_confirmed = True
    elif depth_reduction > 5:
        print("✅ CONFIRMED: PAC-compliant has significantly more structure")
        print(f"   PAC depth: {pac_depth}, Mean violation: {np.mean(violation_depths):.2f}")
        print(f"   Reduction: {depth_reduction:.2f} levels")
        necessity_confirmed = True
    else:
        print("⚠️  INCONCLUSIVE: Need more sensitive metrics")
        print(f"   Depth reduction: {depth_reduction:.2f}")
        necessity_confirmed = False
    
    print()
    
    # Specific violation analysis
    print("=" * 70)
    print("VIOLATION-SPECIFIC ANALYSIS")
    print("=" * 70)
    print()
    
    for r in results[1:]:
        name = r['name']
        dev = r['pac_deviation']['total']
        depth = r['metrics']['structure_depth']
        ratio = r['metrics']['final_ratio']
        
        if r['metrics']['converged']:
            ratio_desc = f"→ {ratio:.4f}"
            phi_error = abs(ratio - PHI) / PHI * 100
            if phi_error < 1:
                ratio_desc += f" (φ error: {phi_error:.2f}%)"
        else:
            ratio_desc = "DIVERGED"
        
        print(f"{name}")
        print(f"  PAC deviation: {dev:.4f}")
        print(f"  Structure depth: {depth}")
        print(f"  Ratio: {ratio_desc}")
        print()
    
    # Save results
    output = {
        'experiment': 'exp_26_pac_violation',
        'timestamp': datetime.now().isoformat(),
        'hypothesis': 'Breaking PAC breaks structure - PAC is necessary',
        'parameters': {
            'depth': depth,
            'num_tests': len(tests)
        },
        'results': results,
        'analysis': {
            'depth_correlation': float(depth_corr),
            'depth_p_value': float(depth_p),
            'convergence_correlation': float(conv_corr),
            'convergence_p_value': float(conv_p),
            'pac_depth': int(pac_depth),
            'mean_violation_depth': float(np.mean(violation_depths)),
            'depth_reduction': float(depth_reduction),
            'necessity_confirmed': bool(necessity_confirmed)
        },
        'conclusion': 'PAC deviation correlates with structure loss' if necessity_confirmed 
                     else 'Results inconclusive - need refined metrics'
    }
    
    # Save to results folder
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(results_dir, f'exp_26_pac_violation_{timestamp}.json')
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return output


if __name__ == '__main__':
    main()

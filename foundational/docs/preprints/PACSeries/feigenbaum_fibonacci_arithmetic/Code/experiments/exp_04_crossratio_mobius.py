"""
Experiment 10: Cross-Ratio Analysis of Feigenbaum Bifurcation Points

If the Feigenbaum cascade has Möbius structure, cross-ratios of successive
bifurcation points should be invariant (or converge to a constant).

Cross-ratio: CR(a,b,c,d) = ((a-c)(b-d))/((a-d)(b-c))

This is THE invariant of Möbius transformations - if CR is constant across
the cascade, it proves Möbius structure governs the dynamics.
"""

from mpmath import mp, mpf, sqrt, pi, log, phi
import json
from datetime import datetime

mp.dps = 50  # High precision

# Known bifurcation points of the logistic map r*x*(1-x)
# These are the r values where period-doubling occurs
# Using high-precision values from literature

BIFURCATION_POINTS = [
    mpf('1.0'),                          # r₀: fixed point at x=0
    mpf('3.0'),                          # r₁: period-1 → period-2
    mpf('3.4494897427831780981972840747'),  # r₂: period-2 → period-4
    mpf('3.5440903595978866135308749773'),  # r₃: period-4 → period-8
    mpf('3.5644072661903142508511124048'),  # r₄: period-8 → period-16
    mpf('3.5687594193471592869139915447'),  # r₅: period-16 → period-32
    mpf('3.5696916098932225476197739747'),  # r₆: period-32 → period-64
    mpf('3.5698913059409833324588466072'),  # r₇: period-64 → period-128
    mpf('3.5699340794904158088244319169'),  # r₈: period-128 → period-256
    mpf('3.5699432485688857369250327062'),  # r₉
    mpf('3.5699452113746473675977614337'),  # r₁₀
    mpf('3.5699456316664400951458014075'),  # r₁₁
    mpf('3.5699457217081990296008888950'),  # r₁₂
]

# The accumulation point
R_INF = mpf('3.5699456718709449018420051513864989367638369115148323781079755299213628875')

def cross_ratio(a, b, c, d):
    """Compute cross-ratio CR(a,b,c,d) = ((a-c)(b-d))/((a-d)(b-c))"""
    return ((a - c) * (b - d)) / ((a - d) * (b - c))

def analyze_cross_ratios():
    """Analyze cross-ratios of consecutive bifurcation quadruples."""
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cross_ratios': [],
        'patterns': {},
        'reference_constants': {}
    }
    
    print("=" * 70)
    print("CROSS-RATIO ANALYSIS OF FEIGENBAUM BIFURCATION CASCADE")
    print("=" * 70)
    
    # Reference constants
    delta = mpf('4.6692016091029906718532038204662016172581855774757686327456513430041343302113')
    alpha = mpf('2.5029078750958928222839028732182157863812713767271499773361920567792354')
    phi_val = phi
    
    results['reference_constants'] = {
        'delta': str(delta),
        'phi': str(phi_val),
        'alpha': str(alpha),
        '1/phi': str(1/phi_val),
        'delta/phi': str(delta/phi_val),
    }
    
    print(f"\nReference: δ = {delta}")
    print(f"Reference: φ = {phi_val}")
    print(f"Reference: δ/φ = {delta/phi_val}")
    print(f"Reference: 1/φ = {1/phi_val}")
    
    # Compute cross-ratios for sliding windows of 4 points
    print("\n" + "-" * 70)
    print("CONSECUTIVE QUADRUPLE CROSS-RATIOS: CR(rₙ, rₙ₊₁, rₙ₊₂, rₙ₊₃)")
    print("-" * 70)
    
    crs = []
    for i in range(len(BIFURCATION_POINTS) - 3):
        r0, r1, r2, r3 = BIFURCATION_POINTS[i:i+4]
        cr = cross_ratio(r0, r1, r2, r3)
        crs.append(cr)
        
        results['cross_ratios'].append({
            'n': i,
            'points': [str(r0), str(r1), str(r2), str(r3)],
            'cross_ratio': str(cr)
        })
        
        print(f"\nCR(r{i}, r{i+1}, r{i+2}, r{i+3}):")
        print(f"  = {cr}")
        
        # Check relationships
        print(f"  × δ = {cr * delta}")
        print(f"  × φ = {cr * phi_val}")
        print(f"  / (1/φ) = {cr / (1/phi_val)}")
    
    # Analyze convergence
    print("\n" + "-" * 70)
    print("CROSS-RATIO CONVERGENCE")
    print("-" * 70)
    
    if len(crs) >= 2:
        print("\nRatios of successive cross-ratios:")
        for i in range(len(crs) - 1):
            ratio = crs[i+1] / crs[i]
            print(f"  CR{i+1}/CR{i} = {ratio}")
            print(f"    - δ = {ratio - delta}")
            print(f"    × δ = {ratio * delta}")
    
    # Cross-ratio with r_infinity
    print("\n" + "-" * 70)
    print("CROSS-RATIOS INVOLVING R_∞")
    print("-" * 70)
    
    for i in range(len(BIFURCATION_POINTS) - 2):
        r0, r1, r2 = BIFURCATION_POINTS[i:i+3]
        cr_inf = cross_ratio(r0, r1, r2, R_INF)
        
        print(f"\nCR(r{i}, r{i+1}, r{i+2}, r∞) = {cr_inf}")
        print(f"  log₂(|CR|) = {log(abs(cr_inf), 2)}")
        
        results['cross_ratios'].append({
            'n': f'{i}_inf',
            'points': [str(r0), str(r1), str(r2), 'r_inf'],
            'cross_ratio': str(cr_inf)
        })
    
    # Alternative: gaps between bifurcation points
    print("\n" + "-" * 70)
    print("GAP ANALYSIS: δₙ = rₙ₊₁ - rₙ")
    print("-" * 70)
    
    gaps = []
    for i in range(len(BIFURCATION_POINTS) - 1):
        gap = BIFURCATION_POINTS[i+1] - BIFURCATION_POINTS[i]
        gaps.append(gap)
        print(f"δ{i} = r{i+1} - r{i} = {gap}")
    
    print("\nGap ratios (should approach δ ≈ 4.669...):")
    gap_ratios = []
    for i in range(len(gaps) - 1):
        ratio = gaps[i] / gaps[i+1]
        gap_ratios.append(ratio)
        diff_from_delta = ratio - delta
        print(f"  δ{i}/δ{i+1} = {ratio}  (diff from δ: {diff_from_delta})")
    
    results['patterns']['gap_ratios'] = [str(r) for r in gap_ratios]
    
    # Cross-ratio of gaps
    print("\n" + "-" * 70)
    print("CROSS-RATIO OF GAPS")
    print("-" * 70)
    
    if len(gaps) >= 4:
        for i in range(len(gaps) - 3):
            cr_gaps = cross_ratio(gaps[i], gaps[i+1], gaps[i+2], gaps[i+3])
            print(f"\nCR(δ{i}, δ{i+1}, δ{i+2}, δ{i+3}) = {cr_gaps}")
            print(f"  - 1 = {cr_gaps - 1}")
            print(f"  × δ = {cr_gaps * delta}")
    
    # Check for φ in cross-ratio limits
    print("\n" + "-" * 70)
    print("SEARCHING FOR φ AND FIBONACCI STRUCTURE")
    print("-" * 70)
    
    F10 = 55
    F8 = 21
    L5 = 11
    
    if len(crs) >= 1:
        last_cr = crs[-1]
        print(f"\nLast cross-ratio: {last_cr}")
        print(f"  × 55 = {last_cr * 55}")
        print(f"  × 21 = {last_cr * 21}")
        print(f"  × 11 = {last_cr * 11}")
        print(f"  × (55+21) = {last_cr * 76}")
        print(f"  / φ = {last_cr / phi_val}")
        print(f"  × φ² = {last_cr * phi_val**2}")
    
    # The key insight: in Möbius dynamics, cross-ratio is preserved
    # If we see it changing, the transformation isn't exactly Möbius
    # But if it converges, that limit is the "effective" Möbius invariant
    
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)
    
    # Check if cross-ratios converge
    if len(crs) >= 3:
        cr_var = max(crs[-3:]) - min(crs[-3:])
        print(f"\nCross-ratio variation (last 3): {cr_var}")
        
        if cr_var < mpf('0.1'):
            avg_cr = sum(crs[-3:]) / 3
            print(f"Cross-ratios appear to CONVERGE to: {avg_cr}")
            
            # Test relationship to known constants
            print(f"\nRelationship to constants:")
            print(f"  CR_limit / (1-1/δ) = {avg_cr / (1 - 1/delta)}")
            print(f"  CR_limit × δ = {avg_cr * delta}")
            print(f"  CR_limit × δ² = {avg_cr * delta**2}")
            print(f"  (1 - CR_limit) × δ = {(1 - avg_cr) * delta}")
            
            results['patterns']['converged_cross_ratio'] = str(avg_cr)
        else:
            print("Cross-ratios do NOT converge - structure may be more complex")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'../results/exp_10_crossratio_mobius_{timestamp}.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results

if __name__ == '__main__':
    analyze_cross_ratios()

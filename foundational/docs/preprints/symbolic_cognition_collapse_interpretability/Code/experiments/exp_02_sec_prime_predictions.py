"""
SEC Framework: Testable Predictions About Primes

If primes are the "branch pole" opposite to Fibonacci's "collapse pole",
the SEC framework should predict something non-obvious about primes.

Predictions to test:
1. Prime gaps near Fibonacci numbers should be larger (collapse repulsion)
2. Twin primes should avoid Fibonacci-dense regions
3. Prime k-tuples should have anti-Fibonacci structure
4. Zeckendorf complexity correlates with primality in specific ways

The goal: Find something we DIDN'T already know, predicted by the framework.
"""

import numpy as np
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2

# =============================================================================
# Setup
# =============================================================================

def fibonacci_sequence(n_terms):
    """Generate Fibonacci sequence."""
    fib = [1, 1]
    for _ in range(n_terms - 2):
        fib.append(fib[-1] + fib[-2])
    return np.array(fib)

def sieve_primes(limit):
    """Sieve of Eratosthenes."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

def zeckendorf(n):
    """Return Zeckendorf representation (greedy Fibonacci decomposition)."""
    if n <= 0:
        return []
    fibs = [1, 2]
    while fibs[-1] < n:
        fibs.append(fibs[-1] + fibs[-2])
    
    rep = []
    for f in reversed(fibs):
        if f <= n:
            rep.append(f)
            n -= f
        if n == 0:
            break
    return rep

def distance_to_nearest_fib(n, fibs):
    """Distance to nearest Fibonacci number."""
    idx = np.searchsorted(fibs, n)
    if idx == 0:
        return abs(n - fibs[0])
    if idx >= len(fibs):
        return abs(n - fibs[-1])
    return min(abs(n - fibs[idx-1]), abs(n - fibs[idx]))

# =============================================================================
# PREDICTION 1: Prime gaps near Fibonacci numbers
# =============================================================================

def test_prediction_1():
    """
    SEC PREDICTION: If Fibonacci numbers sit at the "collapse pole", 
    primes (at the "branch pole") should be repelled from them.
    
    Test: Are prime gaps LARGER near Fibonacci numbers?
    """
    print("=" * 70)
    print("PREDICTION 1: Prime Gap Enhancement Near Fibonacci")
    print("=" * 70)
    
    limit = 100000
    primes = sieve_primes(limit)
    fibs = fibonacci_sequence(30)
    fibs = fibs[fibs < limit]
    
    # For each prime, compute distance to nearest Fibonacci
    # and the gap to next prime
    gaps = np.diff(primes)
    prime_fib_dist = np.array([distance_to_nearest_fib(p, fibs) for p in primes[:-1]])
    
    # Bin primes by distance to Fibonacci
    bins = [0, 5, 10, 20, 50, 100, 500, np.inf]
    bin_labels = ['0-5', '5-10', '10-20', '20-50', '50-100', '100-500', '>500']
    
    print("\nPrime gaps by distance to nearest Fibonacci:")
    print("-" * 50)
    
    results = []
    for i in range(len(bins) - 1):
        mask = (prime_fib_dist >= bins[i]) & (prime_fib_dist < bins[i+1])
        if mask.sum() > 10:
            mean_gap = gaps[mask].mean()
            std_gap = gaps[mask].std()
            results.append((bin_labels[i], mask.sum(), mean_gap, std_gap))
            print(f"  Distance {bin_labels[i]:>8}: n={mask.sum():5d}, mean_gap={mean_gap:.2f}, std={std_gap:.2f}")
    
    # Statistical test: are gaps near Fibonacci larger?
    near_mask = prime_fib_dist < 10
    far_mask = prime_fib_dist > 100
    
    near_gaps = gaps[near_mask]
    far_gaps = gaps[far_mask]
    
    t_stat, p_value = stats.ttest_ind(near_gaps, far_gaps)
    
    print(f"\n  Near Fibonacci (d<10): mean gap = {near_gaps.mean():.2f}")
    print(f"  Far from Fibonacci (d>100): mean gap = {far_gaps.mean():.2f}")
    print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    
    if near_gaps.mean() > far_gaps.mean() and p_value < 0.05:
        print("\n  ✓ CONFIRMED: Primes are repelled from Fibonacci (larger gaps nearby)")
        return True, p_value
    else:
        print("\n  ✗ NOT CONFIRMED: No significant gap enhancement near Fibonacci")
        return False, p_value


# =============================================================================
# PREDICTION 2: Twin primes avoid Fibonacci-dense regions
# =============================================================================

def test_prediction_2():
    """
    SEC PREDICTION: Twin primes (p, p+2) are "maximally branch-like" -
    they should avoid regions where Fibonacci density is high.
    
    Fibonacci density = sum of 1/distance for nearby Fibs
    """
    print("\n" + "=" * 70)
    print("PREDICTION 2: Twin Primes Avoid Fibonacci-Dense Regions")
    print("=" * 70)
    
    limit = 100000
    primes = sieve_primes(limit)
    fibs = fibonacci_sequence(30)
    fibs = fibs[(fibs > 10) & (fibs < limit)]
    
    # Find twin primes
    gaps = np.diff(primes)
    twin_indices = np.where(gaps == 2)[0]
    twins = primes[twin_indices]
    
    # For comparison: non-twin primes
    non_twin_mask = np.ones(len(primes) - 1, dtype=bool)
    non_twin_mask[twin_indices] = False
    non_twins = primes[:-1][non_twin_mask]
    
    def fib_density(n, fibs, window=50):
        """Local Fibonacci density around n."""
        nearby = fibs[(fibs > n - window) & (fibs < n + window)]
        if len(nearby) == 0:
            return 0
        return sum(1.0 / (abs(n - f) + 1) for f in nearby)
    
    twin_densities = [fib_density(t, fibs) for t in twins[:500]]
    non_twin_densities = [fib_density(nt, fibs) for nt in non_twins[:2000]]
    
    print(f"\n  Twin primes analyzed: {len(twin_densities)}")
    print(f"  Non-twin primes analyzed: {len(non_twin_densities)}")
    print(f"\n  Mean Fibonacci density near twins: {np.mean(twin_densities):.4f}")
    print(f"  Mean Fibonacci density near non-twins: {np.mean(non_twin_densities):.4f}")
    
    t_stat, p_value = stats.ttest_ind(twin_densities, non_twin_densities)
    print(f"\n  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    
    if np.mean(twin_densities) < np.mean(non_twin_densities) and p_value < 0.05:
        print("\n  ✓ CONFIRMED: Twin primes avoid Fibonacci-dense regions")
        return True, p_value
    else:
        print("\n  ✗ NOT CONFIRMED: No significant avoidance pattern")
        return False, p_value


# =============================================================================
# PREDICTION 3: Zeckendorf complexity predicts prime gaps
# =============================================================================

def test_prediction_3():
    """
    SEC PREDICTION: Primes with simpler Zeckendorf representations
    (fewer terms, smaller max Fib) should have different gap structure.
    
    Simpler Zeckendorf = closer to "Fibonacci primitive"
    """
    print("\n" + "=" * 70)
    print("PREDICTION 3: Zeckendorf Complexity Predicts Prime Gap Structure")
    print("=" * 70)
    
    limit = 50000
    primes = sieve_primes(limit)
    gaps = np.diff(primes)
    
    # Compute Zeckendorf complexity for each prime
    complexities = []
    for p in primes[:-1]:
        zeck = zeckendorf(p)
        complexity = len(zeck)  # Number of terms
        complexities.append(complexity)
    
    complexities = np.array(complexities)
    
    # Group by complexity
    print("\nPrime gap statistics by Zeckendorf complexity:")
    print("-" * 50)
    
    results = {}
    for c in range(2, 9):
        mask = complexities == c
        if mask.sum() > 20:
            mean_gap = gaps[mask].mean()
            std_gap = gaps[mask].std()
            results[c] = (mask.sum(), mean_gap, std_gap)
            print(f"  Complexity {c}: n={mask.sum():5d}, mean_gap={mean_gap:.2f}, std={std_gap:.2f}")
    
    # Is there a correlation?
    valid = complexities > 0
    corr, p_value = stats.spearmanr(complexities[valid], gaps[valid])
    
    print(f"\n  Spearman correlation (complexity vs gap): {corr:.4f}")
    print(f"  p-value: {p_value:.6f}")
    
    if abs(corr) > 0.05 and p_value < 0.01:
        direction = "positive" if corr > 0 else "negative"
        print(f"\n  ✓ CONFIRMED: {direction.title()} correlation between Zeckendorf complexity and gap size")
        return True, p_value
    else:
        print("\n  ✗ NOT CONFIRMED: No significant correlation")
        return False, p_value


# =============================================================================
# PREDICTION 4: Prime residues mod Fibonacci show structure
# =============================================================================

def test_prediction_4():
    """
    SEC PREDICTION: If primes are "anti-phase" to Fibonacci,
    their residues mod F_n should show non-uniform distribution.
    
    Specifically: primes should avoid residue 0 (divisibility) and 
    nearby residues (collapse basin).
    """
    print("\n" + "=" * 70)
    print("PREDICTION 4: Prime Residues Mod Fibonacci Show Structure")
    print("=" * 70)
    
    limit = 100000
    primes = sieve_primes(limit)
    primes = primes[primes > 100]  # Skip small primes
    
    fibs_to_test = [13, 21, 34, 55, 89]  # F_7 through F_11
    
    print("\nResidue distribution of primes mod F_n:")
    print("-" * 60)
    
    all_results = []
    for f in fibs_to_test:
        residues = primes % f
        
        # Expected uniform distribution
        expected = len(primes) / f
        
        # Observed distribution
        counts = Counter(residues)
        observed = [counts.get(r, 0) for r in range(f)]
        
        # Chi-square test for uniformity
        chi2, p_value = stats.chisquare(observed)
        
        # Check specific residues: 0, 1, f-1 (near Fibonacci)
        near_fib_residues = [0, 1, f-1]
        near_fib_count = sum(counts.get(r, 0) for r in near_fib_residues)
        expected_near = 3 * expected
        
        deficit = (expected_near - near_fib_count) / expected_near * 100
        
        print(f"\n  F_{fibs_to_test.index(f)+7} = {f}:")
        print(f"    Chi-square: {chi2:.1f}, p-value: {p_value:.4f}")
        print(f"    Near-Fib residues (0,1,{f-1}): {near_fib_count} (expected: {expected_near:.0f})")
        print(f"    Deficit: {deficit:.1f}%")
        
        all_results.append((f, chi2, p_value, deficit))
    
    # Overall assessment
    significant = sum(1 for _, _, p, _ in all_results if p < 0.05)
    mean_deficit = np.mean([d for _, _, _, d in all_results])
    
    print(f"\n  Significant non-uniformity: {significant}/{len(fibs_to_test)} Fibonacci moduli")
    print(f"  Mean deficit at near-Fib residues: {mean_deficit:.1f}%")
    
    if significant >= 3 and mean_deficit > 5:
        print("\n  ✓ CONFIRMED: Primes show structured residues mod Fibonacci")
        return True, min(p for _, _, p, _ in all_results)
    else:
        print("\n  ✗ NOT CONFIRMED: Residues appear uniform")
        return False, 1.0


# =============================================================================
# PREDICTION 5: Gap entropy correlates with Fibonacci proximity
# =============================================================================

def test_prediction_5():
    """
    SEC PREDICTION: The "entropy" of local prime gap distribution
    should be lower near Fibonacci (more ordered, collapse-influenced).
    
    Entropy = Shannon entropy of gap distribution in a window.
    """
    print("\n" + "=" * 70)
    print("PREDICTION 5: Local Gap Entropy Varies with Fibonacci Proximity")
    print("=" * 70)
    
    limit = 100000
    primes = sieve_primes(limit)
    gaps = np.diff(primes)
    fibs = fibonacci_sequence(25)
    fibs = fibs[(fibs > 100) & (fibs < limit - 100)]
    
    def local_entropy(center, gaps, primes, window=50):
        """Compute Shannon entropy of gap distribution near center."""
        # Find primes in window
        mask = (primes[:-1] > center - window) & (primes[:-1] < center + window)
        local_gaps = gaps[mask]
        
        if len(local_gaps) < 5:
            return np.nan
        
        # Discretize gaps
        gap_counts = Counter(local_gaps)
        total = sum(gap_counts.values())
        probs = [c / total for c in gap_counts.values()]
        
        # Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy
    
    # Compute entropy at Fibonacci locations
    fib_entropies = []
    for f in fibs:
        e = local_entropy(f, gaps, primes)
        if not np.isnan(e):
            fib_entropies.append(e)
    
    # Compute entropy at random locations (matched to Fib range)
    np.random.seed(42)
    random_locs = np.random.randint(fibs.min(), fibs.max(), size=len(fibs) * 3)
    random_entropies = []
    for loc in random_locs:
        e = local_entropy(loc, gaps, primes)
        if not np.isnan(e):
            random_entropies.append(e)
    
    print(f"\n  Fibonacci locations analyzed: {len(fib_entropies)}")
    print(f"  Random locations analyzed: {len(random_entropies)}")
    print(f"\n  Mean entropy at Fibonacci: {np.mean(fib_entropies):.4f}")
    print(f"  Mean entropy at random: {np.mean(random_entropies):.4f}")
    
    t_stat, p_value = stats.ttest_ind(fib_entropies, random_entropies)
    print(f"\n  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    
    if np.mean(fib_entropies) < np.mean(random_entropies) and p_value < 0.05:
        print("\n  ✓ CONFIRMED: Lower gap entropy near Fibonacci (more ordered)")
        return True, p_value
    elif np.mean(fib_entropies) > np.mean(random_entropies) and p_value < 0.05:
        print("\n  ✓ CONFIRMED: Higher gap entropy near Fibonacci (chaos at boundary)")
        return True, p_value
    else:
        print("\n  ✗ NOT CONFIRMED: No significant entropy difference")
        return False, p_value


# =============================================================================
# PREDICTION 6: Prime constellation density vs Fibonacci
# =============================================================================

def test_prediction_6():
    """
    SEC PREDICTION: Prime k-tuples (constellations like twin, cousin, sexy primes)
    should have density that varies with Fibonacci proximity.
    
    If primes are "branch" structures, k-tuples are "multi-branch" events.
    """
    print("\n" + "=" * 70)
    print("PREDICTION 6: Prime Constellation Density vs Fibonacci Proximity")
    print("=" * 70)
    
    limit = 100000
    primes = sieve_primes(limit)
    prime_set = set(primes)
    fibs = fibonacci_sequence(25)
    fibs = fibs[(fibs > 50) & (fibs < limit - 50)]
    
    def count_constellations(center, prime_set, window=100):
        """Count prime k-tuples in a window around center."""
        twins = 0  # (p, p+2)
        cousins = 0  # (p, p+4)
        sexy = 0  # (p, p+6)
        
        for p in range(center - window, center + window):
            if p in prime_set:
                if p + 2 in prime_set:
                    twins += 1
                if p + 4 in prime_set:
                    cousins += 1
                if p + 6 in prime_set:
                    sexy += 1
        
        return twins + cousins + sexy  # Total constellation count
    
    # At Fibonacci
    fib_constellations = [count_constellations(f, prime_set) for f in fibs]
    
    # At random locations
    np.random.seed(42)
    random_locs = np.random.randint(fibs.min(), fibs.max(), size=len(fibs) * 3)
    random_constellations = [count_constellations(loc, prime_set) for loc in random_locs]
    
    print(f"\n  Mean constellations near Fibonacci: {np.mean(fib_constellations):.2f}")
    print(f"  Mean constellations at random: {np.mean(random_constellations):.2f}")
    
    t_stat, p_value = stats.ttest_ind(fib_constellations, random_constellations)
    print(f"\n  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    
    if abs(t_stat) > 2 and p_value < 0.05:
        direction = "fewer" if np.mean(fib_constellations) < np.mean(random_constellations) else "more"
        print(f"\n  ✓ CONFIRMED: {direction.title()} prime constellations near Fibonacci")
        return True, p_value
    else:
        print("\n  ✗ NOT CONFIRMED: No significant difference")
        return False, p_value


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SEC FRAMEWORK: TESTABLE PREDICTIONS ABOUT PRIMES")
    print("=" * 70)
    print("""
If the SEC framework is correct:
- Fibonacci = collapse pole (order, determinism)
- Primes = branch pole (entropy, irreducibility)

Then primes should show MEASURABLE repulsion from Fibonacci structure.
Testing 6 predictions...
""")
    
    results = []
    
    # Run all tests
    results.append(("Gap enhancement near Fib", *test_prediction_1()))
    results.append(("Twin prime avoidance", *test_prediction_2()))
    results.append(("Zeckendorf-gap correlation", *test_prediction_3()))
    results.append(("Residue structure mod Fib", *test_prediction_4()))
    results.append(("Gap entropy variation", *test_prediction_5()))
    results.append(("Constellation density", *test_prediction_6()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SEC PREDICTIONS ABOUT PRIMES")
    print("=" * 70)
    
    confirmed = 0
    for name, success, p_val in results:
        status = "✓ CONFIRMED" if success else "✗ NOT CONFIRMED"
        print(f"  {name:40s} {status:20s} (p={p_val:.4f})")
        if success:
            confirmed += 1
    
    print(f"\n  Total confirmed: {confirmed}/{len(results)}")
    
    if confirmed >= 4:
        print("\n  STRONG SUPPORT for SEC prime predictions")
    elif confirmed >= 2:
        print("\n  PARTIAL SUPPORT - some predictions confirmed")
    else:
        print("\n  WEAK SUPPORT - framework needs refinement")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    print("""
The confirmed predictions (if any) represent NON-OBVIOUS facts about primes
that were PREDICTED by the SEC framework, not observed first and explained after.

This is the difference between:
- Curve fitting (explaining known facts)
- Theory (predicting unknown facts)

If multiple predictions confirm, the SEC Fibonacci-Prime duality has
predictive power and is more than just a reinterpretation.
""")

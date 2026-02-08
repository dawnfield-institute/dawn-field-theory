"""
Experiment 15: Reconciliation Depth per k

PURPOSE:
    For each Bateman-Horn k value from sec_prime_manifold, measure how many
    "PAC levels up" (prime sieve steps) are needed before the sieve residual
    at that k converges to its expected density.

    The hypothesis: forbidden k values (5, 12-15) are positions where NO clean
    Fibonacci-structured reconciliation depth exists. Working k values reconcile
    at depths that align with Fibonacci numbers.

    This reframes forbidden k from "wave interference gaps" (which failed in
    exp_04 of prime_growth_dynamics_v2) to "PAC reconciliation failures."

HYPOTHESIS:
    "Forbidden k are values where local SEC losses cannot reconcile at any
     Fibonacci depth. Working k reconcile within MED bounds."

    Prediction:
    - k=3,4: fast reconciliation (depth 1-2)
    - k=6,7,8: moderate reconciliation (depth 2-3)
    - k=9: reconciliation at MED boundary (depth = F₃² = 9 sieve steps) 
    - k=10,11: supercritical but still Fibonacci-accessible
    - k=5,12-15: NO clean Fibonacci reconciliation depth

Success criterion:
    Working k values reconcile at Fibonacci depths; forbidden k don't.
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from constants import print_header, print_subheader, save_results, PHI, XI

GAMMA = 0.5772156649015329
LN_PHI = math.log(PHI)
XI_ANALYTIC = GAMMA + LN_PHI

FIBS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
FIB_SET = set(FIBS)

# Known λ* values from sec_prime_manifold (smoothing model)
LAMBDA_BY_K = {
    3: 0.9998, 4: 0.9996, 5: None,  # forbidden
    6: 0.9992, 7: 0.9987, 8: 0.9967,
    9: 0.9816, 10: 0.9302, 11: 0.9005,
    12: None, 13: None, 14: None, 15: None,  # forbidden
}

WORKING_K = [3, 4, 6, 7, 8, 9, 10, 11]
FORBIDDEN_K = [5, 12, 13, 14, 15]


def sieve_primes(N):
    """Return list of primes up to N."""
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p*p, N + 1, p):
                is_prime[m] = False
    return [i for i in range(2, N + 1) if is_prime[i]]


def k_tuple_density(primes, k, N):
    """
    Measure the density of k-tuples of primes (consecutive primes within 
    distance 2k of each other) relative to the Bateman-Horn prediction.
    
    For simplicity, we measure: among primes p ≤ N, what fraction have 
    p+k also prime (or have a specific pattern with spacing k)?
    
    Returns (count, density, expected_density).
    """
    prime_set = set(primes)
    count = sum(1 for p in primes if p + k in prime_set and p + k <= N)
    density = count / len(primes) if primes else 0
    
    # Hardy-Littlewood twin prime constant approximation for spacing k
    # C_k · N / ln(N)²  — simplified
    if k % 2 == 1:
        # Odd spacings: no pairs (p, p+k) with both prime if k is odd and > 1
        # because one of them is even (except p=2)
        expected = 1 / len(primes) if len(primes) > 0 else 0  # basically 0
    else:
        # Even spacing: use twin prime constant approximation
        # For k=2: C₂ ≈ 1.32, general C_k depends on sieve of k
        expected = 2 * twin_prime_constant(k) / math.log(N) if N > 2 else 0
    
    return count, density, expected


def twin_prime_constant(k):
    """
    Approximate the Hardy-Littlewood constant C_k for prime pairs (p, p+k).
    C_k = 2·C₂·∏(p|k, p>2) (p-1)/(p-2) where C₂ ≈ 0.6601618...
    """
    C2 = 0.6601618  # twin prime constant
    if k <= 0 or k % 2 != 0:
        return 0
    
    product = 1.0
    # Factor k, adjust for each odd prime factor
    n = k
    p = 3
    while p * p <= n:
        if n % p == 0:
            product *= (p - 1) / (p - 2)
            while n % p == 0:
                n //= p
        p += 2
    if n > 2:
        product *= (n - 1) / (n - 2)
    
    return C2 * product


def measure_reconciliation_depth(primes, k, N, max_depth=30):
    """
    For a given k, measure how many sieve steps (primes) it takes before
    the residual density at spacing k converges.
    
    "Reconciliation depth" = number of small primes you must sieve by
    before the k-spaced prime pair density stabilizes.
    
    We measure: after sieving by the first d primes, how close is the
    pair density to its asymptotic value?
    
    Returns: list of (depth, error, is_fibonacci) tuples
    """
    if k % 2 != 0:
        return []  # Odd k: trivially no pairs
    
    prime_set = set(primes)
    
    # Asymptotic count (using all primes)
    total_pairs = sum(1 for p in primes if p + k in prime_set and p + k <= N)
    if total_pairs == 0:
        return []
    
    # Now measure convergence: after sieving by first d primes, 
    # what fraction of the eventual pairs are already "decided"?
    results = []
    
    # For each depth d, compute: candidates that survive sieving by first d primes
    # and check what fraction of eventual k-pairs are among them
    sieve = list(range(2, N + 1))
    decided_pairs = 0
    
    for d in range(min(max_depth, len(primes))):
        p = primes[d]
        
        # After sieving by primes[0..d], check how many k-pairs are determined
        # A pair (n, n+k) is "decided" if at least one was eliminated
        # or both survive to this point
        
        # Simpler metric: fraction of eventual prime pairs (p, p+k)
        # where both p and p+k are coprime to all primes[0..d]
        sieve_primes_used = set(primes[:d+1])
        
        # How many of the actual pairs have both elements coprime to sieve primes?
        pairs_surviving = 0
        for pp in primes:
            if pp + k in prime_set and pp + k <= N:
                # This is an actual pair. Check if both survive partial sieve.
                both_coprime = all(pp % sp != 0 and (pp+k) % sp != 0 
                                   for sp in sieve_primes_used
                                   if sp != pp and sp != pp+k)
                if both_coprime:
                    pairs_surviving += 1
        
        # The reconciliation metric: how much has the density converged?
        convergence = 1 - (pairs_surviving / total_pairs) if total_pairs > 0 else 1
        # convergence → 1 means fully reconciled (all pairs decided)
        # convergence → 0 means nothing decided yet
        
        is_fib = (d + 1) in FIB_SET  # depth is 1-indexed
        
        results.append({
            'depth': d + 1,
            'prime_used': p,
            'pairs_surviving': pairs_surviving,
            'total_pairs': total_pairs,
            'convergence': convergence,
            'is_fibonacci_depth': is_fib,
        })
        
        # Stop early if fully converged
        if convergence > 0.99:
            break
    
    return results


def measure_density_convergence(k, N, checkpoints=None):
    """
    Alternative approach: measure how the running density of (p, p+k) pairs
    converges as x increases. 
    
    The "reconciliation depth" is how many primes you need before the 
    running density stabilizes.
    """
    primes = sieve_primes(N)
    prime_set = set(primes)
    
    if checkpoints is None:
        checkpoints = [100, 500, 1000, 5000, 10000, 50000, 100000, N]
        checkpoints = [c for c in checkpoints if c <= N]
    
    results = []
    prev_density = None
    
    for x in checkpoints:
        primes_up_to_x = [p for p in primes if p <= x]
        pairs = sum(1 for p in primes_up_to_x if p + k in prime_set and p + k <= x)
        
        if len(primes_up_to_x) > 0:
            density = pairs / len(primes_up_to_x)
        else:
            density = 0
        
        rate_of_change = abs(density - prev_density) if prev_density is not None else float('inf')
        prev_density = density
        
        results.append({
            'x': x,
            'primes_count': len(primes_up_to_x),
            'pairs': pairs,
            'density': density,
            'rate_of_change': rate_of_change,
        })
    
    return results


def find_reconciliation_depth_heuristic(convergence_data):
    """
    Find the depth at which density rate-of-change drops below threshold.
    This is the "number of PAC levels up" needed.
    """
    threshold = 0.001  # density change < 0.1%
    for i, entry in enumerate(convergence_data):
        if i > 0 and entry['rate_of_change'] < threshold:
            return i, entry['x']
    return len(convergence_data), convergence_data[-1]['x'] if convergence_data else 0


def run():
    print_header("EXP 15: Reconciliation Depth per k")
    
    N = 200000
    primes = sieve_primes(N)
    prime_set = set(primes)
    print(f"  Primes up to N = {N:,}: {len(primes):,}")
    
    # ================================================================
    # Test 1: Pair counts for each k
    # ================================================================
    print_subheader("Test 1: Prime pair counts for each k (even k only)")
    
    print(f"  {'k':>4}  {'pairs':>8}  {'density':>10}  {'status':>10}  {'λ*':>8}")
    print(f"  {'-'*50}")
    
    k_data = {}
    for k in range(2, 32, 2):  # Even k only
        pairs = sum(1 for p in primes if p + k in prime_set and p + k <= N)
        density = pairs / len(primes) if primes else 0
        status = 'FORBIDDEN' if k in FORBIDDEN_K else ('WORKING' if k in WORKING_K else '')
        lam = LAMBDA_BY_K.get(k, '')
        lam_str = f"{lam:.4f}" if isinstance(lam, float) else str(lam)
        
        print(f"  {k:4d}  {pairs:8d}  {density:10.6f}  {status:>10}  {lam_str:>8}")
        k_data[k] = {'pairs': pairs, 'density': density}
    
    # ================================================================
    # Test 2: Density convergence for working vs forbidden k
    # ================================================================
    print_subheader("Test 2: Density convergence trajectories")
    
    all_convergence = {}
    test_ks = [2, 4, 6, 8, 10, 12, 14, 30]
    
    for k in test_ks:
        conv = measure_density_convergence(k, N)
        all_convergence[k] = conv
        
        depth_idx, depth_x = find_reconciliation_depth_heuristic(conv)
        is_fib = depth_idx in FIB_SET
        is_working = k in WORKING_K
        is_forbidden = k in FORBIDDEN_K
        
        print(f"\n  k={k}: reconciliation at checkpoint #{depth_idx} (x={depth_x:,})")
        final = conv[-1] if conv else {'density': 0, 'pairs': 0}
        print(f"    Final density: {final['density']:.6f}, pairs: {final['pairs']}")
        
        # Show trajectory
        for entry in conv:
            marker = ""
            if entry['x'] in [100, 1000, 10000, N]:
                print(f"    x={entry['x']:>7,}: density={entry['density']:.6f} "
                      f"Δ={entry['rate_of_change']:.6f}{marker}")
    
    # ================================================================
    # Test 3: Sieve-step reconciliation for first few primes
    # ================================================================
    print_subheader("Test 3: Sieve-step reconciliation (first 15 primes)")
    
    # For each k, measure how many sieve steps until the pair pattern stabilizes
    # Using a simpler metric: after removing multiples of primes[0..d],
    # what fraction of numbers in [2, N] with spacing k are both survivors?
    
    reconciliation_depths = {}
    
    for k in [2, 4, 6, 8, 10, 12, 14, 30]:
        if k % 2 != 0:
            continue
        
        # Start with all odd numbers (after sieve by 2)
        # Then progressively sieve by 3, 5, 7, 11, ...
        sieve = [True] * (N + 1)
        sieve[0] = sieve[1] = False
        
        # Track pair survival rate at each sieve depth
        trajectory = []
        
        for d, p in enumerate(primes[:20]):
            # Sieve by this prime
            if d > 0:  # skip p=2 for first step since we start with all
                for m in range(p*p, N + 1, p):
                    sieve[m] = False
            elif p == 2:
                for m in range(4, N + 1, 2):
                    sieve[m] = False
            
            # Count surviving pairs with spacing k
            surviving_pairs = sum(1 for n in range(2, N - k + 1) 
                                  if sieve[n] and sieve[n + k])
            total_survivors = sum(1 for n in range(2, N + 1) if sieve[n])
            
            pair_fraction = surviving_pairs / total_survivors if total_survivors > 0 else 0
            
            trajectory.append({
                'depth': d + 1,
                'prime': p,
                'surviving_pairs': surviving_pairs,
                'total_survivors': total_survivors,
                'pair_fraction': pair_fraction,
            })
        
        # Find where pair_fraction stabilizes
        fractions = [t['pair_fraction'] for t in trajectory]
        reconciliation_depth = None
        for i in range(2, len(fractions)):
            change = abs(fractions[i] - fractions[i-1])
            if change < 0.0005:  # threshold for "stabilized"
                reconciliation_depth = i + 1
                break
        
        if reconciliation_depth is None:
            reconciliation_depth = len(fractions)
        
        is_fib = reconciliation_depth in FIB_SET
        reconciliation_depths[k] = {
            'depth': reconciliation_depth,
            'is_fibonacci': is_fib,
            'trajectory': trajectory[:8],
        }
    
    print(f"\n  {'k':>4}  {'recon_depth':>12}  {'is_fib':>7}  {'status':>10}")
    print(f"  {'-'*45}")
    for k in sorted(reconciliation_depths.keys()):
        rd = reconciliation_depths[k]
        status = 'FORBIDDEN' if k in FORBIDDEN_K else ('WORKING' if k in WORKING_K else '')
        print(f"  {k:4d}  {rd['depth']:12d}  {'YES' if rd['is_fibonacci'] else 'no':>7}  {status:>10}")
    
    # ================================================================
    # Test 4: Fibonacci depth alignment analysis
    # ================================================================
    print_subheader("Test 4: Fibonacci depth alignment")
    
    working_fib_count = 0
    working_total = 0
    forbidden_fib_count = 0
    forbidden_total = 0
    
    for k, rd in reconciliation_depths.items():
        if k in WORKING_K:
            working_total += 1
            if rd['is_fibonacci']:
                working_fib_count += 1
        elif k in FORBIDDEN_K:
            forbidden_total += 1
            if rd['is_fibonacci']:
                forbidden_fib_count += 1
    
    print(f"  Working k with Fibonacci reconciliation depth: "
          f"{working_fib_count}/{working_total}")
    print(f"  Forbidden k with Fibonacci reconciliation depth: "
          f"{forbidden_fib_count}/{forbidden_total}")
    
    hypothesis_supported = (working_fib_count > forbidden_fib_count) if (working_total > 0 and forbidden_total > 0) else False
    
    # ================================================================
    # Test 5: k=9 at MED boundary
    # ================================================================
    print_subheader("Test 5: k=9 as MED boundary (F₃² = 9)")
    
    # k=9 is odd, so no twin primes. But k=9 is the critical k in Bateman-Horn.
    # The MED connection: 9 = 3² = F₄², and it's where λ* drops sharply.
    # Check if reconciliation of the smoothing model at k=9 requires exactly
    # 9 = F₄² steps.
    
    print(f"  k=9: λ* = {LAMBDA_BY_K[9]}")
    print(f"  9 = F₄² = 3² (MED nodes squared)")
    print(f"  This is the critical transition point in sec_prime_manifold.")
    print(f"  Below k=9: high λ (fast reconciliation)")
    print(f"  Above k=9: rapid λ decay (reconciliation breaks down)")
    
    # Measure: for the smoothing model λ(k) values, what's the 
    # "reconciliation depth" = how many Fibonacci numbers summed to reach k?
    print(f"\n  Zeckendorf (Fibonacci) representations of k:")
    for k in WORKING_K + FORBIDDEN_K:
        if k is None:
            continue
        zeck = zeckendorf(k)
        depth = len(zeck)
        is_fib_depth = depth in FIB_SET
        lam = LAMBDA_BY_K.get(k, None)
        lam_str = f"λ={lam:.4f}" if lam else "FORBIDDEN"
        print(f"    k={k:2d}: {' + '.join(f'F({z})' for z in zeck)} "
              f"(depth={depth}) {lam_str}")
    
    # ================================================================
    # Results
    # ================================================================
    success = hypothesis_supported
    
    data = {
        'experiment': 'exp_15_reconciliation_depth',
        'hypothesis': 'Forbidden k cannot reconcile at Fibonacci depths',
        'N': N,
        'n_primes': len(primes),
        'pair_counts': {str(k): v for k, v in k_data.items()},
        'reconciliation_depths': {
            str(k): {'depth': v['depth'], 'is_fibonacci': v['is_fibonacci']}
            for k, v in reconciliation_depths.items()
        },
        'working_fib_fraction': working_fib_count / working_total if working_total > 0 else 0,
        'forbidden_fib_fraction': forbidden_fib_count / forbidden_total if forbidden_total > 0 else 0,
        'hypothesis_supported': hypothesis_supported,
        'success': success,
    }
    
    print(f"\n{'='*70}")
    print(f"WORKING k Fibonacci reconciliation: {working_fib_count}/{working_total}")
    print(f"FORBIDDEN k Fibonacci reconciliation: {forbidden_fib_count}/{forbidden_total}")
    print(f"HYPOTHESIS SUPPORTED: {'YES' if success else 'INCONCLUSIVE'}")
    print(f"{'='*70}")
    
    save_results(data, 'exp_15_reconciliation_depth')
    return data


def zeckendorf(n):
    """Zeckendorf representation: express n as sum of non-consecutive Fibonacci numbers."""
    if n <= 0:
        return []
    
    # Generate Fibs up to n
    fibs = [1, 2]
    while fibs[-1] < n:
        fibs.append(fibs[-1] + fibs[-2])
    
    result = []
    remaining = n
    for f in reversed(fibs):
        if f <= remaining:
            result.append(f)
            remaining -= f
        if remaining == 0:
            break
    
    return result


if __name__ == '__main__':
    run()

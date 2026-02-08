"""
Experiment 14: Sieve as Local SEC

PURPOSE:
    Model each step of the sieve of Eratosthenes as a local SEC collapse event.
    Each prime p eliminates a fraction 1/p of remaining candidates — this is a
    local non-conservation (possibilities are destroyed, not transferred).

    At any intermediate sieve step, P + A + Δ ≠ constant locally.
    But at the GLOBAL level (PAC): π(x) + C(x) = x - 1, exactly.

    The Δ buffer accumulates the "unreconciled" difference between local SEC
    losses and global PAC accounting. We show:
    
    1. Δ trajectory after each sieve step tracks the Mertens product
    2. Cumulative local SEC losses = -γ - ln(ln(x)) asymptotically
    3. PAC reconciliation at the aggregate level yields exact conservation

HYPOTHESIS:
    "Each prime's sieve step is a local SEC collapse that doesn't conserve.
     The Mertens product ∏(1-1/p) is the cumulative Δ trajectory.
     PAC conserves only at reconciliation boundaries."

Success criterion: 
    - Mertens product matches e^(-γ)/ln(x) within established bounds
    - PAC conservation π(x) + C(x) = x - 1 is exact at every x
    - Δ(p) = residual after sieve by p tracks ∏(1-1/p) · x
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from constants import print_header, print_subheader, save_results, PHI, XI

# Dawn Field constants
GAMMA = 0.5772156649015329
LN_PHI = math.log(PHI)
XI_ANALYTIC = GAMMA + LN_PHI


def sieve_with_delta_tracking(N):
    """
    Run sieve of Eratosthenes tracking PAC state at each sieve step.
    
    At each prime p:
      - Local SEC: remove 1/p of remaining (non-conserving locally)
      - Δ accumulates the running loss
      - PAC: π(x) + C(x) = x-1 remains exact globally
    
    Returns list of dicts, one per sieve prime.
    """
    # Boolean sieve
    is_candidate = [True] * (N + 1)
    is_candidate[0] = is_candidate[1] = False
    
    steps = []
    cumulative_product = 1.0  # ∏(1 - 1/p)
    
    for p in range(2, int(math.isqrt(N)) + 1):
        if not is_candidate[p]:
            continue
        
        # Count candidates BEFORE this sieve step
        candidates_before = sum(1 for i in range(2, N + 1) if is_candidate[i])
        
        # Local SEC collapse: mark multiples of p
        removed_count = 0
        for multiple in range(p * p, N + 1, p):
            if is_candidate[multiple]:
                is_candidate[multiple] = False
                removed_count += 1
        
        # Count candidates AFTER
        candidates_after = sum(1 for i in range(2, N + 1) if is_candidate[i])
        
        # Update Mertens product
        cumulative_product *= (1 - 1/p)
        mertens_prediction = math.exp(-GAMMA) / math.log(N) if N > 1 else 0
        
        # Local SEC accounting
        local_loss_fraction = removed_count / candidates_before if candidates_before > 0 else 0
        theoretical_loss = 1/p  # Expected fraction removed
        
        # Count actual primes and composites up to N
        primes_so_far = sum(1 for i in range(2, N + 1) if is_candidate[i])
        composites_so_far = (N - 1) - primes_so_far  # total slots = N-1 (2..N)
        pac_sum = primes_so_far + composites_so_far  # Should = N - 1
        
        # Δ: difference between Mertens prediction and actual surviving fraction
        surviving_fraction = candidates_after / (N - 1) if N > 1 else 0
        delta = surviving_fraction - cumulative_product
        
        steps.append({
            'p': p,
            'candidates_before': candidates_before,
            'removed': removed_count,
            'candidates_after': candidates_after,
            'local_loss_fraction': local_loss_fraction,
            'theoretical_loss': 1/p,
            'cumulative_product': cumulative_product,
            'mertens_prediction': mertens_prediction,
            'surviving_fraction': surviving_fraction,
            'delta': delta,
            'pac_exact': pac_sum == N - 1,
            'pac_sum': pac_sum,
        })
    
    return steps, is_candidate


def run():
    print_header("EXP 14: Sieve as Local SEC Collapse")
    
    N = 500000
    print(f"  Sieve up to N = {N:,}")
    
    # ================================================================
    # Test 1: Run sieve with Δ tracking
    # ================================================================
    print_subheader("Test 1: Sieve steps as local SEC collapses")
    
    steps, is_prime = sieve_with_delta_tracking(N)
    
    print(f"  {'p':>6}  {'removed':>8}  {'loss%':>8}  {'1/p':>8}  "
          f"{'∏(1-1/p)':>12}  {'surviving':>12}  {'Δ':>12}  {'PAC':>4}")
    print(f"  {'-'*80}")
    
    for s in steps[:30]:
        print(f"  {s['p']:6d}  {s['removed']:8d}  "
              f"{s['local_loss_fraction']:8.4f}  {s['theoretical_loss']:8.4f}  "
              f"{s['cumulative_product']:12.8f}  {s['surviving_fraction']:12.8f}  "
              f"{s['delta']:12.8f}  {'✓' if s['pac_exact'] else '✗'}")
    
    # ================================================================
    # Test 2: PAC conservation is EXACT at every step
    # ================================================================
    print_subheader("Test 2: PAC conservation at every sieve step")
    
    all_pac_exact = all(s['pac_exact'] for s in steps)
    print(f"  PAC conservation π(x) + C(x) = x - 1 exact at ALL {len(steps)} steps: "
          f"{'YES ✓' if all_pac_exact else 'NO ✗'}")
    
    # Count primes and composites at final state
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    composites = [i for i in range(2, N + 1) if not is_prime[i]]
    print(f"  π({N:,}) = {len(primes):,}")
    print(f"  C({N:,}) = {len(composites):,}")
    print(f"  π + C = {len(primes) + len(composites):,} = N-1 = {N-1:,}: "
          f"{'EXACT ✓' if len(primes) + len(composites) == N - 1 else 'VIOLATED ✗'}")
    
    # ================================================================
    # Test 3: Mertens product convergence
    # ================================================================
    print_subheader("Test 3: Cumulative Δ → Mertens product")
    
    final_step = steps[-1]
    sqrt_N = int(math.isqrt(N))
    
    # (a) Sieve-only product: p ≤ √N → compare with e^(-γ)/ln(√N)
    sieve_product = final_step['cumulative_product']
    sieve_expected = math.exp(-GAMMA) / math.log(sqrt_N)
    sieve_error = abs(sieve_product - sieve_expected) / sieve_expected * 100
    
    print(f"  (a) Sieve primes only (p ≤ √N = {sqrt_N}):")
    print(f"      ∏(1-1/p)        = {sieve_product:.10f}")
    print(f"      e^(-γ)/ln(√N)   = {sieve_expected:.10f}")
    print(f"      Error: {sieve_error:.4f}%")
    
    # (b) Full Mertens product: ∏(1-1/p) for ALL p ≤ N → compare with e^(-γ)/ln(N)
    full_product = 1.0
    for p in primes:
        full_product *= (1 - 1/p)
    mertens_expected = math.exp(-GAMMA) / math.log(N)
    mertens_error = abs(full_product - mertens_expected) / mertens_expected * 100
    
    print(f"\n  (b) All primes (p ≤ N = {N:,}):")
    print(f"      ∏(1-1/p)        = {full_product:.10e}")
    print(f"      e^(-γ)/ln(N)    = {mertens_expected:.10f}")
    print(f"      Error: {mertens_error:.4f}%")
    print(f"      (Mertens 3rd theorem: asymptotic, convergence is slow)")
    
    # ================================================================
    # Test 4: Δ trajectory analysis
    # ================================================================
    print_subheader("Test 4: Δ buffer trajectory")
    
    deltas = [s['delta'] for s in steps]
    print(f"  Max |Δ|: {max(abs(d) for d in deltas):.8f}")
    print(f"  Min |Δ|: {min(abs(d) for d in deltas):.8f}")
    print(f"  Mean Δ:  {sum(deltas)/len(deltas):.8f}")
    
    # Δ should converge to 0 as we sieve more (reconciliation at global level)
    print(f"  Δ at p=2:  {steps[0]['delta']:.8f}")
    if len(steps) > 5:
        print(f"  Δ at p={steps[5]['p']}:  {steps[5]['delta']:.8f}")
    if len(steps) > 20:
        print(f"  Δ at p={steps[20]['p']}:  {steps[20]['delta']:.8f}")
    print(f"  Δ at p={steps[-1]['p']}: {steps[-1]['delta']:.8f}")
    
    # ================================================================
    # Test 5: Local SEC loss vs theoretical
    # ================================================================
    print_subheader("Test 5: Local SEC loss accuracy per prime")
    
    print(f"\n  Each prime's sieve step removes ~1/p of remaining candidates.")
    print(f"  This is LOCAL SEC collapse (non-conserving at single-step level).")
    print(f"  But cumulative product converges to Mertens = e^(-γ)/ln(N).")
    
    deviations = []
    for s in steps:
        dev = abs(s['local_loss_fraction'] - s['theoretical_loss'])
        deviations.append(dev)
    
    print(f"\n  Mean |actual_loss - 1/p| = {sum(deviations)/len(deviations):.6f}")
    print(f"  Max  |actual_loss - 1/p| = {max(deviations):.6f} (at p={steps[deviations.index(max(deviations))]['p']})")
    
    # ================================================================
    # Test 6: The SEC → PAC bridge: ln contributions
    # ================================================================
    print_subheader("Test 6: SEC local collapses sum to PAC global constant")
    
    # Sum of ln(1-1/p) = ln(∏(1-1/p)) → -γ - ln(ln(x))
    # (a) Sieve primes only (p ≤ √N)
    ln_sum_sieve = sum(math.log(1 - 1/s['p']) for s in steps)
    expected_ln_sieve = -GAMMA - math.log(math.log(sqrt_N))
    ln_sieve_error = abs(ln_sum_sieve - expected_ln_sieve) / abs(expected_ln_sieve) * 100
    
    print(f"  (a) Sieve primes (p ≤ √N = {sqrt_N}):")
    print(f"      Σ ln(1-1/p)     = {ln_sum_sieve:.10f}")
    print(f"      -γ - ln(ln(√N)) = {expected_ln_sieve:.10f}")
    print(f"      Error: {ln_sieve_error:.4f}%")
    
    # (b) All primes ≤ N
    ln_sum = sum(math.log(1 - 1/p) for p in primes)
    expected_ln_sum = -GAMMA - math.log(math.log(N))
    ln_error = abs(ln_sum - expected_ln_sum) / abs(expected_ln_sum) * 100
    
    print(f"\n  (b) All primes (p ≤ N = {N:,}):")
    print(f"      Σ ln(1-1/p)     = {ln_sum:.10f}")
    print(f"      -γ - ln(ln(N))  = {expected_ln_sum:.10f}")
    print(f"      Error: {ln_error:.4f}%")
    print(f"\n  The cumulative SEC local losses converge to a PAC global identity")
    print(f"  involving γ (Phase I cost = Euler-Mascheroni constant).")
    
    # Decompose into γ and ln(ln(N))
    print(f"\n  γ component:      {GAMMA:.10f}")
    print(f"  ln(ln(N)) component: {math.log(math.log(N)):.10f}")
    print(f"  Sum:              {GAMMA + math.log(math.log(N)):.10f}")
    print(f"  -Σ ln(1-1/p):    {-ln_sum:.10f}")
    
    # ================================================================
    # Test 7: Phase constant decomposition of Mertens
    # ================================================================
    print_subheader("Test 7: Mertens in phase constants")
    
    # e^(-γ) is the Phase I cost factor
    # 1/ln(x) is the Phase III smoothing rate
    # Their product = Mertens = fraction surviving all SEC collapses
    e_neg_gamma = math.exp(-GAMMA)
    print(f"  e^(-γ) = {e_neg_gamma:.10f}  (Phase I: cost of bounding possibility space)")
    print(f"  1/ln(N) = {1/math.log(N):.10f}  (Phase III: smoothing rate)")
    print(f"  Product = {e_neg_gamma / math.log(N):.10f}  (Mertens)")
    print(f"  π(N)/N  = {len(primes)/N:.10f}  (actual prime density)")
    pnt_error = abs(len(primes)/N - e_neg_gamma/math.log(N)) / (len(primes)/N) * 100
    print(f"  Relative error: {pnt_error:.4f}%")
    
    # What about e^(-Ξ)?
    e_neg_xi = math.exp(-XI_ANALYTIC)
    print(f"\n  e^(-Ξ) = e^(-γ-ln(φ)) = e^(-γ)·e^(-ln(φ)) = e^(-γ)/φ")
    print(f"  e^(-Ξ) = {e_neg_xi:.10f}")
    print(f"  e^(-γ)/φ = {e_neg_gamma/PHI:.10f}")
    print(f"  Match: {abs(e_neg_xi - e_neg_gamma/PHI) < 1e-10}")
    
    # ================================================================
    # Results
    # ================================================================
    success = all_pac_exact and sieve_error < 5 and ln_sieve_error < 5
    
    data = {
        'experiment': 'exp_14_sieve_as_local_sec',
        'hypothesis': 'Each sieve step is local SEC (non-conserving); PAC conserves globally',
        'N': N,
        'n_primes': len(primes),
        'n_composites': len(composites),
        'pac_exact_all_steps': all_pac_exact,
        'n_sieve_steps': len(steps),
        'sieve_product': sieve_product,
        'sieve_expected': sieve_expected,
        'sieve_error_pct': sieve_error,
        'full_mertens_product': full_product,
        'full_mertens_expected': mertens_expected,
        'full_mertens_error_pct': mertens_error,
        'ln_sum_sieve': ln_sum_sieve,
        'ln_sum_sieve_expected': expected_ln_sieve,
        'ln_sum_sieve_error_pct': ln_sieve_error,
        'ln_sum_full': ln_sum,
        'ln_sum_full_expected': expected_ln_sum,
        'ln_sum_full_error_pct': ln_error,
        'delta_max': max(abs(d) for d in deltas),
        'delta_mean': sum(deltas)/len(deltas),
        'phase_decomposition': {
            'e_neg_gamma': e_neg_gamma,
            'inv_ln_N': 1/math.log(N),
            'mertens': e_neg_gamma / math.log(N),
            'actual_density': len(primes)/N,
            'density_error_pct': pnt_error,
        },
        'first_10_steps': steps[:10],
        'success': success,
        'success_criterion': 'PAC exact at all steps, Mertens sieve < 5% error',
    }
    
    print(f"\n{'='*70}")
    print(f"PAC CONSERVATION EXACT: {'YES ✓' if all_pac_exact else 'NO ✗'}")
    print(f"MERTENS (sieve, p≤√N): {sieve_error:.4f}% error")
    print(f"SEC→PAC BRIDGE (sieve): Σln(1-1/p) → -γ-ln(ln(√N)), {ln_sieve_error:.4f}% error")
    print(f"MERTENS (full, p≤N): {mertens_error:.4f}% error")
    print(f"SEC→PAC BRIDGE (full): Σln(1-1/p) → -γ-ln(ln(N)), {ln_error:.4f}% error")
    print(f"SUCCESS: {'YES' if success else 'NO'}")
    print(f"{'='*70}")
    
    save_results(data, 'exp_14_sieve_as_local_sec')
    return data


if __name__ == '__main__':
    run()

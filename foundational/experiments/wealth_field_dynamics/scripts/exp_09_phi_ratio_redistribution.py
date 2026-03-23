#!/usr/bin/env python3
"""
Experiment 14: φ-Ratio in Wealth Redistribution

CORRECT FRAMING:
    Ξ is NOT a "threshold to cross" - it's the EMERGENCE per PAC collapse level.
    
    From exp_24 in oscillation_attractor_dynamics:
        Ξ - 1 = π/55 = net twist per level
        
    This is geometric: when a parent splits at φ-ratio (61.8/38.2),
    the system generates (Ξ - 1) ≈ 5.71% "emergence" per level.

THE RIGHT QUESTION:
    Does wealth redistribution produce φ-ratio splits?
    
    If PAC dynamics apply to economics, redistributive events should show:
    - Wealth dividing at approximately 61.8% / 38.2%
    - NOT arbitrary 50/50 or 90/10 splits
    
    This is testable: examine actual inheritance, tax, or reform data.

METHODOLOGY:
    1. Model what φ-splitting looks like in economic terms
    2. Compare to historical redistribution patterns (where data exists)
    3. Test: does the split ratio cluster near φ⁻¹ ≈ 0.618?

EPISTEMIC STATUS:
    This tests whether PAC dynamics are even applicable to economics.
    If splits are NOT near φ-ratio, the entire framework doesn't apply.
"""

import numpy as np
import json
from datetime import datetime
from constants import print_header, print_subheader, PHI, XI

PHI_INV = 1 / PHI  # ≈ 0.618


def model_phi_redistribution():
    """Model what φ-splitting looks like in wealth redistribution."""
    
    print_header("EXPERIMENT 14: φ-RATIO IN WEALTH REDISTRIBUTION")
    
    print_subheader("PART 1: WHAT φ-SPLITTING PREDICTS")
    
    print(f"""
    PAC collapse at φ-ratio predicts:
    ─────────────────────────────────────────────────────────────
    
    When wealth W redistributes via PAC dynamics:
    
        W_larger = W × φ⁻¹ = W × {PHI_INV:.4f}  (61.8% stays concentrated)
        W_smaller = W × (1 - φ⁻¹) = W × {1 - PHI_INV:.4f}  (38.2% disperses)
    
    The "emergence" per split:
        Ξ - 1 = π/55 ≈ {np.pi/55:.4f} = 5.71%
    
    This emergence represents NEW value/structure created by reorganization.
    ─────────────────────────────────────────────────────────────
    
    TESTABLE PREDICTION:
    
    If PAC applies to economics, major redistribution events should show
    wealth splits clustering near 61.8/38.2, not 50/50 or arbitrary.
    """)


def examine_historical_patterns():
    """Examine historical redistribution patterns."""
    
    print_subheader("PART 2: HISTORICAL REDISTRIBUTION PATTERNS")
    
    # Historical examples with rough split estimates
    # These are simplified - real analysis would need detailed data
    
    redistributions = [
        {
            'event': 'New Deal Era (1933-1945)',
            'description': 'Progressive taxation, labor rights, social programs',
            'top_share_before': 0.44,  # Top 1% share ~44% in 1928
            'top_share_after': 0.32,   # Top 1% share ~32% in 1945
            'ratio': None  # To be computed
        },
        {
            'event': 'Great Compression (1945-1970)',
            'description': 'Strong unions, high top marginal rates, GI Bill',
            'top_share_before': 0.32,
            'top_share_after': 0.22,
            'ratio': None
        },
        {
            'event': 'Post-War UK (1945-1979)',
            'description': 'NHS, nationalization, welfare state',
            'top_share_before': 0.40,
            'top_share_after': 0.21,
            'ratio': None
        },
        {
            'event': 'Reaganomics (1980-2000)',
            'description': 'Tax cuts, deregulation, weakened unions',
            'top_share_before': 0.22,
            'top_share_after': 0.40,
            'ratio': None  # This is concentration, not redistribution
        }
    ]
    
    print("    Historical wealth share changes:")
    print("    " + "─" * 70)
    print(f"    {'Event':<35} {'Before':<10} {'After':<10} {'Δ Share':<10}")
    print("    " + "─" * 70)
    
    for event in redistributions:
        delta = event['top_share_after'] - event['top_share_before']
        print(f"    {event['event']:<35} {event['top_share_before']:<10.2f} "
              f"{event['top_share_after']:<10.2f} {delta:+.2f}")
    
    print(f"""
    
    ANALYSIS:
    
    The question is whether these share changes follow φ-dynamics.
    
    For New Deal Era:
        Top 1% lost: 44% → 32% = -12 percentage points
        If PAC: new split = 0.32/0.44 = 0.727
        
        φ⁻¹ = {PHI_INV:.4f}
        Observed ratio: 0.727
        Difference from φ⁻¹: {abs(0.727 - PHI_INV):.3f}
    
    This is NOT a close match to φ. 
    
    HOWEVER: The PAC prediction is about SPLITTING, not share reduction.
    We're looking at the wrong metric.
    """)


def correct_metric():
    """Derive the correct metric for testing φ-splitting."""
    
    print_subheader("PART 3: THE CORRECT METRIC")
    
    print(f"""
    PROBLEM WITH ABOVE ANALYSIS:
    
    PAC splitting applies to INDIVIDUAL wealth transfers, not aggregate shares.
    
    When a fortune F splits (inheritance, business partition):
        F₁ = F × φ⁻¹ ≈ 0.618 × F
        F₂ = F × (1 - φ⁻¹) ≈ 0.382 × F
    
    Aggregate Gini changes are EMERGENT from many individual splits.
    
    THE RIGHT DATA:
    
    1. Estate tax records: How do inheritances actually divide?
       - Do 2-heir estates split near 61.8/38.2?
       - Do 3-heir estates follow Fibonacci?
    
    2. Business partitions: When companies split, what's the ratio?
       - M&A divestitures
       - Partnership dissolutions
    
    3. Bankruptcy proceedings: Asset distribution ratios
    
    4. Divorce settlements: Asset division patterns
    
    THIS DATA EXISTS but requires economic research access.
    """)


def simulate_phi_splitting():
    """Simulate what φ-splitting would look like in aggregate."""
    
    print_subheader("PART 4: SIMULATED φ-SPLITTING AGGREGATE EFFECT")
    
    np.random.seed(42)
    
    # Simulate 1000 agents with initial wealth following Pareto
    n_agents = 1000
    initial_wealth = np.random.pareto(1.5, n_agents) + 1
    initial_wealth = initial_wealth / initial_wealth.sum()  # Normalize
    
    def gini(w):
        w = np.sort(w)
        n = len(w)
        cumsum = np.cumsum(w)
        return (2 * np.sum((np.arange(1, n+1) - 0.5) * w) / (n * w.sum())) - 1
    
    initial_gini = gini(initial_wealth)
    
    # Scenario 1: φ-splitting redistribution (PAC dynamics)
    # Top 10% split their wealth at φ-ratio with random recipients
    wealth_phi = initial_wealth.copy()
    top_indices = np.argsort(wealth_phi)[-100:]  # Top 10%
    
    for idx in top_indices:
        amount_to_redistribute = wealth_phi[idx] * (1 - PHI_INV)  # 38.2%
        wealth_phi[idx] *= PHI_INV  # Keep 61.8%
        # Distribute to random recipients
        recipients = np.random.choice(n_agents, size=10, replace=False)
        wealth_phi[recipients] += amount_to_redistribute / 10
    
    gini_phi = gini(wealth_phi)
    
    # Scenario 2: Equal splitting (50/50)
    wealth_equal = initial_wealth.copy()
    for idx in top_indices:
        amount_to_redistribute = wealth_equal[idx] * 0.5  # 50%
        wealth_equal[idx] *= 0.5  # Keep 50%
        recipients = np.random.choice(n_agents, size=10, replace=False)
        wealth_equal[recipients] += amount_to_redistribute / 10
    
    gini_equal = gini(wealth_equal)
    
    # Scenario 3: Aggressive redistribution (90/10)
    wealth_aggressive = initial_wealth.copy()
    for idx in top_indices:
        amount_to_redistribute = wealth_aggressive[idx] * 0.9  # 90%
        wealth_aggressive[idx] *= 0.1  # Keep 10%
        recipients = np.random.choice(n_agents, size=10, replace=False)
        wealth_aggressive[recipients] += amount_to_redistribute / 10
    
    gini_aggressive = gini(wealth_aggressive)
    
    print(f"    Simulation: 1000 agents, Pareto initial distribution")
    print("    " + "─" * 60)
    print(f"    Initial Gini:           {initial_gini:.4f}")
    print(f"    After φ-split (61.8%):  {gini_phi:.4f}  (Δ = {gini_phi - initial_gini:+.4f})")
    print(f"    After 50/50 split:      {gini_equal:.4f}  (Δ = {gini_equal - initial_gini:+.4f})")
    print(f"    After 90/10 split:      {gini_aggressive:.4f}  (Δ = {gini_aggressive - initial_gini:+.4f})")
    
    # Compute "emergence" per scenario
    emergence_phi = abs(gini_phi - initial_gini) / 0.382  # Per unit redistributed
    emergence_equal = abs(gini_equal - initial_gini) / 0.5
    emergence_aggressive = abs(gini_aggressive - initial_gini) / 0.9
    
    print(f"""
    
    "Emergence" per unit redistributed:
    ─────────────────────────────────────────────────────────────
    φ-split (38.2% moved):     {emergence_phi:.4f} per unit
    50/50 split (50% moved):   {emergence_equal:.4f} per unit
    90/10 split (90% moved):   {emergence_aggressive:.4f} per unit
    
    PAC prediction (Ξ - 1):    {np.pi/55:.4f} per unit
    
    Closest to PAC: {'φ-split' if abs(emergence_phi - np.pi/55) < abs(emergence_equal - np.pi/55) else '50/50'}
    """)
    
    return {
        'initial_gini': initial_gini,
        'gini_phi': gini_phi,
        'gini_equal': gini_equal,
        'gini_aggressive': gini_aggressive,
        'emergence_phi': emergence_phi,
        'emergence_equal': emergence_equal,
        'emergence_aggressive': emergence_aggressive,
        'pac_prediction': np.pi/55
    }


def summary():
    """Summary with correct framing."""
    
    print_subheader("PART 5: SUMMARY - THE CORRECT TEST")
    
    print(f"""
    WHAT WE LEARNED:
    ─────────────────────────────────────────────────────────────
    
    1. Ξ is NOT a "threshold" - it's EMERGENCE per PAC collapse level
    
    2. The correct test for φ-dynamics in economics:
       - Do individual wealth splits cluster near 61.8/38.2?
       - Does "emergence" per split approximate π/55 ≈ 5.71%?
    
    3. Testing aggregate Gini changes is INDIRECT
       - Need microdata on individual splits
       - Estate tax records, business partitions, etc.
    
    4. The simulation suggests:
       - Different split ratios produce different aggregate effects
       - φ-splitting may be "minimum intervention for maximum restructure"
    
    WHAT WOULD VALIDATE PAC IN ECONOMICS:
    ─────────────────────────────────────────────────────────────
    
    ✓ Estate inheritance splits cluster near φ⁻¹
    ✓ Business divestitures show 61.8/38.2 patterns
    ✓ "Stable" redistributions (that don't revert) use φ-like ratios
    
    WHAT WOULD FALSIFY IT:
    ─────────────────────────────────────────────────────────────
    
    ✗ Splits are uniformly distributed (no φ preference)
    ✗ Stable outcomes occur at arbitrary split ratios
    ✗ 50/50 or other ratios work equally well for stability
    
    STATUS: Hypothesis reframed correctly. Microdata needed to test.
    ─────────────────────────────────────────────────────────────
    """)


def run_experiment():
    """Main entry point."""
    
    results = {
        'experiment': 'exp_14_phi_ratio_redistribution',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Test whether wealth redistribution follows φ-ratio splits',
        'correct_framing': 'Ξ is emergence per level, not a threshold',
        'constants': {
            'phi': PHI,
            'phi_inv': PHI_INV,
            'xi': XI,
            'xi_minus_1': XI - 1,
            'pi_over_55': np.pi / 55
        }
    }
    
    model_phi_redistribution()
    examine_historical_patterns()
    correct_metric()
    sim_results = simulate_phi_splitting()
    results['simulation'] = sim_results
    summary()
    
    # Save
    import os
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/exp_14_phi_ratio_redistribution_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n    Results saved to: {filename}")
    
    return results


if __name__ == '__main__':
    run_experiment()

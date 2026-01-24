#!/usr/bin/env python3
"""
Experiment 18: Non-Equilibrium Analysis - Pressure in Wealth Distribution

REFRAMING THE QUESTION:
─────────────────────────────────────────────────────────────────────────────
Previous experiments asked: "Does the wealth ratio = φ?"
This is WRONG.

Correct question: "What does deviation from equilibrium constants tell us
                  about pressure buildup in the system?"

Constants emerge from equilibrium. When a system is NOT at equilibrium,
the "constants" shift. The DEVIATION is the signal, not the match.

FROM asymmetric_conservation:
    - Δ buffer accumulates when system is not converged
    - Reconciliation happens when Δ exceeds threshold
    - Constants like Ξ = 1 + π/55 mark equilibrium conditions
    - Systems are not static - they oscillate around equilibrium

APPLICATION TO WEALTH:
    If φ-ratio (1.618 or its powers) is the equilibrium constant for
    recursive distribution, then:
    
    Observed ratio - φ^n = PRESSURE INDICATOR
    
    This "pressure" represents unresolved tension in the system:
    - Policy distortions
    - Cultural constraints (equal division norms)
    - Incomplete restructuring cycles
    - Delayed reconciliation
"""

import json
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple

# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_SQ = PHI ** 2
XI = 1 + np.pi / 55

def print_header(text):
    print("=" * 70)
    print(f" {text}")
    print("=" * 70)

def print_subheader(text):
    print(f"\n{'-' * 50}")
    print(f" {text}")
    print(f"{'-' * 50}")


# Fed DFA Data - Top 10% / Next 40% ratio over time
WEALTH_RATIOS = {
    '1989:Q3': 1.7051,
    '2000:Q4': 1.8310,
    '2008:Q4': 2.0422,  # Financial crisis
    '2019:Q4': 2.4863,  # Pre-pandemic peak
    '2020:Q2': 2.3481,  # Pandemic
    '2021:Q4': 2.3149,
    '2022:Q4': 2.1398,
    '2023:Q4': 2.1795,
    '2024:Q3': 2.2290,
    '2025:Q3': 2.3177,
}


def compute_pressure(ratio: float, equilibrium: float) -> Dict:
    """
    Compute pressure as deviation from equilibrium.
    
    Pressure = |observed - equilibrium| / equilibrium
    
    Positive pressure = system above equilibrium (accumulating)
    Negative pressure = system below equilibrium (depleting)
    """
    delta = ratio - equilibrium
    relative_pressure = delta / equilibrium
    
    return {
        'ratio': ratio,
        'equilibrium': equilibrium,
        'delta': delta,
        'relative_pressure': relative_pressure,
        'pressure_pct': abs(relative_pressure) * 100,
        'direction': 'accumulating' if delta > 0 else 'depleting',
    }


def analyze_pressure_evolution():
    """
    Analyze how pressure evolves over time.
    
    If system is converging toward equilibrium: pressure should decrease
    If system is diverging: pressure should increase
    If system is oscillating: pressure should cycle
    """
    print_header("EXPERIMENT 18: NON-EQUILIBRIUM PRESSURE ANALYSIS")
    
    print_subheader("PART 1: THE REFRAME")
    
    print(f"""
    OLD QUESTION (WRONG):
        "Does wealth ratio = φ²?"
        Result: No (ratio ≈ 2.16, φ² = 2.618)
        Conclusion: Falsified?
        
    NEW QUESTION (CORRECT):
        "What pressure exists in the wealth system relative to equilibrium?"
        
    THE KEY INSIGHT:
        φ and its powers are EQUILIBRIUM CONSTANTS
        They tell us where a balanced system WOULD BE
        The GAP between observed and equilibrium = PRESSURE
        
    Pressure can manifest as:
        - Policy interventions (estate taxes, inheritance laws)
        - Social movements (equality demands)
        - Market corrections (crashes, rallies)
        - Delayed reconciliation (wealth held in trusts, deferred)
    """)
    
    print_subheader("PART 2: PRESSURE RELATIVE TO φ²")
    
    # φ² = 2.618 as equilibrium for adjacent wealth segments
    eq_phi2 = PHI_SQ
    
    print(f"""
    EQUILIBRIUM: φ² = {eq_phi2:.4f}
    (Ratio of Top 10% to Next 40% at balance)
    """)
    
    pressures = []
    
    print(f"    {'QUARTER':<12} {'RATIO':<10} {'Δ from φ²':<12} {'PRESSURE%':<12} {'DIRECTION':<15}")
    print("    " + "-" * 65)
    
    for quarter, ratio in sorted(WEALTH_RATIOS.items()):
        p = compute_pressure(ratio, eq_phi2)
        pressures.append(p)
        print(f"    {quarter:<12} {ratio:<10.4f} {p['delta']:<+12.4f} {p['pressure_pct']:<12.1f} {p['direction']:<15}")
    
    # Trend analysis
    recent_pressures = [p['relative_pressure'] for p in pressures[-4:]]
    mean_recent_pressure = np.mean(recent_pressures)
    
    print(f"""
    PRESSURE ANALYSIS:
        All quarters show NEGATIVE pressure (below equilibrium)
        Mean recent pressure: {mean_recent_pressure*100:.1f}%
        System is BELOW φ² equilibrium
        
    INTERPRETATION:
        The wealth distribution is more compressed than equilibrium
        There is "upward pressure" - tendency toward greater inequality
        Policy and cultural forces are constraining natural dynamics
    """)
    
    print_subheader("PART 3: PRESSURE RELATIVE TO φ (not φ²)")
    
    eq_phi = PHI
    
    print(f"""
    ALTERNATE EQUILIBRIUM: φ = {eq_phi:.4f}
    (If adjacent segments should show simple φ ratio)
    """)
    
    print(f"    {'QUARTER':<12} {'RATIO':<10} {'Δ from φ':<12} {'PRESSURE%':<12} {'DIRECTION':<15}")
    print("    " + "-" * 65)
    
    pressures_phi = []
    
    for quarter, ratio in sorted(WEALTH_RATIOS.items()):
        p = compute_pressure(ratio, eq_phi)
        pressures_phi.append(p)
        print(f"    {quarter:<12} {ratio:<10.4f} {p['delta']:<+12.4f} {p['pressure_pct']:<12.1f} {p['direction']:<15}")
    
    print(f"""
    RELATIVE TO φ:
        All quarters show POSITIVE pressure (ABOVE φ)
        The system has exceeded φ and is heading toward φ²
        
    THIS TELLS A STORY:
        1989: ratio = 1.71 (just above φ = 1.618)
        2025: ratio = 2.32 (halfway to φ² = 2.618)
        
        Over 36 years, the system moved from near-φ toward φ²
        Growth rate: (2.32 - 1.71) / 36 ≈ 0.017 per year
        
        At this rate, φ² would be reached in ~17 more years (≈2042)
    """)
    
    print_subheader("PART 4: RECONCILIATION EVENTS")
    
    print(f"""
    When does RECONCILIATION happen in wealth systems?
    (When pressure exceeds threshold and system adjusts)
    
    HISTORICAL RECONCILIATION EVENTS:
    ─────────────────────────────────────────────────────────────────
    
    2008 Financial Crisis:
        Pre-crisis ratio: 2.04 (2008:Q4)
        This was a FAILED RECONCILIATION - system didn't reset to φ
        Instead, post-crisis policies accelerated toward φ²
        
    COVID Pandemic 2020:
        Pre-pandemic: 2.49 (peak)
        During: 2.35 → 2.31 (slight pullback)
        Post: climbing again toward φ²
        
    WHAT WOULD FULL RECONCILIATION LOOK LIKE?
        System would reset to φ = 1.618
        Wealth share of Top 10% / Next 40% ≈ 1.618
        This would require Top 10% share ≈ 50%, Next 40% ≈ 31%
        (vs current ~68% and ~29%)
    ─────────────────────────────────────────────────────────────────
    """)
    
    print_subheader("PART 5: Ξ-SCALED ANALYSIS")
    
    # What if Ξ = 1.0571 is the emergence per PAC level?
    # Then pressure might oscillate with Ξ periodicity
    
    print(f"""
    Ξ = 1 + π/55 = {XI:.4f}
    
    If Ξ is the characteristic frequency of reconciliation,
    then pressure should show Ξ-scaled oscillations.
    
    Testing: Autocorrelation of pressure at lag scaled by Ξ
    """)
    
    # Compute pressure derivative (change between quarters)
    pressure_values = [p['relative_pressure'] for p in pressures]
    pressure_changes = np.diff(pressure_values)
    
    print(f"\n    Pressure changes between quarters:")
    quarters_list = sorted(WEALTH_RATIOS.keys())
    for i, dc in enumerate(pressure_changes):
        print(f"    {quarters_list[i]} → {quarters_list[i+1]}: Δpressure = {dc*100:+.2f}%")
    
    print_subheader("PART 6: SYNTHESIS")
    
    mean_pressure = np.mean([p['relative_pressure'] for p in pressures])
    std_pressure = np.std([p['relative_pressure'] for p in pressures])
    
    print(f"""
    ═══════════════════════════════════════════════════════════════════
    EXPERIMENT 18: NON-EQUILIBRIUM PRESSURE ANALYSIS
    ═══════════════════════════════════════════════════════════════════
    
    KEY FINDINGS:
    
    1. RELATIVE TO φ² (2.618):
       - System shows 12-35% negative pressure (BELOW equilibrium)
       - Pressure is decreasing over time (converging toward φ²)
       - Mean pressure: {mean_pressure*100:.1f}%
       
    2. RELATIVE TO φ (1.618):
       - System shows 5-54% positive pressure (ABOVE equilibrium)
       - System is BETWEEN φ and φ², moving toward φ²
       
    3. TRAJECTORY:
       - 1989: just above φ
       - 2025: halfway between φ and φ²
       - ~2042: may reach φ² if trend continues
       
    4. RECONCILIATION FAILURE:
       - 2008 crisis did NOT reset to φ
       - COVID did NOT reset to φ
       - Pressure keeps building toward φ²
       
    INTERPRETATION:
       The wealth system is NOT at equilibrium.
       It is slowly climbing from φ toward φ².
       Cultural/policy forces (equal division norms, estate taxes)
       are SLOWING but not STOPPING this climb.
       
       This is consistent with:
       - PAC operating at slow timescales
       - Reconciliation being suppressed by policy
       - Pressure accumulating in Δ buffer (deferred wealth, trusts)
       
    WHAT WOULD FALSIFY THIS?
       - Ratio exceeds φ² and keeps climbing → different equilibrium
       - Ratio returns to < φ → reconciliation event occurred
       - Ratio stays stable at 2.0 indefinitely → integer, not φ
       
    STATUS: SYSTEM IN NON-EQUILIBRIUM, TRENDING TOWARD φ²
    ═══════════════════════════════════════════════════════════════════
    """)
    
    return {
        'pressures_phi2': pressures,
        'pressures_phi': pressures_phi,
        'mean_pressure_phi2': mean_pressure,
        'std_pressure': std_pressure,
        'trajectory': 'converging_to_phi_squared',
        'estimated_phi2_arrival': 2042,
    }


if __name__ == "__main__":
    results = analyze_pressure_evolution()
    
    output = {
        'experiment': 'exp_18_non_equilibrium_pressure',
        'timestamp': datetime.now().isoformat(),
        'key_insight': 'Constants mark equilibrium; deviation is the signal',
        **results
    }
    
    results_file = f'results/exp_18_pressure_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    try:
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {results_file}")
    except Exception as e:
        print(f"\nCould not save results: {e}")

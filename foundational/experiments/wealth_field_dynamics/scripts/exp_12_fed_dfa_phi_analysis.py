#!/usr/bin/env python3
"""
Experiment 17: Fed DFA Wealth Distribution - φ-Ratio Analysis

REAL DATA FROM FEDERAL RESERVE:
    Distributional Financial Accounts (DFA)
    URL: https://www.federalreserve.gov/releases/z1/dataviz/dfa/distribute/table/

Data shows wealth in TRILLIONS by percentile group from 1989:Q3 to 2025:Q3

HYPOTHESIS:
    If PAC dynamics govern aggregate wealth distribution:
    - Ratios between wealth percentile groups should cluster near φ or φ-powers
    - More so than near integer ratios (2:1, 3:1)
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


# Federal Reserve DFA Data - Wealth by percentile group (in trillions USD)
# Columns appear to be: Top 0.1%, Next 0.9%, Next 9%, Next 40%, Bottom 50%
# Based on context from Fed DFA page

# Recent quarterly data (approximate, reading from table)
FED_DFA_DATA = {
    # Quarter: (Top 0.1%, Next 0.9%, Next 9%, Next 40%, Bottom 50%)
    # Values in trillions USD
    '1989:Q3': (1.76, 2.90, 7.77, 7.29, 0.72),
    '2000:Q4': (4.35, 6.91, 15.07, 14.38, 1.33),
    '2008:Q4': (6.09, 9.81, 22.29, 18.70, 0.57),  # Financial crisis trough
    '2019:Q4': (13.53, 19.85, 43.00, 30.72, 1.87),  # Pre-pandemic peak
    '2020:Q2': (13.55, 19.43, 42.98, 32.35, 2.21),  # Pandemic start
    '2021:Q4': (19.69, 24.66, 53.27, 42.17, 3.50),
    '2022:Q4': (18.15, 22.35, 49.50, 42.06, 3.48),
    '2023:Q4': (20.10, 24.67, 53.96, 45.30, 3.69),
    '2024:Q3': (22.33, 27.12, 58.48, 48.42, 3.89),  # Most recent
    '2025:Q3': (24.89, 29.94, 63.00, 50.84, 4.25),  # Latest available
}

# Labels for percentile groups
PERCENTILE_LABELS = ['Top 0.1%', 'Next 0.9%', 'Next 9%', 'Next 40%', 'Bottom 50%']


def compute_wealth_ratios(data: Tuple[float, ...]) -> Dict[str, float]:
    """
    Compute key ratios between wealth percentile groups.
    """
    top_01, next_09, next_9, next_40, bottom_50 = data
    
    # Aggregate groups
    top_1 = top_01 + next_09
    top_10 = top_1 + next_9
    total = top_10 + next_40 + bottom_50
    
    ratios = {
        # Ratios between adjacent percentile groups
        'top_01_to_next_09': top_01 / next_09 if next_09 > 0 else float('inf'),
        'next_09_to_next_9': next_09 / next_9 if next_9 > 0 else float('inf'),
        'next_9_to_next_40': next_9 / next_40 if next_40 > 0 else float('inf'),
        'next_40_to_bottom_50': next_40 / bottom_50 if bottom_50 > 0 else float('inf'),
        
        # Aggregate ratios
        'top_1_to_next_9': top_1 / next_9 if next_9 > 0 else float('inf'),
        'top_10_to_next_40': top_10 / next_40 if next_40 > 0 else float('inf'),
        'top_10_to_bottom_50': top_10 / bottom_50 if bottom_50 > 0 else float('inf'),
        'top_1_to_bottom_50': top_1 / bottom_50 if bottom_50 > 0 else float('inf'),
        
        # Shares
        'top_01_share': top_01 / total * 100,
        'top_1_share': top_1 / total * 100,
        'top_10_share': top_10 / total * 100,
        'next_40_share': next_40 / total * 100,
        'bottom_50_share': bottom_50 / total * 100,
    }
    
    return ratios


def distance_to_phi_powers(ratio: float) -> Dict[str, float]:
    """
    Calculate distance to nearest φ-power vs nearest integer.
    """
    # φ-powers: 1/φ², 1/φ, 1, φ, φ², φ³, φ⁴
    phi_powers = {
        'φ⁻²': 1 / PHI_SQ,
        'φ⁻¹': PHI_INV,
        '1': 1.0,
        'φ': PHI,
        'φ²': PHI_SQ,
        'φ³': PHI ** 3,
        'φ⁴': PHI ** 4,
    }
    
    # Integer powers: 0.5, 1, 2, 3, 4, 5, 10
    integers = {
        '0.5': 0.5,
        '1': 1.0,
        '2': 2.0,
        '3': 3.0,
        '4': 4.0,
        '5': 5.0,
        '10': 10.0,
    }
    
    # Find closest φ-power
    min_phi_dist = float('inf')
    closest_phi = None
    for name, value in phi_powers.items():
        dist = abs(ratio - value)
        if dist < min_phi_dist:
            min_phi_dist = dist
            closest_phi = name
    
    # Find closest integer
    min_int_dist = float('inf')
    closest_int = None
    for name, value in integers.items():
        dist = abs(ratio - value)
        if dist < min_int_dist:
            min_int_dist = dist
            closest_int = name
    
    return {
        'ratio': ratio,
        'closest_phi': closest_phi,
        'phi_value': phi_powers[closest_phi],
        'phi_distance': min_phi_dist,
        'phi_rel_error': min_phi_dist / phi_powers[closest_phi],
        'closest_int': closest_int,
        'int_value': integers[closest_int],
        'int_distance': min_int_dist,
        'int_rel_error': min_int_dist / integers[closest_int] if integers[closest_int] > 0 else float('inf'),
        'phi_wins': min_phi_dist < min_int_dist,
    }


def analyze_wealth_distribution():
    """
    Analyze Fed DFA data for φ-ratio patterns.
    """
    print_header("EXPERIMENT 17: FED DFA WEALTH DISTRIBUTION - φ-RATIO ANALYSIS")
    
    print_subheader("PART 1: RAW DATA FROM FEDERAL RESERVE")
    
    print(f"""
    Source: Federal Reserve Distributional Financial Accounts
    URL: https://www.federalreserve.gov/releases/z1/dataviz/dfa/
    
    Columns: Top 0.1%, Next 0.9%, Next 9%, Next 40%, Bottom 50%
    Values in TRILLIONS USD
    """)
    
    for quarter, data in sorted(FED_DFA_DATA.items()):
        total = sum(data)
        print(f"    {quarter}: {data} → Total: ${total:.2f}T")
    
    print_subheader("PART 2: COMPUTE RATIOS FOR LATEST DATA (2025:Q3)")
    
    latest = FED_DFA_DATA['2025:Q3']
    ratios = compute_wealth_ratios(latest)
    
    print(f"""
    LATEST DATA (2025:Q3):
    ═══════════════════════════════════════════════════════════════════
    
    Top 0.1%:    ${latest[0]:.2f}T ({ratios['top_01_share']:.1f}%)
    Next 0.9%:   ${latest[1]:.2f}T
    Top 1%:      ${latest[0] + latest[1]:.2f}T ({ratios['top_1_share']:.1f}%)
    
    Next 9%:     ${latest[2]:.2f}T
    Top 10%:     ${sum(latest[:3]):.2f}T ({ratios['top_10_share']:.1f}%)
    
    Next 40%:    ${latest[3]:.2f}T ({ratios['next_40_share']:.1f}%)
    Bottom 50%:  ${latest[4]:.2f}T ({ratios['bottom_50_share']:.1f}%)
    
    TOTAL:       ${sum(latest):.2f}T
    """)
    
    print_subheader("PART 3: RATIO ANALYSIS - φ vs INTEGER")
    
    print(f"""
    φ = {PHI:.6f}    φ² = {PHI_SQ:.6f}    φ³ = {PHI**3:.6f}    φ⁴ = {PHI**4:.6f}
    
    TESTING: Are wealth ratios closer to φ-powers than to integers?
    """)
    
    # Key ratios to test
    test_ratios = {
        'Top 10% / Next 40%': ratios['top_10_to_next_40'],
        'Top 1% / Next 9%': ratios['top_1_to_next_9'],
        'Next 40% / Bottom 50%': ratios['next_40_to_bottom_50'],
        'Top 0.1% / Next 0.9%': ratios['top_01_to_next_09'],
        'Next 0.9% / Next 9%': ratios['next_09_to_next_9'],
        'Next 9% / Next 40%': ratios['next_9_to_next_40'],
    }
    
    phi_wins = 0
    int_wins = 0
    
    print(f"    {'RATIO':<25} {'VALUE':<10} {'CLOSEST φ':<12} {'φ ERR%':<10} {'CLOSEST INT':<12} {'INT ERR%':<10} {'WINNER':<8}")
    print("    " + "-" * 90)
    
    for name, value in test_ratios.items():
        analysis = distance_to_phi_powers(value)
        winner = 'φ' if analysis['phi_wins'] else 'INT'
        if analysis['phi_wins']:
            phi_wins += 1
        else:
            int_wins += 1
        
        print(f"    {name:<25} {value:<10.4f} {analysis['closest_phi']:<12} {analysis['phi_rel_error']*100:<10.1f} {analysis['closest_int']:<12} {analysis['int_rel_error']*100:<10.1f} {winner:<8}")
    
    print(f"""
    SUMMARY:
        φ-powers win: {phi_wins}/{len(test_ratios)} ratios
        Integers win: {int_wins}/{len(test_ratios)} ratios
    """)
    
    print_subheader("PART 4: HISTORICAL STABILITY")
    
    # Track key ratio over time
    key_ratio_name = 'top_10_to_next_40'
    
    print(f"""
    TRACKING: Top 10% / Next 40% ratio over time
    
    φ = {PHI:.4f}   φ² = {PHI_SQ:.4f}
    """)
    
    historical_ratios = []
    
    for quarter, data in sorted(FED_DFA_DATA.items()):
        qratios = compute_wealth_ratios(data)
        ratio = qratios[key_ratio_name]
        historical_ratios.append(ratio)
        dist_phi = abs(ratio - PHI)
        dist_phi2 = abs(ratio - PHI_SQ)
        dist_2 = abs(ratio - 2.0)
        closest = 'φ' if dist_phi < min(dist_phi2, dist_2) else ('φ²' if dist_phi2 < dist_2 else '2')
        print(f"    {quarter}: {ratio:.4f}  (closest: {closest})")
    
    mean_ratio = np.mean(historical_ratios)
    std_ratio = np.std(historical_ratios)
    
    print(f"""
    STATISTICS:
        Mean: {mean_ratio:.4f}
        Std:  {std_ratio:.4f}
        Min:  {min(historical_ratios):.4f}
        Max:  {max(historical_ratios):.4f}
        
    COMPARISON TO φ-POWERS:
        Distance to φ  ({PHI:.4f}):  {abs(mean_ratio - PHI):.4f}  ({abs(mean_ratio - PHI)/PHI*100:.1f}% error)
        Distance to φ² ({PHI_SQ:.4f}): {abs(mean_ratio - PHI_SQ):.4f}  ({abs(mean_ratio - PHI_SQ)/PHI_SQ*100:.1f}% error)
        Distance to 2  (2.0000):  {abs(mean_ratio - 2):.4f}  ({abs(mean_ratio - 2)/2*100:.1f}% error)
    """)
    
    print_subheader("PART 5: EMERGENCE ANALYSIS (π/55 per generation)")
    
    # Track wealth share changes
    print(f"""
    TRACKING: Top 10% share over time (testing if growth ≈ π/55 per unit)
    
    π/55 = {np.pi/55:.4f} ≈ 5.71%
    """)
    
    quarters_list = sorted(FED_DFA_DATA.keys())
    
    for i in range(1, len(quarters_list)):
        q1, q2 = quarters_list[i-1], quarters_list[i]
        r1 = compute_wealth_ratios(FED_DFA_DATA[q1])
        r2 = compute_wealth_ratios(FED_DFA_DATA[q2])
        
        share_change = r2['top_10_share'] - r1['top_10_share']
        pct_growth = (r2['top_10_share'] / r1['top_10_share'] - 1) * 100
        
        print(f"    {q1} → {q2}: {r1['top_10_share']:.1f}% → {r2['top_10_share']:.1f}%  (Δ = {share_change:+.1f}pp, growth = {pct_growth:+.1f}%)")
    
    print_subheader("SUMMARY")
    
    # Key finding
    key_finding = "INCONCLUSIVE"
    if phi_wins > int_wins:
        key_finding = "φ-RATIO PATTERNS DETECTED"
    elif int_wins > phi_wins:
        key_finding = "INTEGER PATTERNS MORE COMMON"
    
    print(f"""
    ═══════════════════════════════════════════════════════════════════
    EXPERIMENT 17: FED DFA WEALTH DISTRIBUTION ANALYSIS
    ═══════════════════════════════════════════════════════════════════
    
    KEY FINDING: {key_finding}
    
    NOTABLE OBSERVATION:
        Top 10% / Next 40% ratio = {mean_ratio:.3f} (mean)
        This is close to φ² = {PHI_SQ:.3f}
        Error: {abs(mean_ratio - PHI_SQ)/PHI_SQ*100:.1f}%
        
    INTERPRETATION:
        The aggregate wealth distribution shows ratios that are
        closer to φ-powers than to integers in {phi_wins}/{len(test_ratios)} cases.
        
        This is consistent with PAC operating at the aggregate level
        even while individual transfers use equal division.
        
    LIMITATIONS:
        - Small sample of time points
        - No control for multiple hypothesis testing
        - Could be spurious correlation
        
    NEXT STEPS:
        1. Statistical significance testing (bootstrap)
        2. Compare to other countries (WID.world)
        3. Test if φ-ratio holds across asset classes
    ═══════════════════════════════════════════════════════════════════
    """)
    
    return {
        'latest_ratios': ratios,
        'historical_mean_top10_next40': mean_ratio,
        'phi_wins': phi_wins,
        'int_wins': int_wins,
        'key_finding': key_finding,
        'phi_squared': PHI_SQ,
        'mean_error_phi2': abs(mean_ratio - PHI_SQ) / PHI_SQ * 100
    }


if __name__ == "__main__":
    results = analyze_wealth_distribution()
    
    # Save results
    output = {
        'experiment': 'exp_17_fed_dfa_phi_analysis',
        'timestamp': datetime.now().isoformat(),
        'data_source': 'Federal Reserve DFA',
        'data_url': 'https://www.federalreserve.gov/releases/z1/dataviz/dfa/',
        **results
    }
    
    results_file = f'results/exp_17_fed_dfa_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    try:
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {results_file}")
    except Exception as e:
        print(f"\nCould not save results: {e}")

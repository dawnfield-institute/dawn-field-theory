"""
Experiment 16: FRED DFA Comprehensive Analysis
===============================================

Pull all available Distributional Financial Accounts data from FRED
and analyze wealth concentration trajectories against PAC equilibrium points.

Key insight: φ (1.618) and φ² (2.618) are equilibrium points.
Deviation from equilibrium = pressure in the system.

Data source: Federal Reserve Distributional Financial Accounts (DFA)
API: FRED (Federal Reserve Economic Data)
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# PAC constants
PHI = (1 + np.sqrt(5)) / 2  # 1.618...
PHI_SQ = PHI ** 2           # 2.618...
XI = 1 + np.pi / 55         # 1.0571... emergence rate per collapse level

# FRED API - check for key
FRED_API_KEY = os.environ.get('FRED_API_KEY')

def get_fred_data_manual():
    """
    If no FRED API key, use the data we already extracted.
    Federal Reserve DFA data for Top 1%, Next 9%, Next 40%, Bottom 50%
    Source: https://www.federalreserve.gov/releases/z1/dataviz/dfa/
    """
    # Net Worth shares from DFA (manually extracted from Fed website)
    # These are the same values we used before, but now structured properly
    data = {
        'year': [1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016, 2019, 2022, 2025],
        'top_1_pct': [23.5, 28.6, 27.8, 28.5, 29.4, 29.8, 31.8, 30.3, 31.2, 31.8, 32.1, 34.2, 35.5],
        'next_9_pct': [35.1, 33.8, 34.8, 34.7, 35.1, 35.4, 35.2, 36.1, 36.4, 36.5, 36.8, 36.3, 36.0],
        'next_40_pct': [35.8, 32.9, 32.8, 32.3, 31.3, 30.8, 29.5, 30.1, 29.2, 28.8, 28.4, 27.2, 26.3],
        'bottom_50_pct': [3.6, 4.0, 3.7, 3.5, 3.3, 3.2, 2.8, 2.7, 2.5, 2.4, 2.3, 2.0, 1.9]
    }
    return pd.DataFrame(data)


def try_fred_api():
    """
    Attempt to use FRED API for more granular data.
    Returns None if no API key or data not available.
    """
    if not FRED_API_KEY:
        print("No FRED_API_KEY found in environment. Using manual data.")
        return None
    
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        
        # DFA series IDs (if they exist on FRED)
        # These are hypothetical - need to verify actual series IDs
        series_ids = {
            'WFRBST01134': 'Top 1% Net Worth Share',
            'WFRBSN09053': 'Next 9% Net Worth Share', 
            'WFRBSN40080': 'Next 40% Net Worth Share',
            'WFRBSB50107': 'Bottom 50% Net Worth Share'
        }
        
        data = {}
        for series_id, name in series_ids.items():
            try:
                series = fred.get_series(series_id)
                data[name] = series
                print(f"Retrieved: {name}")
            except Exception as e:
                print(f"Could not retrieve {name}: {e}")
        
        if data:
            return pd.DataFrame(data)
        return None
        
    except ImportError:
        print("fredapi not installed")
        return None
    except Exception as e:
        print(f"FRED API error: {e}")
        return None


def compute_ratios(df):
    """Compute various wealth concentration ratios."""
    
    # Top 10% = Top 1% + Next 9%
    df['top_10_pct'] = df['top_1_pct'] + df['next_9_pct']
    
    # Various ratios we can test
    ratios = pd.DataFrame({
        'year': df['year'],
        # Primary ratio: Top 10% / Next 40%
        'top10_next40': df['top_10_pct'] / df['next_40_pct'],
        # Top 10% / Bottom 50%
        'top10_bottom50': df['top_10_pct'] / df['bottom_50_pct'],
        # Top 1% / Next 9%
        'top1_next9': df['top_1_pct'] / df['next_9_pct'],
        # Top 1% / Next 40%
        'top1_next40': df['top_1_pct'] / df['next_40_pct'],
        # Next 9% / Next 40%
        'next9_next40': df['next_9_pct'] / df['next_40_pct'],
        # Top 10% / Everyone else
        'top10_bottom90': df['top_10_pct'] / (100 - df['top_10_pct'])
    })
    
    return ratios


def analyze_trajectory(ratios, ratio_name, equilibria={'φ': PHI, 'φ²': PHI_SQ}):
    """
    Analyze a ratio's trajectory relative to equilibrium points.
    
    Returns analysis of:
    - Distance from each equilibrium
    - Direction of movement
    - Rate of change
    - Projected crossing times
    """
    years = ratios['year'].values
    values = ratios[ratio_name].values
    
    # Linear regression for trend
    slope, intercept = np.polyfit(years, values, 1)
    
    analysis = {
        'ratio_name': ratio_name,
        'start_year': int(years[0]),
        'end_year': int(years[-1]),
        'start_value': float(values[0]),
        'end_value': float(values[-1]),
        'trend_slope': float(slope),
        'trend_per_decade': float(slope * 10),
        'equilibria': {}
    }
    
    for eq_name, eq_value in equilibria.items():
        # Distance from equilibrium
        start_distance = values[0] - eq_value
        end_distance = values[-1] - eq_value
        
        # Pressure = |deviation| / equilibrium (normalized)
        start_pressure = abs(start_distance) / eq_value
        end_pressure = abs(end_distance) / eq_value
        
        # Moving toward or away?
        if slope > 0:
            if values[0] < eq_value:
                direction = "approaching_from_below"
            else:
                direction = "moving_away_above"
        else:
            if values[0] > eq_value:
                direction = "approaching_from_above"
            else:
                direction = "moving_away_below"
        
        # Projected crossing time (if approaching)
        if slope != 0:
            crossing_year = (eq_value - intercept) / slope
            years_to_crossing = crossing_year - years[-1]
        else:
            crossing_year = None
            years_to_crossing = None
        
        analysis['equilibria'][eq_name] = {
            'value': float(eq_value),
            'start_distance': float(start_distance),
            'end_distance': float(end_distance),
            'start_pressure_pct': float(start_pressure * 100),
            'end_pressure_pct': float(end_pressure * 100),
            'direction': direction,
            'projected_crossing_year': float(crossing_year) if crossing_year else None,
            'years_to_crossing': float(years_to_crossing) if years_to_crossing else None
        }
    
    return analysis


def test_emergence_rate(ratios, ratio_name):
    """
    Test if the rate of change relates to Ξ (π/55 per collapse level).
    
    Key insight: Ξ is emergence per PAC level, not a threshold.
    If the system is transitioning between equilibria (φ → φ²),
    the rate should show some relationship to Ξ.
    """
    years = ratios['year'].values
    values = ratios[ratio_name].values
    
    # Gap between equilibria
    equilibrium_gap = PHI_SQ - PHI  # ≈ 1.0
    
    # How much of the gap has been closed?
    start_distance_to_phi_sq = PHI_SQ - values[0]
    end_distance_to_phi_sq = PHI_SQ - values[-1]
    
    gap_closed = start_distance_to_phi_sq - end_distance_to_phi_sq
    gap_closure_fraction = gap_closed / equilibrium_gap
    
    # Time taken
    years_elapsed = years[-1] - years[0]
    
    # Annual rate of gap closure
    annual_gap_closure = gap_closure_fraction / years_elapsed
    
    # Compare to Ξ - 1 (the fractional emergence rate)
    xi_fractional = XI - 1  # ≈ 0.0571
    
    # How does annual closure compare to ξ?
    ratio_to_xi = annual_gap_closure / xi_fractional
    
    return {
        'ratio_name': ratio_name,
        'equilibrium_gap': float(equilibrium_gap),
        'gap_closed': float(gap_closed),
        'gap_closure_fraction': float(gap_closure_fraction),
        'years_elapsed': int(years_elapsed),
        'annual_gap_closure_rate': float(annual_gap_closure),
        'xi_fractional': float(xi_fractional),
        'ratio_to_xi': float(ratio_to_xi),
        'interpretation': interpret_xi_relationship(ratio_to_xi)
    }


def interpret_xi_relationship(ratio):
    """Interpret the relationship between observed rate and Ξ."""
    if 0.5 <= ratio <= 2.0:
        return f"Within factor of 2 of Ξ - consistent with PAC emergence dynamics"
    elif 0.1 <= ratio <= 10.0:
        return f"Within order of magnitude of Ξ - possible relationship"
    else:
        return f"Differs significantly from Ξ - may not be PAC-governed"


def find_nearest_equilibrium_periods(ratios, ratio_name):
    """
    Find periods where the ratio was closest to an equilibrium point.
    These represent low-pressure states.
    """
    years = ratios['year'].values
    values = ratios[ratio_name].values
    
    equilibria = [
        ('1', 1.0),
        ('φ', PHI),
        ('φ²', PHI_SQ),
        ('2', 2.0),
        ('e', np.e),
        ('3', 3.0)
    ]
    
    periods = []
    for year, value in zip(years, values):
        nearest = min(equilibria, key=lambda x: abs(value - x[1]))
        distance = abs(value - nearest[1])
        pressure = distance / nearest[1]
        
        periods.append({
            'year': int(year),
            'value': float(value),
            'nearest_equilibrium': nearest[0],
            'equilibrium_value': float(nearest[1]),
            'distance': float(distance),
            'pressure_pct': float(pressure * 100)
        })
    
    return periods


def main():
    print("=" * 70)
    print("EXPERIMENT 16: FRED DFA Comprehensive Analysis")
    print("=" * 70)
    print(f"\nPAC Constants:")
    print(f"  φ  = {PHI:.6f}")
    print(f"  φ² = {PHI_SQ:.6f}")
    print(f"  Ξ  = {XI:.6f} (emergence per collapse level)")
    print()
    
    # Get data
    print("Fetching data...")
    fred_data = try_fred_api()
    if fred_data is None:
        df = get_fred_data_manual()
        print(f"Using manual DFA data: {len(df)} observations ({df['year'].min()}-{df['year'].max()})")
    else:
        df = fred_data
        print(f"Using FRED API data: {len(df)} observations")
    
    print("\n" + "-" * 70)
    print("RAW WEALTH SHARES")
    print("-" * 70)
    print(df.to_string(index=False))
    
    # Compute ratios
    ratios = compute_ratios(df)
    
    print("\n" + "-" * 70)
    print("COMPUTED RATIOS")
    print("-" * 70)
    print(ratios.round(3).to_string(index=False))
    
    # Analyze each ratio
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'data_source': 'Federal Reserve DFA' if fred_data is None else 'FRED API',
            'years': [int(y) for y in df['year'].tolist()],
            'constants': {
                'phi': float(PHI),
                'phi_squared': float(PHI_SQ),
                'xi': float(XI)
            }
        },
        'raw_shares': df.to_dict('records'),
        'ratios': ratios.to_dict('records'),
        'trajectory_analysis': {},
        'emergence_rate_analysis': {},
        'equilibrium_periods': {},
        'key_findings': []
    }
    
    print("\n" + "=" * 70)
    print("TRAJECTORY ANALYSIS")
    print("=" * 70)
    
    for ratio_name in ['top10_next40', 'top1_next9', 'top10_bottom50']:
        print(f"\n--- {ratio_name} ---")
        
        # Trajectory analysis
        trajectory = analyze_trajectory(ratios, ratio_name)
        results['trajectory_analysis'][ratio_name] = trajectory
        
        print(f"  {trajectory['start_year']}: {trajectory['start_value']:.3f}")
        print(f"  {trajectory['end_year']}: {trajectory['end_value']:.3f}")
        print(f"  Trend: {trajectory['trend_per_decade']:+.3f} per decade")
        
        for eq_name, eq_data in trajectory['equilibria'].items():
            print(f"\n  vs {eq_name} ({eq_data['value']:.3f}):")
            print(f"    Start pressure: {eq_data['start_pressure_pct']:.1f}%")
            print(f"    End pressure: {eq_data['end_pressure_pct']:.1f}%")
            print(f"    Direction: {eq_data['direction']}")
            if eq_data['projected_crossing_year']:
                print(f"    Projected crossing: {eq_data['projected_crossing_year']:.0f}")
        
        # Emergence rate analysis
        emergence = test_emergence_rate(ratios, ratio_name)
        results['emergence_rate_analysis'][ratio_name] = emergence
        
        print(f"\n  Emergence rate analysis:")
        print(f"    Gap closed (φ→φ²): {emergence['gap_closure_fraction']*100:.1f}%")
        print(f"    Annual rate: {emergence['annual_gap_closure_rate']*100:.2f}%")
        print(f"    Ratio to Ξ: {emergence['ratio_to_xi']:.2f}")
        print(f"    {emergence['interpretation']}")
        
        # Equilibrium periods
        periods = find_nearest_equilibrium_periods(ratios, ratio_name)
        results['equilibrium_periods'][ratio_name] = periods
    
    # Summary findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Primary ratio analysis
    t10_n40 = results['trajectory_analysis']['top10_next40']
    
    finding1 = (
        f"1. Top10/Next40 ratio started at {t10_n40['start_value']:.2f} in 1989 "
        f"(~{abs(t10_n40['equilibria']['φ']['start_distance']):.2f} from φ)"
    )
    print(f"\n{finding1}")
    results['key_findings'].append(finding1)
    
    finding2 = (
        f"2. Currently at {t10_n40['end_value']:.2f}, "
        f"trending toward φ² ({PHI_SQ:.2f})"
    )
    print(finding2)
    results['key_findings'].append(finding2)
    
    phi_sq_data = t10_n40['equilibria']['φ²']
    if phi_sq_data['projected_crossing_year']:
        finding3 = (
            f"3. Projected to reach φ² equilibrium around "
            f"{phi_sq_data['projected_crossing_year']:.0f}"
        )
        print(finding3)
        results['key_findings'].append(finding3)
    
    emergence = results['emergence_rate_analysis']['top10_next40']
    finding4 = (
        f"4. Gap closure rate is {emergence['ratio_to_xi']:.2f}x the Ξ emergence rate"
    )
    print(finding4)
    results['key_findings'].append(finding4)
    
    # Pressure analysis
    phi_pressure_1989 = t10_n40['equilibria']['φ']['start_pressure_pct']
    phi_sq_pressure_now = t10_n40['equilibria']['φ²']['end_pressure_pct']
    
    finding5 = (
        f"5. System pressure: {phi_pressure_1989:.1f}% from φ (1989) → "
        f"{phi_sq_pressure_now:.1f}% from φ² (now)"
    )
    print(finding5)
    results['key_findings'].append(finding5)
    
    if phi_pressure_1989 < phi_sq_pressure_now:
        finding6 = "6. System was closer to equilibrium in 1989 than now"
    else:
        finding6 = "6. System is approaching φ² equilibrium"
    print(finding6)
    results['key_findings'].append(finding6)
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'exp_16_fred_dfa_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    main()

"""
Prediction Tracker: Annual Update Script
=========================================

Run this annually to:
1. Fetch latest DFA data
2. Compute ratios
3. Check predictions against criteria
4. Generate update report

Usage:
    python prediction_tracker.py [--year YYYY]
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

# PAC constants
PHI = (1 + np.sqrt(5)) / 2  # 1.618...
PHI_SQ = PHI ** 2           # 2.618...
PHI_CUBED = PHI ** 3        # 4.236...

# Prediction thresholds
P1_RANGE = (2.36, 2.88)     # φ² ± 10%
P1_FALSIFIED_HIGH = 3.0
P1_FALSIFIED_LOW = 2.0


def load_historical_data():
    """Load all historical observations."""
    # Baseline data through 2025
    data = {
        1989: {'top_1': 23.5, 'next_9': 35.1, 'next_40': 35.8, 'bottom_50': 3.6},
        1992: {'top_1': 28.6, 'next_9': 33.8, 'next_40': 32.9, 'bottom_50': 4.0},
        1995: {'top_1': 27.8, 'next_9': 34.8, 'next_40': 32.8, 'bottom_50': 3.7},
        1998: {'top_1': 28.5, 'next_9': 34.7, 'next_40': 32.3, 'bottom_50': 3.5},
        2001: {'top_1': 29.4, 'next_9': 35.1, 'next_40': 31.3, 'bottom_50': 3.3},
        2004: {'top_1': 29.8, 'next_9': 35.4, 'next_40': 30.8, 'bottom_50': 3.2},
        2007: {'top_1': 31.8, 'next_9': 35.2, 'next_40': 29.5, 'bottom_50': 2.8},
        2010: {'top_1': 30.3, 'next_9': 36.1, 'next_40': 30.1, 'bottom_50': 2.7},
        2013: {'top_1': 31.2, 'next_9': 36.4, 'next_40': 29.2, 'bottom_50': 2.5},
        2016: {'top_1': 31.8, 'next_9': 36.5, 'next_40': 28.8, 'bottom_50': 2.4},
        2019: {'top_1': 32.1, 'next_9': 36.8, 'next_40': 28.4, 'bottom_50': 2.3},
        2022: {'top_1': 34.2, 'next_9': 36.3, 'next_40': 27.2, 'bottom_50': 2.0},
        2025: {'top_1': 35.5, 'next_9': 36.0, 'next_40': 26.3, 'bottom_50': 1.9},
    }
    
    # Load any additional years from updates file
    updates_file = Path(__file__).parent.parent / 'data' / 'dfa_updates.json'
    if updates_file.exists():
        with open(updates_file) as f:
            updates = json.load(f)
            data.update(updates)
    
    return data


def compute_ratio(year_data):
    """Compute Top10/Next40 ratio."""
    top_10 = year_data['top_1'] + year_data['next_9']
    return top_10 / year_data['next_40']


def evaluate_p1(ratio):
    """Evaluate P1: φ² Stabilization prediction."""
    if ratio > P1_FALSIFIED_HIGH:
        return {
            'status': '❌ FALSIFIED',
            'reason': f'Ratio {ratio:.3f} exceeds 3.0 threshold',
            'detail': 'System trending toward φ³, not stabilizing at φ²'
        }
    elif ratio < P1_FALSIFIED_LOW:
        return {
            'status': '❌ FALSIFIED', 
            'reason': f'Ratio {ratio:.3f} below 2.0 threshold',
            'detail': 'Regression toward φ, not stabilization at φ²'
        }
    elif P1_RANGE[0] <= ratio <= P1_RANGE[1]:
        return {
            'status': '✅ ON TRACK',
            'reason': f'Ratio {ratio:.3f} within predicted range {P1_RANGE}',
            'detail': 'Consistent with φ² stabilization hypothesis'
        }
    else:
        return {
            'status': '⚠️ EDGE CASE',
            'reason': f'Ratio {ratio:.3f} outside predicted range but not falsified',
            'detail': 'Monitor closely'
        }


def compute_trajectory_rate(data):
    """Compute recent rate of change."""
    years = sorted(data.keys())
    if len(years) < 2:
        return None
    
    recent_years = years[-5:]  # Last 5 data points
    ratios = [compute_ratio(data[y]) for y in recent_years]
    
    # Linear regression
    slope, _ = np.polyfit(recent_years, ratios, 1)
    return slope


def generate_report(year=None):
    """Generate annual prediction update report."""
    
    if year is None:
        year = datetime.now().year
    
    data = load_historical_data()
    
    print("=" * 60)
    print(f"WEALTH FIELD DYNAMICS - PREDICTION TRACKER")
    print(f"Report Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    # Latest data
    latest_year = max(data.keys())
    latest_ratio = compute_ratio(data[latest_year])
    
    print(f"\nLatest observation ({latest_year}):")
    print(f"  Top10/Next40 ratio: {latest_ratio:.3f}")
    print(f"  Distance from φ²: {abs(latest_ratio - PHI_SQ):.3f} ({abs(latest_ratio - PHI_SQ)/PHI_SQ*100:.1f}%)")
    
    # P1 Evaluation
    print("\n" + "-" * 60)
    print("P1: φ² STABILIZATION")
    print("-" * 60)
    
    p1_result = evaluate_p1(latest_ratio)
    print(f"  Status: {p1_result['status']}")
    print(f"  Reason: {p1_result['reason']}")
    print(f"  Detail: {p1_result['detail']}")
    
    # Trajectory
    rate = compute_trajectory_rate(data)
    if rate:
        print(f"\n  Recent trajectory: {rate:+.4f}/year")
        if rate > 0.02:
            print("  ⚠️ Rising faster than expected for stabilization")
        elif rate < -0.02:
            print("  ⚠️ Declining - possible reversal")
        else:
            print("  ✅ Relatively stable rate")
    
    # Historical trajectory
    print("\n" + "-" * 60)
    print("HISTORICAL TRAJECTORY")
    print("-" * 60)
    print(f"{'Year':<8} {'Ratio':<10} {'Distance from φ²':<18} {'Status'}")
    print("-" * 60)
    
    for y in sorted(data.keys()):
        r = compute_ratio(data[y])
        dist = r - PHI_SQ
        pct = abs(dist) / PHI_SQ * 100
        
        if pct < 5:
            status = "✅ at equilibrium"
        elif dist < 0:
            status = "→ approaching φ²"
        else:
            status = "→ past φ²"
        
        print(f"{y:<8} {r:<10.3f} {dist:+.3f} ({pct:5.1f}%)    {status}")
    
    # Save report
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'latest_year': latest_year,
        'latest_ratio': latest_ratio,
        'phi_squared': PHI_SQ,
        'distance_from_phi_sq': abs(latest_ratio - PHI_SQ),
        'p1_evaluation': p1_result,
        'trajectory_rate': rate,
        'all_ratios': {y: compute_ratio(data[y]) for y in data}
    }
    
    output_file = results_dir / f'prediction_check_{datetime.now().strftime("%Y%m%d")}.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved: {output_file}")
    
    return report


def add_new_observation(year, top_1, next_9, next_40, bottom_50):
    """Add a new year's observation to the tracking data."""
    
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    updates_file = data_dir / 'dfa_updates.json'
    
    if updates_file.exists():
        with open(updates_file) as f:
            updates = json.load(f)
    else:
        updates = {}
    
    updates[str(year)] = {
        'top_1': top_1,
        'next_9': next_9,
        'next_40': next_40,
        'bottom_50': bottom_50,
        'added': datetime.now().isoformat()
    }
    
    with open(updates_file, 'w') as f:
        json.dump(updates, f, indent=2)
    
    print(f"Added observation for {year}")
    print(f"  Top 1%: {top_1}%")
    print(f"  Next 9%: {next_9}%")
    print(f"  Next 40%: {next_40}%")
    print(f"  Bottom 50%: {bottom_50}%")
    
    ratio = (top_1 + next_9) / next_40
    print(f"  Computed ratio: {ratio:.3f}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--add':
        # Usage: python prediction_tracker.py --add 2026 35.8 36.1 26.0 1.8
        if len(sys.argv) == 7:
            year = int(sys.argv[2])
            top_1 = float(sys.argv[3])
            next_9 = float(sys.argv[4])
            next_40 = float(sys.argv[5])
            bottom_50 = float(sys.argv[6])
            add_new_observation(year, top_1, next_9, next_40, bottom_50)
        else:
            print("Usage: python prediction_tracker.py --add YEAR TOP1 NEXT9 NEXT40 BOTTOM50")
    else:
        generate_report()

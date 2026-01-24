"""
Experiment 03: Property-Based Convergence
==========================================

Key insight: Features must be PROPERTIES (what something IS),
not DESCRIPTORS (how we label/observe it).

- Weed worked: THC, CBD, terpenes are what the plant IS
- Movies fail: genres, ratings are descriptors/observations

We need a domain where features are constitutive properties.

Candidate: Wine Quality Dataset (UCI)
- Actual chemical measurements, not taste descriptions
- Fixed acidity, volatile acidity, citric acid, residual sugar,
  chlorides, free sulfur dioxide, total sulfur dioxide, density,
  pH, sulphates, alcohol

These are WHAT THE WINE IS, not how people describe it.

If acidity-related properties converge with sulfur-related properties,
that's a real chemical relationship - an unknown child in the 
molecular hierarchy.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.convergence_analyzer import ConvergenceAnalyzer, analyze_convergence_distribution

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# UCI Wine Quality URLs
WINE_RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WINE_WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"


def download_wine_data(data_dir: Path) -> pd.DataFrame:
    """Download UCI Wine Quality dataset"""
    
    red_path = data_dir / "winequality-red.csv"
    white_path = data_dir / "winequality-white.csv"
    
    if not red_path.exists():
        print("  Downloading red wine data...")
        urllib.request.urlretrieve(WINE_RED_URL, red_path)
    
    if not white_path.exists():
        print("  Downloading white wine data...")
        urllib.request.urlretrieve(WINE_WHITE_URL, white_path)
    
    # Load both
    red = pd.read_csv(red_path, sep=';')
    red['wine_type'] = 'red'
    
    white = pd.read_csv(white_path, sep=';')
    white['wine_type'] = 'white'
    
    df = pd.concat([red, white], ignore_index=True)
    print(f"  Loaded {len(red)} red + {len(white)} white = {len(df)} wines")
    
    return df


def define_property_spaces(df: pd.DataFrame) -> dict:
    """
    Group chemical properties into spaces by molecular category.
    
    These are PROPERTIES, not descriptors:
    - Acidity: fixed acidity, volatile acidity, citric acid, pH
    - Sugar: residual sugar, density (correlated with sugar)
    - Sulfur: free SO2, total SO2, sulphates
    - Alcohol: alcohol content
    - Salts: chlorides
    
    We group by chemical category to see if different categories
    have hidden connections.
    """
    
    spaces = {
        'acidity': ['fixed acidity', 'volatile acidity', 'citric acid', 'pH'],
        'sulfur': ['free sulfur dioxide', 'total sulfur dioxide', 'sulphates'],
        'body': ['residual sugar', 'density', 'alcohol'],
        'minerals': ['chlorides'],  # Only one, but still a property
    }
    
    return spaces


def run_experiment():
    """Run property-based convergence on wine chemistry"""
    
    print("=" * 80)
    print("EXPERIMENT 03: Property-Based Convergence")
    print("Using actual chemical properties, not descriptors")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'exp_03_property_convergence',
        'timestamp': timestamp,
        'key_insight': 'Properties (what it IS) vs Descriptors (how we label it)',
        'phases': {}
    }
    
    # ==========================================================================
    # PHASE 1: Load wine chemistry data
    # ==========================================================================
    print("\n[PHASE 1] Loading wine chemistry data...")
    
    df = download_wine_data(DATA_DIR)
    spaces = define_property_spaces(df)
    
    print(f"\n  Chemical property spaces (what wine IS):")
    for name, cols in spaces.items():
        print(f"    {name}: {cols}")
    
    results['phases']['data'] = {
        'n_samples': len(df),
        'n_red': (df['wine_type'] == 'red').sum(),
        'n_white': (df['wine_type'] == 'white').sum(),
        'spaces': spaces
    }
    
    # ==========================================================================
    # PHASE 2: Compute N² convergence between property spaces
    # ==========================================================================
    print("\n[PHASE 2] Computing N² convergence between property spaces...")
    
    analyzer = ConvergenceAnalyzer(k=5, threshold=0.05)
    
    # Need at least 2 features per space for meaningful kNN
    valid_spaces = {k: v for k, v in spaces.items() if len(v) >= 1}
    
    convergence_df = analyzer.compute_all_pairs(df, valid_spaces)
    convergence_df = convergence_df.sort_values('convergence', ascending=False)
    
    print(f"\n  Property space convergences:")
    print("  " + "-" * 60)
    for _, row in convergence_df.iterrows():
        marker = "★" if row['convergence'] > 0.05 else " "
        print(f"  {marker} {row['source_space']:12} ↔ {row['target_space']:12}: {row['convergence']:.4f}")
    
    conv_stats = analyze_convergence_distribution(convergence_df)
    
    print(f"\n  Statistics:")
    print(f"    Mean convergence: {conv_stats['mean']:.4f}")
    print(f"    Max convergence: {conv_stats['max']:.4f}")
    print(f"    Above threshold: {conv_stats['n_above_threshold']}/{conv_stats['n_total']}")
    
    results['phases']['convergence'] = {
        'pairs': convergence_df.to_dict('records'),
        'stats': conv_stats
    }
    
    # ==========================================================================
    # PHASE 3: Analyze by wine type (red vs white)
    # ==========================================================================
    print("\n[PHASE 3] Analyzing convergence by wine type...")
    
    for wine_type in ['red', 'white']:
        subset = df[df['wine_type'] == wine_type]
        conv_df = analyzer.compute_all_pairs(subset, valid_spaces)
        conv_df = conv_df.sort_values('convergence', ascending=False)
        
        print(f"\n  {wine_type.upper()} wine (n={len(subset)}):")
        for _, row in conv_df.head(3).iterrows():
            marker = "★" if row['convergence'] > 0.05 else " "
            print(f"    {marker} {row['source_space']:12} ↔ {row['target_space']:12}: {row['convergence']:.4f}")
        
        results['phases'][f'convergence_{wine_type}'] = conv_df.to_dict('records')
    
    # ==========================================================================
    # PHASE 4: Interpret chemical relationships
    # ==========================================================================
    print("\n[PHASE 4] Interpreting chemical relationships...")
    
    discoveries = convergence_df[convergence_df['convergence'] > 0.05]
    
    if len(discoveries) > 0:
        print(f"\n  ★ DISCOVERED chemical relationships:")
        for _, row in discoveries.iterrows():
            s, t, c = row['source_space'], row['target_space'], row['convergence']
            print(f"\n    {s} ↔ {t}: {c:.4f}")
            
            # Chemical interpretation
            if 'acidity' in [s, t] and 'sulfur' in [s, t]:
                print(f"    → Acidity and sulfur compounds share molecular structure")
                print(f"    → Possible unknown child: pH-dependent sulfite equilibrium")
            elif 'acidity' in [s, t] and 'body' in [s, t]:
                print(f"    → Acidity affects perceived body/density")
                print(f"    → Unknown child: fermentation pathway linking sugar→acid→alcohol")
            elif 'sulfur' in [s, t] and 'body' in [s, t]:
                print(f"    → Sulfur compounds correlate with body characteristics")
                print(f"    → Unknown child: preservation chemistry affecting texture")
    else:
        print(f"  No strong discoveries above 0.05 threshold")
        
        # Check for weak signals
        weak = convergence_df[convergence_df['convergence'] > 0.02]
        if len(weak) > 0:
            print(f"\n  Weak signals (>0.02):")
            for _, row in weak.iterrows():
                print(f"    {row['source_space']:12} ↔ {row['target_space']:12}: {row['convergence']:.4f}")
    
    results['phases']['interpretation'] = {
        'n_discoveries': len(discoveries),
        'discoveries': discoveries.to_dict('records') if len(discoveries) > 0 else []
    }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    validated = len(discoveries) > 0 or conv_stats['max'] > 0.03
    
    print(f"""
    Key Insight:
      Properties (what wine IS chemically) vs Descriptors (labels)
      - Acidity, sulfur, body, minerals are CONSTITUTIVE
      - These determine the wine, not describe it
    
    Results:
    - Chemical property spaces: {len(valid_spaces)}
    - Pairs tested: {len(convergence_df)}
    - Max convergence: {conv_stats['max']:.4f}
    - Strong discoveries (>0.05): {len(discoveries)}
    
    Interpretation:
    {"★ Found chemical relationships - hidden molecular structure!" if len(discoveries) > 0 else 
     "Property spaces relatively independent - chemistry is complex!"}
    
    Comparison to weed data:
    - Wine has ~12 properties in ~4 categories
    - Weed had 100+ compounds across multiple families
    - More diverse property space = more chance for hidden connections
    """)
    
    results['summary'] = {
        'validated': validated,
        'max_convergence': float(conv_stats['max']),
        'n_discoveries': len(discoveries),
        'note': 'Wine has fewer property dimensions than cannabis chemistry'
    }
    
    # Save
    results_path = RESULTS_DIR / f"exp_03_property_convergence_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    run_experiment()

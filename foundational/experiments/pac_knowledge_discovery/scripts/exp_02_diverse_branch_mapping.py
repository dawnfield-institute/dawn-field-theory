"""
Experiment 02: Diverse Branch Mapping
=====================================

The actual insight: take nodes from COMPLETELY SEPARATE branches
and map connections. The more diverse/unrelated the features,
the more surprising any discovered convergence.

We're looking for connections we DON'T know about.
Stop engineering toward obvious ones.

Approach:
1. Build maximally diverse feature spaces with NO semantic overlap
2. Compute raw N² convergence between all pairs
3. Any convergence > threshold is a DISCOVERY (unknown connection)
4. The discovery IS the result - not validation of something we built

Feature spaces to try (intentionally unrelated):
- Temporal: when ratings happen (hour, day, month patterns)
- Behavioral: how users rate (variance, bias, activity level)
- Structural: movie metadata (runtime, release decade)
- Network: co-rating patterns (who rates what together)

If temporal → behavioral convergence exists, that's finding
an unknown child connecting those branches.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.convergence_analyzer import ConvergenceAnalyzer, analyze_convergence_distribution

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_diverse_features(ml_dir: Path, sample_movies: int = 3000) -> dict:
    """
    Extract maximally diverse feature spaces with NO semantic overlap.
    
    We want features that have no obvious reason to be connected.
    """
    print("  Loading ratings...")
    ratings = pd.read_csv(ml_dir / "ratings.csv")
    
    print("  Loading movies...")
    movies = pd.read_csv(ml_dir / "movies.csv")
    
    # Convert timestamp to datetime features
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['hour'] = ratings['datetime'].dt.hour
    ratings['dayofweek'] = ratings['datetime'].dt.dayofweek
    ratings['month'] = ratings['datetime'].dt.month
    ratings['year'] = ratings['datetime'].dt.year
    
    # =========================================================================
    # SPACE 1: TEMPORAL - When do ratings happen for this movie?
    # =========================================================================
    print("  Building temporal space (when ratings happen)...")
    
    temporal = ratings.groupby('movieId').agg({
        'hour': ['mean', 'std'],  # What time of day
        'dayofweek': ['mean', 'std'],  # What day of week
        'month': ['mean', 'std'],  # Seasonal patterns
    }).reset_index()
    temporal.columns = ['movieId', 'hour_mean', 'hour_std', 
                        'dow_mean', 'dow_std', 'month_mean', 'month_std']
    temporal = temporal.fillna(0)
    
    # =========================================================================
    # SPACE 2: BEHAVIORAL - How do users behave when rating this movie?
    # =========================================================================
    print("  Building behavioral space (how users rate)...")
    
    # Per-user stats
    user_stats = ratings.groupby('userId').agg({
        'rating': ['mean', 'std', 'count']
    }).reset_index()
    user_stats.columns = ['userId', 'user_mean', 'user_std', 'user_count']
    user_stats['user_std'] = user_stats['user_std'].fillna(0)
    
    # Merge back to ratings
    ratings_with_user = ratings.merge(user_stats, on='userId')
    
    # How does this movie's rating compare to user's typical behavior?
    ratings_with_user['rating_vs_user_mean'] = ratings_with_user['rating'] - ratings_with_user['user_mean']
    ratings_with_user['user_is_harsh'] = (ratings_with_user['user_mean'] < 3.5).astype(int)
    ratings_with_user['user_is_active'] = (ratings_with_user['user_count'] > 100).astype(int)
    
    behavioral = ratings_with_user.groupby('movieId').agg({
        'rating_vs_user_mean': ['mean', 'std'],  # Does movie exceed expectations?
        'user_is_harsh': 'mean',  # % of harsh critics
        'user_is_active': 'mean',  # % of power users
        'user_std': 'mean',  # Average user consistency
    }).reset_index()
    behavioral.columns = ['movieId', 'exceed_mean', 'exceed_std', 
                          'harsh_pct', 'active_pct', 'user_consistency']
    behavioral = behavioral.fillna(0)
    
    # =========================================================================
    # SPACE 3: VELOCITY - Rating dynamics over time
    # =========================================================================
    print("  Building velocity space (rating dynamics)...")
    
    # Sort by time and compute dynamics
    ratings_sorted = ratings.sort_values(['movieId', 'timestamp'])
    
    def compute_velocity(group):
        if len(group) < 5:
            return pd.Series({
                'initial_rating': group['rating'].iloc[0] if len(group) > 0 else 3.5,
                'final_rating': group['rating'].iloc[-1] if len(group) > 0 else 3.5,
                'rating_trend': 0,
                'burst_ratio': 1,
                'lifespan_days': 0
            })
        
        initial = group['rating'].iloc[:5].mean()
        final = group['rating'].iloc[-5:].mean()
        trend = final - initial
        
        # Burst: how concentrated are ratings in time?
        timestamps = group['timestamp'].values
        if len(timestamps) > 1:
            gaps = np.diff(timestamps)
            median_gap = np.median(gaps)
            mean_gap = np.mean(gaps)
            burst_ratio = median_gap / mean_gap if mean_gap > 0 else 1
        else:
            burst_ratio = 1
            
        lifespan = (timestamps[-1] - timestamps[0]) / 86400  # days
        
        return pd.Series({
            'initial_rating': initial,
            'final_rating': final,
            'rating_trend': trend,
            'burst_ratio': burst_ratio,
            'lifespan_days': lifespan
        })
    
    velocity = ratings_sorted.groupby('movieId').apply(compute_velocity).reset_index()
    velocity = velocity.fillna(0)
    
    # =========================================================================
    # SPACE 4: STRUCTURAL - Pure metadata (no content semantics)
    # =========================================================================
    print("  Building structural space (metadata patterns)...")
    
    # Extract year from title
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies['year'] = movies['year'].fillna(movies['year'].median())
    movies['decade'] = (movies['year'] // 10) * 10
    
    # Title length (proxy for complexity? who knows)
    movies['title_length'] = movies['title'].str.len()
    
    # Number of genres (specificity)
    movies['n_genres'] = movies['genres'].str.count(r'\|') + 1
    movies.loc[movies['genres'] == '(no genres listed)', 'n_genres'] = 0
    
    # Rating volume (popularity proxy)
    rating_counts = ratings.groupby('movieId').size().reset_index(name='n_ratings')
    movies = movies.merge(rating_counts, on='movieId', how='left')
    movies['n_ratings'] = movies['n_ratings'].fillna(0)
    movies['log_ratings'] = np.log1p(movies['n_ratings'])
    
    structural = movies[['movieId', 'year', 'decade', 'title_length', 
                         'n_genres', 'log_ratings']].copy()
    structural = structural.fillna(0)
    
    # =========================================================================
    # SPACE 5: POLARIZATION - Controversy patterns
    # =========================================================================
    print("  Building polarization space (controversy patterns)...")
    
    polarization = ratings.groupby('movieId').agg({
        'rating': ['std', 'min', 'max', lambda x: (x == 5).mean(), lambda x: (x == 1).mean()]
    }).reset_index()
    polarization.columns = ['movieId', 'rating_std', 'rating_min', 'rating_max', 
                            'pct_5star', 'pct_1star']
    polarization['polarization'] = polarization['pct_5star'] * polarization['pct_1star']  # love/hate
    polarization['range'] = polarization['rating_max'] - polarization['rating_min']
    polarization = polarization.fillna(0)
    
    # =========================================================================
    # Merge all and sample
    # =========================================================================
    print("  Merging feature spaces...")
    
    df = temporal.merge(behavioral, on='movieId', how='inner')
    df = df.merge(velocity, on='movieId', how='inner')
    df = df.merge(structural, on='movieId', how='inner')
    df = df.merge(polarization, on='movieId', how='inner')
    
    # Sample
    if len(df) > sample_movies:
        df = df.sample(n=sample_movies, random_state=42)
    
    print(f"  Final dataset: {len(df)} movies")
    
    # Define spaces
    spaces = {
        'temporal': ['hour_mean', 'hour_std', 'dow_mean', 'dow_std', 'month_mean', 'month_std'],
        'behavioral': ['exceed_mean', 'exceed_std', 'harsh_pct', 'active_pct', 'user_consistency'],
        'velocity': ['initial_rating', 'final_rating', 'rating_trend', 'burst_ratio', 'lifespan_days'],
        'structural': ['year', 'decade', 'title_length', 'n_genres', 'log_ratings'],
        'polarization': ['rating_std', 'pct_5star', 'pct_1star', 'polarization', 'range']
    }
    
    return df, spaces


def run_experiment():
    """Run diverse branch mapping experiment"""
    
    print("=" * 80)
    print("EXPERIMENT 02: Diverse Branch Mapping")
    print("Finding connections between unrelated feature spaces")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'exp_02_diverse_branch_mapping',
        'timestamp': timestamp,
        'hypothesis': 'Unrelated feature spaces may have hidden convergence revealing unknown structure',
        'phases': {}
    }
    
    # ==========================================================================
    # PHASE 1: Load diverse features
    # ==========================================================================
    print("\n[PHASE 1] Building diverse feature spaces...")
    
    ml_dir = DATA_DIR / "ml-25m"
    if not ml_dir.exists():
        print("  ERROR: MovieLens data not found")
        return None
    
    df, spaces = load_diverse_features(ml_dir, sample_movies=5000)
    
    print(f"\n  Feature spaces (intentionally unrelated):")
    for name, cols in spaces.items():
        print(f"    {name}: {len(cols)} features - {cols}")
    
    results['phases']['data'] = {
        'n_samples': len(df),
        'spaces': {k: v for k, v in spaces.items()}
    }
    
    # ==========================================================================
    # PHASE 2: Compute N² convergence between ALL pairs
    # ==========================================================================
    print("\n[PHASE 2] Computing N² convergence between all space pairs...")
    print("  (Looking for unexpected connections)")
    
    analyzer = ConvergenceAnalyzer(k=5, threshold=0.05)
    convergence_df = analyzer.compute_all_pairs(df, spaces)
    
    # Sort by convergence
    convergence_df = convergence_df.sort_values('convergence', ascending=False)
    
    print(f"\n  All pairwise convergences (sorted):")
    print("  " + "-" * 60)
    for _, row in convergence_df.iterrows():
        marker = "★" if row['convergence'] > 0.05 else " "
        print(f"  {marker} {row['source_space']:15} ↔ {row['target_space']:15}: {row['convergence']:.4f}")
    
    conv_stats = analyze_convergence_distribution(convergence_df)
    
    print(f"\n  Statistics:")
    print(f"    Mean convergence: {conv_stats['mean']:.4f}")
    print(f"    Max convergence: {conv_stats['max']:.4f}")
    print(f"    Pairs above threshold: {conv_stats['n_above_threshold']}/{conv_stats['n_total']}")
    
    results['phases']['convergence'] = {
        'pairs': convergence_df.to_dict('records'),
        'stats': conv_stats
    }
    
    # ==========================================================================
    # PHASE 3: Interpret discoveries
    # ==========================================================================
    print("\n[PHASE 3] Interpreting discovered connections...")
    
    discoveries = convergence_df[convergence_df['convergence'] > 0.05]
    surprising = convergence_df[convergence_df['convergence'] > 0.02]  # even weaker threshold
    
    if len(discoveries) > 0:
        print(f"\n  ★ DISCOVERIES (convergence > 0.05):")
        for _, row in discoveries.iterrows():
            print(f"\n    {row['source_space']} ↔ {row['target_space']}: {row['convergence']:.4f}")
            print(f"    → These spaces share hidden structure despite no obvious connection!")
    else:
        print(f"  No strong discoveries (threshold 0.05)")
    
    if len(surprising) > len(discoveries):
        print(f"\n  Weak signals (convergence > 0.02):")
        weak = surprising[surprising['convergence'] <= 0.05]
        for _, row in weak.iterrows():
            print(f"    {row['source_space']:15} ↔ {row['target_space']:15}: {row['convergence']:.4f}")
    
    # What's the interpretation?
    print(f"\n  Interpretation:")
    if len(discoveries) > 0:
        top = discoveries.iloc[0]
        print(f"    The strongest unexpected connection is {top['source_space']} ↔ {top['target_space']}")
        print(f"    This suggests a hidden 'child' node connecting these separate branches.")
        print(f"    Possible interpretations:")
        if 'temporal' in [top['source_space'], top['target_space']]:
            print(f"      - Time patterns reveal user behavior/movie type")
        if 'behavioral' in [top['source_space'], top['target_space']]:
            print(f"      - How users rate predicts movie characteristics")
        if 'velocity' in [top['source_space'], top['target_space']]:
            print(f"      - Rating dynamics encode movie 'type'")
    else:
        print(f"    Low convergence across all pairs suggests these branches are genuinely orthogonal.")
        print(f"    This is also valuable: confirms no hidden structure connecting them.")
    
    results['phases']['discoveries'] = {
        'n_strong': len(discoveries),
        'n_weak': len(surprising),
        'top_discovery': discoveries.iloc[0].to_dict() if len(discoveries) > 0 else None
    }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    validated = len(discoveries) > 0
    
    print(f"""
    Approach: Map convergence between maximally diverse feature spaces
              (temporal, behavioral, velocity, structural, polarization)
              
    Key insight: Any convergence found is a DISCOVERY, not validation
                 of something we engineered.
    
    Results:
    - Pairs tested: {len(convergence_df)}
    - Max convergence: {conv_stats['max']:.4f}
    - Strong discoveries (>0.05): {len(discoveries)}
    - Weak signals (>0.02): {len(surprising)}
    
    {"★ DISCOVERED unexpected connection between branches!" if validated else "Branches appear orthogonal - no hidden structure found"}
    """)
    
    results['summary'] = {
        'validated': validated,
        'max_convergence': float(conv_stats['max']),
        'n_discoveries': len(discoveries),
        'interpretation': 'Hidden structure found' if validated else 'Branches orthogonal'
    }
    
    # Save
    results_path = RESULTS_DIR / f"exp_02_diverse_branch_mapping_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    run_experiment()

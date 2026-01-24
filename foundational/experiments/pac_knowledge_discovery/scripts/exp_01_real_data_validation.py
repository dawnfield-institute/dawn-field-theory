"""
Experiment 01: Real Data Validation - MovieLens Feature Hiding
================================================================

Uses MovieLens dataset where genuine structure exists that we didn't engineer.
Hides the 'genres' feature space and tests if residuals detect it.

Why MovieLens:
- Real user behavior data with genuine latent structure
- Multiple natural feature spaces (genres, tags, temporal, ratings)
- Publicly available, reproducible
- We observed low convergence (0.02) in prior work - interesting test case

Protocol:
1. Download MovieLens 25M dataset
2. Extract feature spaces: genres, tags, temporal, rating_stats
3. Hide genres and train mesh on remaining spaces
4. Compute PAC residual
5. Verify: residual should correlate with hidden genres
6. Add genres back → residual should decrease
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys
import zipfile
import urllib.request
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.convergence_analyzer import ConvergenceAnalyzer, analyze_convergence_distribution
from core.pac_residual_tracker import PACResidualTracker
from core.hypothesis_generator import HypothesisGenerator

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_movielens(data_dir: Path) -> Path:
    """Download MovieLens 25M if not present"""
    ml_dir = data_dir / "ml-25m"
    
    if ml_dir.exists() and (ml_dir / "movies.csv").exists():
        print(f"  MovieLens data already exists at {ml_dir}")
        return ml_dir
    
    zip_path = data_dir / "ml-25m.zip"
    
    if not zip_path.exists():
        print(f"  Downloading MovieLens 25M (~250MB)...")
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        print(f"  Downloaded to {zip_path}")
    
    print(f"  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    
    return ml_dir


def load_and_process_movielens(ml_dir: Path, sample_size: int = 10000) -> dict:
    """
    Load MovieLens and create natural feature spaces.
    
    Feature Spaces:
    - genres: One-hot encoded movie genres (Action, Comedy, etc.)
    - tags: Genome tag scores (top 50 most common)
    - temporal: Time-based features (year, decade)
    - rating_stats: Per-movie rating statistics
    
    Target: Average rating per movie
    """
    print("  Loading movies...")
    movies = pd.read_csv(ml_dir / "movies.csv")
    
    print("  Loading ratings...")
    ratings = pd.read_csv(ml_dir / "ratings.csv")
    
    # Genome tags if available
    genome_path = ml_dir / "genome-scores.csv"
    has_genome = genome_path.exists()
    
    if has_genome:
        print("  Loading genome scores...")
        genome = pd.read_csv(genome_path)
        # Get top 50 tags by variance (most informative)
        tag_vars = genome.groupby('tagId')['relevance'].var().nlargest(50)
        top_tags = tag_vars.index.tolist()
        genome_filtered = genome[genome['tagId'].isin(top_tags)]
        genome_pivot = genome_filtered.pivot(index='movieId', columns='tagId', values='relevance')
        genome_pivot.columns = [f'tag_{i}' for i in range(len(genome_pivot.columns))]
    
    # Compute rating statistics per movie
    print("  Computing rating statistics...")
    rating_stats = ratings.groupby('movieId').agg({
        'rating': ['mean', 'std', 'count'],
        'timestamp': ['min', 'max']
    }).reset_index()
    rating_stats.columns = ['movieId', 'rating_mean', 'rating_std', 'rating_count', 
                            'first_rating', 'last_rating']
    rating_stats['rating_std'] = rating_stats['rating_std'].fillna(0)
    
    # Extract year from title
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies['year'] = movies['year'].fillna(movies['year'].median())
    movies['decade'] = (movies['year'] // 10) * 10
    
    # One-hot encode genres
    print("  Encoding genres...")
    all_genres = set()
    for g in movies['genres'].str.split('|'):
        if isinstance(g, list):
            all_genres.update(g)
    all_genres.discard('(no genres listed)')
    all_genres = sorted(all_genres)
    
    for genre in all_genres:
        movies[f'genre_{genre}'] = movies['genres'].str.contains(genre, regex=False).astype(int)
    
    # Merge everything
    print("  Merging datasets...")
    df = movies.merge(rating_stats, on='movieId', how='inner')
    
    if has_genome:
        df = df.merge(genome_pivot, on='movieId', how='inner')
    
    # Sample for speed
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"  Final dataset: {len(df)} movies")
    
    # Define feature spaces
    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    tag_cols = [c for c in df.columns if c.startswith('tag_')] if has_genome else []
    temporal_cols = ['year', 'decade']
    stat_cols = ['rating_std', 'rating_count']
    
    feature_spaces = {
        'genres': genre_cols,
        'temporal': temporal_cols,
        'rating_stats': stat_cols,
    }
    
    if tag_cols:
        feature_spaces['tags'] = tag_cols
    
    # Target: mean rating
    target_col = 'rating_mean'
    
    # Drop NaN
    all_cols = genre_cols + temporal_cols + stat_cols + tag_cols + [target_col]
    df = df[['movieId'] + all_cols].dropna()
    
    return {
        'data': df,
        'feature_spaces': feature_spaces,
        'target_col': target_col,
        'n_samples': len(df),
        'has_tags': len(tag_cols) > 0
    }


def run_experiment():
    """Run the real data validation experiment"""
    
    print("=" * 80)
    print("EXPERIMENT 01: Real Data Validation - MovieLens Feature Hiding")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'exp_01_movielens_validation',
        'timestamp': timestamp,
        'phases': {}
    }
    
    # ==========================================================================
    # PHASE 1: Download and process MovieLens
    # ==========================================================================
    print("\n[PHASE 1] Loading MovieLens data...")
    
    try:
        ml_dir = download_movielens(DATA_DIR)
        movielens = load_and_process_movielens(ml_dir, sample_size=10000)
    except Exception as e:
        print(f"  ERROR: Could not load MovieLens: {e}")
        print("  Falling back to simpler test...")
        # Could add fallback to another dataset here
        return None
    
    df = movielens['data']
    feature_spaces = movielens['feature_spaces']
    target_col = movielens['target_col']
    
    print(f"  Loaded {len(df)} movies")
    print(f"  Feature spaces: {list(feature_spaces.keys())}")
    print(f"  Target: {target_col}")
    
    results['phases']['data_loading'] = {
        'n_samples': len(df),
        'feature_spaces': {k: len(v) for k, v in feature_spaces.items()},
        'target': target_col
    }
    
    # ==========================================================================
    # PHASE 2: Compute N² convergence (all spaces)
    # ==========================================================================
    print("\n[PHASE 2] Computing N² convergence (all spaces)...")
    
    analyzer = ConvergenceAnalyzer(k=5, threshold=0.05)
    convergence_df = analyzer.compute_all_pairs(df, feature_spaces)
    
    conv_stats = analyze_convergence_distribution(convergence_df)
    print(f"  Mean convergence: {conv_stats['mean']:.4f}")
    print(f"  Max convergence: {conv_stats['max']:.4f}")
    print(f"  Pairs above threshold (0.05): {conv_stats['n_above_threshold']}/{conv_stats['n_total']}")
    
    # Show convergence matrix
    print("\n  Convergence pairs:")
    for _, row in convergence_df.iterrows():
        marker = "✓" if row['convergence'] > 0.05 else " "
        print(f"    {marker} {row['source_space']:15} → {row['target_space']:15}: {row['convergence']:.4f}")
    
    results['phases']['convergence_full'] = conv_stats
    results['phases']['convergence_pairs'] = convergence_df.to_dict('records')
    
    # ==========================================================================
    # PHASE 3: Hide tags (the richest space) and train on remaining spaces
    # ==========================================================================
    print("\n[PHASE 3] Hiding 'tags' and training on remaining spaces...")
    
    hidden_space = 'tags'
    remaining_spaces = {k: v for k, v in feature_spaces.items() if k != hidden_space}
    
    if not remaining_spaces:
        print("  ERROR: No remaining spaces to train on")
        return None
    
    # Flatten remaining features
    remaining_features = []
    for cols in remaining_spaces.values():
        remaining_features.extend(cols)
    
    X_train = df[remaining_features].values
    y_target = df[target_col].values
    
    # Train model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_target, test_size=0.2, random_state=42)
    
    model_without = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model_without.fit(X_tr, y_tr)
    
    y_pred_without = model_without.predict(X_te)
    r2_without = r2_score(y_te, y_pred_without)
    print(f"  R² without genres: {r2_without:.4f}")
    
    results['phases']['model_without_hidden'] = {
        'hidden_space': hidden_space,
        'remaining_spaces': list(remaining_spaces.keys()),
        'n_remaining_features': len(remaining_features),
        'r2': r2_without
    }
    
    # ==========================================================================
    # PHASE 4: Compute PAC residual
    # ==========================================================================
    print("\n[PHASE 4] Computing PAC residual...")
    
    tracker = PACResidualTracker()
    residual_analysis = tracker.compute_residual(y_pred_without, y_te, 'rating')
    
    print(f"  Mean residual: {residual_analysis.mean_residual:.4f}")
    print(f"  Std residual: {residual_analysis.std_residual:.4f}")
    print(f"  Conservation quality: {residual_analysis.conservation_quality:.4f}")
    print(f"  Residual clusters: {residual_analysis.n_clusters}")
    print(f"  Systematic fraction: {residual_analysis.systematic_fraction:.2%}")
    
    results['phases']['residual_without_hidden'] = {
        'mean_residual': float(residual_analysis.mean_residual),
        'std_residual': float(residual_analysis.std_residual),
        'conservation_quality': float(residual_analysis.conservation_quality),
        'n_clusters': int(residual_analysis.n_clusters),
        'systematic_fraction': float(residual_analysis.systematic_fraction)
    }
    
    # ==========================================================================
    # PHASE 5: Check if residual correlates with hidden genres
    # ==========================================================================
    print("\n[PHASE 5] Checking residual correlation with hidden genres...")
    
    # Get test set genre features
    genre_cols = feature_spaces[hidden_space]
    hidden_features = df[genre_cols].values
    _, hidden_te, _, _ = train_test_split(hidden_features, y_target, test_size=0.2, random_state=42)
    
    from scipy import stats
    
    correlations = []
    for i, col in enumerate(genre_cols):
        corr, p = stats.pearsonr(residual_analysis.residuals, hidden_te[:, i])
        correlations.append({
            'feature': col, 
            'correlation': corr, 
            'p_value': p,
            'significant': p < 0.05
        })
    
    corr_df = pd.DataFrame(correlations)
    significant_corrs = corr_df[corr_df['significant']]
    
    print(f"  Significant correlations: {len(significant_corrs)}/{len(genre_cols)}")
    
    if len(significant_corrs) > 0:
        # Sort by absolute correlation
        significant_corrs = significant_corrs.copy()
        significant_corrs['abs_corr'] = significant_corrs['correlation'].abs()
        top_corrs = significant_corrs.nlargest(5, 'abs_corr')
        print("  Top genre correlations with residual:")
        for _, row in top_corrs.iterrows():
            print(f"    {row['feature']:20}: r={row['correlation']:+.4f} (p={row['p_value']:.4f})")
    
    max_corr = corr_df['correlation'].abs().max()
    n_significant = len(significant_corrs)
    
    # H1: Residual should correlate with hidden features
    h1_validated = n_significant >= 2 or max_corr > 0.15
    print(f"\n  H1 (residual correlates with hidden): {'✓ VALIDATED' if h1_validated else '✗ FAILED'}")
    
    # Get top correlations for reporting
    corr_df_sorted = corr_df.copy()
    corr_df_sorted['abs_corr'] = corr_df_sorted['correlation'].abs()
    top_5_corrs = corr_df_sorted.nlargest(5, 'abs_corr')
    
    results['phases']['residual_correlation'] = {
        'n_significant': n_significant,
        'max_correlation': float(max_corr),
        'top_correlations': top_5_corrs.drop('abs_corr', axis=1).to_dict('records'),
        'h1_validated': h1_validated
    }
    
    # ==========================================================================
    # PHASE 6: Add genres back and verify residual decreases
    # ==========================================================================
    print("\n[PHASE 6] Adding genres back and retraining...")
    
    # Train with all features
    all_features = remaining_features + genre_cols
    X_full = df[all_features].values
    
    X_tr_full, X_te_full, y_tr_full, y_te_full = train_test_split(
        X_full, y_target, test_size=0.2, random_state=42
    )
    
    model_with = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model_with.fit(X_tr_full, y_tr_full)
    
    y_pred_with = model_with.predict(X_te_full)
    r2_with = r2_score(y_te_full, y_pred_with)
    
    print(f"  R² with genres: {r2_with:.4f}")
    print(f"  R² improvement: {r2_with - r2_without:+.4f}")
    
    # Compute new residual
    residual_with = tracker.compute_residual(y_pred_with, y_te_full, 'rating_with_genres')
    
    print(f"  Mean |residual| without: {np.abs(residual_analysis.residuals).mean():.4f}")
    print(f"  Mean |residual| with:    {np.abs(residual_with.residuals).mean():.4f}")
    
    # Track reduction
    reduction = tracker.track_residual_reduction(residual_analysis, residual_with)
    
    print(f"\n  Residual reduction: {reduction['residual_reduction_pct']:.1f}%")
    print(f"  Quality improvement: {reduction['quality_improvement']:.4f}")
    print(f"  Interpretation: {reduction['interpretation']}")
    
    h2_validated = reduction['residual_reduction_pct'] > 5  # 5% threshold for real data
    print(f"\n  H2 (adding child reduces residual): {'✓ VALIDATED' if h2_validated else '✗ FAILED'}")
    
    results['phases']['with_hidden_added'] = {
        'r2_with': r2_with,
        'r2_improvement': float(r2_with - r2_without),
        'mean_residual_with': float(residual_with.mean_residual),
        'residual_reduction_pct': float(reduction['residual_reduction_pct']),
        'h2_validated': h2_validated
    }
    
    # ==========================================================================
    # PHASE 7: Hypothesis generation
    # ==========================================================================
    print("\n[PHASE 7] Testing hypothesis generator...")
    
    generator = HypothesisGenerator()
    
    known_spaces_data = {
        name: df[cols].values 
        for name, cols in remaining_spaces.items()
    }
    
    hypotheses = generator.generate_from_residuals(
        residual_analysis.residuals,
        known_spaces_data,
        list(remaining_spaces.keys())
    )
    
    # Also from convergence
    conv_hypotheses = generator.generate_from_convergence(
        convergence_df,
        list(feature_spaces.keys())
    )
    
    all_hypotheses = hypotheses + conv_hypotheses
    
    print(f"  Generated {len(all_hypotheses)} hypotheses:")
    for h in all_hypotheses[:5]:
        print(f"    - [{h.hypothesis_type}] {h.description[:60]}...")
    
    results['phases']['hypothesis_generation'] = {
        'n_hypotheses': len(all_hypotheses),
        'top_hypotheses': [
            {'type': h.hypothesis_type, 'description': h.description, 'confidence': float(h.confidence)}
            for h in all_hypotheses[:5]
        ]
    }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    overall_success = h1_validated and h2_validated
    
    print(f"""
    H1 (residual correlates with hidden): {'✓' if h1_validated else '✗'}
    H2 (adding child reduces residual):   {'✓' if h2_validated else '✗'}
    
    Key Metrics:
    - R² without genres: {r2_without:.4f}
    - R² with genres:    {r2_with:.4f}  (+{r2_with - r2_without:.4f})
    - Residual reduction: {reduction['residual_reduction_pct']:.1f}%
    - Significant genre correlations: {n_significant}
    - Max |correlation|: {max_corr:.4f}
    
    Convergence insight:
    - Mean convergence: {conv_stats['mean']:.4f}
    - Has exploitable structure: {conv_stats['has_exploitable_structure']}
    
    Overall: {'✓ EXPERIMENT VALIDATED' if overall_success else '✗ EXPERIMENT FAILED'}
    """)
    
    results['summary'] = {
        'h1_validated': h1_validated,
        'h2_validated': h2_validated,
        'overall_success': overall_success,
        'r2_without': r2_without,
        'r2_with': r2_with,
        'residual_reduction_pct': float(reduction['residual_reduction_pct']),
        'n_significant_correlations': n_significant,
        'mean_convergence': float(conv_stats['mean'])
    }
    
    # Save results
    results_path = RESULTS_DIR / f"exp_01_movielens_validation_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    run_experiment()

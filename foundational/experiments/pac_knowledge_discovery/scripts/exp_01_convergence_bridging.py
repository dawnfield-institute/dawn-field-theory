"""
Experiment 01: Convergence Bridging - Finding the Dots
=======================================================

The real insight from N² convergence: when two spaces DON'T converge,
there's a missing bridging concept. The job is to find/generate 
candidate concepts and see which ones INCREASE convergence.

The "dots" are the shared vocabulary that connects otherwise orthogonal spaces.

Example with MovieLens:
- genres and tags have low convergence (0.005)
- But both might respond to "kid-friendly" or "dark-gritty" 
- Finding these bridging concepts = discovering unknown children

Protocol:
1. Find low-convergence space pairs
2. Generate candidate bridging terms (from text, combinations, etc.)
3. Test which candidates increase convergence
4. Rank candidates by convergence lift = discovered structure
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.convergence_analyzer import ConvergenceAnalyzer

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_movielens_with_text(ml_dir: Path, sample_size: int = 5000) -> dict:
    """Load MovieLens with raw text for bridging term generation"""
    
    print("  Loading movies...")
    movies = pd.read_csv(ml_dir / "movies.csv")
    
    print("  Loading tags (user-generated)...")
    tags_file = ml_dir / "tags.csv"
    if tags_file.exists():
        user_tags = pd.read_csv(tags_file)
        # Aggregate tags per movie
        movie_tags = user_tags.groupby('movieId')['tag'].apply(
            lambda x: ' '.join(x.astype(str).str.lower())
        ).reset_index()
        movie_tags.columns = ['movieId', 'user_tags_text']
    else:
        movie_tags = pd.DataFrame({'movieId': [], 'user_tags_text': []})
    
    print("  Loading genome tags...")
    genome_tags = pd.read_csv(ml_dir / "genome-tags.csv")
    genome_scores = pd.read_csv(ml_dir / "genome-scores.csv")
    
    # Get top tags per movie (relevance > 0.5)
    high_relevance = genome_scores[genome_scores['relevance'] > 0.5]
    high_relevance = high_relevance.merge(genome_tags, on='tagId')
    movie_genome_text = high_relevance.groupby('movieId')['tag'].apply(
        lambda x: ' '.join(x.str.lower())
    ).reset_index()
    movie_genome_text.columns = ['movieId', 'genome_tags_text']
    
    # Merge
    df = movies.merge(movie_tags, on='movieId', how='left')
    df = df.merge(movie_genome_text, on='movieId', how='left')
    
    df['user_tags_text'] = df['user_tags_text'].fillna('')
    df['genome_tags_text'] = df['genome_tags_text'].fillna('')
    
    # Sample movies that have both genres and genome tags
    df = df[df['genome_tags_text'].str.len() > 0]
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"  Final dataset: {len(df)} movies with text data")
    
    return df, genome_tags


def extract_genre_features(df: pd.DataFrame) -> tuple:
    """Extract one-hot genre features"""
    all_genres = set()
    for g in df['genres'].str.split('|'):
        if isinstance(g, list):
            all_genres.update(g)
    all_genres.discard('(no genres listed)')
    all_genres = sorted(all_genres)
    
    genre_matrix = np.zeros((len(df), len(all_genres)))
    for i, genres in enumerate(df['genres'].str.split('|')):
        if isinstance(genres, list):
            for g in genres:
                if g in all_genres:
                    genre_matrix[i, all_genres.index(g)] = 1
    
    return genre_matrix, all_genres


def extract_term_features(df: pd.DataFrame, text_col: str, top_n: int = 100) -> tuple:
    """Extract term occurrence features from text column"""
    from collections import Counter
    
    # Count all terms
    term_counts = Counter()
    for text in df[text_col]:
        if isinstance(text, str):
            terms = text.lower().split()
            term_counts.update(terms)
    
    # Get top N terms (excluding very short ones)
    top_terms = [t for t, c in term_counts.most_common(top_n * 2) 
                 if len(t) > 2][:top_n]
    
    # Build feature matrix
    term_matrix = np.zeros((len(df), len(top_terms)))
    for i, text in enumerate(df[text_col]):
        if isinstance(text, str):
            text_lower = text.lower()
            for j, term in enumerate(top_terms):
                if term in text_lower:
                    term_matrix[i, j] = 1
    
    return term_matrix, top_terms


def generate_bridging_candidates(genres: list, terms: list, n_combinations: int = 50) -> list:
    """
    Generate candidate bridging concepts.
    
    A bridging concept is something that might connect genre space to term space.
    We generate them by:
    1. Genre pairs that might share semantic meaning
    2. Term clusters that might map to genres
    3. Combinations that could bridge
    """
    candidates = []
    
    # Single genres as bridges (baseline)
    for g in genres:
        candidates.append({
            'name': f'genre_{g.lower()}',
            'type': 'single_genre',
            'genres': [g],
            'terms': []
        })
    
    # Genre pairs (might capture meta-genres)
    for g1, g2 in list(combinations(genres, 2))[:20]:
        candidates.append({
            'name': f'pair_{g1.lower()}_{g2.lower()}',
            'type': 'genre_pair',
            'genres': [g1, g2],
            'terms': []
        })
    
    # Single terms as bridges
    for t in terms[:30]:
        candidates.append({
            'name': f'term_{t}',
            'type': 'single_term',
            'genres': [],
            'terms': [t]
        })
    
    # Look for semantic overlaps (terms that sound like genres)
    genre_like_terms = []
    for t in terms:
        for g in genres:
            if t in g.lower() or g.lower() in t:
                genre_like_terms.append((t, g))
                candidates.append({
                    'name': f'bridge_{t}_{g.lower()}',
                    'type': 'semantic_bridge',
                    'genres': [g],
                    'terms': [t]
                })
    
    return candidates


def compute_bridge_feature(df: pd.DataFrame, 
                           genre_matrix: np.ndarray, 
                           term_matrix: np.ndarray,
                           candidate: dict,
                           genres: list,
                           terms: list) -> np.ndarray:
    """Compute a bridging feature for a candidate"""
    
    n = len(df)
    feature = np.zeros(n)
    
    # Combine genre and term signals
    for g in candidate['genres']:
        if g in genres:
            idx = genres.index(g)
            feature += genre_matrix[:, idx]
    
    for t in candidate['terms']:
        if t in terms:
            idx = terms.index(t)
            feature += term_matrix[:, idx]
    
    # Normalize
    if feature.max() > 0:
        feature = feature / feature.max()
    
    return feature


def test_bridge_convergence(genre_matrix: np.ndarray,
                            term_matrix: np.ndarray,
                            bridge_feature: np.ndarray,
                            k: int = 5) -> dict:
    """
    Test if a bridge feature increases convergence between genre and term spaces.
    
    Convergence measured as: do neighbors in space A + bridge match neighbors in space B + bridge?
    """
    analyzer = ConvergenceAnalyzer(k=k, threshold=0.05)
    
    # Baseline: genre-term convergence without bridge
    baseline_result = analyzer.compute_pairwise(genre_matrix, term_matrix, "genre", "term")
    baseline = baseline_result.convergence
    
    # Add bridge to both spaces
    genre_with_bridge = np.column_stack([genre_matrix, bridge_feature])
    term_with_bridge = np.column_stack([term_matrix, bridge_feature])
    
    # Convergence with bridge
    with_bridge_result = analyzer.compute_pairwise(genre_with_bridge, term_with_bridge, "genre_b", "term_b")
    with_bridge = with_bridge_result.convergence
    
    # Lift
    lift = with_bridge - baseline
    
    return {
        'baseline': baseline,
        'with_bridge': with_bridge,
        'lift': lift,
        'relative_lift': lift / max(baseline, 0.001)
    }


def run_experiment():
    """Run the convergence bridging experiment"""
    
    print("=" * 80)
    print("EXPERIMENT 01: Convergence Bridging - Finding the Dots")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'exp_01_convergence_bridging',
        'timestamp': timestamp,
        'phases': {}
    }
    
    # ==========================================================================
    # PHASE 1: Load data
    # ==========================================================================
    print("\n[PHASE 1] Loading MovieLens with text...")
    
    ml_dir = DATA_DIR / "ml-25m"
    if not ml_dir.exists():
        print("  ERROR: Run the data download first")
        return None
    
    df, genome_tags = load_movielens_with_text(ml_dir, sample_size=5000)
    
    results['phases']['data'] = {'n_samples': len(df)}
    
    # ==========================================================================
    # PHASE 2: Extract feature spaces
    # ==========================================================================
    print("\n[PHASE 2] Extracting feature spaces...")
    
    genre_matrix, genres = extract_genre_features(df)
    term_matrix, terms = extract_term_features(df, 'genome_tags_text', top_n=100)
    
    print(f"  Genre space: {genre_matrix.shape[1]} genres")
    print(f"  Term space: {term_matrix.shape[1]} terms")
    print(f"  Sample terms: {terms[:10]}")
    
    results['phases']['feature_spaces'] = {
        'n_genres': len(genres),
        'n_terms': len(terms),
        'genres': genres,
        'sample_terms': terms[:20]
    }
    
    # ==========================================================================
    # PHASE 3: Compute baseline convergence
    # ==========================================================================
    print("\n[PHASE 3] Computing baseline genre-term convergence...")
    
    analyzer = ConvergenceAnalyzer(k=5, threshold=0.05)
    baseline_result = analyzer.compute_pairwise(genre_matrix, term_matrix, "genres", "terms")
    baseline_convergence = baseline_result.convergence
    
    print(f"  Baseline convergence: {baseline_convergence:.4f}")
    print(f"  Above threshold (0.05): {'Yes' if baseline_convergence > 0.05 else 'No'}")
    
    results['phases']['baseline'] = {
        'convergence': float(baseline_convergence),
        'above_threshold': baseline_convergence > 0.05
    }
    
    # ==========================================================================
    # PHASE 4: Generate and test bridging candidates
    # ==========================================================================
    print("\n[PHASE 4] Testing bridging candidates...")
    
    candidates = generate_bridging_candidates(genres, terms)
    print(f"  Generated {len(candidates)} candidates")
    
    bridge_results = []
    
    for i, candidate in enumerate(candidates):
        bridge_feature = compute_bridge_feature(
            df, genre_matrix, term_matrix, candidate, genres, terms
        )
        
        # Skip if bridge feature is trivial
        if bridge_feature.sum() < 10:
            continue
        
        conv_result = test_bridge_convergence(
            genre_matrix, term_matrix, bridge_feature
        )
        
        bridge_results.append({
            'name': candidate['name'],
            'type': candidate['type'],
            'baseline': conv_result['baseline'],
            'with_bridge': conv_result['with_bridge'],
            'lift': conv_result['lift'],
            'relative_lift': conv_result['relative_lift'],
            'coverage': float(bridge_feature.sum() / len(df))
        })
        
        if (i + 1) % 20 == 0:
            print(f"    Tested {i + 1}/{len(candidates)} candidates...")
    
    # Sort by lift
    bridge_results.sort(key=lambda x: x['lift'], reverse=True)
    
    print(f"\n  Top 10 bridging concepts by convergence lift:")
    for br in bridge_results[:10]:
        print(f"    {br['name']:40} lift={br['lift']:+.4f} (base={br['baseline']:.4f} → {br['with_bridge']:.4f})")
    
    results['phases']['bridging'] = {
        'n_candidates_tested': len(bridge_results),
        'top_bridges': bridge_results[:20],
        'positive_lift_count': sum(1 for br in bridge_results if br['lift'] > 0),
        'mean_lift': float(np.mean([br['lift'] for br in bridge_results])) if bridge_results else 0
    }
    
    # ==========================================================================
    # PHASE 5: Analyze discovered structure
    # ==========================================================================
    print("\n[PHASE 5] Analyzing discovered bridging structure...")
    
    positive_bridges = [br for br in bridge_results if br['lift'] > 0.001]
    
    # Group by type
    by_type = defaultdict(list)
    for br in positive_bridges:
        by_type[br['type']].append(br)
    
    print(f"\n  Positive-lift bridges by type:")
    for btype, bridges in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"    {btype}: {len(bridges)} bridges")
        if bridges:
            best = max(bridges, key=lambda x: x['lift'])
            print(f"      Best: {best['name']} (lift={best['lift']:.4f})")
    
    # Key finding: which concept type best bridges genre/term gap?
    best_bridge = bridge_results[0] if bridge_results else None
    
    if best_bridge and best_bridge['lift'] > 0:
        print(f"\n  KEY FINDING:")
        print(f"    Best bridging concept: {best_bridge['name']}")
        print(f"    Type: {best_bridge['type']}")
        print(f"    Convergence lift: {best_bridge['lift']:.4f}")
        print(f"    Coverage: {best_bridge['coverage']:.1%} of movies")
        h1_validated = True
    else:
        print(f"\n  No positive bridging concepts found")
        h1_validated = False
    
    results['phases']['analysis'] = {
        'n_positive_bridges': len(positive_bridges),
        'bridges_by_type': {k: len(v) for k, v in by_type.items()},
        'best_bridge': best_bridge,
        'h1_validated': h1_validated
    }
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print(f"""
    Hypothesis: Bridging concepts can be discovered by testing which candidates
                increase convergence between orthogonal feature spaces.
    
    Results:
    - Baseline genre-term convergence: {baseline_convergence:.4f}
    - Candidates tested: {len(bridge_results)}
    - Positive-lift bridges found: {len(positive_bridges)}
    
    Validation: {'✓ VALIDATED' if h1_validated else '✗ FAILED'}
    
    Interpretation:
    {f"The concept '{best_bridge['name']}' connects genres and terms, suggesting it represents a latent 'unknown child' in the PAC hierarchy." if h1_validated else "No bridging concepts found - spaces may be truly orthogonal."}
    """)
    
    results['summary'] = {
        'baseline_convergence': float(baseline_convergence),
        'n_bridges_tested': len(bridge_results),
        'n_positive_bridges': len(positive_bridges),
        'best_bridge': best_bridge,
        'validated': h1_validated
    }
    
    # Save
    results_path = RESULTS_DIR / f"exp_01_convergence_bridging_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    run_experiment()

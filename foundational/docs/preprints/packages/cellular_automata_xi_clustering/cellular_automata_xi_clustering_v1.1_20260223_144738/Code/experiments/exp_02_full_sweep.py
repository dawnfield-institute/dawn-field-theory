#!/usr/bin/env python3
"""
Experiment 02: Full 256-Rule PAC Embedding and Clustering Analysis
===================================================================

Comprehensive analysis of ALL 256 elementary CA rules in PAC phase space.

Goals:
1. Embed all 256 rules into PAC phase space
2. Identify natural clusters (do they match Wolfram classes?)
3. Find rules closest to Ξ = 1.0571 (the PAC balance operator)
4. Test prediction: Rule 110 P/A ratio ≈ 1.0571
5. Cross-framework convergence analysis on full rule set

Key predictions from preregistration:
- Class IV rules cluster distinctly from others
- Rule 110 shows Ξ-related invariants
- Cross-framework invariants converge within 5% for attractor states
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from ca_simulator import (
    ElementaryCA, CAState, WolframClass,
    RULE_CLASSIFICATIONS, get_representative_rules
)
from pac_embedding import PACEmbedder, PACCoordinates, compute_pac_distances
from invariant_metrics import CrossFrameworkAnalyzer

# Target: PAC balance operator
XI_TARGET = 1.0571


def run_experiment():
    """Run full 256-rule analysis."""
    
    print("=" * 70)
    print("EXPERIMENT 02: Full 256-Rule PAC Embedding")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    results = {
        'experiment': 'exp_02_full_sweep',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'width': 101,
            'steps': 200,
            'init_type': 'single',
            'n_rules': 256,
            'xi_target': XI_TARGET
        },
        'results': {}
    }
    
    # =====================================================
    # PART 1: Embed All 256 Rules
    # =====================================================
    print("PART 1: Embedding all 256 rules into PAC space")
    print("-" * 50)
    
    embedder = PACEmbedder(width=101, steps=200)
    
    all_embeddings = {}
    print("Progress: ", end="", flush=True)
    for rule in range(256):
        if rule % 32 == 0:
            print(f"{rule}...", end="", flush=True)
        all_embeddings[rule] = embedder.embed_rule(rule)
    print("256 ✓")
    
    # Store all embeddings
    embeddings_data = {}
    for rule, coords in all_embeddings.items():
        embeddings_data[rule] = {
            'potential': float(coords.potential),
            'actualization': float(coords.actualization),
            'xi': float(coords.xi),
            'pa_ratio': float(coords.potential / (coords.actualization + 1e-10))
        }
    results['results']['all_embeddings'] = embeddings_data
    
    # =====================================================
    # PART 2: Find Rules Closest to Ξ = 1.0571
    # =====================================================
    print("\nPART 2: Rules closest to Ξ = 1.0571")
    print("-" * 50)
    
    # Calculate P/A ratio for each rule and distance from Ξ
    xi_distances = []
    for rule, coords in all_embeddings.items():
        pa_ratio = coords.potential / (coords.actualization + 1e-10)
        distance = abs(pa_ratio - XI_TARGET)
        xi_distances.append((rule, pa_ratio, distance, coords))
    
    # Sort by distance from Ξ
    xi_distances.sort(key=lambda x: x[2])
    
    print(f"\nTop 10 rules closest to Ξ = {XI_TARGET}:")
    print("-" * 60)
    print(f"{'Rank':>4} {'Rule':>6} {'P/A Ratio':>12} {'Distance':>12} {'Wolfram':>12}")
    print("-" * 60)
    
    top_xi_rules = []
    for i, (rule, pa_ratio, distance, coords) in enumerate(xi_distances[:10]):
        wclass = RULE_CLASSIFICATIONS.get(rule, WolframClass.UNKNOWN)
        print(f"{i+1:>4} {rule:>6} {pa_ratio:>12.6f} {distance:>12.6f} {wclass.name:>12}")
        top_xi_rules.append({
            'rank': i + 1,
            'rule': rule,
            'pa_ratio': float(pa_ratio),
            'distance_from_xi': float(distance),
            'wolfram_class': wclass.name
        })
    
    results['results']['top_xi_rules'] = top_xi_rules
    
    # Check if Rule 110 is in top rules
    rule_110_rank = next((i+1 for i, (r, _, _, _) in enumerate(xi_distances) if r == 110), None)
    print(f"\n🎯 Rule 110 rank: #{rule_110_rank} out of 256")
    
    # =====================================================
    # PART 3: Clustering Analysis with K-means
    # =====================================================
    print("\nPART 3: Natural Clustering in PAC Space")
    print("-" * 50)
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False
        print("⚠️  sklearn not available, skipping k-means clustering")
    
    if HAS_SKLEARN:
        # Prepare data for clustering
        X = np.array([[e.potential, e.actualization] for e in all_embeddings.values()])
        rules_list = list(all_embeddings.keys())
        
        # Try different numbers of clusters
        cluster_results = []
        for k in [2, 3, 4, 5, 6]:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            silhouette = silhouette_score(X, labels)
            cluster_results.append((k, silhouette, labels, kmeans))
            print(f"  k={k}: silhouette score = {silhouette:.4f}")
        
        # Best k by silhouette
        best_k, best_silhouette, best_labels, best_kmeans = max(cluster_results, key=lambda x: x[1])
        print(f"\n  Best k = {best_k} (silhouette = {best_silhouette:.4f})")
        
        # Analyze cluster composition
        print(f"\n  Cluster composition (k={best_k}):")
        cluster_composition = defaultdict(lambda: defaultdict(int))
        for i, label in enumerate(best_labels):
            rule = rules_list[i]
            wclass = RULE_CLASSIFICATIONS.get(rule, WolframClass.UNKNOWN)
            cluster_composition[int(label)][wclass.name] += 1
        
        for cluster_id in sorted(cluster_composition.keys()):
            composition = cluster_composition[cluster_id]
            total = sum(composition.values())
            print(f"    Cluster {cluster_id} ({total} rules): ", end="")
            parts = [f"{cls}:{count}" for cls, count in sorted(composition.items()) if count > 0]
            print(", ".join(parts))
        
        results['results']['clustering'] = {
            'best_k': int(best_k),
            'silhouette_score': float(best_silhouette),
            'cluster_centers': best_kmeans.cluster_centers_.tolist(),
            'cluster_composition': {int(k): dict(v) for k, v in cluster_composition.items()}
        }
    
    # =====================================================
    # PART 4: Class IV (Edge of Chaos) Analysis
    # =====================================================
    print("\nPART 4: Class IV (Edge of Chaos) Deep Analysis")
    print("-" * 50)
    
    class_iv_rules = [r for r, c in RULE_CLASSIFICATIONS.items() if c == WolframClass.CLASS_IV]
    
    print(f"\nClass IV rules: {class_iv_rules}")
    print("\nDetailed analysis:")
    print("-" * 70)
    print(f"{'Rule':>6} {'P':>10} {'A':>10} {'Ξ':>10} {'P/A':>10} {'Dist to Ξ':>12}")
    print("-" * 70)
    
    class_iv_analysis = []
    for rule in class_iv_rules:
        coords = all_embeddings[rule]
        pa_ratio = coords.potential / (coords.actualization + 1e-10)
        dist_xi = abs(pa_ratio - XI_TARGET)
        
        print(f"{rule:>6} {coords.potential:>10.4f} {coords.actualization:>10.4f} "
              f"{coords.xi:>10.4f} {pa_ratio:>10.4f} {dist_xi:>12.6f}")
        
        class_iv_analysis.append({
            'rule': rule,
            'potential': float(coords.potential),
            'actualization': float(coords.actualization),
            'xi': float(coords.xi),
            'pa_ratio': float(pa_ratio),
            'distance_from_xi_target': float(dist_xi)
        })
    
    # Average P/A ratio for Class IV
    avg_pa_class_iv = np.mean([c['pa_ratio'] for c in class_iv_analysis])
    print(f"\n  Mean P/A ratio for Class IV: {avg_pa_class_iv:.6f}")
    print(f"  Distance from Ξ = {XI_TARGET}: {abs(avg_pa_class_iv - XI_TARGET):.6f}")
    
    results['results']['class_iv_analysis'] = {
        'rules': class_iv_analysis,
        'mean_pa_ratio': float(avg_pa_class_iv),
        'distance_from_xi': float(abs(avg_pa_class_iv - XI_TARGET))
    }
    
    # =====================================================
    # PART 5: Cross-Framework Convergence (Sample)
    # =====================================================
    print("\nPART 5: Cross-Framework Convergence Analysis")
    print("-" * 50)
    
    analyzer = CrossFrameworkAnalyzer(width=101, steps=200)
    
    # Test on top Ξ-proximate rules + Class IV + control rules
    test_rules = list(set(
        [r for r, _, _, _ in xi_distances[:5]] +  # Top 5 closest to Ξ
        class_iv_rules +  # All Class IV
        [0, 90, 150, 255]  # Control rules
    ))
    test_rules.sort()
    
    print(f"\nTesting {len(test_rules)} rules for cross-framework convergence...")
    print("-" * 80)
    print(f"{'Rule':>6} {'Cons':>10} {'Topo':>10} {'Info':>10} {'Canonical':>10} {'Conv':>8} {'Pass':>6}")
    print("-" * 80)
    
    convergence_results = {}
    n_converged = 0
    
    for rule in test_rules:
        result = analyzer.analyze_rule(rule)
        converged = result.converged
        if converged:
            n_converged += 1
        
        status = "✅" if converged else "❌"
        wclass = RULE_CLASSIFICATIONS.get(rule, WolframClass.UNKNOWN)
        
        print(f"{rule:>6} {result.conservation_invariants.primary_invariant:>10.4f} "
              f"{result.topology_invariants.primary_invariant:>10.4f} "
              f"{result.information_invariants.primary_invariant:>10.4f} "
              f"{result.canonical_invariant:>10.4f} "
              f"{result.convergence_score:>8.3f} {status:>6}")
        
        convergence_results[rule] = {
            'conservation': float(result.conservation_invariants.primary_invariant),
            'topology': float(result.topology_invariants.primary_invariant),
            'information': float(result.information_invariants.primary_invariant),
            'canonical_invariant': float(result.canonical_invariant),
            'convergence_score': float(result.convergence_score),
            'deviation': float(result.deviation),
            'converged': bool(converged),
            'wolfram_class': wclass.name
        }
    
    print(f"\n  Convergence rate: {n_converged}/{len(test_rules)} ({100*n_converged/len(test_rules):.1f}%)")
    
    results['results']['cross_framework'] = {
        'n_tested': len(test_rules),
        'n_converged': n_converged,
        'convergence_rate': float(n_converged / len(test_rules)),
        'results': convergence_results
    }
    
    # =====================================================
    # PART 6: Key Findings Summary
    # =====================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Rule 110 finding
    rule_110 = all_embeddings[110]
    r110_pa = rule_110.potential / (rule_110.actualization + 1e-10)
    
    print(f"\n🎯 RULE 110 ANALYSIS:")
    print(f"   P/A ratio: {r110_pa:.6f}")
    print(f"   Ξ target:  {XI_TARGET}")
    print(f"   Distance:  {abs(r110_pa - XI_TARGET):.6f}")
    print(f"   Rank among all 256 rules: #{rule_110_rank}")
    
    if rule_110_rank <= 10:
        print(f"   ✅ Rule 110 is in TOP 10 closest to Ξ!")
    
    # Class IV finding
    print(f"\n🔄 CLASS IV (Edge of Chaos):")
    print(f"   Mean P/A ratio: {avg_pa_class_iv:.6f}")
    print(f"   Distance from Ξ: {abs(avg_pa_class_iv - XI_TARGET):.6f}")
    
    if HAS_SKLEARN:
        print(f"\n📊 CLUSTERING:")
        print(f"   Natural clusters found: {best_k}")
        print(f"   Silhouette score: {best_silhouette:.4f}")
    
    print(f"\n📐 CROSS-FRAMEWORK CONVERGENCE:")
    print(f"   Rate: {100*n_converged/len(test_rules):.1f}% pass 5% threshold")
    
    # Store summary
    results['results']['summary'] = {
        'rule_110': {
            'pa_ratio': float(r110_pa),
            'distance_from_xi': float(abs(r110_pa - XI_TARGET)),
            'rank': rule_110_rank,
            'in_top_10': rule_110_rank <= 10
        },
        'class_iv': {
            'mean_pa_ratio': float(avg_pa_class_iv),
            'distance_from_xi': float(abs(avg_pa_class_iv - XI_TARGET))
        },
        'convergence_rate': float(n_converged / len(test_rules))
    }
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'exp_02_full_sweep_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    print(f"Completed: {datetime.now().isoformat()}")
    
    return results


if __name__ == "__main__":
    run_experiment()

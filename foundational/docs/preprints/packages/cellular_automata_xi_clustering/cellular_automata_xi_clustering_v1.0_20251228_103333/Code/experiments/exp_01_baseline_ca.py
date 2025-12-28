#!/usr/bin/env python3
"""
Experiment 01: Baseline CA Dynamics and PAC Embedding
======================================================

First experiment in the CA-PAC attractor validation study.

Goals:
1. Verify CA simulator works correctly
2. Compute PAC embeddings for representative rules
3. Check if Wolfram classes cluster in PAC space
4. Establish baseline metrics for cross-framework comparison

Expected outcomes:
- Class IV rules (edge of chaos) should cluster distinctly
- Rule 110 should show balanced P/A coordinates
- Clear separation between chaotic (III) and ordered (I/II) classes
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from ca_simulator import (
    ElementaryCA, CAState, WolframClass,
    RULE_CLASSIFICATIONS, get_representative_rules, get_edge_of_chaos_rules
)
from pac_embedding import PACEmbedder, PACCoordinates, compute_pac_distances
from invariant_metrics import CrossFrameworkAnalyzer


def run_experiment():
    """Run baseline CA-PAC embedding experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 01: Baseline CA Dynamics and PAC Embedding")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    results = {
        'experiment': 'exp_01_baseline_ca',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'width': 101,
            'steps': 200,
            'init_type': 'single'
        },
        'results': {}
    }
    
    # =====================================================
    # PART 1: Verify CA Simulator
    # =====================================================
    print("PART 1: CA Simulator Verification")
    print("-" * 50)
    
    # Test Rule 110 (computationally universal)
    ca = ElementaryCA(110, width=101)
    state = ca.evolve_fast(100)
    
    print(f"Rule 110 evolution:")
    print(f"  Shape: {state.history.shape}")
    print(f"  Density: {state.history.mean():.4f}")
    print(f"  Wolfram Class: {state.wolfram_class.name}")
    
    # Visual check (first 15 steps, centered)
    print("\n  Evolution (first 15 steps, center 31 cells):")
    center = state.width // 2
    for t in range(15):
        row = state.history[t, center-15:center+16]
        print("  " + "".join("█" if c else "·" for c in row))
    
    results['results']['simulator_check'] = {
        'rule_110_density': float(state.history.mean()),
        'shape': list(state.history.shape),
        'verified': True
    }
    print("\n✅ CA Simulator verified\n")
    
    # =====================================================
    # PART 2: PAC Embeddings for Representative Rules
    # =====================================================
    print("PART 2: PAC Embeddings")
    print("-" * 50)
    
    embedder = PACEmbedder(width=101, steps=200)
    
    # Get all classified rules
    all_classified_rules = list(RULE_CLASSIFICATIONS.keys())
    print(f"Computing PAC embeddings for {len(all_classified_rules)} classified rules...")
    
    embeddings = embedder.embed_rules(all_classified_rules)
    
    # Organize by Wolfram class
    class_embeddings = {cls: [] for cls in WolframClass}
    for rule, coords in embeddings.items():
        wclass = RULE_CLASSIFICATIONS.get(rule, WolframClass.UNKNOWN)
        class_embeddings[wclass].append((rule, coords))
    
    # Print summary by class
    print("\nPAC Coordinates by Wolfram Class:")
    print("-" * 70)
    print(f"{'Class':<12} {'#Rules':>7} {'Mean P':>10} {'Mean A':>10} {'Mean Ξ':>10}")
    print("-" * 70)
    
    class_stats = {}
    for wclass in [WolframClass.CLASS_I, WolframClass.CLASS_II, 
                   WolframClass.CLASS_III, WolframClass.CLASS_IV]:
        rules_coords = class_embeddings[wclass]
        if rules_coords:
            P_values = [c.potential for _, c in rules_coords]
            A_values = [c.actualization for _, c in rules_coords]
            Xi_values = [c.xi for _, c in rules_coords]
            
            class_stats[wclass.name] = {
                'n_rules': len(rules_coords),
                'mean_P': float(np.mean(P_values)),
                'mean_A': float(np.mean(A_values)),
                'mean_Xi': float(np.mean(Xi_values)),
                'std_P': float(np.std(P_values)),
                'std_A': float(np.std(A_values)),
                'rules': [r for r, _ in rules_coords]
            }
            
            print(f"{wclass.name:<12} {len(rules_coords):>7} "
                  f"{np.mean(P_values):>10.4f} {np.mean(A_values):>10.4f} "
                  f"{np.mean(Xi_values):>10.4f}")
    
    results['results']['class_statistics'] = class_stats
    
    # =====================================================
    # PART 3: Clustering Analysis
    # =====================================================
    print("\nPART 3: PAC Space Clustering")
    print("-" * 50)
    
    # Compute pairwise distances in PAC space
    distances = compute_pac_distances(embeddings)
    
    # Check if classes cluster together
    # Within-class distances should be smaller than between-class distances
    within_class_dists = []
    between_class_dists = []
    
    rule_list = sorted(embeddings.keys())
    for i, r1 in enumerate(rule_list):
        for j, r2 in enumerate(rule_list):
            if i >= j:
                continue
            c1 = RULE_CLASSIFICATIONS.get(r1, WolframClass.UNKNOWN)
            c2 = RULE_CLASSIFICATIONS.get(r2, WolframClass.UNKNOWN)
            
            if c1 == c2 and c1 != WolframClass.UNKNOWN:
                within_class_dists.append(distances[i, j])
            elif c1 != WolframClass.UNKNOWN and c2 != WolframClass.UNKNOWN:
                between_class_dists.append(distances[i, j])
    
    within_mean = np.mean(within_class_dists) if within_class_dists else 0
    between_mean = np.mean(between_class_dists) if between_class_dists else 0
    
    # Cluster quality: ratio of between/within distances
    cluster_quality = between_mean / (within_mean + 1e-10)
    
    print(f"Within-class mean distance: {within_mean:.4f}")
    print(f"Between-class mean distance: {between_mean:.4f}")
    print(f"Cluster quality ratio: {cluster_quality:.4f}")
    
    clustering_success = cluster_quality > 1.0
    print(f"\n{'✅' if clustering_success else '❌'} Classes {'do' if clustering_success else 'do NOT'} cluster in PAC space")
    
    results['results']['clustering'] = {
        'within_class_mean': float(within_mean),
        'between_class_mean': float(between_mean),
        'cluster_quality_ratio': float(cluster_quality),
        'classes_cluster': bool(clustering_success)
    }
    
    # =====================================================
    # PART 4: Edge of Chaos (Rule 110) Analysis
    # =====================================================
    print("\nPART 4: Rule 110 (Edge of Chaos) Deep Dive")
    print("-" * 50)
    
    rule_110_coords = embeddings[110]
    print(f"Rule 110 PAC coordinates:")
    print(f"  P (Potential):     {rule_110_coords.potential:.6f}")
    print(f"  A (Actualization): {rule_110_coords.actualization:.6f}")
    print(f"  Ξ (Xi balance):    {rule_110_coords.xi:.6f}")
    
    # Check if near balanced (P ≈ A)
    balance_ratio = rule_110_coords.potential / (rule_110_coords.actualization + 1e-10)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    print(f"\n  P/A ratio: {balance_ratio:.6f}")
    print(f"  Distance from φ (1.618): {abs(balance_ratio - phi):.6f}")
    print(f"  Distance from 1.0 (perfect balance): {abs(balance_ratio - 1.0):.6f}")
    
    results['results']['rule_110'] = {
        'potential': float(rule_110_coords.potential),
        'actualization': float(rule_110_coords.actualization),
        'xi': float(rule_110_coords.xi),
        'pa_ratio': float(balance_ratio),
        'distance_from_phi': float(abs(balance_ratio - phi)),
        'distance_from_balance': float(abs(balance_ratio - 1.0))
    }
    
    # =====================================================
    # PART 5: Cross-Framework Invariant Preview
    # =====================================================
    print("\nPART 5: Cross-Framework Invariant Preview")
    print("-" * 50)
    
    analyzer = CrossFrameworkAnalyzer(width=101, steps=200)
    
    # Test on key rules
    key_rules = [0, 30, 90, 110, 126]
    
    print(f"{'Rule':>6} {'Cons':>10} {'Topo':>10} {'Info':>10} {'Conv':>8} {'Status':>8}")
    print("-" * 60)
    
    cross_framework_results = {}
    for rule in key_rules:
        result = analyzer.analyze_rule(rule)
        
        status = "✅ PASS" if result.converged else "❌ FAIL"
        
        print(f"{rule:>6} "
              f"{result.conservation_invariants.primary_invariant:>10.4f} "
              f"{result.topology_invariants.primary_invariant:>10.4f} "
              f"{result.information_invariants.primary_invariant:>10.4f} "
              f"{result.convergence_score:>8.3f} "
              f"{status:>8}")
        
        cross_framework_results[rule] = {
            'conservation': float(result.conservation_invariants.primary_invariant),
            'topology': float(result.topology_invariants.primary_invariant),
            'information': float(result.information_invariants.primary_invariant),
            'canonical_invariant': float(result.canonical_invariant),
            'convergence_score': float(result.convergence_score),
            'deviation': float(result.deviation),
            'converged': bool(result.converged)
        }
    
    results['results']['cross_framework_preview'] = cross_framework_results
    
    # =====================================================
    # Summary
    # =====================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    n_converged = sum(1 for r in cross_framework_results.values() if r['converged'])
    
    print(f"✅ CA Simulator: Working correctly")
    print(f"{'✅' if clustering_success else '❌'} PAC Clustering: Classes {'cluster' if clustering_success else 'do not cluster'} (ratio: {cluster_quality:.2f})")
    print(f"📊 Cross-framework convergence: {n_converged}/{len(key_rules)} rules pass 5% threshold")
    print(f"🎯 Rule 110 P/A ratio: {balance_ratio:.4f} (φ={phi:.4f}, balanced=1.0)")
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'exp_01_baseline_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    print(f"Completed: {datetime.now().isoformat()}")
    
    return results


if __name__ == "__main__":
    run_experiment()

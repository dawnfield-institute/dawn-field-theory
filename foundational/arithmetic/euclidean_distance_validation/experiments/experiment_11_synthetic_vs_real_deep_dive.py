"""
Experiment 11: Deep Dive - Synthetic vs Real Embeddings

Goal: Understand what synthetic embeddings got RIGHT vs what was artifact.

Key questions:
1. What geometric properties do synthetic embeddings preserve by construction?
2. What do real embeddings show that's different?
3. Is there a deeper relationship we missed?
4. Does PAC structure predict ANYTHING about real embeddings?

Hypothesis: Synthetic embeddings preserve PAC conservation by design.
Real embeddings don't have perfect E=mc², but may show other patterns.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.pac_hierarchy import PACNode, PACHierarchy
from core.embedding_generator import EmbeddingGenerator
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
import json


class SyntheticVsRealAnalysis:
    """Compare synthetic and real embeddings in detail."""
    
    def __init__(self):
        self.synthetic_hierarchy = None
        self.real_hierarchy = None
        
    def create_test_hierarchy(self, prefix: str) -> PACHierarchy:
        """Create identical hierarchy structure for both tests."""
        root = PACNode(id=f"{prefix}_root", value=100.0)
        root.metadata['text'] = f"Root concept: organizational structure"
        hierarchy = PACHierarchy(root)
        
        # Level 1: Departments
        l1_values = [30.0, 25.0, 20.0, 15.0, 10.0]
        l1_texts = ["Engineering", "Sales", "Marketing", "Operations", "Finance"]
        level1_nodes = []
        
        for i, (val, text) in enumerate(zip(l1_values, l1_texts)):
            node = PACNode(id=f"{prefix}_L1_{i}", value=val)
            node.metadata['text'] = f"Department: {text}"
            hierarchy.add_node(node, parent_id=root.id, ownership_weight=val/100.0)
            level1_nodes.append(node)
        
        # Level 2: Teams within departments
        level2_nodes = []
        for i, parent in enumerate(level1_nodes):
            n_children = 3 + (i % 3)
            child_value = parent.value / n_children
            team_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
            
            for j in range(n_children):
                node = PACNode(id=f"{prefix}_L2_{i}_{j}", value=child_value)
                node.metadata['text'] = f"Team: {l1_texts[i]} {team_names[j]}"
                hierarchy.add_node(node, parent_id=parent.id, ownership_weight=1.0/n_children)
                level2_nodes.append(node)
        
        # Level 3: Individual contributors (leaves)
        person_id = 0
        for i, parent in enumerate(level2_nodes):
            n_children = 2 + (i % 4)
            child_value = parent.value / n_children
            
            for j in range(n_children):
                node = PACNode(id=f"{prefix}_L3_{i}_{j}", value=child_value)
                node.metadata['text'] = f"Person {person_id}: individual contributor"
                hierarchy.add_node(node, parent_id=parent.id, ownership_weight=1.0/n_children)
                person_id += 1
        
        return hierarchy
    
    def analyze_energy_relationships(self, hierarchy: PACHierarchy, name: str) -> Dict:
        """Comprehensive analysis of f(v) vs ||e||² relationships."""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {name}")
        print(f"{'='*60}")
        
        # Separate leaves and parents
        leaf_nodes = [n for n in hierarchy.nodes.values() if not n.children]
        parent_nodes = [n for n in hierarchy.nodes.values() if n.children]
        
        results = {}
        
        # 1. Leaf analysis
        print(f"\n--- LEAF NODES (n={len(leaf_nodes)}) ---")
        leaf_masses = np.array([n.value for n in leaf_nodes if n.embedding is not None])
        leaf_energies = np.array([np.linalg.norm(n.embedding)**2 for n in leaf_nodes if n.embedding is not None])
        
        if len(leaf_masses) > 1:
            slope, intercept, r, p, stderr = linregress(leaf_masses, leaf_energies)
            print(f"  f(v) range: [{leaf_masses.min():.3f}, {leaf_masses.max():.3f}]")
            print(f"  ||e||² range: [{leaf_energies.min():.3f}, {leaf_energies.max():.3f}]")
            print(f"  Mean ||e||²: {leaf_energies.mean():.3f} ± {leaf_energies.std():.3f}")
            print(f"  R²: {r**2:.6f}, p={p:.2e}")
            print(f"  Slope (c²): {slope:.4f}")
            print(f"  Intercept: {intercept:.4f}")
            
            # Check if normalized
            is_normalized = np.allclose(leaf_energies, 1.0, atol=0.1)
            print(f"  Normalized (~1.0): {is_normalized}")
            
            results['leaves'] = {
                'n': len(leaf_masses),
                'r_squared': r**2,
                'p_value': p,
                'slope': slope,
                'intercept': intercept,
                'mean_energy': leaf_energies.mean(),
                'std_energy': leaf_energies.std(),
                'is_normalized': is_normalized,
                'masses': leaf_masses,
                'energies': leaf_energies
            }
        
        # 2. Parent analysis
        print(f"\n--- PARENT NODES (n={len(parent_nodes)}) ---")
        parent_masses = np.array([n.value for n in parent_nodes if n.embedding is not None])
        parent_energies = np.array([np.linalg.norm(n.embedding)**2 for n in parent_nodes if n.embedding is not None])
        
        if len(parent_masses) > 1:
            slope, intercept, r, p, stderr = linregress(parent_masses, parent_energies)
            print(f"  f(v) range: [{parent_masses.min():.3f}, {parent_masses.max():.3f}]")
            print(f"  ||e||² range: [{parent_energies.min():.3f}, {parent_energies.max():.3f}]")
            print(f"  Mean ||e||²: {parent_energies.mean():.3f} ± {parent_energies.std():.3f}")
            print(f"  R²: {r**2:.6f}, p={p:.2e}")
            print(f"  Slope: {slope:.4f}")
            
            results['parents'] = {
                'n': len(parent_masses),
                'r_squared': r**2,
                'p_value': p,
                'slope': slope,
                'intercept': intercept,
                'mean_energy': parent_energies.mean(),
                'std_energy': parent_energies.std(),
                'masses': parent_masses,
                'energies': parent_energies
            }
        
        # 3. Conservation test: Parent energy vs sum of children energies
        print(f"\n--- CONSERVATION TEST ---")
        conservation_errors = []
        binding_energies = []
        
        for parent in parent_nodes:
            if parent.embedding is not None and all(c.embedding is not None for c in parent.children):
                E_parent = np.linalg.norm(parent.embedding) ** 2
                E_children = sum(np.linalg.norm(c.embedding)**2 for c in parent.children)
                
                error = abs(E_parent - E_children) / E_children if E_children > 0 else 0
                binding = (E_children - E_parent) / E_children if E_children > 0 else 0
                
                conservation_errors.append(error)
                binding_energies.append(binding)
        
        if conservation_errors:
            conservation_errors = np.array(conservation_errors)
            binding_energies = np.array(binding_energies)
            
            print(f"  Mean conservation error: {conservation_errors.mean()*100:.2f}%")
            print(f"  Max conservation error: {conservation_errors.max()*100:.2f}%")
            print(f"  Perfect conservation: {(conservation_errors < 0.01).sum()}/{len(conservation_errors)}")
            print(f"  Mean binding energy: {binding_energies.mean()*100:.2f}%")
            
            results['conservation'] = {
                'mean_error': float(conservation_errors.mean()),
                'max_error': float(conservation_errors.max()),
                'perfect_count': int((conservation_errors < 0.01).sum()),
                'mean_binding': float(binding_energies.mean()),
                'errors': conservation_errors.tolist(),
                'bindings': binding_energies.tolist()
            }
        
        # 4. Distance-based analysis
        print(f"\n--- DISTANCE ANALYSIS ---")
        # Check if PAC distance correlates with embedding distance
        pac_distances = []
        emb_distances = []
        
        nodes_with_emb = [n for n in hierarchy.nodes.values() if n.embedding is not None]
        for i, n1 in enumerate(nodes_with_emb):
            for n2 in nodes_with_emb[i+1:]:
                # PAC distance (simplified: based on common ancestor depth)
                pac_dist = abs(n1.value - n2.value)  # Simplified
                emb_dist = np.linalg.norm(n1.embedding - n2.embedding)
                
                pac_distances.append(pac_dist)
                emb_distances.append(emb_dist)
        
        pac_distances = np.array(pac_distances[:100])  # Limit for speed
        emb_distances = np.array(emb_distances[:100])
        
        if len(pac_distances) > 1:
            r, p = pearsonr(pac_distances, emb_distances)
            print(f"  PAC-Embedding distance correlation: r={r:.4f}, p={p:.2e}")
            
            results['distance_correlation'] = {
                'r': float(r),
                'p_value': float(p)
            }
        
        return results
    
    def compare_results(self, synthetic_results: Dict, real_results: Dict):
        """Compare synthetic vs real results."""
        print(f"\n{'='*60}")
        print("SYNTHETIC vs REAL COMPARISON")
        print(f"{'='*60}")
        
        print(f"\nLEAF NODES - E=mc² relationship:")
        print(f"  Synthetic R²: {synthetic_results['leaves']['r_squared']:.6f}")
        print(f"  Real R²:      {real_results['leaves']['r_squared']:.6f}")
        print(f"  Difference:   {abs(synthetic_results['leaves']['r_squared'] - real_results['leaves']['r_squared']):.6f}")
        
        print(f"\nLEAF NODES - Energy normalization:")
        print(f"  Synthetic: mean={synthetic_results['leaves']['mean_energy']:.3f}, std={synthetic_results['leaves']['std_energy']:.3f}")
        print(f"  Real:      mean={real_results['leaves']['mean_energy']:.3f}, std={real_results['leaves']['std_energy']:.3f}")
        
        print(f"\nCONSERVATION - Parent = Sum(Children):")
        print(f"  Synthetic error: {synthetic_results['conservation']['mean_error']*100:.2f}%")
        print(f"  Real error:      {real_results['conservation']['mean_error']*100:.2f}%")
        print(f"  Synthetic perfect: {synthetic_results['conservation']['perfect_count']}/{len(synthetic_results['conservation']['errors'])}")
        print(f"  Real perfect:      {real_results['conservation']['perfect_count']}/{len(real_results['conservation']['errors'])}")
        
        print(f"\nBINDING ENERGY:")
        print(f"  Synthetic: {synthetic_results['conservation']['mean_binding']*100:.2f}%")
        print(f"  Real:      {real_results['conservation']['mean_binding']*100:.2f}%")
        
        print(f"\nDISTANCE CORRELATION:")
        print(f"  Synthetic: r={synthetic_results['distance_correlation']['r']:.4f}")
        print(f"  Real:      r={real_results['distance_correlation']['r']:.4f}")
        
        # Key insights
        print(f"\n{'='*60}")
        print("KEY INSIGHTS")
        print(f"{'='*60}")
        
        if synthetic_results['leaves']['is_normalized'] and not real_results['leaves']['is_normalized']:
            print("\n✓ CONFIRMED: Synthetic embeddings are normalized (~1.0)")
            print("  This explains the perfect R²=1.0 for leaves with f(v)≈1.0")
        
        if synthetic_results['conservation']['mean_error'] < 0.01:
            print("\n✓ CONFIRMED: Synthetic embeddings preserve PAC conservation")
            print("  Parent energy = Sum(children energies) by construction")
        
        if real_results['conservation']['mean_error'] > 0.1:
            print("\n✓ REAL embeddings do NOT preserve PAC conservation")
            print("  Semantic composition ≠ vector addition")
        
        if synthetic_results['distance_correlation']['r'] > real_results['distance_correlation']['r']:
            print(f"\n✓ Synthetic embeddings better preserve PAC distance structure")
        
    def visualize_comparison(self, synthetic_results: Dict, real_results: Dict):
        """Create comprehensive visualizations."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Row 1: Leaf E vs m
        ax1 = fig.add_subplot(gs[0, 0])
        syn_m = synthetic_results['leaves']['masses']
        syn_e = synthetic_results['leaves']['energies']
        ax1.scatter(syn_m, syn_e, alpha=0.6, s=50, label='Synthetic')
        m_range = np.linspace(syn_m.min(), syn_m.max(), 100)
        slope = synthetic_results['leaves']['slope']
        ax1.plot(m_range, slope * m_range + synthetic_results['leaves']['intercept'], 
                'r--', label=f'R²={synthetic_results["leaves"]["r_squared"]:.4f}')
        ax1.set_xlabel('f(v) [mass]')
        ax1.set_ylabel('||e||² [energy]')
        ax1.set_title('SYNTHETIC: Leaf E=mc²')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        real_m = real_results['leaves']['masses']
        real_e = real_results['leaves']['energies']
        ax2.scatter(real_m, real_e, alpha=0.6, s=50, color='green', label='Real')
        slope_real = real_results['leaves']['slope']
        ax2.plot(m_range, slope_real * m_range + real_results['leaves']['intercept'],
                'r--', label=f'R²={real_results["leaves"]["r_squared"]:.4f}')
        ax2.set_xlabel('f(v) [mass]')
        ax2.set_ylabel('||e||² [energy]')
        ax2.set_title('REAL: Leaf E=mc²')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Side-by-side
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(syn_m, syn_e, alpha=0.5, s=40, label='Synthetic', color='blue')
        ax3.scatter(real_m, real_e, alpha=0.5, s=40, label='Real', color='green')
        ax3.set_xlabel('f(v) [mass]')
        ax3.set_ylabel('||e||² [energy]')
        ax3.set_title('Synthetic vs Real Comparison')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Row 2: Energy distributions
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(syn_e, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(syn_e.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={syn_e.mean():.3f}')
        ax4.axvline(1.0, color='green', linestyle=':', linewidth=2, label='||e||²=1.0')
        ax4.set_xlabel('||e||²')
        ax4.set_ylabel('Frequency')
        ax4.set_title('SYNTHETIC: Energy Distribution')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(real_e, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax5.axvline(real_e.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={real_e.mean():.3f}')
        ax5.set_xlabel('||e||²')
        ax5.set_ylabel('Frequency')
        ax5.set_title('REAL: Energy Distribution')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # Conservation errors
        ax6 = fig.add_subplot(gs[1, 2])
        syn_cons = np.array(synthetic_results['conservation']['errors']) * 100
        real_cons = np.array(real_results['conservation']['errors']) * 100
        ax6.boxplot([syn_cons, real_cons], labels=['Synthetic', 'Real'])
        ax6.set_ylabel('Conservation Error (%)')
        ax6.set_title('Energy Conservation: Parent vs Sum(Children)')
        ax6.grid(alpha=0.3, axis='y')
        
        # Row 3: Binding energy
        ax7 = fig.add_subplot(gs[2, 0])
        syn_bind = np.array(synthetic_results['conservation']['bindings']) * 100
        ax7.hist(syn_bind, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax7.axvline(syn_bind.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={syn_bind.mean():.1f}%')
        ax7.set_xlabel('Binding Energy (%)')
        ax7.set_ylabel('Frequency')
        ax7.set_title('SYNTHETIC: Binding Energy')
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        ax8 = fig.add_subplot(gs[2, 1])
        real_bind = np.array(real_results['conservation']['bindings']) * 100
        ax8.hist(real_bind, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax8.axvline(real_bind.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={real_bind.mean():.1f}%')
        ax8.set_xlabel('Binding Energy (%)')
        ax8.set_ylabel('Frequency')
        ax8.set_title('REAL: Binding Energy')
        ax8.legend()
        ax8.grid(alpha=0.3)
        
        # Summary comparison
        ax9 = fig.add_subplot(gs[2, 2])
        metrics = ['R² (leaves)', 'Cons. Error', 'Binding %']
        syn_vals = [
            synthetic_results['leaves']['r_squared'],
            synthetic_results['conservation']['mean_error'],
            abs(synthetic_results['conservation']['mean_binding'])
        ]
        real_vals = [
            real_results['leaves']['r_squared'],
            real_results['conservation']['mean_error'],
            abs(real_results['conservation']['mean_binding'])
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax9.bar(x - width/2, syn_vals, width, label='Synthetic', alpha=0.8, color='blue')
        ax9.bar(x + width/2, real_vals, width, label='Real', alpha=0.8, color='green')
        ax9.set_xticks(x)
        ax9.set_xticklabels(metrics, rotation=45, ha='right')
        ax9.set_ylabel('Value')
        ax9.set_title('Key Metrics Comparison')
        ax9.legend()
        ax9.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        import os
        os.makedirs('euclidean_distance_validation/results', exist_ok=True)
        plt.savefig('euclidean_distance_validation/results/experiment_11_synthetic_vs_real.png',
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved to results/experiment_11_synthetic_vs_real.png")


def main():
    """Run deep dive analysis."""
    
    print("="*60)
    print("EXPERIMENT 11: SYNTHETIC VS REAL DEEP DIVE")
    print("="*60)
    
    analyzer = SyntheticVsRealAnalysis()
    
    # 1. Create and embed synthetic hierarchy
    print("\n[1/4] Creating synthetic hierarchy...")
    analyzer.synthetic_hierarchy = analyzer.create_test_hierarchy("syn")
    emb_syn = EmbeddingGenerator(model='synthetic', dimension=128, seed=42)
    emb_syn.embed_hierarchy(analyzer.synthetic_hierarchy)
    print(f"Created: {len(analyzer.synthetic_hierarchy.nodes)} nodes")
    
    # 2. Create and embed real hierarchy (same structure)
    print("\n[2/4] Creating real embeddings hierarchy...")
    analyzer.real_hierarchy = analyzer.create_test_hierarchy("real")
    emb_real = EmbeddingGenerator(model='sentence-transformers', 
                                  model_name='all-MiniLM-L6-v2')
    emb_real.embed_hierarchy(analyzer.real_hierarchy)
    print(f"Created: {len(analyzer.real_hierarchy.nodes)} nodes")
    
    # 3. Analyze both
    print("\n[3/4] Analyzing both hierarchies...")
    synthetic_results = analyzer.analyze_energy_relationships(
        analyzer.synthetic_hierarchy, "SYNTHETIC EMBEDDINGS")
    real_results = analyzer.analyze_energy_relationships(
        analyzer.real_hierarchy, "REAL EMBEDDINGS")
    
    # 4. Compare
    print("\n[4/4] Comparing results...")
    analyzer.compare_results(synthetic_results, real_results)
    
    # Visualize
    analyzer.visualize_comparison(synthetic_results, real_results)
    
    # Save results
    import os
    os.makedirs('euclidean_distance_validation/results', exist_ok=True)
    with open('euclidean_distance_validation/results/experiment_11_results.json', 'w') as f:
        output = {
            'synthetic': {k: v for k, v in synthetic_results.items() 
                         if k not in ['masses', 'energies']},
            'real': {k: v for k, v in real_results.items() 
                    if k not in ['masses', 'energies']}
        }
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/experiment_11_results.json")
    
    return synthetic_results, real_results


if __name__ == "__main__":
    synthetic_results, real_results = main()

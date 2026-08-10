"""
Experiment 13: Geometric E=mc² Equivalence

TEST: Are two geometric properties of embeddings proportional?
E_geometric = c²(context) · m_geometric

Both E and m derived FROM embedding geometry (not external values).

Key insight: The embedding space IS the PAC tree. E=mc² should relate
geometric properties measured in the SAME space from different perspectives.

This tests:
1. Do geometric properties correlate? (E ∝ m)
2. Is c² context-dependent? (varies by subtree/level)
3. Does this show relativity? (same node, different c² from different frames)
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
from sklearn.neighbors import NearestNeighbors
import json


class GeometricEquivalence:
    """Test geometric E=mc² in embedding space."""
    
    def __init__(self, hierarchy: PACHierarchy):
        self.hierarchy = hierarchy
        self.embedding_matrix = None
        self.node_list = None
        
    def prepare_embedding_matrix(self):
        """Create matrix of all embeddings for efficient computation."""
        self.node_list = [n for n in self.hierarchy.nodes.values() 
                         if n.embedding is not None]
        self.embedding_matrix = np.array([n.embedding for n in self.node_list])
        print(f"Embedding matrix: {self.embedding_matrix.shape}")
        
    def compute_energy_local_density(self, k: int = 5) -> Dict[str, float]:
        """
        E_geometric: Local density (average distance to k nearest neighbors).
        
        Interpretation: How "energetic" is this position - dense regions have
        high energy (many nearby nodes).
        """
        nbrs = NearestNeighbors(n_neighbors=min(k+1, len(self.node_list))).fit(self.embedding_matrix)
        distances, indices = nbrs.kneighbors(self.embedding_matrix)
        
        # Skip first neighbor (self), average the rest
        avg_distances = distances[:, 1:].mean(axis=1)
        
        # Energy = 1/distance (higher in dense regions)
        energies = 1.0 / (avg_distances + 1e-10)
        
        return {n.id: energies[i] for i, n in enumerate(self.node_list)}
    
    def compute_energy_centrality(self) -> Dict[str, float]:
        """
        E_geometric: Betweenness centrality in embedding k-NN graph.
        
        Interpretation: How "important" is this node as a hub/connector.
        """
        from sklearn.neighbors import kneighbors_graph
        import networkx as nx
        
        # Build k-NN graph
        k = min(10, len(self.node_list) - 1)
        A = kneighbors_graph(self.embedding_matrix, k, mode='distance')
        G = nx.from_scipy_sparse_array(A)
        
        # Compute betweenness centrality
        centrality = nx.betweenness_centrality(G, normalized=True)
        
        return {n.id: centrality.get(i, 0.0) for i, n in enumerate(self.node_list)}
    
    def compute_energy_neighborhood_volume(self, k: int = 5) -> Dict[str, float]:
        """
        E_geometric: Volume of k-ball around node (determinant of covariance).
        
        Interpretation: Local manifold volume - how "spacious" is this region.
        """
        nbrs = NearestNeighbors(n_neighbors=min(k+1, len(self.node_list))).fit(self.embedding_matrix)
        distances, indices = nbrs.kneighbors(self.embedding_matrix)
        
        volumes = {}
        for i, node in enumerate(self.node_list):
            neighbor_indices = indices[i][1:]  # Skip self
            if len(neighbor_indices) > 0:
                neighbors = self.embedding_matrix[neighbor_indices]
                # Volume ≈ geometric mean of distances
                volume = np.exp(np.mean(np.log(distances[i][1:] + 1e-10)))
                volumes[node.id] = volume
            else:
                volumes[node.id] = 0.0
        
        return volumes
    
    def compute_mass_depth(self) -> Dict[str, float]:
        """
        m_geometric: Depth in hierarchy (distance from root).
        
        Interpretation: How "massive" in terms of hierarchical position.
        """
        masses = {}
        for node in self.node_list:
            depth = 0
            current = node
            while current.parent is not None:
                depth += 1
                current = current.parent
            masses[node.id] = float(depth + 1)  # +1 so root has mass
        return masses
    
    def compute_mass_subtree_size(self) -> Dict[str, float]:
        """
        m_geometric: Number of descendants (subtree size).
        
        Interpretation: How much "mass" under this node.
        """
        def count_descendants(node: PACNode) -> int:
            if not node.children:
                return 1
            return 1 + sum(count_descendants(c) for c in node.children)
        
        return {n.id: float(count_descendants(n)) for n in self.node_list}
    
    def compute_mass_branching(self) -> Dict[str, float]:
        """
        m_geometric: Branching factor (number of children).
        
        Interpretation: Local "mass" from connectivity.
        """
        return {n.id: float(len(n.children) + 1) for n in self.node_list}
    
    def compute_mass_embedding_norm(self) -> Dict[str, float]:
        """
        m_geometric: Euclidean norm of embedding.
        
        Interpretation: "Mass" from embedding magnitude (for non-normalized).
        """
        return {n.id: float(np.linalg.norm(n.embedding)) for n in self.node_list}
    
    def test_equivalence(self, E_values: Dict[str, float], m_values: Dict[str, float],
                        E_name: str, m_name: str) -> Dict:
        """Test if E = c² · m with correlation analysis."""
        # Get common nodes
        common_ids = set(E_values.keys()) & set(m_values.keys())
        
        E = np.array([E_values[id] for id in common_ids])
        m = np.array([m_values[id] for id in common_ids])
        
        # Remove any zeros or invalid values
        valid = (E > 0) & (m > 0) & np.isfinite(E) & np.isfinite(m)
        E = E[valid]
        m = m[valid]
        
        if len(E) < 2:
            return {'valid': False}
        
        # Linear regression E = c² · m (through origin)
        c_squared = np.sum(E * m) / np.sum(m ** 2)
        predicted = c_squared * m
        
        # R² calculation
        ss_res = np.sum((E - predicted) ** 2)
        ss_tot = np.sum((E - E.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Pearson correlation
        r, p = pearsonr(m, E)
        
        # Also try log-log (power law)
        log_E = np.log(E)
        log_m = np.log(m)
        slope, intercept, r_log, p_log, _ = linregress(log_m, log_E)
        
        return {
            'valid': True,
            'n': len(E),
            'c_squared': c_squared,
            'r_squared': r_squared,
            'r': r,
            'p_value': p,
            'log_slope': slope,
            'r_log': r_log,
            'p_log': p_log,
            'E_mean': E.mean(),
            'E_std': E.std(),
            'm_mean': m.mean(),
            'm_std': m.std(),
            'E': E,
            'm': m,
            'predicted': predicted
        }
    
    def test_context_dependence(self, E_values: Dict[str, float], 
                                m_values: Dict[str, float]) -> Dict:
        """Test if c² varies by context (level, subtree)."""
        results = {}
        
        # Group by depth
        by_depth = {}
        for node in self.node_list:
            depth = 0
            current = node
            while current.parent is not None:
                depth += 1
                current = current.parent
            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(node.id)
        
        # Compute c² for each depth
        c_squared_by_depth = {}
        for depth, ids in by_depth.items():
            if len(ids) < 3:
                continue
            E_subset = np.array([E_values[id] for id in ids if id in E_values and id in m_values])
            m_subset = np.array([m_values[id] for id in ids if id in E_values and id in m_values])
            
            valid = (E_subset > 0) & (m_subset > 0)
            if valid.sum() < 3:
                continue
                
            E_subset = E_subset[valid]
            m_subset = m_subset[valid]
            
            c_sq = np.sum(E_subset * m_subset) / np.sum(m_subset ** 2)
            c_squared_by_depth[depth] = c_sq
        
        if len(c_squared_by_depth) > 1:
            c_values = list(c_squared_by_depth.values())
            variation = np.std(c_values) / (np.mean(c_values) + 1e-10)
            results['context_variation'] = variation
            results['c_squared_by_depth'] = c_squared_by_depth
        else:
            results['context_variation'] = 0.0
            results['c_squared_by_depth'] = {}
        
        return results
    
    def run_full_analysis(self) -> Dict:
        """Run complete geometric equivalence analysis."""
        print(f"\n{'='*60}")
        print("GEOMETRIC E=mc² EQUIVALENCE TEST")
        print(f"{'='*60}")
        
        self.prepare_embedding_matrix()
        
        # Compute all energy measures
        print("\n[1/3] Computing energy-like properties...")
        E_density = self.compute_energy_local_density(k=5)
        E_centrality = self.compute_energy_centrality()
        E_volume = self.compute_energy_neighborhood_volume(k=5)
        print(f"  Local density: {len(E_density)} nodes")
        print(f"  Centrality: {len(E_centrality)} nodes")
        print(f"  Volume: {len(E_volume)} nodes")
        
        # Compute all mass measures
        print("\n[2/3] Computing mass-like properties...")
        m_depth = self.compute_mass_depth()
        m_subtree = self.compute_mass_subtree_size()
        m_branching = self.compute_mass_branching()
        m_norm = self.compute_mass_embedding_norm()
        print(f"  Depth: {len(m_depth)} nodes")
        print(f"  Subtree size: {len(m_subtree)} nodes")
        print(f"  Branching: {len(m_branching)} nodes")
        print(f"  Norm: {len(m_norm)} nodes")
        
        # Test all combinations
        print("\n[3/3] Testing E=mc² equivalence...")
        
        E_measures = [
            ('Local Density', E_density),
            ('Centrality', E_centrality),
            ('Neighborhood Volume', E_volume)
        ]
        
        m_measures = [
            ('Depth', m_depth),
            ('Subtree Size', m_subtree),
            ('Branching', m_branching),
            ('Norm', m_norm)
        ]
        
        results = {}
        best_r2 = 0
        best_pair = None
        
        for E_name, E_vals in E_measures:
            for m_name, m_vals in m_measures:
                key = f"{E_name} vs {m_name}"
                result = self.test_equivalence(E_vals, m_vals, E_name, m_name)
                
                if result['valid']:
                    results[key] = result
                    print(f"\n  {key}:")
                    print(f"    R² = {result['r_squared']:.6f}")
                    print(f"    c² = {result['c_squared']:.4f}")
                    print(f"    r = {result['r']:.4f} (p={result['p_value']:.2e})")
                    print(f"    log-log: slope={result['log_slope']:.3f}, R²={result['r_log']**2:.4f}")
                    
                    if result['r_squared'] > best_r2:
                        best_r2 = result['r_squared']
                        best_pair = (key, E_name, m_name, E_vals, m_vals)
        
        # Context analysis for best pair
        if best_pair:
            key, E_name, m_name, E_vals, m_vals = best_pair
            print(f"\n{'='*60}")
            print(f"CONTEXT-DEPENDENCE TEST (best pair: {key})")
            print(f"{'='*60}")
            
            context_results = self.test_context_dependence(E_vals, m_vals)
            results['best_pair'] = {
                'pair': key,
                'r_squared': best_r2,
                **context_results
            }
            
            if context_results['c_squared_by_depth']:
                print(f"\nc² by depth (context):")
                for depth, c_sq in sorted(context_results['c_squared_by_depth'].items()):
                    print(f"  Depth {depth}: c² = {c_sq:.4f}")
                print(f"\nContext variation: {context_results['context_variation']:.4f}")
                
                if context_results['context_variation'] > 0.3:
                    print("✅ STRONG context-dependence (c² varies significantly)")
                elif context_results['context_variation'] > 0.1:
                    print("✓ Moderate context-dependence")
                else:
                    print("⚠️  Weak context-dependence")
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        if best_r2 > 0.8:
            print(f"✅ STRONG geometric equivalence found")
            print(f"   Best: {best_pair[0]}")
            print(f"   R² = {best_r2:.6f}")
        elif best_r2 > 0.6:
            print(f"✓ Moderate geometric equivalence")
            print(f"   Best: {best_pair[0]}")
            print(f"   R² = {best_r2:.6f}")
        else:
            print(f"⚠️  Weak geometric equivalence")
            print(f"   May need different geometric properties")
        
        return results
    
    def visualize_results(self, results: Dict):
        """Visualize geometric E=mc² results."""
        # Find best result
        best_key = None
        best_r2 = 0
        for key, res in results.items():
            if key != 'best_pair' and res.get('valid') and res['r_squared'] > best_r2:
                best_r2 = res['r_squared']
                best_key = key
        
        if not best_key:
            print("No valid results to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        best_res = results[best_key]
        
        # 1. E vs m scatter
        ax = axes[0, 0]
        ax.scatter(best_res['m'], best_res['E'], alpha=0.6, s=50)
        ax.plot(best_res['m'], best_res['predicted'], 'r--', linewidth=2,
               label=f'c²={best_res["c_squared"]:.3f}')
        ax.set_xlabel('m (geometric mass)')
        ax.set_ylabel('E (geometric energy)')
        ax.set_title(f'{best_key}\nR²={best_res["r_squared"]:.4f}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Log-log plot
        ax = axes[0, 1]
        ax.scatter(np.log(best_res['m']), np.log(best_res['E']), alpha=0.6, s=50)
        log_m = np.log(best_res['m'])
        log_pred = best_res['log_slope'] * log_m + np.log(best_res['E']).mean() - best_res['log_slope'] * log_m.mean()
        ax.plot(log_m, log_pred, 'r--', linewidth=2,
               label=f'slope={best_res["log_slope"]:.3f}')
        ax.set_xlabel('log(m)')
        ax.set_ylabel('log(E)')
        ax.set_title(f'Power Law Test\nR²={best_res["r_log"]**2:.4f}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Residuals
        ax = axes[0, 2]
        residuals = best_res['E'] - best_res['predicted']
        ax.scatter(best_res['m'], residuals, alpha=0.6, s=50)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('m (geometric mass)')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Analysis')
        ax.grid(alpha=0.3)
        
        # 4. c² by context
        ax = axes[1, 0]
        if 'best_pair' in results and results['best_pair']['c_squared_by_depth']:
            depths = sorted(results['best_pair']['c_squared_by_depth'].keys())
            c_sqs = [results['best_pair']['c_squared_by_depth'][d] for d in depths]
            ax.plot(depths, c_sqs, 'o-', markersize=10, linewidth=2)
            ax.set_xlabel('Depth (context)')
            ax.set_ylabel('c²')
            ax.set_title(f'Context-Dependent c²\nVariation={results["best_pair"]["context_variation"]:.4f}')
            ax.grid(alpha=0.3)
        
        # 5. R² comparison for all pairs
        ax = axes[1, 1]
        valid_results = {k: v for k, v in results.items() if k != 'best_pair' and v.get('valid')}
        if valid_results:
            pairs = list(valid_results.keys())
            r2s = [valid_results[p]['r_squared'] for p in pairs]
            colors = ['green' if r2 > 0.8 else 'orange' if r2 > 0.6 else 'gray' for r2 in r2s]
            y_pos = np.arange(len(pairs))
            ax.barh(y_pos, r2s, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([p.replace(' vs ', '\nvs\n') for p in pairs], fontsize=8)
            ax.set_xlabel('R²')
            ax.set_title('All E-m Pair Comparisons')
            ax.axvline(0.8, color='green', linestyle=':', alpha=0.5)
            ax.axvline(0.6, color='orange', linestyle=':', alpha=0.5)
            ax.grid(alpha=0.3, axis='x')
        
        # 6. Predicted vs Actual
        ax = axes[1, 2]
        ax.scatter(best_res['predicted'], best_res['E'], alpha=0.6, s=50)
        min_val = min(best_res['predicted'].min(), best_res['E'].min())
        max_val = max(best_res['predicted'].max(), best_res['E'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('Predicted E')
        ax.set_ylabel('Actual E')
        ax.set_title(f'Prediction Quality\nR²={best_res["r_squared"]:.4f}')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        import os
        os.makedirs('euclidean_distance_validation/results', exist_ok=True)
        plt.savefig('euclidean_distance_validation/results/experiment_13_geometric_equivalence.png',
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved to results/experiment_13_geometric_equivalence.png")


def main():
    """Run Experiment 13: Geometric E=mc²."""
    
    print("="*60)
    print("EXPERIMENT 13: GEOMETRIC E=mc² EQUIVALENCE")
    print("="*60)
    print("\nGoal: Test if geometric properties satisfy E = c²·m")
    print("where BOTH E and m are derived from embedding geometry.")
    print()
    
    # Test with both synthetic and real
    for embedding_type in ['synthetic', 'real']:
        print(f"\n{'#'*60}")
        print(f"# {embedding_type.upper()} EMBEDDINGS")
        print(f"{'#'*60}")
        
        # Create hierarchy
        root = PACNode(id=f"{embedding_type}_root", value=100.0)
        root.metadata['text'] = "Root: organizational structure"
        hierarchy = PACHierarchy(root)
        
        # Build (same structure for both)
        level1_values = [30.0, 25.0, 20.0, 15.0, 10.0]
        level1_texts = ["Engineering", "Sales", "Marketing", "Operations", "Finance"]
        level1_nodes = []
        
        for i, (val, text) in enumerate(zip(level1_values, level1_texts)):
            node = PACNode(id=f"{embedding_type}_L1_{i}", value=val)
            node.metadata['text'] = f"Department: {text}"
            hierarchy.add_node(node, parent_id=root.id, ownership_weight=val/100.0)
            level1_nodes.append(node)
        
        level2_nodes = []
        for i, parent in enumerate(level1_nodes):
            n_children = 3 + (i % 3)
            child_value = parent.value / n_children
            for j in range(n_children):
                node = PACNode(id=f"{embedding_type}_L2_{i}_{j}", value=child_value)
                node.metadata['text'] = f"Team: {level1_texts[i]} {j}"
                hierarchy.add_node(node, parent_id=parent.id, ownership_weight=1.0/n_children)
                level2_nodes.append(node)
        
        for i, parent in enumerate(level2_nodes):
            n_children = 2 + (i % 4)
            child_value = parent.value / n_children
            for j in range(n_children):
                node = PACNode(id=f"{embedding_type}_L3_{i}_{j}", value=child_value)
                node.metadata['text'] = f"Person {i}_{j}"
                hierarchy.add_node(node, parent_id=parent.id, ownership_weight=1.0/n_children)
        
        print(f"Created hierarchy: {len(hierarchy.nodes)} nodes")
        
        # Generate embeddings
        if embedding_type == 'synthetic':
            emb_gen = EmbeddingGenerator(model='synthetic', dimension=128, seed=42)
        else:
            emb_gen = EmbeddingGenerator(model='sentence-transformers',
                                         model_name='all-MiniLM-L6-v2')
        emb_gen.embed_hierarchy(hierarchy)
        
        # Run analysis
        analyzer = GeometricEquivalence(hierarchy)
        results = analyzer.run_full_analysis()
        
        # Visualize
        analyzer.visualize_results(results)
        
        # Save results
        import os
        os.makedirs('euclidean_distance_validation/results', exist_ok=True)
        with open(f'euclidean_distance_validation/results/experiment_13_{embedding_type}_results.json', 'w') as f:
            output = {}
            for k, v in results.items():
                if isinstance(v, dict) and 'E' in v:
                    # Remove numpy arrays for JSON
                    output[k] = {kk: vv for kk, vv in v.items() if kk not in ['E', 'm', 'predicted']}
                else:
                    output[k] = v
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to results/experiment_13_{embedding_type}_results.json")


if __name__ == "__main__":
    main()

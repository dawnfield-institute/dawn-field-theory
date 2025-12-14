"""
Experiment 8: Null Hypothesis Tests for E=mc² Relationship

Tests whether R²=1.0 result from experiment_06 could occur by chance.
Uses multiple null hypothesis strategies to establish statistical significance.

Null Hypotheses to Test:
1. H0_shuffle: Randomly shuffling embeddings produces similar R²
2. H0_random_hierarchy: Breaking parent-child structure doesn't affect R²
3. H0_permute_values: Permuting f(v) values produces similar R²
4. H0_noise: Adding noise to embeddings doesn't change R²
5. H0_independent: Completely independent random embeddings produce similar R²

If PAC theory is valid, original R² should significantly exceed all nulls.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.pac_hierarchy import PACNode, PACHierarchy
from core.embedding_generator import EmbeddingGenerator
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, ttest_ind
from scipy.optimize import curve_fit
import json


class NullHypothesisTester:
    """Tests null hypotheses for E=mc² relationship."""
    
    def __init__(self, hierarchy: PACHierarchy, n_iterations: int = 100):
        self.hierarchy = hierarchy
        self.n_iterations = n_iterations
        self.embedding_gen = EmbeddingGenerator(model='sentence-transformers', 
                                                 model_name='all-MiniLM-L6-v2')
        self.original_embeddings = {}
        
    def generate_real_embeddings(self):
        """Generate REAL embeddings using sentence-transformers."""
        print("Generating real embeddings with sentence-transformers...")
        self.embedding_gen.embed_hierarchy(self.hierarchy)
        
        # Store original embeddings
        for node in self.hierarchy.nodes.values():
            if node.embedding is not None:
                self.original_embeddings[node.id] = node.embedding.copy()
        
        print(f"Generated embeddings for {len(self.original_embeddings)} nodes")
    
    def compute_r_squared(self, nodes: List[PACNode] = None) -> Dict:
        """Compute R² for E=mc² relationship (leaf nodes only)."""
        if nodes is None:
            nodes = [n for n in self.hierarchy.nodes.values() if not n.children]
        
        masses = []
        energies = []
        
        for node in nodes:
            if node.embedding is not None:
                masses.append(node.value)
                energies.append(np.linalg.norm(node.embedding) ** 2)
        
        if len(masses) < 2:
            return {'r_squared': 0.0, 'n': 0}
        
        masses = np.array(masses)
        energies = np.array(energies)
        
        # Linear fit
        slope, intercept, r, p_value, std_err = linregress(masses, energies)
        
        return {
            'r_squared': r**2,
            'r': r,
            'p_value': p_value,
            'slope': slope,
            'intercept': intercept,
            'n': len(masses)
        }
    
    def test_h0_shuffle_embeddings(self) -> Dict:
        """
        H0: Shuffling embeddings across nodes produces similar R².
        
        If PAC structure matters, shuffling should destroy correlation.
        """
        print("\n=== H0_shuffle: Shuffling embeddings ===")
        
        # Get original R²
        original_r2 = self.compute_r_squared()
        print(f"Original R²: {original_r2['r_squared']:.6f}")
        
        # Run shuffle iterations
        null_r2_values = []
        leaf_nodes = [n for n in self.hierarchy.nodes.values() if not n.children]
        
        for i in range(self.n_iterations):
            # Shuffle embeddings randomly across leaf nodes
            embeddings = [self.original_embeddings[n.id].copy() for n in leaf_nodes]
            np.random.shuffle(embeddings)
            
            for node, emb in zip(leaf_nodes, embeddings):
                node.embedding = emb
            
            result = self.compute_r_squared(leaf_nodes)
            null_r2_values.append(result['r_squared'])
        
        # Restore original embeddings
        for node in leaf_nodes:
            node.embedding = self.original_embeddings[node.id].copy()
        
        null_mean = np.mean(null_r2_values)
        null_std = np.std(null_r2_values)
        p_value = np.mean(np.array(null_r2_values) >= original_r2['r_squared'])
        
        print(f"Null R² mean: {null_mean:.6f} ± {null_std:.6f}")
        print(f"p-value: {p_value:.6f}")
        
        return {
            'test': 'shuffle_embeddings',
            'original_r2': original_r2['r_squared'],
            'null_mean': null_mean,
            'null_std': null_std,
            'null_r2_values': null_r2_values,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def test_h0_permute_values(self) -> Dict:
        """
        H0: Permuting f(v) values produces similar R².
        
        If E=mc² is real, permuting masses should destroy correlation.
        """
        print("\n=== H0_permute: Permuting f(v) values ===")
        
        leaf_nodes = [n for n in self.hierarchy.nodes.values() if not n.children]
        original_values = {n.id: n.value for n in leaf_nodes}
        
        # Get original R²
        original_r2 = self.compute_r_squared(leaf_nodes)
        print(f"Original R²: {original_r2['r_squared']:.6f}")
        
        # Run permutation iterations
        null_r2_values = []
        values = [n.value for n in leaf_nodes]
        
        for i in range(self.n_iterations):
            # Permute values
            shuffled_values = values.copy()
            np.random.shuffle(shuffled_values)
            
            for node, val in zip(leaf_nodes, shuffled_values):
                node.value = val
            
            result = self.compute_r_squared(leaf_nodes)
            null_r2_values.append(result['r_squared'])
        
        # Restore original values
        for node in leaf_nodes:
            node.value = original_values[node.id]
        
        null_mean = np.mean(null_r2_values)
        null_std = np.std(null_r2_values)
        p_value = np.mean(np.array(null_r2_values) >= original_r2['r_squared'])
        
        print(f"Null R² mean: {null_mean:.6f} ± {null_std:.6f}")
        print(f"p-value: {p_value:.6f}")
        
        return {
            'test': 'permute_values',
            'original_r2': original_r2['r_squared'],
            'null_mean': null_mean,
            'null_std': null_std,
            'null_r2_values': null_r2_values,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def test_h0_independent_random(self) -> Dict:
        """
        H0: Completely random embeddings produce similar R².
        
        Generate fresh random embeddings independent of structure.
        """
        print("\n=== H0_random: Independent random embeddings ===")
        
        leaf_nodes = [n for n in self.hierarchy.nodes.values() if not n.children]
        
        # Get original R²
        original_r2 = self.compute_r_squared(leaf_nodes)
        print(f"Original R²: {original_r2['r_squared']:.6f}")
        
        # Get embedding dimension
        dim = self.original_embeddings[leaf_nodes[0].id].shape[0]
        
        # Run random iterations
        null_r2_values = []
        
        for i in range(self.n_iterations):
            # Generate random embeddings
            for node in leaf_nodes:
                node.embedding = np.random.randn(dim)
                node.embedding /= np.linalg.norm(node.embedding)  # Normalize
            
            result = self.compute_r_squared(leaf_nodes)
            null_r2_values.append(result['r_squared'])
        
        # Restore original embeddings
        for node in leaf_nodes:
            node.embedding = self.original_embeddings[node.id].copy()
        
        null_mean = np.mean(null_r2_values)
        null_std = np.std(null_r2_values)
        p_value = np.mean(np.array(null_r2_values) >= original_r2['r_squared'])
        
        print(f"Null R² mean: {null_mean:.6f} ± {null_std:.6f}")
        print(f"p-value: {p_value:.6f}")
        
        return {
            'test': 'independent_random',
            'original_r2': original_r2['r_squared'],
            'null_mean': null_mean,
            'null_std': null_std,
            'null_r2_values': null_r2_values,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def run_all_tests(self) -> Dict:
        """Run all null hypothesis tests."""
        print("=" * 60)
        print("NULL HYPOTHESIS TESTING FOR E=mc² R²=1.0 RESULT")
        print("=" * 60)
        
        results = {
            'shuffle': self.test_h0_shuffle_embeddings(),
            'permute': self.test_h0_permute_values(),
            'random': self.test_h0_independent_random()
        }
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        for test_name, result in results.items():
            print(f"\n{test_name.upper()}:")
            print(f"  Original R²: {result['original_r2']:.6f}")
            print(f"  Null mean:   {result['null_mean']:.6f} ± {result['null_std']:.6f}")
            print(f"  p-value:     {result['p_value']:.6f}")
            print(f"  Significant: {'✅ YES' if result['significant'] else '❌ NO'}")
        
        # Overall conclusion
        all_significant = all(r['significant'] for r in results.values())
        
        print("\n" + "=" * 60)
        if all_significant:
            print("✅ CONCLUSION: Original R² is statistically significant")
            print("   All null hypotheses rejected (p < 0.05)")
            print("   PAC structure genuinely predicts embedding geometry")
        else:
            print("⚠️  CONCLUSION: Some null hypotheses NOT rejected")
            print("   R²=1.0 may be partially due to chance")
            print("   Need to investigate further")
        
        return results
    
    def visualize_results(self, results: Dict):
        """Create visualizations of null hypothesis tests."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        test_names = list(results.keys())
        
        # Plot 1-3: Distributions for each test
        for idx, (test_name, result) in enumerate(results.items()):
            if idx >= 3:
                break
            
            ax = axes[idx // 2, idx % 2]
            
            null_r2 = result['null_r2_values']
            original_r2 = result['original_r2']
            
            ax.hist(null_r2, bins=30, alpha=0.7, color='gray', 
                   edgecolor='black', label='Null distribution')
            ax.axvline(original_r2, color='red', linewidth=3, 
                      linestyle='--', label=f'Original R²={original_r2:.4f}')
            ax.axvline(result['null_mean'], color='blue', linewidth=2,
                      linestyle=':', label=f'Null mean={result["null_mean"]:.4f}')
            
            ax.set_xlabel('R² value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{test_name.upper()}: p={result["p_value"]:.4f}')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Plot 4: Summary comparison
        ax = axes[1, 1]
        
        test_labels = [t.capitalize() for t in test_names]
        original_r2_vals = [results[t]['original_r2'] for t in test_names]
        null_means = [results[t]['null_mean'] for t in test_names]
        null_stds = [results[t]['null_std'] for t in test_names]
        
        x = np.arange(len(test_labels))
        width = 0.35
        
        ax.bar(x - width/2, original_r2_vals, width, label='Original', 
               alpha=0.8, color='red', edgecolor='black')
        ax.bar(x + width/2, null_means, width, yerr=null_stds, label='Null',
               alpha=0.8, color='gray', edgecolor='black', capsize=5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(test_labels, rotation=45, ha='right')
        ax.set_ylabel('R² value')
        ax.set_title('Original vs Null Hypothesis Comparison')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        import os
        os.makedirs('euclidean_distance_validation/results', exist_ok=True)
        plt.savefig('euclidean_distance_validation/results/experiment_08_null_hypothesis_tests.png',
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved to results/experiment_08_null_hypothesis_tests.png")


def main():
    """Run Experiment 8: Null Hypothesis Tests."""
    
    # Build test hierarchy (same as exp_06)
    root = PACNode(id="root", value=100.0)
    hierarchy = PACHierarchy(root)
    
    # Create hierarchy
    level1_values = [30.0, 25.0, 20.0, 15.0, 10.0]
    level1_nodes = []
    for i, val in enumerate(level1_values):
        node = PACNode(id=f"L1_{i}", value=val)
        hierarchy.add_node(node, parent_id=root.id, ownership_weight=val/100.0)
        level1_nodes.append(node)
    
    level2_nodes = []
    for i, parent in enumerate(level1_nodes):
        n_children = 3 + (i % 3)
        child_value = parent.value / n_children
        for j in range(n_children):
            node = PACNode(id=f"L2_{i}_{j}", value=child_value)
            hierarchy.add_node(node, parent_id=parent.id, ownership_weight=1.0/n_children)
            level2_nodes.append(node)
    
    for i, parent in enumerate(level2_nodes):
        n_children = 2 + (i % 4)
        child_value = parent.value / n_children
        for j in range(n_children):
            node = PACNode(id=f"L3_{i}_{j}", value=child_value)
            hierarchy.add_node(node, parent_id=parent.id, ownership_weight=1.0/n_children)
    
    print(f"Created hierarchy: {len(hierarchy.nodes)} nodes")
    leaf_count = len([n for n in hierarchy.nodes.values() if not n.children])
    print(f"Leaf nodes: {leaf_count}")
    print()
    
    # Run tests
    tester = NullHypothesisTester(hierarchy, n_iterations=100)
    tester.generate_real_embeddings()
    results = tester.run_all_tests()
    
    # Visualize
    tester.visualize_results(results)
    
    # Save results
    import os
    os.makedirs('euclidean_distance_validation/results', exist_ok=True)
    with open('euclidean_distance_validation/results/experiment_08_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for k, v in results.items():
            json_results[k] = {
                'test': v['test'],
                'original_r2': float(v['original_r2']),
                'null_mean': float(v['null_mean']),
                'null_std': float(v['null_std']),
                'p_value': float(v['p_value']),
                'significant': bool(v['significant'])
            }
        json.dump(json_results, f, indent=2)
    
    print("\nResults saved to results/experiment_08_results.json")
    
    return results


if __name__ == "__main__":
    results = main()

"""
Experiment 4: Context-Relative Distance Invariance

Tests Axiom 3: Distance ratios are preserved within shared collapse contexts.

Key insight from experimental failures:
- Distance is relative to collapse history (Einstein-like relativity in info space)
- Ratios are invariant WITHIN the same SEC context, not universally
- Cross-context divergence is expected and correct

This experiment validates:
1. Within-context invariance: nodes with shared collapse history show low ratio variance
2. Cross-context divergence: nodes from different contexts can have high variance
3. Context = information = memory of SEC recursion depth
"""

import numpy as np
from typing import List, Tuple, Dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.pac_hierarchy import PACNode, PACHierarchy
from core.embedding_generator import EmbeddingGenerator
import matplotlib.pyplot as plt
from scipy.stats import variation


class ContextGroup:
    """Group of nodes sharing collapse history to specified depth."""
    
    def __init__(self, nodes: List[PACNode], shared_depth: int, context_root: PACNode):
        self.nodes = nodes
        self.shared_depth = shared_depth  # How many levels of shared ancestry
        self.context_root = context_root  # Most recent common ancestor
        
    def compute_ratio_variance(self) -> float:
        """Compute coefficient of variation for all pairwise distance ratios."""
        if len(self.nodes) < 3:
            return np.nan
            
        ratios = []
        for i, A in enumerate(self.nodes):
            for j, B in enumerate(self.nodes):
                if i >= j:
                    continue
                for k, C in enumerate(self.nodes):
                    if k == i or k == j:
                        continue
                    
                    d_AB = A.distance_to(B)
                    d_AC = A.distance_to(C)
                    
                    if d_AC > 1e-10:  # Avoid division by zero
                        ratio = d_AB / d_AC
                        ratios.append(ratio)
        
        if len(ratios) < 5:
            return np.nan
            
        return variation(ratios)  # Coefficient of variation = std/mean


class ContextRelativeInvarianceAnalyzer:
    """Analyzes distance ratio invariance conditioned on collapse context."""
    
    def __init__(self, hierarchy: PACHierarchy):
        self.hierarchy = hierarchy
        self.embedding_gen = EmbeddingGenerator(model='synthetic', dimension=128)
        
    def generate_embeddings(self):
        """Generate embeddings for all nodes."""
        self.embedding_gen.embed_hierarchy(self.hierarchy)
        
    def find_context_groups(self, min_group_size: int = 4) -> List[ContextGroup]:
        """
        Identify groups of nodes sharing collapse history.
        
        Returns groups organized by shared ancestry depth:
        - Siblings (depth 1): share immediate parent
        - Cousins (depth 2): share grandparent
        - Etc.
        """
        groups = []
        
        # Find all internal nodes (potential context roots)
        internal_nodes = [n for n in self.hierarchy.nodes.values() 
                         if n.children]
        
        for root in internal_nodes:
            # Get all descendants at various depths
            descendants_by_level = self._get_descendants_by_level(root)
            
            for depth, descendants in descendants_by_level.items():
                if len(descendants) >= min_group_size:
                    group = ContextGroup(
                        nodes=descendants,
                        shared_depth=depth,
                        context_root=root
                    )
                    groups.append(group)
        
        return groups
    
    def _get_descendants_by_level(self, root: PACNode, max_depth: int = 4) -> Dict[int, List[PACNode]]:
        """Get descendants organized by tree depth from root."""
        levels = {i: [] for i in range(1, max_depth + 1)}
        
        def traverse(node: PACNode, depth: int):
            if depth > max_depth:
                return
            if node.children:
                for child in node.children:  # children is a list of PACNode objects
                    levels[depth].append(child)
                    traverse(child, depth + 1)
        
        traverse(root, 1)
        return {d: nodes for d, nodes in levels.items() if nodes}
    
    def test_within_context_invariance(self, groups: List[ContextGroup]) -> Dict:
        """
        Test Hypothesis 4A: Within shared context, distance ratios should be invariant.
        
        Expected: CV < 0.20 for most groups (low variance)
        """
        results = {
            'group_cvs': [],
            'group_sizes': [],
            'group_depths': [],
            'mean_cv': None,
            'pass_rate': None
        }
        
        for group in groups:
            cv = group.compute_ratio_variance()
            if not np.isnan(cv):
                results['group_cvs'].append(cv)
                results['group_sizes'].append(len(group.nodes))
                results['group_depths'].append(group.shared_depth)
        
        if results['group_cvs']:
            results['mean_cv'] = np.mean(results['group_cvs'])
            results['pass_rate'] = np.mean([cv < 0.20 for cv in results['group_cvs']])
        
        return results
    
    def test_cross_context_divergence(self, groups: List[ContextGroup], n_samples: int = 100) -> Dict:
        """
        Test Hypothesis 4B: Across different contexts, ratios CAN diverge.
        
        Expected: CV_cross > CV_within (context matters!)
        """
        if len(groups) < 2:
            return {'cross_context_cv': np.nan}
        
        cross_ratios = []
        
        for _ in range(n_samples):
            # Sample nodes from different context groups
            g1, g2 = np.random.choice(groups, size=2, replace=False)
            
            if len(g1.nodes) < 2 or len(g2.nodes) < 2:
                continue
            
            A = np.random.choice(g1.nodes)
            B = np.random.choice(g1.nodes)
            C = np.random.choice(g2.nodes)  # From different context!
            
            d_AB = A.distance_to(B)
            d_AC = A.distance_to(C)
            
            if d_AC > 1e-10:
                ratio = d_AB / d_AC
                cross_ratios.append(ratio)
        
        results = {
            'cross_context_cv': variation(cross_ratios) if cross_ratios else np.nan,
            'n_samples': len(cross_ratios)
        }
        
        return results
    
    def run_full_analysis(self) -> Dict:
        """Execute complete context-relative invariance test."""
        print("=" * 60)
        print("EXPERIMENT 4: Context-Relative Distance Invariance")
        print("=" * 60)
        print("\nHypothesis 4A: Within shared collapse context, distance ratios are invariant")
        print("Hypothesis 4B: Across contexts, ratios can diverge (Einstein-like relativity)")
        print()
        
        # Generate embeddings
        print("Generating synthetic embeddings...")
        self.generate_embeddings()
        print(f"OK - Embeddings generated for {len(self.hierarchy.nodes)} nodes")
        print()
        
        # Find context groups
        print("Identifying context groups...")
        groups = self.find_context_groups(min_group_size=4)
        print(f"OK - Found {len(groups)} context groups")
        
        # Show group distribution
        depth_counts = {}
        for g in groups:
            depth_counts[g.shared_depth] = depth_counts.get(g.shared_depth, 0) + 1
        print("\nGroups by shared depth:")
        for depth in sorted(depth_counts.keys()):
            print(f"  Depth {depth}: {depth_counts[depth]} groups")
        print()
        
        # Test within-context invariance
        print("Testing within-context invariance...")
        within_results = self.test_within_context_invariance(groups)
        
        print(f"\nWithin-Context Results:")
        print(f"  Mean CV: {within_results['mean_cv']:.4f}")
        print(f"  Pass rate (CV < 0.20): {within_results['pass_rate']*100:.1f}%")
        print(f"  Groups tested: {len(within_results['group_cvs'])}")
        
        # Test cross-context divergence
        print("\nTesting cross-context divergence...")
        cross_results = self.test_cross_context_divergence(groups)
        
        print(f"\nCross-Context Results:")
        print(f"  Cross-context CV: {cross_results['cross_context_cv']:.4f}")
        print(f"  Samples: {cross_results['n_samples']}")
        
        # Compare
        if within_results['mean_cv'] and not np.isnan(cross_results['cross_context_cv']):
            ratio = cross_results['cross_context_cv'] / within_results['mean_cv']
            print(f"\nContext Sensitivity:")
            print(f"  CV_cross / CV_within = {ratio:.2f}x")
            
            if ratio > 2.0:
                print("  Strong context-dependence detected (Einstein-like relativity)")
            elif ratio > 1.5:
                print("  Moderate context-dependence")
            else:
                print("  Weak context-dependence")
        
        # Final verdict
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        success_4a = within_results['pass_rate'] > 0.70 if within_results['pass_rate'] else False
        success_4b = (cross_results['cross_context_cv'] > within_results['mean_cv'] 
                     if within_results['mean_cv'] and not np.isnan(cross_results['cross_context_cv']) 
                     else False)
        
        print(f"\nHypothesis 4A (within-context invariance): {'PASS' if success_4a else 'FAIL'}")
        print(f"Hypothesis 4B (cross-context divergence): {'PASS' if success_4b else 'FAIL'}")
        
        if success_4a and success_4b:
            print("\nAXIOM 3 VALIDATED")
            print("Distance is relative to collapse history (context matters!)")
        elif success_4a:
            print("\nPartial validation: invariance within context confirmed")
        else:
            print("\nAxiom 3 not validated by this embedding strategy")
        
        return {
            'within_context': within_results,
            'cross_context': cross_results,
            'groups': groups,
            'success_4a': success_4a,
            'success_4b': success_4b
        }
    
    def visualize_results(self, results: Dict):
        """Create visualizations of context-relative invariance."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        within = results['within_context']
        
        # Plot 1: CV distribution within contexts
        ax = axes[0, 0]
        ax.hist(within['group_cvs'], bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(0.20, color='red', linestyle='--', label='Threshold (0.20)')
        ax.axvline(within['mean_cv'], color='blue', linestyle='--', label=f'Mean ({within["mean_cv"]:.3f})')
        ax.set_xlabel('Coefficient of Variation')
        ax.set_ylabel('Number of Groups')
        ax.set_title('Within-Context Ratio Variance')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: CV vs group size
        ax = axes[0, 1]
        ax.scatter(within['group_sizes'], within['group_cvs'], alpha=0.6)
        ax.axhline(0.20, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Group Size (nodes)')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Invariance vs Group Size')
        ax.grid(alpha=0.3)
        
        # Plot 3: CV vs shared depth
        ax = axes[1, 0]
        depth_cvs = {}
        for depth, cv in zip(within['group_depths'], within['group_cvs']):
            if depth not in depth_cvs:
                depth_cvs[depth] = []
            depth_cvs[depth].append(cv)
        
        depths = sorted(depth_cvs.keys())
        means = [np.mean(depth_cvs[d]) for d in depths]
        stds = [np.std(depth_cvs[d]) for d in depths]
        
        ax.errorbar(depths, means, yerr=stds, marker='o', capsize=5, capthick=2)
        ax.axhline(0.20, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.set_xlabel('Shared Collapse Depth')
        ax.set_ylabel('Mean CV')
        ax.set_title('Invariance vs Collapse History Depth')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 4: Context comparison
        ax = axes[1, 1]
        comparison_data = [
            within['mean_cv'],
            results['cross_context']['cross_context_cv']
        ]
        labels = ['Within\nContext', 'Cross\nContext']
        colors = ['green', 'orange']
        
        bars = ax.bar(labels, comparison_data, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0.20, color='red', linestyle='--', alpha=0.5, label='Within threshold')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Context Sensitivity Comparison')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # Add ratio annotation
        if comparison_data[1] and comparison_data[0]:
            ratio = comparison_data[1] / comparison_data[0]
            ax.text(0.5, max(comparison_data)*0.9, f'{ratio:.1f}x', 
                   ha='center', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        import os
        os.makedirs('euclidean_distance_validation/results', exist_ok=True)
        plt.savefig('euclidean_distance_validation/results/experiment_04_context_invariance.png', 
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved to results/experiment_04_context_invariance.png")


def main():
    """Run Experiment 4: Context-Relative Distance Invariance."""
    
    # Build test hierarchy
    root = PACNode(id="root", value=100.0, metadata={"level": 0})
    hierarchy = PACHierarchy(root)
    
    # Level 1: 4 main branches (different contexts)
    level1_nodes = []
    for i in range(4):
        node = PACNode(id=f"L1_{i}", value=25.0, metadata={"level": 1, "branch": i})
        hierarchy.add_node(node, parent_id=root.id, ownership_weight=0.25)
        level1_nodes.append(node)
    
    # Level 2: Each L1 has 4 children (sibling groups = shared context)
    level2_nodes = []
    for i, parent in enumerate(level1_nodes):
        for j in range(4):
            node = PACNode(id=f"L2_{i}_{j}", value=6.25, metadata={"level": 2, "branch": i})
            hierarchy.add_node(node, parent_id=parent.id, ownership_weight=0.25)
            level2_nodes.append(node)
    
    # Level 3: Each L2 has 3-5 children (leaf groups)
    for i, parent in enumerate(level2_nodes):
        n_children = np.random.randint(3, 6)
        child_value = parent.value / n_children
        for j in range(n_children):
            node = PACNode(id=f"L3_{i}_{j}", value=child_value, 
                         metadata={"level": 3, "parent_idx": i})
            hierarchy.add_node(node, parent_id=parent.id, ownership_weight=1.0/n_children)
    
    print(f"Created test hierarchy: {len(hierarchy.nodes)} nodes")
    print(f"  Level 0: 1 root")
    print(f"  Level 1: 4 branches")
    print(f"  Level 2: {len(level2_nodes)} nodes")
    print(f"  Level 3: {len([n for n in hierarchy.nodes.values() if n.metadata.get('level') == 3])} leaves")
    print()
    
    # Run analysis
    analyzer = ContextRelativeInvarianceAnalyzer(hierarchy)
    results = analyzer.run_full_analysis()
    
    # Visualize
    analyzer.visualize_results(results)
    
    return results


if __name__ == "__main__":
    results = main()

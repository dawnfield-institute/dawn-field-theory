"""
Experiment 3: Depth-Width Distance Tradeoff

Tests the Complexity Symmetry Principle in distance space.

Hypothesis 3: Distance Spread Correlation
    Variance(d(Cᵢ, P)) ∝ Depth_recursive(P)

From PAC theory:
- Deep parent (compressed complexity) → children spread widely in distance
- Shallow parent (explicit complexity) → children clustered nearby
- Total "distance complexity" conserved across transformation

Measures:
- Recursive depth of each parent
- Distance spread (std deviation) of children
- Correlation between depth and spread
- Depth-width "conservation" coefficient
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.pac_hierarchy import PACNode, PACHierarchy
from core.embedding_generator import EmbeddingGenerator


@dataclass
class DepthWidthResult:
    """Results for single parent node."""
    parent_id: str
    recursive_depth: int
    num_children: int
    children_distance_spread: float  # std dev
    children_distance_mean: float
    parent_depth: int  # depth from root


class DepthWidthAnalyzer:
    """
    Analyzes depth-width tradeoff in distance space.
    
    Tests whether recursive depth correlates with distance spread.
    """
    
    def __init__(self, hierarchy: PACHierarchy):
        """
        Initialize analyzer.
        
        Args:
            hierarchy: Hierarchy to analyze
        """
        self.hierarchy = hierarchy
        self.results: List[DepthWidthResult] = []
    
    def compute_recursive_depth(self, node: PACNode) -> int:
        """
        Compute maximum recursive depth from node to leaves.
        
        Args:
            node: Node to measure from
        
        Returns:
            Maximum path length to any leaf descendant
        """
        if not node.children:
            return 0
        return 1 + max(self.compute_recursive_depth(child) for child in node.children)
    
    def compute_children_distance_stats(
        self, 
        parent: PACNode
    ) -> Tuple[float, float]:
        """
        Compute distance statistics for children relative to parent.
        
        Args:
            parent: Parent node
        
        Returns:
            Tuple of (mean_distance, std_distance)
        """
        if not parent.children or parent.embedding is None:
            return 0.0, 0.0
        
        distances = []
        for child in parent.children:
            if child.embedding is not None:
                dist = np.linalg.norm(child.embedding - parent.embedding)
                distances.append(dist)
        
        if not distances:
            return 0.0, 0.0
        
        return np.mean(distances), np.std(distances)
    
    def analyze_node(self, node: PACNode) -> DepthWidthResult:
        """
        Analyze single parent node.
        
        Args:
            node: Parent node to analyze
        
        Returns:
            DepthWidthResult with measurements
        """
        recursive_depth = self.compute_recursive_depth(node)
        mean_dist, std_dist = self.compute_children_distance_stats(node)
        
        return DepthWidthResult(
            parent_id=node.id,
            recursive_depth=recursive_depth,
            num_children=len(node.children),
            children_distance_spread=std_dist,
            children_distance_mean=mean_dist,
            parent_depth=node.depth
        )
    
    def run(self) -> Dict:
        """
        Run full analysis across all parent nodes.
        
        Returns:
            Dictionary with summary statistics and detailed results
        """
        self.results = []
        
        for parent in self.hierarchy.get_all_parents():
            try:
                result = self.analyze_node(parent)
                # Only include if has meaningful data
                if result.num_children > 1:  # Need at least 2 children for spread
                    self.results.append(result)
            except Exception as e:
                print(f"Warning: Failed to analyze parent {parent.id}: {e}")
        
        return self._compute_summary()
    
    def _compute_summary(self) -> Dict:
        """Compute summary statistics from results."""
        if not self.results:
            return {'error': 'No results collected'}
        
        recursive_depths = [r.recursive_depth for r in self.results]
        distance_spreads = [r.children_distance_spread for r in self.results]
        
        # Correlation analysis
        pearson_r, pearson_p = pearsonr(recursive_depths, distance_spreads)
        spearman_r, spearman_p = spearmanr(recursive_depths, distance_spreads)
        
        # Group by recursive depth
        depth_groups = {}
        for r in self.results:
            if r.recursive_depth not in depth_groups:
                depth_groups[r.recursive_depth] = []
            depth_groups[r.recursive_depth].append(r.children_distance_spread)
        
        depth_group_stats = {
            depth: {
                'mean': np.mean(spreads),
                'std': np.std(spreads),
                'count': len(spreads)
            }
            for depth, spreads in depth_groups.items()
        }
        
        summary = {
            'num_parents': len(self.results),
            'recursive_depth': {
                'mean': np.mean(recursive_depths),
                'std': np.std(recursive_depths),
                'min': np.min(recursive_depths),
                'max': np.max(recursive_depths)
            },
            'distance_spread': {
                'mean': np.mean(distance_spreads),
                'std': np.std(distance_spreads),
                'min': np.min(distance_spreads),
                'max': np.max(distance_spreads)
            },
            'correlation': {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            },
            'depth_groups': depth_group_stats,
            'validation': self._interpret_results(pearson_r, pearson_p)
        }
        
        return summary
    
    def _interpret_results(self, correlation: float, p_value: float) -> str:
        """
        Interpret validation results.
        
        Args:
            correlation: Pearson correlation coefficient
            p_value: Statistical significance
        
        Returns:
            Interpretation string
        """
        if p_value > 0.05:
            return "NO SIGNIFICANT CORRELATION: Depth-width hypothesis not supported"
        
        if correlation > 0.7:
            return "STRONG POSITIVE: Deep structures show wider distance spread"
        elif correlation > 0.5:
            return "MODERATE POSITIVE: Some depth-width correlation present"
        elif correlation > 0.3:
            return "WEAK POSITIVE: Slight depth-width relationship"
        elif correlation < -0.3:
            return "NEGATIVE: Deeper structures show tighter clustering (unexpected)"
        else:
            return "WEAK/NONE: Little relationship between depth and spread"
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Visualize depth-width analysis.
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.results:
            print("No results to plot. Run analysis first.")
            return
        
        recursive_depths = [r.recursive_depth for r in self.results]
        distance_spreads = [r.children_distance_spread for r in self.results]
        num_children = [r.num_children for r in self.results]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Scatter with regression line
        ax = axes[0, 0]
        ax.scatter(recursive_depths, distance_spreads, alpha=0.6, s=50)
        ax.set_xlabel('Recursive Depth')
        ax.set_ylabel('Children Distance Spread (std)')
        ax.set_title('Depth-Width Tradeoff in Distance Space')
        
        # Fit line
        if len(recursive_depths) > 1:
            z = np.polyfit(recursive_depths, distance_spreads, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(recursive_depths), max(recursive_depths), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, 
                   label=f'Fit: y={z[0]:.4f}x+{z[1]:.4f}')
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Boxplot by depth
        ax = axes[0, 1]
        depth_groups = {}
        for r in self.results:
            if r.recursive_depth not in depth_groups:
                depth_groups[r.recursive_depth] = []
            depth_groups[r.recursive_depth].append(r.children_distance_spread)
        
        depths_sorted = sorted(depth_groups.keys())
        spreads_by_depth = [depth_groups[d] for d in depths_sorted]
        
        ax.boxplot(spreads_by_depth, labels=depths_sorted)
        ax.set_xlabel('Recursive Depth')
        ax.set_ylabel('Children Distance Spread')
        ax.set_title('Distance Spread Distribution by Depth')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: By number of children
        ax = axes[1, 0]
        scatter = ax.scatter(
            recursive_depths, 
            distance_spreads,
            c=num_children,
            s=100,
            alpha=0.6,
            cmap='viridis'
        )
        ax.set_xlabel('Recursive Depth')
        ax.set_ylabel('Children Distance Spread')
        ax.set_title('Colored by Number of Children')
        plt.colorbar(scatter, ax=ax, label='Num Children')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Mean distance vs spread
        ax = axes[1, 1]
        mean_distances = [r.children_distance_mean for r in self.results]
        ax.scatter(mean_distances, distance_spreads, alpha=0.6, s=50)
        ax.set_xlabel('Mean Distance (Parent to Children)')
        ax.set_ylabel('Distance Spread (std)')
        ax.set_title('Mean vs Spread of Children Distances')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()


def run_experiment(
    hierarchy: Optional[PACHierarchy] = None,
    depth: int = 5,
    branching: int = 3,
    dimension: int = 128,
    seed: int = 42
) -> Dict:
    """
    Run Experiment 3: Depth-Width Distance Tradeoff.
    
    Args:
        hierarchy: Optional pre-built hierarchy (creates synthetic if None)
        depth: Depth for synthetic hierarchy
        branching: Branching factor for synthetic hierarchy
        dimension: Embedding dimension
        seed: Random seed
    
    Returns:
        Analysis results dictionary
    """
    print("=" * 80)
    print("EXPERIMENT 3: DEPTH-WIDTH DISTANCE TRADEOFF")
    print("=" * 80)
    
    # Create or use provided hierarchy
    if hierarchy is None:
        print(f"\nGenerating synthetic hierarchy (depth={depth}, branching={branching})...")
        from core.embedding_generator import create_synthetic_hierarchy_with_embeddings
        hierarchy = create_synthetic_hierarchy_with_embeddings(
            depth=depth,
            branching=branching,
            dimension=dimension,
            seed=seed
        )
    
    print(f"Hierarchy: {len(hierarchy)} nodes, max depth {hierarchy.get_max_depth()}")
    
    # Run analysis
    print("\nAnalyzing depth-width tradeoff...")
    analyzer = DepthWidthAnalyzer(hierarchy)
    results = analyzer.run()
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nParents analyzed: {results['num_parents']}")
    
    print(f"\nRecursive Depth Statistics:")
    print(f"  Mean: {results['recursive_depth']['mean']:.2f}")
    print(f"  Range: [{results['recursive_depth']['min']}, {results['recursive_depth']['max']}]")
    
    print(f"\nDistance Spread Statistics:")
    print(f"  Mean: {results['distance_spread']['mean']:.4f}")
    print(f"  Std:  {results['distance_spread']['std']:.4f}")
    print(f"  Range: [{results['distance_spread']['min']:.4f}, {results['distance_spread']['max']:.4f}]")
    
    print(f"\nCorrelation Analysis:")
    print(f"  Pearson r:  {results['correlation']['pearson_r']:.4f} "
          f"(p={results['correlation']['pearson_p']:.4e})")
    print(f"  Spearman r: {results['correlation']['spearman_r']:.4f} "
          f"(p={results['correlation']['spearman_p']:.4e})")
    
    print(f"\nBy Recursive Depth:")
    for depth_val in sorted(results['depth_groups'].keys()):
        stats = results['depth_groups'][depth_val]
        print(f"  Depth {depth_val}: spread = {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"(n={stats['count']})")
    
    print(f"\nValidation: {results['validation']}")
    
    # Interpretation
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print("=" * 80)
    
    r = results['correlation']['pearson_r']
    p = results['correlation']['pearson_p']
    
    if p < 0.05:
        if r > 0:
            print("\n✅ COMPLEXITY SYMMETRY SUPPORTED:")
            print("   Deeper recursive structures → wider distance spread")
            print("   Parent's compressed depth → children's explicit width")
        else:
            print("\n⚠️  NEGATIVE CORRELATION (Unexpected):")
            print("   Deeper structures → tighter clustering")
            print("   May indicate specific embedding properties")
    else:
        print("\n❌ NO SIGNIFICANT CORRELATION:")
        print("   Depth and spread appear independent")
        print("   Complexity Symmetry not evident in distance space")
    
    # Visualization
    print(f"\n{'=' * 80}")
    print("Generating visualization...")
    analyzer.plot_results()
    
    return results


if __name__ == "__main__":
    # Run with synthetic data
    results = run_experiment(
        depth=5,
        branching=3,
        dimension=128,
        seed=42
    )

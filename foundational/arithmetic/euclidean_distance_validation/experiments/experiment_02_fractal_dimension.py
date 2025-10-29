"""
Experiment 2: Fractal Dimension Measurement

Tests whether PAC hierarchies exhibit fractal scaling in distance space.

Hypothesis 2: Power Law Distance Scaling
    d(level_k) ∼ λᵏ · d(level_0)
    
    Fractal dimension: D = log(N) / log(1/λ)
    
    Where:
    - k is decomposition depth
    - λ is scaling factor
    - N is branching factor

Measures:
- Average distance from root at each level
- Log-log scaling relationship
- Fractal dimension D
- R² goodness of fit
- Consistency across subtrees
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import linregress

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.pac_hierarchy import PACNode, PACHierarchy
from core.embedding_generator import EmbeddingGenerator


@dataclass
class FractalResult:
    """Results from fractal dimension measurement."""
    hierarchy_id: str
    fractal_dimension: float
    scaling_factor: float
    r_squared: float
    branching_factor: float
    max_depth: int
    avg_distances: List[float]
    std_distances: List[float]


class FractalDimensionAnalyzer:
    """
    Analyzes fractal scaling properties of PAC hierarchies.
    
    Measures how distances scale across hierarchical levels.
    """
    
    def __init__(self, hierarchy: PACHierarchy):
        """
        Initialize analyzer.
        
        Args:
            hierarchy: Hierarchy to analyze
        """
        self.hierarchy = hierarchy
        self.root = hierarchy.root
        self.results: Optional[FractalResult] = None
    
    def measure_level_distances(self) -> Tuple[List[float], List[float]]:
        """
        Measure average distance from root at each level.
        
        Returns:
            Tuple of (avg_distances, std_distances) for each level
        """
        if self.root.embedding is None:
            raise ValueError("Root node has no embedding")
        
        levels = self.hierarchy.get_levels()
        avg_distances = []
        std_distances = []
        
        for level in levels:
            if not level:
                continue
            
            distances = []
            for node in level:
                if node.embedding is not None:
                    dist = np.linalg.norm(node.embedding - self.root.embedding)
                    distances.append(dist)
            
            if distances:
                avg_distances.append(np.mean(distances))
                std_distances.append(np.std(distances))
            else:
                avg_distances.append(0.0)
                std_distances.append(0.0)
        
        return avg_distances, std_distances
    
    def fit_power_law(
        self, 
        avg_distances: List[float]
    ) -> Tuple[float, float, float]:
        """
        Fit power law to distance scaling.
        
        Args:
            avg_distances: Average distance at each level
        
        Returns:
            Tuple of (scaling_factor λ, intercept, R²)
        """
        # Remove level 0 (root has distance 0)
        if len(avg_distances) < 2:
            raise ValueError("Need at least 2 levels for power law fit")
        
        # Use levels 1+ (skip root)
        k_values = np.arange(1, len(avg_distances))
        distances = np.array(avg_distances[1:])
        
        # Filter out zero or negative distances
        valid_mask = distances > 0
        k_values = k_values[valid_mask]
        distances = distances[valid_mask]
        
        if len(k_values) < 2:
            raise ValueError("Not enough valid distance measurements")
        
        # Fit: log(d) = log(λ) * k + log(d0)
        log_distances = np.log(distances)
        slope, intercept, r_value, _, _ = linregress(k_values, log_distances)
        
        scaling_factor = np.exp(slope)
        r_squared = r_value ** 2
        
        return scaling_factor, intercept, r_squared
    
    def compute_fractal_dimension(
        self, 
        scaling_factor: float, 
        branching_factor: float
    ) -> float:
        """
        Compute fractal dimension from scaling factor.
        
        D = log(N) / log(1/λ)
        
        Args:
            scaling_factor: λ from power law fit
            branching_factor: Average number of children
        
        Returns:
            Fractal dimension D
        """
        if scaling_factor <= 0:
            return np.nan
        
        # If scaling_factor > 1, distances are growing (converging to limit)
        # This is actually expected for embeddings where children spread out
        # In this case, use inverse: D = log(N) / log(λ)
        if scaling_factor >= 1:
            # Converging case
            fractal_dim = np.log(branching_factor) / np.log(scaling_factor)
        else:
            # Diverging case (original formula)
            fractal_dim = np.log(branching_factor) / np.log(1 / scaling_factor)
        
        return fractal_dim
    
    def analyze(self) -> FractalResult:
        """
        Perform complete fractal analysis.
        
        Returns:
            FractalResult with all measurements
        """
        # Measure distances
        avg_distances, std_distances = self.measure_level_distances()
        
        # Fit power law
        scaling_factor, intercept, r_squared = self.fit_power_law(avg_distances)
        
        # Compute branching factor
        parents = self.hierarchy.get_all_parents()
        if not parents:
            branching_factor = 0.0
        else:
            branching_factor = np.mean([len(p.children) for p in parents])
        
        # Compute fractal dimension
        fractal_dimension = self.compute_fractal_dimension(
            scaling_factor, 
            branching_factor
        )
        
        self.results = FractalResult(
            hierarchy_id=self.root.id,
            fractal_dimension=fractal_dimension,
            scaling_factor=scaling_factor,
            r_squared=r_squared,
            branching_factor=branching_factor,
            max_depth=self.hierarchy.get_max_depth(),
            avg_distances=avg_distances,
            std_distances=std_distances
        )
        
        return self.results
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Visualize fractal scaling.
        
        Args:
            save_path: Optional path to save figure
        """
        if self.results is None:
            print("No results to plot. Run analysis first.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Linear scale
        ax = axes[0]
        levels = np.arange(len(self.results.avg_distances))
        ax.errorbar(
            levels, 
            self.results.avg_distances,
            yerr=self.results.std_distances,
            fmt='o-',
            capsize=5,
            label='Measured'
        )
        ax.set_xlabel('Depth Level k')
        ax.set_ylabel('Average Distance from Root')
        ax.set_title('Distance Scaling (Linear)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Log-log scale
        ax = axes[1]
        # Skip level 0 (root)
        levels_nonzero = levels[1:]
        distances_nonzero = np.array(self.results.avg_distances[1:])
        valid_mask = distances_nonzero > 0
        
        if np.any(valid_mask):
            ax.loglog(
                levels_nonzero[valid_mask],
                distances_nonzero[valid_mask],
                'ro',
                markersize=8,
                label='Measured'
            )
            
            # Fit line
            k_fit = np.linspace(1, max(levels_nonzero[valid_mask]), 100)
            log_distances_fit = (
                np.log(self.results.scaling_factor) * k_fit + 
                np.log(distances_nonzero[valid_mask][0]) - 
                np.log(self.results.scaling_factor)
            )
            distances_fit = np.exp(log_distances_fit)
            ax.loglog(
                k_fit, 
                distances_fit, 
                'b--', 
                alpha=0.7,
                label=f'Fit: λ={self.results.scaling_factor:.3f}'
            )
        
        ax.set_xlabel('Depth Level k')
        ax.set_ylabel('Average Distance from Root')
        ax.set_title(f'Fractal Scaling (Log-Log), R²={self.results.r_squared:.3f}')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()
        
        # Plot 3: Summary statistics
        ax = axes[2]
        ax.axis('off')
        
        summary_text = f"""
        FRACTAL ANALYSIS RESULTS
        ========================
        
        Fractal Dimension (D): {self.results.fractal_dimension:.3f}
        Scaling Factor (λ):    {self.results.scaling_factor:.3f}
        Branching Factor (N):  {self.results.branching_factor:.2f}
        
        R² (fit quality):      {self.results.r_squared:.4f}
        Max Depth:             {self.results.max_depth}
        
        Interpretation:
        {'D ≈ 1: Linear/chain structure' if self.results.fractal_dimension < 1.5 else ''}
        {'D ≈ 2: Tree/planar structure' if 1.5 <= self.results.fractal_dimension < 2.5 else ''}
        {'D > 2: High-dimensional structure' if self.results.fractal_dimension >= 2.5 else ''}
        
        Quality: {'EXCELLENT (R² > 0.95)' if self.results.r_squared > 0.95 else 
                  'GOOD (R² > 0.9)' if self.results.r_squared > 0.9 else
                  'MODERATE (R² > 0.8)' if self.results.r_squared > 0.8 else
                  'POOR (R² < 0.8)'}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()


def compare_subtrees(hierarchy: PACHierarchy, min_depth: int = 2) -> Dict:
    """
    Analyze fractal consistency across different subtrees.
    
    Tests whether fractal dimension is consistent across the hierarchy.
    
    Args:
        hierarchy: Full hierarchy
        min_depth: Minimum depth for subtree to be analyzed
    
    Returns:
        Dictionary with comparison results
    """
    results = []
    
    # Find suitable subtree roots (not too shallow)
    for node in hierarchy.nodes.values():
        if node.depth > 0 and _get_subtree_depth(node) >= min_depth:
            # Create subtree hierarchy
            subtree = _extract_subtree(node)
            
            # Analyze
            try:
                analyzer = FractalDimensionAnalyzer(subtree)
                result = analyzer.analyze()
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to analyze subtree {node.id}: {e}")
    
    if not results:
        return {'error': 'No suitable subtrees found'}
    
    # Compute statistics
    fractal_dims = [r.fractal_dimension for r in results if not np.isnan(r.fractal_dimension)]
    scaling_factors = [r.scaling_factor for r in results]
    r_squareds = [r.r_squared for r in results]
    
    if not fractal_dims:
        return {
            'num_subtrees': len(results),
            'error': 'No valid fractal dimensions computed',
            'note': 'All subtrees had invalid scaling factors or insufficient depth'
        }
    
    return {
        'num_subtrees': len(results),
        'fractal_dimension': {
            'mean': np.mean(fractal_dims),
            'std': np.std(fractal_dims),
            'min': np.min(fractal_dims),
            'max': np.max(fractal_dims)
        },
        'scaling_factor': {
            'mean': np.mean(scaling_factors),
            'std': np.std(scaling_factors)
        },
        'r_squared': {
            'mean': np.mean(r_squareds),
            'min': np.min(r_squareds)
        },
        'consistency': np.std(fractal_dims) / np.mean(fractal_dims) if fractal_dims else np.nan
    }


def _get_subtree_depth(node: PACNode) -> int:
    """Get maximum depth of subtree rooted at node."""
    if not node.children:
        return 0
    return 1 + max(_get_subtree_depth(child) for child in node.children)


def _extract_subtree(root: PACNode) -> PACHierarchy:
    """Extract subtree as independent hierarchy."""
    # Reset depths relative to new root
    def reset_depths(node: PACNode, depth: int = 0):
        node.depth = depth
        for child in node.children:
            reset_depths(child, depth + 1)
    
    reset_depths(root, 0)
    
    subtree = PACHierarchy(root)
    
    # Collect nodes
    def collect(node: PACNode):
        subtree.nodes[node.id] = node
        for child in node.children:
            collect(child)
    
    collect(root)
    
    return subtree


def run_experiment(
    hierarchy: Optional[PACHierarchy] = None,
    depth: int = 5,
    branching: int = 3,
    dimension: int = 128,
    seed: int = 42,
    compare_subtrees_flag: bool = True
) -> Dict:
    """
    Run Experiment 2: Fractal Dimension Measurement.
    
    Args:
        hierarchy: Optional pre-built hierarchy (creates synthetic if None)
        depth: Depth for synthetic hierarchy
        branching: Branching factor for synthetic hierarchy
        dimension: Embedding dimension
        seed: Random seed
        compare_subtrees_flag: Whether to compare subtrees
    
    Returns:
        Analysis results dictionary
    """
    print("=" * 80)
    print("EXPERIMENT 2: FRACTAL DIMENSION MEASUREMENT")
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
    print("\nAnalyzing fractal scaling...")
    analyzer = FractalDimensionAnalyzer(hierarchy)
    result = analyzer.analyze()
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\nFractal Dimension (D): {result.fractal_dimension:.4f}")
    print(f"Scaling Factor (λ):    {result.scaling_factor:.4f}")
    print(f"Branching Factor (N):  {result.branching_factor:.2f}")
    print(f"R² (fit quality):      {result.r_squared:.4f}")
    print(f"Max Depth:             {result.max_depth}")
    
    print(f"\nDistance by Level:")
    for i, (avg, std) in enumerate(zip(result.avg_distances, result.std_distances)):
        print(f"  Level {i}: {avg:.4f} ± {std:.4f}")
    
    # Interpretation
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print("=" * 80)
    
    if result.fractal_dimension < 1.5:
        structure = "LINEAR/CHAIN structure (low complexity)"
    elif result.fractal_dimension < 2.5:
        structure = "TREE/PLANAR structure (moderate complexity)"
    else:
        structure = "HIGH-DIMENSIONAL structure (high complexity)"
    
    print(f"\nStructure Type: {structure}")
    
    if result.r_squared > 0.95:
        quality = "EXCELLENT - Strong power law scaling"
    elif result.r_squared > 0.9:
        quality = "GOOD - Clear power law behavior"
    elif result.r_squared > 0.8:
        quality = "MODERATE - Some power law characteristics"
    else:
        quality = "POOR - Weak power law scaling"
    
    print(f"Fit Quality:    {quality}")
    
    # Theoretical expectation
    theoretical_d = np.log(branching) / np.log(branching)  # Simplistic
    print(f"\nNote: For balanced tree with branching={branching}, D typically in range [1.5, 2.5]")
    print(f"      Observed D = {result.fractal_dimension:.2f} is {'within' if 1.5 <= result.fractal_dimension <= 2.5 else 'outside'} expected range")
    
    # Subtree comparison
    if compare_subtrees_flag and hierarchy.get_max_depth() >= 3:
        print(f"\n{'=' * 80}")
        print("SUBTREE CONSISTENCY ANALYSIS")
        print("=" * 80)
        
        subtree_results = compare_subtrees(hierarchy, min_depth=2)
        
        if 'error' not in subtree_results:
            print(f"\nAnalyzed {subtree_results['num_subtrees']} subtrees")
            print(f"\nFractal Dimension Consistency:")
            print(f"  Mean: {subtree_results['fractal_dimension']['mean']:.4f}")
            print(f"  Std:  {subtree_results['fractal_dimension']['std']:.4f}")
            print(f"  Range: [{subtree_results['fractal_dimension']['min']:.4f}, "
                  f"{subtree_results['fractal_dimension']['max']:.4f}]")
            print(f"  Coefficient of Variation: {subtree_results['consistency']:.4f}")
            
            if subtree_results['consistency'] < 0.1:
                consistency = "EXCELLENT - Very consistent across subtrees"
            elif subtree_results['consistency'] < 0.2:
                consistency = "GOOD - Reasonably consistent"
            else:
                consistency = "VARIABLE - Significant variation across subtrees"
            
            print(f"\nConsistency: {consistency}")
    
    # Visualization
    print(f"\n{'=' * 80}")
    print("Generating visualization...")
    analyzer.plot_results()
    
    return {
        'result': result,
        'subtree_comparison': subtree_results if compare_subtrees_flag else None
    }


if __name__ == "__main__":
    # Run with synthetic data
    results = run_experiment(
        depth=5,
        branching=3,
        dimension=128,
        seed=42,
        compare_subtrees_flag=True
    )

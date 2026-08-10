"""
Experiment 1: Distance Conservation Validation

Tests whether Euclidean distance metrics preserve PAC conservation structure.

Hypothesis 1A: Weighted Distance Conservation
    ||e(P)||² ≈ Σᵢ αᵢ·||e(Cᵢ)||²

Hypothesis 1B: Distance Sum Conservation  
    d(P, ref) ≈ Σᵢ wᵢ·d(Cᵢ, ref)

Measures:
- PAC residual: |f(P) - Σf(C)|
- Distance residual: | ||e(P)||² - Σ||e(C)||² |
- Correlation between residuals
- Distribution of residuals
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
class ConservationResult:
    """Results from distance conservation validation."""
    parent_id: str
    pac_residual: float
    dist_residual_norm: float  # Norm-based
    dist_residual_ref: Optional[float] = None  # Reference-based
    num_children: int = 0
    depth: int = 0


class DistanceConservationValidator:
    """
    Validates PAC conservation through distance metrics.
    
    Tests whether distance relationships mirror value conservation.
    """
    
    def __init__(self, hierarchy: PACHierarchy, reference_node: Optional[str] = None):
        """
        Initialize validator.
        
        Args:
            hierarchy: Hierarchy to validate
            reference_node: Optional reference node ID for distance-based tests
        """
        self.hierarchy = hierarchy
        self.reference_node = (
            hierarchy.nodes[reference_node] if reference_node 
            else self._select_reference_node()
        )
        self.results: List[ConservationResult] = []
    
    def _select_reference_node(self) -> PACNode:
        """Select appropriate reference node (defaults to deepest leaf)."""
        leaves = [n for n in self.hierarchy.nodes.values() if not n.children]
        return max(leaves, key=lambda n: n.depth)
    
    def validate_single_parent(self, parent: PACNode) -> ConservationResult:
        """
        Validate conservation for a single parent-children relationship.
        
        Args:
            parent: Parent node
        
        Returns:
            ConservationResult with residuals
        """
        if not parent.children:
            raise ValueError(f"Node {parent.id} has no children")
        
        # PAC residual
        pac_residual = parent.pac_residual()
        
        # Distance residual (norm-based)
        dist_residual_norm = parent.distance_residual()
        
        # Distance residual (reference-based)
        dist_residual_ref = None
        if self.reference_node and parent.embedding is not None:
            parent_dist = parent.distance_to(self.reference_node)
            
            children_dist_sum = sum(
                child.distance_to(self.reference_node) * 
                child.ownership_weights.get(parent.id, 1.0)
                for child in parent.children
            )
            
            dist_residual_ref = abs(parent_dist - children_dist_sum)
        
        return ConservationResult(
            parent_id=parent.id,
            pac_residual=pac_residual,
            dist_residual_norm=dist_residual_norm,
            dist_residual_ref=dist_residual_ref,
            num_children=len(parent.children),
            depth=parent.depth
        )
    
    def run(self) -> Dict:
        """
        Run full validation across all parent nodes.
        
        Returns:
            Dictionary with summary statistics and detailed results
        """
        self.results = []
        
        for parent in self.hierarchy.get_all_parents():
            try:
                result = self.validate_single_parent(parent)
                self.results.append(result)
            except Exception as e:
                print(f"Warning: Failed to validate parent {parent.id}: {e}")
        
        return self._compute_summary()
    
    def _compute_summary(self) -> Dict:
        """Compute summary statistics from results."""
        if not self.results:
            return {'error': 'No results collected'}
        
        pac_residuals = [r.pac_residual for r in self.results]
        dist_residuals_norm = [r.dist_residual_norm for r in self.results]
        
        # Correlation analysis
        pearson_r, pearson_p = pearsonr(pac_residuals, dist_residuals_norm)
        spearman_r, spearman_p = spearmanr(pac_residuals, dist_residuals_norm)
        
        # Success criteria
        threshold = 0.1
        success_rate = sum(1 for r in dist_residuals_norm if r < threshold) / len(dist_residuals_norm)
        
        summary = {
            'num_parents': len(self.results),
            'pac_residuals': {
                'mean': np.mean(pac_residuals),
                'std': np.std(pac_residuals),
                'median': np.median(pac_residuals),
                'max': np.max(pac_residuals)
            },
            'dist_residuals_norm': {
                'mean': np.mean(dist_residuals_norm),
                'std': np.std(dist_residuals_norm),
                'median': np.median(dist_residuals_norm),
                'max': np.max(dist_residuals_norm)
            },
            'correlation': {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            },
            'success_rate': success_rate,
            'threshold': threshold,
            'validation': self._interpret_results(success_rate, pearson_r)
        }
        
        # Reference-based if available
        if self.results[0].dist_residual_ref is not None:
            dist_residuals_ref = [r.dist_residual_ref for r in self.results]
            summary['dist_residuals_ref'] = {
                'mean': np.mean(dist_residuals_ref),
                'std': np.std(dist_residuals_ref),
                'median': np.median(dist_residuals_ref),
                'max': np.max(dist_residuals_ref)
            }
        
        return summary
    
    def _interpret_results(self, success_rate: float, correlation: float) -> str:
        """
        Interpret validation results.
        
        Args:
            success_rate: Fraction of nodes with residual < threshold
            correlation: Pearson correlation coefficient
        
        Returns:
            Interpretation string
        """
        if success_rate > 0.9 and correlation > 0.7:
            return "STRONG: PAC conservation strongly validated by distance metrics"
        elif success_rate > 0.7 and correlation > 0.5:
            return "MODERATE: PAC conservation moderately supported"
        elif success_rate > 0.5 or correlation > 0.3:
            return "WEAK: Partial evidence for PAC-distance relationship"
        else:
            return "FAILED: Distance metrics do not support PAC conservation"
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Visualize validation results.
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.results:
            print("No results to plot. Run validation first.")
            return
        
        pac_residuals = [r.pac_residual for r in self.results]
        dist_residuals = [r.dist_residual_norm for r in self.results]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Scatter: PAC vs Distance residuals
        ax = axes[0, 0]
        ax.scatter(pac_residuals, dist_residuals, alpha=0.6, s=50)
        ax.set_xlabel('PAC Residual |f(P) - Σf(C)|')
        ax.set_ylabel('Distance Residual ||e(P)||² - Σ||e(C)||²')
        ax.set_title('Distance Conservation vs PAC Conservation')
        
        # Fit line
        z = np.polyfit(pac_residuals, dist_residuals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(pac_residuals), max(pac_residuals), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Fit: y={z[0]:.2f}x+{z[1]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Histogram: Distance residuals
        ax = axes[0, 1]
        ax.hist(dist_residuals, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0.1, color='r', linestyle='--', label='Threshold (0.1)')
        ax.set_xlabel('Distance Residual')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Distance Residuals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # By depth
        ax = axes[1, 0]
        depths = [r.depth for r in self.results]
        ax.scatter(depths, dist_residuals, alpha=0.6, s=50)
        ax.set_xlabel('Depth')
        ax.set_ylabel('Distance Residual')
        ax.set_title('Distance Residual by Depth')
        ax.grid(True, alpha=0.3)
        
        # By number of children
        ax = axes[1, 1]
        num_children = [r.num_children for r in self.results]
        ax.scatter(num_children, dist_residuals, alpha=0.6, s=50)
        ax.set_xlabel('Number of Children')
        ax.set_ylabel('Distance Residual')
        ax.set_title('Distance Residual by Branching Factor')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()


def run_experiment(
    hierarchy: Optional[PACHierarchy] = None,
    depth: int = 4,
    branching: int = 3,
    dimension: int = 128,
    seed: int = 42
) -> Dict:
    """
    Run Experiment 1: Distance Conservation Validation.
    
    Args:
        hierarchy: Optional pre-built hierarchy (creates synthetic if None)
        depth: Depth for synthetic hierarchy
        branching: Branching factor for synthetic hierarchy
        dimension: Embedding dimension
        seed: Random seed
    
    Returns:
        Validation results dictionary
    """
    print("=" * 80)
    print("EXPERIMENT 1: DISTANCE CONSERVATION VALIDATION")
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
    
    # Run validation
    print("\nRunning distance conservation validation...")
    validator = DistanceConservationValidator(hierarchy)
    results = validator.run()
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nParents analyzed: {results['num_parents']}")
    print(f"\nPAC Residuals:")
    print(f"  Mean: {results['pac_residuals']['mean']:.6f}")
    print(f"  Std:  {results['pac_residuals']['std']:.6f}")
    print(f"  Max:  {results['pac_residuals']['max']:.6f}")
    
    print(f"\nDistance Residuals (norm-based):")
    print(f"  Mean: {results['dist_residuals_norm']['mean']:.6f}")
    print(f"  Std:  {results['dist_residuals_norm']['std']:.6f}")
    print(f"  Max:  {results['dist_residuals_norm']['max']:.6f}")
    
    print(f"\nCorrelation Analysis:")
    print(f"  Pearson r:  {results['correlation']['pearson_r']:.4f} (p={results['correlation']['pearson_p']:.4e})")
    print(f"  Spearman r: {results['correlation']['spearman_r']:.4f} (p={results['correlation']['spearman_p']:.4e})")
    
    print(f"\nSuccess Rate: {results['success_rate']*100:.1f}% (residual < {results['threshold']})")
    print(f"\nValidation: {results['validation']}")
    
    # Plot
    print("\nGenerating visualization...")
    validator.plot_results()
    
    return results


if __name__ == "__main__":
    # Run with synthetic data
    results = run_experiment(depth=4, branching=3, dimension=128, seed=42)

"""
Experiment 9: Parameter Sweep with Real Embeddings

Systematically tests E=mc² relationship across multiple:
- Embedding models (sentence-transformers)
- Hierarchy structures (balanced, unbalanced, fibonacci)
- Tree depths (3-6 levels)
- Value distributions (uniform, power-law, fibonacci)

Goal: Determine if R² is consistent across parameters or model-dependent.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.pac_hierarchy import PACNode, PACHierarchy
from core.embedding_generator import EmbeddingGenerator
import matplotlib.pyplot as plt
from scipy.stats import linregress
import json
import pandas as pd


class ParameterSweep:
    """Sweeps parameters to test E=mc² relationship robustness."""
    
    def __init__(self):
        self.results = []
    
    def create_hierarchy(self, structure: str, depth: int, 
                        value_dist: str) -> PACHierarchy:
        """
        Create hierarchy with specified parameters.
        
        Args:
            structure: 'balanced_2', 'balanced_3', 'balanced_4', 'fibonacci', 'random'
            depth: Tree depth (3-6)
            value_dist: 'uniform', 'power_law', 'fibonacci'
        """
        root = PACNode(id="root", value=100.0)
        hierarchy = PACHierarchy(root)
        
        if structure == 'fibonacci':
            # Fibonacci branching: alternates 2, 3
            self._build_fibonacci_tree(hierarchy, root, depth, value_dist)
        elif structure == 'random':
            # Random branching 2-5
            self._build_random_tree(hierarchy, root, depth, value_dist)
        elif structure.startswith('balanced_'):
            # Balanced tree with fixed branching
            branching = int(structure.split('_')[1])
            self._build_balanced_tree(hierarchy, root, depth, branching, value_dist)
        else:
            raise ValueError(f"Unknown structure: {structure}")
        
        return hierarchy
    
    def _get_child_values(self, parent_value: float, n_children: int, 
                          value_dist: str) -> List[float]:
        """Generate child values based on distribution."""
        if value_dist == 'uniform':
            # Equal split
            return [parent_value / n_children] * n_children
        elif value_dist == 'power_law':
            # Power law: first child gets more
            weights = np.array([1.0 / (i + 1) for i in range(n_children)])
            weights /= weights.sum()
            return (parent_value * weights).tolist()
        elif value_dist == 'fibonacci':
            # Fibonacci ratios
            if n_children <= 2:
                weights = np.array([1.0] * n_children)
            else:
                fib = [1, 1]
                for i in range(n_children - 2):
                    fib.append(fib[-1] + fib[-2])
                weights = np.array(fib[:n_children])
            weights /= weights.sum()
            return (parent_value * weights).tolist()
        else:
            raise ValueError(f"Unknown value_dist: {value_dist}")
    
    def _build_balanced_tree(self, hierarchy: PACHierarchy, root: PACNode,
                            depth: int, branching: int, value_dist: str):
        """Build balanced tree with fixed branching factor."""
        current_level = [root]
        
        for d in range(1, depth):
            next_level = []
            for parent in current_level:
                child_values = self._get_child_values(parent.value, branching, value_dist)
                for i, val in enumerate(child_values):
                    child = PACNode(id=f"L{d}_{parent.id}_{i}", value=val)
                    hierarchy.add_node(child, parent_id=parent.id, 
                                     ownership_weight=val/parent.value)
                    next_level.append(child)
            current_level = next_level
    
    def _build_fibonacci_tree(self, hierarchy: PACHierarchy, root: PACNode,
                             depth: int, value_dist: str):
        """Build tree with Fibonacci branching pattern (2, 3, 2, 3, ...)."""
        current_level = [root]
        
        for d in range(1, depth):
            branching = 2 if d % 2 == 1 else 3
            next_level = []
            for parent in current_level:
                child_values = self._get_child_values(parent.value, branching, value_dist)
                for i, val in enumerate(child_values):
                    child = PACNode(id=f"L{d}_{parent.id}_{i}", value=val)
                    hierarchy.add_node(child, parent_id=parent.id,
                                     ownership_weight=val/parent.value)
                    next_level.append(child)
            current_level = next_level
    
    def _build_random_tree(self, hierarchy: PACHierarchy, root: PACNode,
                          depth: int, value_dist: str):
        """Build tree with random branching 2-5."""
        current_level = [root]
        
        for d in range(1, depth):
            next_level = []
            for parent in current_level:
                branching = np.random.randint(2, 6)  # 2-5 children
                child_values = self._get_child_values(parent.value, branching, value_dist)
                for i, val in enumerate(child_values):
                    child = PACNode(id=f"L{d}_{parent.id}_{i}", value=val)
                    hierarchy.add_node(child, parent_id=parent.id,
                                     ownership_weight=val/parent.value)
                    next_level.append(child)
            current_level = next_level
    
    def compute_metrics(self, hierarchy: PACHierarchy) -> Dict:
        """Compute E=mc² metrics for leaf nodes."""
        leaf_nodes = [n for n in hierarchy.nodes.values() if not n.children]
        
        masses = []
        energies = []
        
        for node in leaf_nodes:
            if node.embedding is not None:
                masses.append(node.value)
                energies.append(np.linalg.norm(node.embedding) ** 2)
        
        if len(masses) < 2:
            return None
        
        masses = np.array(masses)
        energies = np.array(energies)
        
        # Linear fit
        slope, intercept, r, p_value, std_err = linregress(masses, energies)
        
        # Fit c² through origin
        c_squared = np.sum(masses * energies) / np.sum(masses ** 2)
        predicted = c_squared * masses
        residuals = energies - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((energies - energies.mean()) ** 2)
        r_squared_origin = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'n_leaves': len(masses),
            'r_squared': r**2,
            'r_squared_origin': r_squared_origin,
            'r': r,
            'p_value': p_value,
            'slope': slope,
            'c_squared': c_squared,
            'intercept': intercept,
            'mean_mass': np.mean(masses),
            'mean_energy': np.mean(energies),
            'std_mass': np.std(masses),
            'std_energy': np.std(energies)
        }
    
    def run_sweep(self, models: List[str], structures: List[str], 
                  depths: List[int], value_dists: List[str],
                  n_replicates: int = 3) -> List[Dict]:
        """
        Run parameter sweep across all combinations.
        
        Args:
            models: List of sentence-transformers model names
            structures: List of hierarchy structures
            depths: List of tree depths
            value_dists: List of value distributions
            n_replicates: Number of replicates per combination
        """
        import itertools
        
        total_runs = (len(models) * len(structures) * len(depths) * 
                     len(value_dists) * n_replicates)
        
        print(f"Starting parameter sweep: {total_runs} total runs")
        print(f"  Models: {len(models)}")
        print(f"  Structures: {len(structures)}")
        print(f"  Depths: {len(depths)}")
        print(f"  Value dists: {len(value_dists)}")
        print(f"  Replicates: {n_replicates}")
        print()
        
        run_num = 0
        
        for model_name in models:
            print(f"\n{'='*60}")
            print(f"MODEL: {model_name}")
            print(f"{'='*60}")
            
            # Initialize embedding generator for this model
            emb_gen = EmbeddingGenerator(model='sentence-transformers',
                                        model_name=model_name)
            
            for structure, depth, value_dist in itertools.product(
                structures, depths, value_dists):
                
                for replicate in range(n_replicates):
                    run_num += 1
                    
                    print(f"\nRun {run_num}/{total_runs}: "
                          f"{structure}, depth={depth}, {value_dist}, rep={replicate}")
                    
                    try:
                        # Create hierarchy
                        hierarchy = self.create_hierarchy(structure, depth, value_dist)
                        n_nodes = len(hierarchy.nodes)
                        n_leaves = len([n for n in hierarchy.nodes.values() if not n.children])
                        
                        print(f"  Nodes: {n_nodes}, Leaves: {n_leaves}")
                        
                        # Generate embeddings
                        emb_gen.embed_hierarchy(hierarchy)
                        
                        # Compute metrics
                        metrics = self.compute_metrics(hierarchy)
                        
                        if metrics is not None:
                            result = {
                                'run': run_num,
                                'model': model_name,
                                'structure': structure,
                                'depth': depth,
                                'value_dist': value_dist,
                                'replicate': replicate,
                                'n_nodes': n_nodes,
                                **metrics
                            }
                            self.results.append(result)
                            
                            print(f"  R²: {metrics['r_squared']:.4f}, "
                                  f"c²: {metrics['c_squared']:.2f}, "
                                  f"p: {metrics['p_value']:.6f}")
                        else:
                            print(f"  ⚠️  Insufficient data")
                    
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
                        continue
        
        return self.results
    
    def analyze_results(self) -> pd.DataFrame:
        """Analyze sweep results."""
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*60)
        print("PARAMETER SWEEP ANALYSIS")
        print("="*60)
        
        # Overall statistics
        print(f"\nOverall Statistics:")
        print(f"  Total runs: {len(df)}")
        print(f"  Mean R²: {df['r_squared'].mean():.4f} ± {df['r_squared'].std():.4f}")
        print(f"  Mean c²: {df['c_squared'].mean():.2f} ± {df['c_squared'].std():.2f}")
        print(f"  Significant (p<0.05): {(df['p_value'] < 0.05).sum()}/{len(df)}")
        
        # By model
        print(f"\nBy Model:")
        model_stats = df.groupby('model').agg({
            'r_squared': ['mean', 'std', 'min', 'max'],
            'c_squared': ['mean', 'std']
        })
        print(model_stats)
        
        # By structure
        print(f"\nBy Structure:")
        struct_stats = df.groupby('structure').agg({
            'r_squared': ['mean', 'std'],
            'c_squared': ['mean', 'std']
        })
        print(struct_stats)
        
        # By depth
        print(f"\nBy Depth:")
        depth_stats = df.groupby('depth').agg({
            'r_squared': ['mean', 'std'],
            'c_squared': ['mean', 'std']
        })
        print(depth_stats)
        
        return df
    
    def visualize_results(self, df: pd.DataFrame):
        """Create visualizations of parameter sweep."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. R² by model
        ax1 = fig.add_subplot(gs[0, 0])
        models = df['model'].unique()
        r2_by_model = [df[df['model'] == m]['r_squared'].values for m in models]
        ax1.boxplot(r2_by_model, labels=[m.split('/')[-1][:15] for m in models])
        ax1.set_ylabel('R²')
        ax1.set_title('R² by Model')
        ax1.grid(alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. c² by model
        ax2 = fig.add_subplot(gs[0, 1])
        c2_by_model = [df[df['model'] == m]['c_squared'].values for m in models]
        ax2.boxplot(c2_by_model, labels=[m.split('/')[-1][:15] for m in models])
        ax2.set_ylabel('c²')
        ax2.set_title('c² by Model')
        ax2.grid(alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. R² by structure
        ax3 = fig.add_subplot(gs[0, 2])
        structures = df['structure'].unique()
        r2_by_struct = [df[df['structure'] == s]['r_squared'].values for s in structures]
        ax3.boxplot(r2_by_struct, labels=structures)
        ax3.set_ylabel('R²')
        ax3.set_title('R² by Structure')
        ax3.grid(alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. R² by depth
        ax4 = fig.add_subplot(gs[1, 0])
        depths = sorted(df['depth'].unique())
        r2_by_depth = [df[df['depth'] == d]['r_squared'].values for d in depths]
        ax4.boxplot(r2_by_depth, labels=depths)
        ax4.set_xlabel('Depth')
        ax4.set_ylabel('R²')
        ax4.set_title('R² by Depth')
        ax4.grid(alpha=0.3)
        
        # 5. R² by value distribution
        ax5 = fig.add_subplot(gs[1, 1])
        value_dists = df['value_dist'].unique()
        r2_by_vd = [df[df['value_dist'] == vd]['r_squared'].values for vd in value_dists]
        ax5.boxplot(r2_by_vd, labels=value_dists)
        ax5.set_ylabel('R²')
        ax5.set_title('R² by Value Distribution')
        ax5.grid(alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 6. Scatter: R² vs c²
        ax6 = fig.add_subplot(gs[1, 2])
        for model in models:
            model_df = df[df['model'] == model]
            ax6.scatter(model_df['r_squared'], model_df['c_squared'], 
                       alpha=0.6, label=model.split('/')[-1][:15], s=50)
        ax6.set_xlabel('R²')
        ax6.set_ylabel('c²')
        ax6.set_title('R² vs c² Relationship')
        ax6.legend(fontsize=8)
        ax6.grid(alpha=0.3)
        
        # 7. Histogram: R² distribution
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.hist(df['r_squared'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax7.axvline(df['r_squared'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean={df["r_squared"].mean():.3f}')
        ax7.set_xlabel('R²')
        ax7.set_ylabel('Frequency')
        ax7.set_title('R² Distribution (All Runs)')
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        # 8. Histogram: c² distribution
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.hist(df['c_squared'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax8.axvline(df['c_squared'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean={df["c_squared"].mean():.1f}')
        ax8.set_xlabel('c²')
        ax8.set_ylabel('Frequency')
        ax8.set_title('c² Distribution (All Runs)')
        ax8.legend()
        ax8.grid(alpha=0.3)
        
        # 9. Heatmap: R² by model x structure
        ax9 = fig.add_subplot(gs[2, 2])
        pivot = df.pivot_table(values='r_squared', index='model', 
                              columns='structure', aggfunc='mean')
        im = ax9.imshow(pivot.values, cmap='viridis', aspect='auto')
        ax9.set_xticks(range(len(pivot.columns)))
        ax9.set_yticks(range(len(pivot.index)))
        ax9.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax9.set_yticklabels([m.split('/')[-1][:15] for m in pivot.index])
        ax9.set_title('Mean R² Heatmap')
        plt.colorbar(im, ax=ax9)
        
        import os
        os.makedirs('euclidean_distance_validation/results', exist_ok=True)
        plt.savefig('euclidean_distance_validation/results/experiment_09_parameter_sweep.png',
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved to results/experiment_09_parameter_sweep.png")


def main():
    """Run Experiment 9: Parameter Sweep."""
    
    sweep = ParameterSweep()
    
    # Define parameter space
    models = [
        'sentence-transformers/all-MiniLM-L6-v2',
        'sentence-transformers/all-mpnet-base-v2',
        'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    ]
    
    structures = ['balanced_2', 'balanced_3', 'fibonacci']
    depths = [3, 4, 5]
    value_dists = ['uniform', 'power_law']
    
    # Run sweep (reduced replicates for speed)
    results = sweep.run_sweep(
        models=models,
        structures=structures,
        depths=depths,
        value_dists=value_dists,
        n_replicates=2
    )
    
    # Analyze
    df = sweep.analyze_results()
    
    # Visualize
    sweep.visualize_results(df)
    
    # Save results
    import os
    os.makedirs('euclidean_distance_validation/results', exist_ok=True)
    
    df.to_csv('euclidean_distance_validation/results/experiment_09_sweep_results.csv', 
              index=False)
    print("\nResults saved to results/experiment_09_sweep_results.csv")
    
    with open('euclidean_distance_validation/results/experiment_09_sweep_summary.json', 'w') as f:
        summary = {
            'total_runs': len(df),
            'mean_r_squared': float(df['r_squared'].mean()),
            'std_r_squared': float(df['r_squared'].std()),
            'mean_c_squared': float(df['c_squared'].mean()),
            'std_c_squared': float(df['c_squared'].std()),
            'models_tested': models,
            'n_significant': int((df['p_value'] < 0.05).sum())
        }
        json.dump(summary, f, indent=2)
    
    print("\nSummary saved to results/experiment_09_sweep_summary.json")
    
    return df


if __name__ == "__main__":
    df = main()

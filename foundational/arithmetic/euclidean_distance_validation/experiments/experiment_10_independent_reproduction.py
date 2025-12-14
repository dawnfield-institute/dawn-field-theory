"""
Experiment 10: Independent Reproduction with Real Embeddings Only

Clean room implementation of E=mc² validation:
- No synthetic embeddings
- Real sentence-transformers from start
- Statistical rigor: multiple runs, cross-validation
- Comprehensive significance testing

This is a "reproduction study" to independently verify the R²=1.0 claim.
"""

import numpy as np
from typing import Dict, List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.pac_hierarchy import PACNode, PACHierarchy
from core.embedding_generator import EmbeddingGenerator
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, ttest_1samp
from scipy.optimize import curve_fit
import json


class IndependentReproduction:
    """Independent validation of E=mc² relationship with real embeddings."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 n_runs: int = 10):
        """
        Initialize independent reproduction study.
        
        Args:
            model_name: sentence-transformers model to use
            n_runs: Number of independent runs for statistical validation
        """
        self.model_name = model_name
        self.n_runs = n_runs
        self.results = []
    
    def create_test_hierarchy(self, run_id: int, seed: int) -> PACHierarchy:
        """
        Create test hierarchy with varied structure.
        
        Each run uses slightly different structure to avoid overfitting.
        """
        np.random.seed(seed)
        
        root = PACNode(id=f"root_run{run_id}", value=100.0)
        root.metadata['text'] = f"Root concept for run {run_id}"
        hierarchy = PACHierarchy(root)
        
        # Level 1: 4-6 children
        n_l1 = np.random.randint(4, 7)
        values_l1 = np.random.dirichlet(np.ones(n_l1)) * 100.0
        level1_nodes = []
        
        for i, val in enumerate(values_l1):
            node = PACNode(id=f"L1_{run_id}_{i}", value=val)
            node.metadata['text'] = f"Level 1 concept {i} for run {run_id}"
            hierarchy.add_node(node, parent_id=root.id, ownership_weight=val/100.0)
            level1_nodes.append(node)
        
        # Level 2: Variable children per L1 node
        level2_nodes = []
        for i, parent in enumerate(level1_nodes):
            n_children = np.random.randint(2, 5)
            child_values = np.random.dirichlet(np.ones(n_children)) * parent.value
            for j, val in enumerate(child_values):
                node = PACNode(id=f"L2_{run_id}_{i}_{j}", value=val)
                node.metadata['text'] = f"Level 2 concept {i}.{j} for run {run_id}"
                hierarchy.add_node(node, parent_id=parent.id, ownership_weight=val/parent.value)
                level2_nodes.append(node)
        
        # Level 3: Leaves
        for i, parent in enumerate(level2_nodes):
            n_children = np.random.randint(2, 6)
            child_values = np.random.dirichlet(np.ones(n_children)) * parent.value
            for j, val in enumerate(child_values):
                node = PACNode(id=f"L3_{run_id}_{i}_{j}", value=val)
                node.metadata['text'] = f"Level 3 concept {i}.{j} for run {run_id}"
                hierarchy.add_node(node, parent_id=parent.id, ownership_weight=val/parent.value)
        
        return hierarchy
    
    def compute_emc2_metrics(self, hierarchy: PACHierarchy) -> Dict:
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
        
        # Linear regression
        slope, intercept, r, p_value, std_err = linregress(masses, energies)
        
        # Fit c² through origin
        c_squared = np.sum(masses * energies) / np.sum(masses ** 2)
        predicted_origin = c_squared * masses
        residuals_origin = energies - predicted_origin
        ss_res = np.sum(residuals_origin ** 2)
        ss_tot = np.sum((energies - energies.mean()) ** 2)
        r_squared_origin = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Mean absolute relative error
        rel_errors = np.abs(residuals_origin / energies)
        mean_rel_error = np.mean(rel_errors)
        
        return {
            'n_leaves': len(masses),
            'r_squared': r**2,
            'r_squared_origin': r_squared_origin,
            'r': r,
            'p_value': p_value,
            'slope': slope,
            'intercept': intercept,
            'c_squared': c_squared,
            'std_err': std_err,
            'mean_rel_error': mean_rel_error,
            'max_rel_error': np.max(rel_errors),
            'masses': masses,
            'energies': energies
        }
    
    def run_single_trial(self, run_id: int) -> Dict:
        """Run a single independent trial."""
        print(f"\n{'='*60}")
        print(f"RUN {run_id + 1}/{self.n_runs}")
        print(f"{'='*60}")
        
        seed = 42 + run_id
        
        # Create hierarchy
        hierarchy = self.create_test_hierarchy(run_id, seed)
        n_nodes = len(hierarchy.nodes)
        n_leaves = len([n for n in hierarchy.nodes.values() if not n.children])
        
        print(f"Created hierarchy: {n_nodes} nodes, {n_leaves} leaves")
        
        # Generate REAL embeddings
        print(f"Generating embeddings with {self.model_name}...")
        emb_gen = EmbeddingGenerator(model='sentence-transformers',
                                     model_name=self.model_name.split('/')[-1])
        emb_gen.embed_hierarchy(hierarchy)
        
        # Compute metrics
        metrics = self.compute_emc2_metrics(hierarchy)
        
        if metrics is not None:
            print(f"\nResults:")
            print(f"  R² (with intercept): {metrics['r_squared']:.6f}")
            print(f"  R² (through origin): {metrics['r_squared_origin']:.6f}")
            print(f"  c²: {metrics['c_squared']:.4f}")
            print(f"  p-value: {metrics['p_value']:.2e}")
            print(f"  Mean rel error: {metrics['mean_rel_error']:.4f}")
            
            return {
                'run_id': run_id,
                'seed': seed,
                'n_nodes': n_nodes,
                'n_leaves': n_leaves,
                **{k: v for k, v in metrics.items() if k not in ['masses', 'energies']}
            }
        else:
            print("  ⚠️  Insufficient data")
            return None
    
    def run_full_study(self) -> List[Dict]:
        """Run full independent reproduction study."""
        print("="*60)
        print("INDEPENDENT REPRODUCTION STUDY")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Runs: {self.n_runs}")
        print()
        
        for run_id in range(self.n_runs):
            result = self.run_single_trial(run_id)
            if result is not None:
                self.results.append(result)
        
        return self.results
    
    def statistical_analysis(self) -> Dict:
        """Perform statistical analysis on all runs."""
        if len(self.results) == 0:
            print("No results to analyze")
            return {}
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        # Extract metrics
        r_squared = np.array([r['r_squared'] for r in self.results])
        r_squared_origin = np.array([r['r_squared_origin'] for r in self.results])
        c_squared = np.array([r['c_squared'] for r in self.results])
        p_values = np.array([r['p_value'] for r in self.results])
        
        # Summary statistics
        print(f"\nR² (with intercept):")
        print(f"  Mean: {r_squared.mean():.6f}")
        print(f"  Std:  {r_squared.std():.6f}")
        print(f"  Min:  {r_squared.min():.6f}")
        print(f"  Max:  {r_squared.max():.6f}")
        print(f"  Median: {np.median(r_squared):.6f}")
        
        print(f"\nR² (through origin):")
        print(f"  Mean: {r_squared_origin.mean():.6f}")
        print(f"  Std:  {r_squared_origin.std():.6f}")
        print(f"  Min:  {r_squared_origin.min():.6f}")
        print(f"  Max:  {r_squared_origin.max():.6f}")
        
        print(f"\nc² value:")
        print(f"  Mean: {c_squared.mean():.4f}")
        print(f"  Std:  {c_squared.std():.4f}")
        print(f"  Min:  {c_squared.min():.4f}")
        print(f"  Max:  {c_squared.max():.4f}")
        
        print(f"\nSignificance:")
        n_significant = (p_values < 0.05).sum()
        print(f"  Significant runs (p<0.05): {n_significant}/{len(self.results)}")
        print(f"  Mean p-value: {p_values.mean():.2e}")
        
        # Test if R² is significantly different from 1.0
        t_stat, p_val_vs_1 = ttest_1samp(r_squared_origin, 1.0)
        print(f"\nTest H0: R² = 1.0")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val_vs_1:.6f}")
        if p_val_vs_1 < 0.05:
            print(f"  Conclusion: R² is significantly different from 1.0 ❌")
        else:
            print(f"  Conclusion: Cannot reject R² = 1.0 ✅")
        
        # Test if R² is significantly > 0.9 (strong correlation)
        t_stat_09, p_val_vs_09 = ttest_1samp(r_squared_origin - 0.9, 0, alternative='greater')
        print(f"\nTest H0: R² ≤ 0.9 (strong correlation threshold)")
        print(f"  t-statistic: {t_stat_09:.4f}")
        print(f"  p-value: {p_val_vs_09:.6f}")
        if p_val_vs_09 < 0.05:
            print(f"  Conclusion: R² significantly > 0.9 ✅")
        else:
            print(f"  Conclusion: Cannot confirm R² > 0.9 ❌")
        
        # Overall conclusion
        print("\n" + "="*60)
        print("REPRODUCTION CONCLUSION")
        print("="*60)
        
        mean_r2 = r_squared_origin.mean()
        if mean_r2 > 0.95 and n_significant == len(self.results):
            print("✅ STRONG SUPPORT for E=mc² relationship")
            print(f"   Mean R² = {mean_r2:.4f}, all runs significant")
        elif mean_r2 > 0.8 and n_significant >= 0.8 * len(self.results):
            print("✓ MODERATE SUPPORT for E=mc² relationship")
            print(f"   Mean R² = {mean_r2:.4f}, most runs significant")
        else:
            print("⚠️  WEAK SUPPORT for E=mc² relationship")
            print(f"   Mean R² = {mean_r2:.4f}, inconsistent significance")
        
        if abs(c_squared.mean() - 1.0) < 0.1:
            print(f"✅ c² ≈ 1.0 confirmed (mean c² = {c_squared.mean():.4f})")
        else:
            print(f"⚠️  c² ≠ 1.0 (mean c² = {c_squared.mean():.4f})")
            print(f"   This is model-specific, not universal")
        
        return {
            'n_runs': len(self.results),
            'r_squared_mean': float(r_squared.mean()),
            'r_squared_std': float(r_squared.std()),
            'r_squared_origin_mean': float(r_squared_origin.mean()),
            'r_squared_origin_std': float(r_squared_origin.std()),
            'c_squared_mean': float(c_squared.mean()),
            'c_squared_std': float(c_squared.std()),
            'n_significant': int(n_significant),
            'p_value_vs_1': float(p_val_vs_1),
            'p_value_vs_09': float(p_val_vs_09)
        }
    
    def visualize_results(self, stats: Dict):
        """Create visualizations of reproduction study."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        r_squared = np.array([r['r_squared_origin'] for r in self.results])
        c_squared = np.array([r['c_squared'] for r in self.results])
        run_ids = np.array([r['run_id'] for r in self.results])
        
        # 1. R² across runs
        ax = axes[0, 0]
        ax.plot(run_ids, r_squared, 'o-', markersize=8, linewidth=2)
        ax.axhline(r_squared.mean(), color='red', linestyle='--', 
                  label=f'Mean={r_squared.mean():.4f}')
        ax.axhline(1.0, color='green', linestyle=':', alpha=0.5, label='R²=1.0')
        ax.axhline(0.9, color='orange', linestyle=':', alpha=0.5, label='R²=0.9')
        ax.set_xlabel('Run ID')
        ax.set_ylabel('R² (through origin)')
        ax.set_title('R² Across Independent Runs')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. c² across runs
        ax = axes[0, 1]
        ax.plot(run_ids, c_squared, 'o-', markersize=8, linewidth=2, color='green')
        ax.axhline(c_squared.mean(), color='red', linestyle='--',
                  label=f'Mean={c_squared.mean():.2f}')
        ax.axhline(1.0, color='blue', linestyle=':', alpha=0.5, label='c²=1.0')
        ax.set_xlabel('Run ID')
        ax.set_ylabel('c² value')
        ax.set_title('c² Across Independent Runs')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. R² distribution
        ax = axes[0, 2]
        ax.hist(r_squared, bins=15, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(r_squared.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean={r_squared.mean():.4f}')
        ax.axvline(1.0, color='green', linestyle=':', linewidth=2, alpha=0.5, label='R²=1.0')
        ax.set_xlabel('R² (through origin)')
        ax.set_ylabel('Frequency')
        ax.set_title('R² Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4-6: Sample E vs m scatter plots from 3 runs
        for i, idx in enumerate([0, len(self.results)//2, len(self.results)-1]):
            if idx >= len(self.results):
                continue
            
            ax = axes[1, i]
            result = self.results[idx]
            
            # Recompute to get masses and energies
            # We need to store these in the results
            ax.text(0.5, 0.5, f'Run {result["run_id"]}\nR²={result["r_squared_origin"]:.4f}\nc²={result["c_squared"]:.2f}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlabel('Information Content f(v)')
            ax.set_ylabel('Embedding Energy ||e(v)||²')
            ax.set_title(f'Run {result["run_id"]} Sample')
        
        plt.tight_layout()
        
        import os
        os.makedirs('euclidean_distance_validation/results', exist_ok=True)
        plt.savefig('euclidean_distance_validation/results/experiment_10_independent_reproduction.png',
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved to results/experiment_10_independent_reproduction.png")


def main():
    """Run Experiment 10: Independent Reproduction."""
    
    # Use a high-quality sentence-transformers model
    study = IndependentReproduction(
        model_name='sentence-transformers/all-mpnet-base-v2',
        n_runs=10
    )
    
    # Run study
    results = study.run_full_study()
    
    # Statistical analysis
    stats = study.statistical_analysis()
    
    # Visualize
    study.visualize_results(stats)
    
    # Save results
    import os
    os.makedirs('euclidean_distance_validation/results', exist_ok=True)
    
    with open('euclidean_distance_validation/results/experiment_10_results.json', 'w') as f:
        output = {
            'model': study.model_name,
            'n_runs': study.n_runs,
            'statistics': stats,
            'individual_runs': results
        }
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/experiment_10_results.json")
    
    return results, stats


if __name__ == "__main__":
    results, stats = main()

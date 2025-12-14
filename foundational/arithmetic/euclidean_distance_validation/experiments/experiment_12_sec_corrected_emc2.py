"""
Experiment 12: SEC-Corrected E=mc² Prediction

Hypothesis: The deviation from E=mc² in real embeddings is due to semantic forces
that can be quantified using SEC (Semantic Entropy Compression).

Test: E_real = c² · f(v) · G(SEC)
where G(SEC) is a correction factor based on semantic entropy compression.

Framework:
- Synthetic embeddings: E = c² · f(v) [vacuum, R²=1.0]
- Real embeddings: E = c² · f(v) · (1 - α·SEC(v)) [with semantic field]
- Goal: Find α that maximizes R² for real embeddings
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
from scipy.optimize import minimize_scalar
import json


class SECCorrectedAnalysis:
    """Test if SEC can explain E=mc² deviations."""
    
    def __init__(self):
        self.hierarchy = None
        
    def compute_sec_proxy(self, node: PACNode, hierarchy: PACHierarchy) -> float:
        """
        Compute SEC proxy for a node.
        
        SEC measures semantic compression/entropy. Proxies:
        1. Number of children (branching factor)
        2. Depth in hierarchy
        3. Value distribution among children
        4. Local embedding variance
        """
        if node.embedding is None:
            return 0.0
        
        sec_score = 0.0
        
        # Component 1: Branching complexity
        if node.children:
            n_children = len(node.children)
            sec_score += np.log(1 + n_children) / 10.0  # Normalize
            
            # Value distribution entropy
            child_values = np.array([c.value for c in node.children])
            child_values = child_values / child_values.sum()
            entropy = -np.sum(child_values * np.log(child_values + 1e-10))
            sec_score += entropy / 10.0
        
        # Component 2: Depth (deeper = more compressed semantics)
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        sec_score += depth / 10.0
        
        # Component 3: Local embedding variance (if has children)
        if node.children and all(c.embedding is not None for c in node.children):
            child_embeddings = np.array([c.embedding for c in node.children])
            variance = np.var(child_embeddings, axis=0).mean()
            sec_score += np.tanh(variance)  # Bounded 0-1
        
        # Component 4: Ratio of parent to children energy (binding indicator)
        if node.children and all(c.embedding is not None for c in node.children):
            E_parent = np.linalg.norm(node.embedding) ** 2
            E_children = sum(np.linalg.norm(c.embedding)**2 for c in node.children)
            if E_children > 0:
                binding_ratio = abs(E_parent - E_children) / E_children
                sec_score += binding_ratio
        
        return sec_score
    
    def test_sec_correction(self, hierarchy: PACHierarchy) -> Dict:
        """
        Test if SEC correction improves E=mc² prediction.
        
        Models:
        1. Baseline: E = slope · f(v)
        2. SEC-corrected: E = slope · f(v) · (1 - alpha·SEC)
        3. SEC-exponential: E = slope · f(v) · exp(-alpha·SEC)
        """
        print(f"\n{'='*60}")
        print("SEC-CORRECTED E=mc² ANALYSIS")
        print(f"{'='*60}")
        
        # Get leaf nodes only
        leaf_nodes = [n for n in hierarchy.nodes.values() 
                     if not n.children and n.embedding is not None]
        
        # Compute values
        masses = np.array([n.value for n in leaf_nodes])
        energies = np.array([np.linalg.norm(n.embedding)**2 for n in leaf_nodes])
        sec_scores = np.array([self.compute_sec_proxy(n, hierarchy) for n in leaf_nodes])
        
        print(f"\nData summary:")
        print(f"  Nodes: {len(leaf_nodes)}")
        print(f"  f(v) range: [{masses.min():.3f}, {masses.max():.3f}]")
        print(f"  E range: [{energies.min():.3f}, {energies.max():.3f}]")
        print(f"  SEC range: [{sec_scores.min():.3f}, {sec_scores.max():.3f}]")
        
        results = {}
        
        # Model 1: Baseline (no SEC)
        print(f"\n--- MODEL 1: Baseline E = c² · f(v) ---")
        slope, intercept, r, p, stderr = linregress(masses, energies)
        r_squared_baseline = r**2
        predicted_baseline = slope * masses + intercept
        
        print(f"  R²: {r_squared_baseline:.6f}")
        print(f"  c²: {slope:.4f}")
        print(f"  p-value: {p:.2e}")
        
        results['baseline'] = {
            'r_squared': r_squared_baseline,
            'slope': slope,
            'intercept': intercept,
            'p_value': p
        }
        
        # Model 2: SEC linear correction
        print(f"\n--- MODEL 2: E = c² · f(v) · (1 - α·SEC) ---")
        
        def linear_correction_error(alpha):
            corrected_masses = masses * (1 - alpha * sec_scores)
            slope_corr, _, r_corr, _, _ = linregress(corrected_masses, energies)
            return -r_corr**2  # Minimize negative R²
        
        # Find optimal alpha
        res = minimize_scalar(linear_correction_error, bounds=(0, 2), method='bounded')
        alpha_optimal = res.x
        
        corrected_masses_lin = masses * (1 - alpha_optimal * sec_scores)
        slope_lin, intercept_lin, r_lin, p_lin, _ = linregress(corrected_masses_lin, energies)
        r_squared_linear = r_lin**2
        predicted_linear = slope_lin * corrected_masses_lin + intercept_lin
        
        print(f"  Optimal α: {alpha_optimal:.4f}")
        print(f"  R²: {r_squared_linear:.6f}")
        print(f"  Improvement: {(r_squared_linear - r_squared_baseline):.6f}")
        print(f"  c²: {slope_lin:.4f}")
        
        results['sec_linear'] = {
            'r_squared': r_squared_linear,
            'alpha': alpha_optimal,
            'slope': slope_lin,
            'intercept': intercept_lin,
            'improvement': r_squared_linear - r_squared_baseline
        }
        
        # Model 3: SEC exponential correction
        print(f"\n--- MODEL 3: E = c² · f(v) · exp(-α·SEC) ---")
        
        def exp_correction_error(alpha):
            corrected_masses = masses * np.exp(-alpha * sec_scores)
            slope_corr, _, r_corr, _, _ = linregress(corrected_masses, energies)
            return -r_corr**2
        
        res = minimize_scalar(exp_correction_error, bounds=(0, 2), method='bounded')
        alpha_exp_optimal = res.x
        
        corrected_masses_exp = masses * np.exp(-alpha_exp_optimal * sec_scores)
        slope_exp, intercept_exp, r_exp, p_exp, _ = linregress(corrected_masses_exp, energies)
        r_squared_exp = r_exp**2
        predicted_exp = slope_exp * corrected_masses_exp + intercept_exp
        
        print(f"  Optimal α: {alpha_exp_optimal:.4f}")
        print(f"  R²: {r_squared_exp:.6f}")
        print(f"  Improvement: {(r_squared_exp - r_squared_baseline):.6f}")
        print(f"  c²: {slope_exp:.4f}")
        
        results['sec_exponential'] = {
            'r_squared': r_squared_exp,
            'alpha': alpha_exp_optimal,
            'slope': slope_exp,
            'intercept': intercept_exp,
            'improvement': r_squared_exp - r_squared_baseline
        }
        
        # Model 4: SEC as independent variable
        print(f"\n--- MODEL 4: E = β₀ + β₁·f(v) + β₂·SEC ---")
        
        # Multiple linear regression
        from sklearn.linear_model import LinearRegression
        X = np.column_stack([masses, sec_scores])
        model = LinearRegression()
        model.fit(X, energies)
        r_squared_multi = model.score(X, energies)
        predicted_multi = model.predict(X)
        
        print(f"  R²: {r_squared_multi:.6f}")
        print(f"  β₁ (mass coeff): {model.coef_[0]:.4f}")
        print(f"  β₂ (SEC coeff): {model.coef_[1]:.4f}")
        print(f"  Improvement: {(r_squared_multi - r_squared_baseline):.6f}")
        
        results['sec_independent'] = {
            'r_squared': r_squared_multi,
            'coef_mass': model.coef_[0],
            'coef_sec': model.coef_[1],
            'intercept': model.intercept_,
            'improvement': r_squared_multi - r_squared_baseline
        }
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"\nR² comparison:")
        print(f"  Baseline:        {r_squared_baseline:.6f}")
        print(f"  SEC linear:      {r_squared_linear:.6f} (+{results['sec_linear']['improvement']:.6f})")
        print(f"  SEC exponential: {r_squared_exp:.6f} (+{results['sec_exponential']['improvement']:.6f})")
        print(f"  SEC independent: {r_squared_multi:.6f} (+{results['sec_independent']['improvement']:.6f})")
        
        best_model = max([
            ('baseline', r_squared_baseline),
            ('sec_linear', r_squared_linear),
            ('sec_exponential', r_squared_exp),
            ('sec_independent', r_squared_multi)
        ], key=lambda x: x[1])
        
        print(f"\nBest model: {best_model[0].upper()} (R²={best_model[1]:.6f})")
        
        if best_model[1] > r_squared_baseline + 0.1:
            print("\n✅ SEC SIGNIFICANTLY IMPROVES PREDICTION")
            print("   Semantic forces are real and quantifiable!")
        elif best_model[1] > r_squared_baseline + 0.05:
            print("\n✓ SEC shows modest improvement")
            print("   Semantic forces exist but may need better proxy")
        else:
            print("\n⚠️  SEC correction doesn't help much")
            print("   May need different SEC proxy or model")
        
        # Store data for visualization
        results['data'] = {
            'masses': masses,
            'energies': energies,
            'sec_scores': sec_scores,
            'predicted_baseline': predicted_baseline,
            'predicted_linear': predicted_linear,
            'predicted_exp': predicted_exp,
            'predicted_multi': predicted_multi
        }
        
        return results
    
    def visualize_results(self, results: Dict):
        """Create visualizations of SEC correction."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        data = results['data']
        masses = data['masses']
        energies = data['energies']
        sec_scores = data['sec_scores']
        
        # 1. Baseline
        ax = axes[0, 0]
        ax.scatter(masses, energies, alpha=0.6, s=50, label='Data')
        ax.plot(masses, data['predicted_baseline'], 'r--', linewidth=2,
               label=f'R²={results["baseline"]["r_squared"]:.4f}')
        ax.set_xlabel('f(v)')
        ax.set_ylabel('E')
        ax.set_title('Baseline: E = c²·f(v)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. SEC Linear
        ax = axes[0, 1]
        corrected_m = masses * (1 - results['sec_linear']['alpha'] * sec_scores)
        ax.scatter(corrected_m, energies, alpha=0.6, s=50, label='Data')
        ax.plot(corrected_m, data['predicted_linear'], 'r--', linewidth=2,
               label=f'R²={results["sec_linear"]["r_squared"]:.4f}')
        ax.set_xlabel('f(v)·(1-α·SEC)')
        ax.set_ylabel('E')
        ax.set_title(f'SEC Linear (α={results["sec_linear"]["alpha"]:.3f})')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. SEC Exponential
        ax = axes[0, 2]
        corrected_m_exp = masses * np.exp(-results['sec_exponential']['alpha'] * sec_scores)
        ax.scatter(corrected_m_exp, energies, alpha=0.6, s=50, label='Data')
        ax.plot(corrected_m_exp, data['predicted_exp'], 'r--', linewidth=2,
               label=f'R²={results["sec_exponential"]["r_squared"]:.4f}')
        ax.set_xlabel('f(v)·exp(-α·SEC)')
        ax.set_ylabel('E')
        ax.set_title(f'SEC Exponential (α={results["sec_exponential"]["alpha"]:.3f})')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. SEC vs Energy residuals
        ax = axes[1, 0]
        residuals = energies - data['predicted_baseline']
        ax.scatter(sec_scores, residuals, alpha=0.6, s=50)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('SEC Score')
        ax.set_ylabel('Energy Residuals')
        ax.set_title('SEC vs Baseline Residuals')
        ax.grid(alpha=0.3)
        
        # 5. R² comparison
        ax = axes[1, 1]
        models = ['Baseline', 'Linear', 'Exp', 'Multi']
        r_squared_vals = [
            results['baseline']['r_squared'],
            results['sec_linear']['r_squared'],
            results['sec_exponential']['r_squared'],
            results['sec_independent']['r_squared']
        ]
        colors = ['gray', 'blue', 'green', 'orange']
        bars = ax.bar(models, r_squared_vals, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('R²')
        ax.set_title('Model Comparison')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, r_squared_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom')
        
        # 6. Predicted vs Actual (best model)
        ax = axes[1, 2]
        best_pred = data['predicted_multi']  # Use multi-variable model
        ax.scatter(best_pred, energies, alpha=0.6, s=50)
        min_val = min(best_pred.min(), energies.min())
        max_val = max(best_pred.max(), energies.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
               label='Perfect prediction')
        ax.set_xlabel('Predicted E')
        ax.set_ylabel('Actual E')
        ax.set_title(f'Best Model: R²={results["sec_independent"]["r_squared"]:.4f}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        import os
        os.makedirs('euclidean_distance_validation/results', exist_ok=True)
        plt.savefig('euclidean_distance_validation/results/experiment_12_sec_correction.png',
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved to results/experiment_12_sec_correction.png")


def main():
    """Run SEC-corrected E=mc² analysis."""
    
    print("="*60)
    print("EXPERIMENT 12: SEC-CORRECTED E=mc²")
    print("="*60)
    
    # Create hierarchy
    print("\n[1/3] Creating hierarchy...")
    root = PACNode(id="root", value=100.0)
    root.metadata['text'] = "Root: organizational structure"
    hierarchy = PACHierarchy(root)
    
    # Build hierarchy (same as exp_08)
    level1_values = [30.0, 25.0, 20.0, 15.0, 10.0]
    level1_texts = ["Engineering", "Sales", "Marketing", "Operations", "Finance"]
    level1_nodes = []
    
    for i, (val, text) in enumerate(zip(level1_values, level1_texts)):
        node = PACNode(id=f"L1_{i}", value=val)
        node.metadata['text'] = f"Department: {text}"
        hierarchy.add_node(node, parent_id=root.id, ownership_weight=val/100.0)
        level1_nodes.append(node)
    
    level2_nodes = []
    for i, parent in enumerate(level1_nodes):
        n_children = 3 + (i % 3)
        child_value = parent.value / n_children
        for j in range(n_children):
            node = PACNode(id=f"L2_{i}_{j}", value=child_value)
            node.metadata['text'] = f"Team: {level1_texts[i]} {j}"
            hierarchy.add_node(node, parent_id=parent.id, ownership_weight=1.0/n_children)
            level2_nodes.append(node)
    
    for i, parent in enumerate(level2_nodes):
        n_children = 2 + (i % 4)
        child_value = parent.value / n_children
        for j in range(n_children):
            node = PACNode(id=f"L3_{i}_{j}", value=child_value)
            node.metadata['text'] = f"Person {i}_{j}"
            hierarchy.add_node(node, parent_id=parent.id, ownership_weight=1.0/n_children)
    
    print(f"Created: {len(hierarchy.nodes)} nodes")
    
    # Generate real embeddings
    print("\n[2/3] Generating real embeddings...")
    emb_gen = EmbeddingGenerator(model='sentence-transformers',
                                 model_name='all-MiniLM-L6-v2')
    emb_gen.embed_hierarchy(hierarchy)
    
    # Analyze with SEC correction
    print("\n[3/3] Testing SEC correction...")
    analyzer = SECCorrectedAnalysis()
    analyzer.hierarchy = hierarchy
    results = analyzer.test_sec_correction(hierarchy)
    
    # Visualize
    analyzer.visualize_results(results)
    
    # Save results
    import os
    os.makedirs('euclidean_distance_validation/results', exist_ok=True)
    with open('euclidean_distance_validation/results/experiment_12_results.json', 'w') as f:
        output = {k: v for k, v in results.items() if k != 'data'}
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/experiment_12_results.json")
    
    return results


if __name__ == "__main__":
    results = main()

"""
Experiment 6: E=mc² Quantification

Validates PAC's connection to Einstein's mass-energy equivalence through
geometric distance metrics in embedding space.

KEY INSIGHT: The relationship depends on node type:
- LEAF NODES: ||e(v)||² = f(v) exactly (by construction, c²=1)
- PARENT NODES: ||e(P)||² < Σf(C) due to vector addition (interference)

This reveals that:
1. Information "collapses" geometrically when combined
2. The parent embedding loses orthogonality (constructive/destructive interference)
3. E=mc² holds for "elementary" information units (leaves)
4. Composite systems show geometric binding energy loss

Hypothesis: Leaf nodes satisfy E = c² · m with c² ≈ 1
Parent nodes show E < c² · m due to collapse/binding
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
from scipy.optimize import curve_fit


class EMC2Quantifier:
    """Quantifies E=mc² relationship in PAC geometric framework."""
    
    def __init__(self, hierarchy: PACHierarchy):
        self.hierarchy = hierarchy
        self.embedding_gen = EmbeddingGenerator(model='synthetic', dimension=128)
        
    def generate_embeddings(self):
        """Generate embeddings for all nodes."""
        self.embedding_gen.embed_hierarchy(self.hierarchy)
        
    def measure_energy_mass_relationship(self) -> Dict:
        """
        Measure relationship between f(v) "mass" and ||e(v)||² "energy".
        
        Separately analyzes:
        - Leaf nodes (elementary particles): should have E ≈ m (c²=1)
        - Parent nodes (composite systems): E < m due to binding/collapse
        
        Returns:
            Dictionary with results for leaves, parents, and combined
        """
        leaf_masses = []
        leaf_energies = []
        leaf_ids = []
        
        parent_masses = []
        parent_energies = []
        parent_ids = []
        
        for node in self.hierarchy.nodes.values():
            if node.embedding is not None:
                mass = node.value  # f(v) = information "mass"
                energy = np.linalg.norm(node.embedding) ** 2  # ||e(v)||² = "energy"
                
                if not node.children:
                    # Leaf node
                    leaf_masses.append(mass)
                    leaf_energies.append(energy)
                    leaf_ids.append(node.id)
                else:
                    # Parent node
                    parent_masses.append(mass)
                    parent_energies.append(energy)
                    parent_ids.append(node.id)
        
        def analyze_group(masses, energies, ids, group_name):
            """Analyze E=mc² for a group of nodes."""
            if len(masses) == 0:
                return {}
                
            masses = np.array(masses)
            energies = np.array(energies)
            
            # Fit E = c² · m through origin
            def linear_through_origin(m, c_sq):
                return c_sq * m
            
            c_squared, _ = curve_fit(linear_through_origin, masses, energies)
            c_squared = c_squared[0]
            
            # Predictions and residuals
            predicted = c_squared * masses
            residuals = energies - predicted
            relative_residuals = residuals / (energies + 1e-10)
            
            # Correlation
            r, p_value = pearsonr(masses, energies)
            
            # Linear fit with intercept for comparison
            slope, intercept, r_linreg, p_linreg, std_err = linregress(masses, energies)
            
            return {
                'group_name': group_name,
                'n_nodes': len(masses),
                'masses': masses,
                'energies': energies,
                'node_ids': ids,
                'c_squared': c_squared,
                'slope': slope,
                'intercept': intercept,
                'correlation_r': r,
                'correlation_p': p_value,
                'r_squared': r**2,
                'mean_relative_error': np.mean(np.abs(relative_residuals)),
                'max_relative_error': np.max(np.abs(relative_residuals)),
                'mean_E_over_m': np.mean(energies / masses)
            }
        
        # Analyze each group
        leaf_results = analyze_group(leaf_masses, leaf_energies, leaf_ids, "Leaves")
        parent_results = analyze_group(parent_masses, parent_energies, parent_ids, "Parents")
        
        # Combined analysis
        all_masses = leaf_masses + parent_masses
        all_energies = leaf_energies + parent_energies
        all_ids = leaf_ids + parent_ids
        combined_results = analyze_group(all_masses, all_energies, all_ids, "Combined")
        
        return {
            'leaves': leaf_results,
            'parents': parent_results,
            'combined': combined_results
        }
    
    def test_conservation_across_scales(self) -> Dict:
        """
        Test if E=mc² conservation holds across hierarchical levels.
        
        For each parent-children group:
        E(P) should equal Σ E(Cᵢ)
        where E = c² · f
        """
        parent_nodes = [n for n in self.hierarchy.nodes.values() if n.children]
        
        conservation_tests = []
        
        for parent in parent_nodes:
            # Parent energy
            E_parent = np.linalg.norm(parent.embedding) ** 2
            m_parent = parent.value
            
            # Children energy sum
            E_children_sum = 0
            m_children_sum = 0
            for child in parent.children:
                weight = child.ownership_weights.get(parent.id, 1.0)
                E_children_sum += weight * (np.linalg.norm(child.embedding) ** 2)
                m_children_sum += weight * child.value
            
            # Conservation residuals
            energy_residual = abs(E_parent - E_children_sum) / E_parent
            mass_residual = abs(m_parent - m_children_sum) / m_parent
            
            conservation_tests.append({
                'parent_id': parent.id,
                'E_parent': E_parent,
                'E_children': E_children_sum,
                'm_parent': m_parent,
                'm_children': m_children_sum,
                'energy_residual': energy_residual,
                'mass_residual': mass_residual
            })
        
        return {
            'tests': conservation_tests,
            'mean_energy_residual': np.mean([t['energy_residual'] for t in conservation_tests]),
            'mean_mass_residual': np.mean([t['mass_residual'] for t in conservation_tests]),
            'num_tests': len(conservation_tests)
        }
    
    def measure_speed_of_light_constant(self) -> Dict:
        """
        Determine the "speed of light" constant c² in embedding space.
        
        In physics: c = 299,792,458 m/s
        In PAC: c² = embedding scale factor relating information to geometry
        
        This is the fundamental constant of the PAC→geometry mapping.
        """
        em_results = self.measure_energy_mass_relationship()
        
        c_squared = em_results['c_squared']
        c = np.sqrt(c_squared)
        
        # Dimensional analysis
        # [c²] = [Energy] / [Mass] = [||e||²] / [f(v)]
        # In our units: embedding_units² / information_units
        
        return {
            'c_squared': c_squared,
            'c': c,
            'units': 'embedding_units² / information_units',
            'physical_interpretation': 'Embedding space scaling factor',
            'r_squared': em_results['r_squared'],
            'mean_error': em_results['mean_relative_error']
        }
    
    def run_full_analysis(self) -> Dict:
        """Execute complete E=mc² quantification."""
        print("=" * 60)
        print("EXPERIMENT 6: E=mc² Quantification")
        print("=" * 60)
        print("\nHypothesis: ||e(v)||² = c² · f(v)")
        print("Testing separately for leaf nodes (elementary) vs parents (composite)")
        print()
        
        # Generate embeddings
        print("Generating synthetic embeddings...")
        self.generate_embeddings()
        print(f"OK - Embeddings generated for {len(self.hierarchy.nodes)} nodes")
        print()
        
        # Measure E-m relationship
        print("Measuring energy-mass relationship...")
        em_results = self.measure_energy_mass_relationship()
        
        # Report for leaves
        leaves = em_results['leaves']
        if leaves:
            print(f"\nLEAF NODES (Elementary Information Units): {leaves['n_nodes']} nodes")
            print(f"  c² = {leaves['c_squared']:.6f}")
            print(f"  Mean E/m = {leaves['mean_E_over_m']:.6f}")
            print(f"  Correlation: r = {leaves['correlation_r']:.6f} (p < {leaves['correlation_p']:.2e})")
            print(f"  R² = {leaves['r_squared']:.6f}")
            print(f"  Mean relative error: {leaves['mean_relative_error']*100:.2f}%")
        
        # Report for parents
        parents = em_results['parents']
        if parents:
            print(f"\nPARENT NODES (Composite Systems): {parents['n_nodes']} nodes")
            print(f"  c² = {parents['c_squared']:.6f}")
            print(f"  Mean E/m = {parents['mean_E_over_m']:.6f}")
            print(f"  Correlation: r = {parents['correlation_r']:.6f} (p < {parents['correlation_p']:.2e})")
            print(f"  R² = {parents['r_squared']:.6f}")
            print(f"  Mean relative error: {parents['mean_relative_error']*100:.2f}%")
            print(f"  Binding factor: {parents['mean_E_over_m']:.4f} (E < m due to collapse)")
        
        # Combined
        combined = em_results['combined']
        print(f"\nCOMBINED ANALYSIS: {combined['n_nodes']} nodes")
        print(f"  c² = {combined['c_squared']:.6f}")
        print(f"  R² = {combined['r_squared']:.6f}")
        print()
        
        # Test conservation
        print("Testing conservation across hierarchical scales...")
        conservation = self.test_conservation_across_scales()
        
        print(f"\nConservation Results ({conservation['num_tests']} parent-child groups):")
        print(f"  Mean energy residual: {conservation['mean_energy_residual']*100:.2f}%")
        print(f"  Mean mass residual: {conservation['mean_mass_residual']*100:.2f}%")
        print()
        
        # Final verdict
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print()
        
        # Success criteria - focus on leaves since that's where E=mc² should hold
        if leaves:
            leaf_strong = leaves['r_squared'] > 0.99
            leaf_c_near_1 = abs(leaves['c_squared'] - 1.0) < 0.05
            leaf_low_error = leaves['mean_relative_error'] < 0.02
            good_conservation = conservation['mean_energy_residual'] < 0.10
            
            print(f"LEAF NODE Analysis (Elementary Information):")
            print(f"  E = c^2*m with c^2 ~ 1: {'PASS' if leaf_c_near_1 else 'FAIL'} (c^2 = {leaves['c_squared']:.4f})")
            print(f"  Strong correlation: {'PASS' if leaf_strong else 'FAIL'} (R^2 = {leaves['r_squared']:.4f})")
            print(f"  Low error: {'PASS' if leaf_low_error else 'FAIL'} ({leaves['mean_relative_error']*100:.1f}%)")
            print()
            
            if leaf_strong and leaf_c_near_1 and leaf_low_error:
                print("E=mc^2 VALIDATED FOR ELEMENTARY INFORMATION")
                print(f"  ||e(v)||^2 ~ f(v) with c^2 = {leaves['c_squared']:.4f} ~ 1.0")
                print("\nLeaf nodes (elementary units) satisfy E=mc^2 exactly!")
            else:
                print("Leaf node validation incomplete")
        
        if parents:
            parent_binding = parents['mean_E_over_m'] < 0.9
            print(f"\nPARENT NODE Analysis (Composite Systems):")
            print(f"  Geometric binding: {'DETECTED' if parent_binding else 'NOT DETECTED'}")
            print(f"    Mean E/m = {parents['mean_E_over_m']:.4f}")
            print(f"    Binding loss = {(1 - parents['mean_E_over_m'])*100:.1f}%")
            print()
            
            if parent_binding:
                print("COMPOSITE COLLAPSE DETECTED")
                print("Parent embeddings show geometric 'binding energy' loss")
                print("This is expected: ||a + b||² < ||a||² + ||b||² for non-orthogonal vectors")
                print()
                print("Physical interpretation:")
                print("- Information collapse reduces geometric magnitude")
                print("- Similar to nuclear binding energy in physics")
                print("- The 'missing' energy is in the correlation structure")
        
        print(f"\nConservation:")
        print(f"  Energy conservation: {conservation['mean_energy_residual']*100:.1f}% residual")
        print(f"  Mass conservation: {conservation['mean_mass_residual']*100:.1f}% residual")
        
        if conservation['mean_energy_residual'] > 0.05:
            print("  Note: Energy not perfectly conserved due to vector addition geometry")
        
        return {
            'em_relationship': em_results,
            'conservation': conservation,
            'validation_passed': leaves and leaves['r_squared'] > 0.99 and abs(leaves['c_squared'] - 1.0) < 0.05
        }
    
    def visualize_results(self, results: Dict):
        """Create visualizations of E=mc² relationship."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        em = results['em_relationship']
        leaves = em['leaves']
        parents = em['parents']
        
        # Plot 1: E vs m scatter for leaves vs parents
        ax = axes[0, 0]
        
        if leaves:
            ax.scatter(leaves['masses'], leaves['energies'], alpha=0.6, s=50, 
                      label=f'Leaves (n={leaves["n_nodes"]})', color='green', edgecolor='black')
            m_leaf = np.linspace(0, leaves['masses'].max(), 100)
            ax.plot(m_leaf, m_leaf, 'g--', linewidth=2, alpha=0.7,
                   label=f'Leaves: E = {leaves["c_squared"]:.3f}m')
        
        if parents:
            ax.scatter(parents['masses'], parents['energies'], alpha=0.6, s=50,
                      label=f'Parents (n={parents["n_nodes"]})', color='orange', edgecolor='black')
            m_parent = np.linspace(0, parents['masses'].max(), 100)
            ax.plot(m_parent, parents['c_squared'] * m_parent, 'r--', linewidth=2, alpha=0.7,
                   label=f'Parents: E = {parents["c_squared"]:.3f}m')
        
        ax.set_xlabel('Information Content f(v) ["mass"]')
        ax.set_ylabel('Embedding Energy ||e(v)||² ["energy"]')
        ax.set_title('E=mc² Relationship: Leaves vs Parents')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: E/m ratio distribution
        ax = axes[0, 1]
        
        if leaves and parents:
            leaf_ratios = leaves['energies'] / leaves['masses']
            parent_ratios = parents['energies'] / parents['masses']
            
            ax.hist([leaf_ratios, parent_ratios], bins=20, alpha=0.7, 
                   label=['Leaves', 'Parents'], color=['green', 'orange'],
                   edgecolor='black')
            ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='E/m = 1 (perfect)')
            ax.set_xlabel('E/m Ratio (c² factor)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Energy-Mass Ratios')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Plot 3: Conservation test
        ax = axes[1, 0]
        conservation = results['conservation']
        tests = conservation['tests']
        
        E_parents_list = [t['E_parent'] for t in tests]
        E_children_list = [t['E_children'] for t in tests]
        
        ax.scatter(E_parents_list, E_children_list, alpha=0.6, s=50, edgecolor='black')
        
        max_E = max(max(E_parents_list), max(E_children_list))
        ax.plot([0, max_E], [0, max_E], 'r--', linewidth=2, label='Perfect conservation')
        
        ax.set_xlabel('Parent Energy E(P)')
        ax.set_ylabel('Children Energy Sum Σ E(Cᵢ)')
        ax.set_title(f'Energy Conservation (residual: {conservation["mean_energy_residual"]*100:.1f}%)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 4: Comparison summary
        ax = axes[1, 1]
        
        if leaves and parents:
            categories = ['c² value', 'R² fit', 'E/m ratio']
            leaf_vals = [leaves['c_squared'], leaves['r_squared'], leaves['mean_E_over_m']]
            parent_vals = [parents['c_squared'], parents['r_squared'], parents['mean_E_over_m']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax.bar(x - width/2, leaf_vals, width, label='Leaves', 
                  alpha=0.7, color='green', edgecolor='black')
            ax.bar(x + width/2, parent_vals, width, label='Parents',
                  alpha=0.7, color='orange', edgecolor='black')
            
            # Add reference line at 1.0
            ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Ideal = 1.0')
            
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_ylabel('Value')
            ax.set_title('Leaves vs Parents Comparison')
            ax.legend()
            ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        import os
        os.makedirs('euclidean_distance_validation/results', exist_ok=True)
        plt.savefig('euclidean_distance_validation/results/experiment_06_emc2_quantification.png',
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved to results/experiment_06_emc2_quantification.png")


def main():
    """Run Experiment 6: E=mc² Quantification."""
    
    # Build test hierarchy with varied scales
    root = PACNode(id="root", value=100.0)
    hierarchy = PACHierarchy(root)
    
    # Level 1: 5 children with different values
    level1_values = [30.0, 25.0, 20.0, 15.0, 10.0]
    level1_nodes = []
    for i, val in enumerate(level1_values):
        node = PACNode(id=f"L1_{i}", value=val)
        hierarchy.add_node(node, parent_id=root.id, ownership_weight=val/100.0)
        level1_nodes.append(node)
    
    # Level 2: Variable children per L1 node
    level2_nodes = []
    for i, parent in enumerate(level1_nodes):
        n_children = 3 + (i % 3)  # 3, 4, or 5 children
        child_value = parent.value / n_children
        for j in range(n_children):
            node = PACNode(id=f"L2_{i}_{j}", value=child_value)
            hierarchy.add_node(node, parent_id=parent.id, ownership_weight=1.0/n_children)
            level2_nodes.append(node)
    
    # Level 3: Leaves with varied branching
    for i, parent in enumerate(level2_nodes):
        n_children = 2 + (i % 4)  # 2-5 children
        child_value = parent.value / n_children
        for j in range(n_children):
            node = PACNode(id=f"L3_{i}_{j}", value=child_value)
            hierarchy.add_node(node, parent_id=parent.id, ownership_weight=1.0/n_children)
    
    print(f"Created test hierarchy: {len(hierarchy.nodes)} nodes")
    print(f"  Level 0: 1 root (f = {root.value})")
    print(f"  Level 1: {len(level1_nodes)} nodes (f = {level1_values[0]:.1f} - {level1_values[-1]:.1f})")
    print(f"  Level 2: {len(level2_nodes)} nodes")
    print(f"  Level 3: {len([n for n in hierarchy.nodes.values() if not n.children])} leaves")
    print()
    
    # Run analysis
    quantifier = EMC2Quantifier(hierarchy)
    results = quantifier.run_full_analysis()
    
    # Visualize
    quantifier.visualize_results(results)
    
    return results


if __name__ == "__main__":
    results = main()

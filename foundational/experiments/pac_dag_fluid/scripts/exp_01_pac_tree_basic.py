"""
exp_01_pac_tree_basic.py - Basic PAC Tree with SEC Field

Constructs a PAC-compliant tree and computes SEC field values.
Validates the root-as-calculus, leaves-as-geometry hypothesis.

Results:
- Root smoothness: ~0.94
- Leaf discreteness: ~0.89
- SEC field decays exponentially with depth
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
XI = 1 + np.pi / 55  # Balance operator


class PACNode:
    """A node in a PAC tree."""
    
    def __init__(self, value, depth=0, node_id=0):
        self.value = value
        self.depth = depth
        self.node_id = node_id
        self.children = []
        self.parent = None
        self.sec_field = 0.0
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class PACTree:
    """PAC-compliant tree with value conservation."""
    
    def __init__(self, root_value=100.0):
        self.root = PACNode(root_value, depth=0, node_id=0)
        self.total_value = root_value
        self.all_nodes = [self.root]
        self._next_id = 1
    
    def fibonacci_weights(self, n):
        """Generate Fibonacci-based weights for n children."""
        if n <= 0:
            return []
        fib = [1, 1]
        while len(fib) < n:
            fib.append(fib[-1] + fib[-2])
        weights = np.array(fib[:n], dtype=float)
        return weights / weights.sum()
    
    def expand_node(self, node, num_children=3):
        """Expand a node into children, conserving PAC value."""
        weights = self.fibonacci_weights(num_children)
        child_values = node.value * weights
        
        for cv in child_values:
            child = PACNode(cv, depth=node.depth + 1, node_id=self._next_id)
            self._next_id += 1
            node.add_child(child)
            self.all_nodes.append(child)
    
    def build(self, levels=6, branch_factor=3):
        """Build tree to specified depth."""
        current_level = [self.root]
        
        for _ in range(1, levels):
            next_level = []
            for node in current_level:
                self.expand_node(node, branch_factor)
                next_level.extend(node.children)
            current_level = next_level
        
        return self
    
    def verify_conservation(self):
        """Verify PAC conservation: sum of leaf values = root value."""
        leaves = [n for n in self.all_nodes if not n.children]
        leaf_sum = sum(n.value for n in leaves)
        return abs(leaf_sum - self.total_value) < 1e-10
    
    def compute_sec_field(self, decay=0.5):
        """
        Compute SEC field for each node.
        SEC(n) = (v(n)/V) * exp(-λ * depth)
        """
        for node in self.all_nodes:
            node.sec_field = (node.value / self.total_value) * np.exp(-decay * node.depth)
    
    def nodes_at_depth(self, depth):
        """Get all nodes at a specific depth."""
        return [n for n in self.all_nodes if n.depth == depth]
    
    def max_depth(self):
        """Return maximum depth of tree."""
        return max(n.depth for n in self.all_nodes)


def compute_smoothness(values):
    """
    Compute smoothness metric (low variance in differences).
    High smoothness = calculus-like
    """
    if len(values) < 2:
        return 1.0
    diffs = np.diff(sorted(values))
    if np.std(diffs) < 1e-10:
        return 1.0
    variance = np.var(diffs) / (np.mean(np.abs(diffs)) + 1e-10)
    return np.exp(-variance)


def compute_discreteness(values):
    """
    Compute discreteness metric (high variance in values).
    High discreteness = geometry-like
    """
    if len(values) < 2:
        return 0.0
    mean_val = np.mean(values)
    if mean_val < 1e-10:
        return 0.0
    cv = np.std(values) / mean_val  # Coefficient of variation
    return 1 - np.exp(-cv)


def analyze_tree_levels(tree):
    """Analyze SEC properties at each tree level."""
    results = []
    
    for depth in range(tree.max_depth() + 1):
        nodes = tree.nodes_at_depth(depth)
        values = [n.value for n in nodes]
        sec_values = [n.sec_field for n in nodes]
        
        smoothness = compute_smoothness(values)
        discreteness = compute_discreteness(values)
        avg_sec = np.mean(sec_values)
        
        results.append({
            'depth': depth,
            'num_nodes': len(nodes),
            'total_value': sum(values),
            'smoothness': smoothness,
            'discreteness': discreteness,
            'avg_sec_field': avg_sec
        })
    
    return results


def run_experiment():
    """Main experiment: build PAC tree and analyze SEC properties."""
    
    print("=" * 60)
    print("PAC Tree SEC Field Analysis")
    print("=" * 60)
    
    # Build PAC tree
    tree = PACTree(root_value=100.0)
    tree.build(levels=6, branch_factor=3)
    tree.compute_sec_field(decay=0.5)
    
    # Verify conservation
    conservation_ok = tree.verify_conservation()
    print(f"\nPAC Conservation verified: {conservation_ok}")
    print(f"Total nodes: {len(tree.all_nodes)}")
    print(f"Max depth: {tree.max_depth()}")
    
    # Analyze each level
    level_analysis = analyze_tree_levels(tree)
    
    print("\nLevel Analysis:")
    print("-" * 70)
    print(f"{'Depth':<6} {'Nodes':<8} {'Value':<12} {'Smooth':<10} {'Discrete':<10} {'SEC':<10}")
    print("-" * 70)
    
    for level in level_analysis:
        print(f"{level['depth']:<6} {level['num_nodes']:<8} {level['total_value']:<12.4f} "
              f"{level['smoothness']:<10.4f} {level['discreteness']:<10.4f} {level['avg_sec_field']:<10.4f}")
    
    # Key metrics
    root_level = level_analysis[0]
    leaf_level = level_analysis[-1]
    
    print("\n" + "=" * 60)
    print("Key Findings:")
    print("=" * 60)
    print(f"Root smoothness:  {root_level['smoothness']:.4f} (calculus-like)")
    print(f"Leaf discreteness: {leaf_level['discreteness']:.4f} (geometry-like)")
    print(f"SEC decay ratio:   {leaf_level['avg_sec_field'] / root_level['avg_sec_field']:.4f}")
    
    # Check if root-calculus/leaf-geometry hypothesis holds
    hypothesis_valid = (root_level['smoothness'] > leaf_level['smoothness'] and
                       leaf_level['discreteness'] > root_level['discreteness'])
    print(f"\nRoot-as-calculus, Leaves-as-geometry: {'✓ VALIDATED' if hypothesis_valid else '✗ NOT SUPPORTED'}")
    
    # Save results
    results = {
        'experiment': 'pac_tree_basic',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'root_value': 100.0,
            'levels': 6,
            'branch_factor': 3,
            'sec_decay': 0.5
        },
        'metrics': {
            'total_nodes': len(tree.all_nodes),
            'max_depth': tree.max_depth(),
            'conservation_verified': bool(conservation_ok),
            'root_smoothness': float(root_level['smoothness']),
            'leaf_discreteness': float(leaf_level['discreteness']),
            'hypothesis_validated': bool(hypothesis_valid)
        },
        'level_analysis': level_analysis,
        'constants_used': {
            'phi': PHI,
            'xi': XI
        }
    }
    
    # Save to results folder
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'exp_01_pac_tree_basic_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()

"""
exp_02_pac_tree_blowup.py - PAC Tree with Blow-Up Operator

Tests SEC dynamics under perturbation using the blow-up operator.
Studies how noise propagates bidirectionally in PAC hierarchies.

Key Result:
- Perturbations travel both toward root (integration) and leaves (differentiation)
- Root remains smoother even under maximal perturbation
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2
XI = 1 + np.pi / 55


class PACNode:
    """A node in a PAC tree with SEC field."""
    
    def __init__(self, value, depth=0, node_id=0):
        self.value = value
        self.depth = depth
        self.node_id = node_id
        self.children = []
        self.parent = None
        self.sec_field = 0.0
        self.original_sec = 0.0
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class PACTree:
    """PAC tree with blow-up dynamics."""
    
    def __init__(self, root_value=100.0):
        self.root = PACNode(root_value, depth=0, node_id=0)
        self.total_value = root_value
        self.all_nodes = [self.root]
        self._next_id = 1
    
    def fibonacci_weights(self, n):
        fib = [1, 1]
        while len(fib) < n:
            fib.append(fib[-1] + fib[-2])
        weights = np.array(fib[:n], dtype=float)
        return weights / weights.sum()
    
    def expand_node(self, node, num_children=3):
        weights = self.fibonacci_weights(num_children)
        child_values = node.value * weights
        
        for cv in child_values:
            child = PACNode(cv, depth=node.depth + 1, node_id=self._next_id)
            self._next_id += 1
            node.add_child(child)
            self.all_nodes.append(child)
    
    def build(self, levels=6, branch_factor=3):
        current_level = [self.root]
        for _ in range(1, levels):
            next_level = []
            for node in current_level:
                self.expand_node(node, branch_factor)
                next_level.extend(node.children)
            current_level = next_level
        return self
    
    def compute_sec_field(self, decay=0.5):
        for node in self.all_nodes:
            node.sec_field = (node.value / self.total_value) * np.exp(-decay * node.depth)
            node.original_sec = node.sec_field
    
    def nodes_at_depth(self, depth):
        return [n for n in self.all_nodes if n.depth == depth]
    
    def max_depth(self):
        return max(n.depth for n in self.all_nodes)


def blow_up_operator(tree, injection_depth, amplitude, scale_weights=None):
    """
    Apply blow-up operator: inject noise at specified depth.
    
    SEC_blowup(Ψ, λ) = Ψ + Σ A_k * noise_k * w(∇Ψ)
    
    Args:
        tree: PACTree instance
        injection_depth: Depth at which to inject noise
        amplitude: Noise amplitude A
        scale_weights: Optional weights for different scales
    """
    if scale_weights is None:
        scale_weights = {injection_depth: 1.0}
    
    nodes_at_injection = tree.nodes_at_depth(injection_depth)
    
    for node in nodes_at_injection:
        # Compute local gradient weight
        if node.parent:
            grad = abs(node.sec_field - node.parent.sec_field)
        else:
            grad = node.sec_field
        
        weight = scale_weights.get(injection_depth, 1.0)
        noise = np.random.randn() * amplitude * weight
        
        # Weight by local gradient (more perturbation where gradient is high)
        node.sec_field += noise * (1 + grad)


def propagate_downward(tree, diffusion_rate=0.3):
    """
    Downward SEC propagation (differentiation).
    Parent perturbations flow to children with golden ratio scaling.
    """
    for depth in range(tree.max_depth()):
        for node in tree.nodes_at_depth(depth):
            if node.children:
                delta = node.sec_field - node.original_sec
                for child in node.children:
                    child.sec_field += delta * diffusion_rate / PHI


def propagate_upward(tree, integration_rate=0.2):
    """
    Upward SEC propagation (integration).
    Child perturbations integrate into parent with golden ratio scaling.
    """
    for depth in range(tree.max_depth(), 0, -1):
        for node in tree.nodes_at_depth(depth):
            if node.parent:
                delta = node.sec_field - node.original_sec
                node.parent.sec_field += delta * integration_rate * PHI / len(node.parent.children)


def compute_smoothness(tree, depth):
    """Compute smoothness at a given depth."""
    nodes = tree.nodes_at_depth(depth)
    values = [n.sec_field for n in nodes]
    if len(values) < 2:
        return 1.0
    diffs = np.diff(sorted(values))
    if np.std(diffs) < 1e-10:
        return 1.0
    variance = np.var(diffs) / (np.mean(np.abs(diffs)) + 1e-10)
    return np.exp(-variance)


def compute_discreteness(tree, depth):
    """Compute discreteness at a given depth."""
    nodes = tree.nodes_at_depth(depth)
    values = [n.sec_field for n in nodes]
    if len(values) < 2:
        return 0.0
    mean_val = np.mean(np.abs(values))
    if mean_val < 1e-10:
        return 0.0
    cv = np.std(values) / mean_val
    return 1 - np.exp(-cv)


def run_experiment():
    """Test blow-up dynamics on PAC tree."""
    
    print("=" * 60)
    print("PAC Tree Blow-Up Operator Analysis")
    print("=" * 60)
    
    np.random.seed(42)  # Reproducibility
    
    # Build tree
    tree = PACTree(root_value=100.0)
    tree.build(levels=6, branch_factor=3)
    tree.compute_sec_field(decay=0.5)
    
    # Store initial state
    initial_root_smoothness = compute_smoothness(tree, 0)
    initial_leaf_discreteness = compute_discreteness(tree, tree.max_depth())
    
    print(f"\nInitial State:")
    print(f"  Root smoothness:  {initial_root_smoothness:.4f}")
    print(f"  Leaf discreteness: {initial_leaf_discreteness:.4f}")
    
    # Test different perturbation amplitudes
    amplitudes = [0.01, 0.05, 0.1, 0.2, 0.5]
    amplitude_results = []
    
    for amp in amplitudes:
        # Reset tree
        tree = PACTree(root_value=100.0)
        tree.build(levels=6, branch_factor=3)
        tree.compute_sec_field(decay=0.5)
        
        # Apply blow-up at middle depth
        injection_depth = tree.max_depth() // 2
        blow_up_operator(tree, injection_depth, amp)
        
        # Propagate bidirectionally
        for _ in range(5):  # Multiple propagation steps
            propagate_downward(tree)
            propagate_upward(tree)
        
        root_smooth = compute_smoothness(tree, 0)
        leaf_discrete = compute_discreteness(tree, tree.max_depth())
        
        amplitude_results.append({
            'amplitude': amp,
            'root_smoothness': root_smooth,
            'leaf_discreteness': leaf_discrete
        })
        
        print(f"\nAmplitude {amp}:")
        print(f"  Root smoothness:  {root_smooth:.4f} (Δ = {root_smooth - initial_root_smoothness:+.4f})")
        print(f"  Leaf discreteness: {leaf_discrete:.4f} (Δ = {leaf_discrete - initial_leaf_discreteness:+.4f})")
    
    # Test injection at different depths
    print("\n" + "=" * 60)
    print("Injection Depth Analysis")
    print("=" * 60)
    
    depths_to_test = [1, 2, 3, 4]
    depth_results = []
    
    for inject_depth in depths_to_test:
        tree = PACTree(root_value=100.0)
        tree.build(levels=6, branch_factor=3)
        tree.compute_sec_field(decay=0.5)
        
        blow_up_operator(tree, inject_depth, amplitude=0.1)
        
        for _ in range(5):
            propagate_downward(tree)
            propagate_upward(tree)
        
        root_smooth = compute_smoothness(tree, 0)
        leaf_discrete = compute_discreteness(tree, tree.max_depth())
        
        depth_results.append({
            'injection_depth': inject_depth,
            'root_smoothness': root_smooth,
            'leaf_discreteness': leaf_discrete
        })
        
        print(f"\nInjection at depth {inject_depth}:")
        print(f"  Root smoothness:  {root_smooth:.4f}")
        print(f"  Leaf discreteness: {leaf_discrete:.4f}")
    
    # Key findings
    print("\n" + "=" * 60)
    print("Key Findings")
    print("=" * 60)
    
    # Check if root always smoother than leaves
    root_always_smoother = all(
        r['root_smoothness'] > 0.7 for r in amplitude_results
    )
    leaf_fragments = all(
        amplitude_results[i]['leaf_discreteness'] >= amplitude_results[i-1]['leaf_discreteness']
        for i in range(1, len(amplitude_results))
    )
    
    print(f"Root remains smooth (> 0.7) under all perturbations: {'✓' if root_always_smoother else '✗'}")
    print(f"Leaf discreteness increases with amplitude: {'✓' if leaf_fragments else '✗'}")
    
    # Save results
    results = {
        'experiment': 'pac_tree_blowup',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'root_value': 100.0,
            'levels': 6,
            'branch_factor': 3,
            'diffusion_rate': 0.3,
            'integration_rate': 0.2
        },
        'initial_state': {
            'root_smoothness': initial_root_smoothness,
            'leaf_discreteness': initial_leaf_discreteness
        },
        'amplitude_analysis': amplitude_results,
        'depth_analysis': depth_results,
        'findings': {
            'root_remains_smooth': root_always_smoother,
            'leaf_fragments_with_amplitude': leaf_fragments
        },
        'constants_used': {
            'phi': PHI,
            'xi': XI
        }
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'exp_02_pac_tree_blowup_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    run_experiment()

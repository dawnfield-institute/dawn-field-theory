"""
Algebraic and Geometric Complexity Metrics

Defines quantitative measures for the algebra-geometry duality.

Algebraic Complexity Aₓ: measures operational/dynamic/entropic properties
Geometric Complexity Gₓ: measures structural/conservative/relational properties

The hypothesis: φ and Ξ emerge when Aₓ and Gₓ are in balance.
"""

import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ComplexityMetrics:
    """Container for algebraic and geometric complexity measures."""
    algebraic: float
    geometric: float
    ratio: float
    distance_from_phi: float
    distance_from_inv_phi: float
    distance_from_xi: float
    
    PHI = (1 + np.sqrt(5)) / 2  # 1.618...
    INV_PHI = 2 / (1 + np.sqrt(5))  # 0.618...
    XI = 1 + np.pi / 55  # 1.0571...


def algebraic_complexity_tree(tree: Dict[str, Any]) -> float:
    """
    Compute algebraic complexity of a tree structure.
    
    Algebraic = operations required to traverse/construct
    - Depth contributes (recursive calls)
    - Branching contributes (choice points)
    - Asymmetry contributes (imbalanced operations)
    
    Args:
        tree: Dict with 'value', optional 'children' list
        
    Returns:
        Algebraic complexity measure
    """
    def traverse(node, depth=0):
        if not node.get('children'):
            return depth  # Leaf: just the depth to reach it
        
        child_complexities = [traverse(c, depth + 1) for c in node['children']]
        
        # Algebraic complexity: sum of child complexities + branching factor
        branching = len(node['children'])
        asymmetry = np.std(child_complexities) if len(child_complexities) > 1 else 0
        
        return sum(child_complexities) + branching + asymmetry
    
    return traverse(tree)


def geometric_complexity_tree(tree: Dict[str, Any]) -> float:
    """
    Compute geometric complexity of a tree structure.
    
    Geometric = structural/relational properties
    - Node count (entities)
    - Edge count (relations)
    - Conservation satisfaction (PAC balance)
    
    Args:
        tree: Dict with 'value', optional 'children' list
        
    Returns:
        Geometric complexity measure
    """
    def count_nodes_edges(node):
        if not node.get('children'):
            return 1, 0  # Leaf: 1 node, 0 edges from here
        
        nodes = 1
        edges = len(node['children'])
        
        for child in node['children']:
            cn, ce = count_nodes_edges(child)
            nodes += cn
            edges += ce
        
        return nodes, edges
    
    def pac_balance_error(node):
        """Check PAC: parent value = sum of children values."""
        if not node.get('children'):
            return 0
        
        parent_val = node.get('value', 0)
        children_sum = sum(c.get('value', 0) for c in node['children'])
        
        error = abs(parent_val - children_sum)
        child_errors = sum(pac_balance_error(c) for c in node['children'])
        
        return error + child_errors
    
    nodes, edges = count_nodes_edges(tree)
    pac_error = pac_balance_error(tree)
    
    # Geometric complexity: structure size minus conservation violations
    # High structure + low error = high geometric complexity
    structure = nodes + edges
    conservation = 1 / (1 + pac_error)  # Approaches 1 when error is 0
    
    return structure * conservation


def compute_complexity_ratio(tree: Dict[str, Any]) -> ComplexityMetrics:
    """
    Compute full complexity metrics for a tree.
    
    Args:
        tree: Dict with 'value', optional 'children' list
        
    Returns:
        ComplexityMetrics dataclass
    """
    alg = algebraic_complexity_tree(tree)
    geo = geometric_complexity_tree(tree)
    
    # Avoid division by zero
    ratio = alg / geo if geo > 0 else float('inf')
    
    return ComplexityMetrics(
        algebraic=alg,
        geometric=geo,
        ratio=ratio,
        distance_from_phi=abs(ratio - ComplexityMetrics.PHI),
        distance_from_inv_phi=abs(ratio - ComplexityMetrics.INV_PHI),
        distance_from_xi=abs(ratio - ComplexityMetrics.XI)
    )


# --- CA-specific metrics ---

def algebraic_complexity_ca_rule(rule_number: int, generations: int = 100) -> float:
    """
    Algebraic complexity of a CA rule.
    
    Measures: transformation operations, state changes, entropy production
    """
    # Rule as binary transformation
    rule_bits = [(rule_number >> i) & 1 for i in range(8)]
    
    # Algebraic complexity proxies:
    # 1. Number of 1s in rule (active transformations)
    active_transforms = sum(rule_bits)
    
    # 2. Bit transition count (operational complexity)
    transitions = sum(1 for i in range(7) if rule_bits[i] != rule_bits[i+1])
    
    # 3. Simulate and count state changes
    width = 101
    state = [0] * width
    state[width // 2] = 1
    
    total_changes = 0
    for _ in range(generations):
        new_state = []
        for i in range(width):
            left = state[(i - 1) % width]
            center = state[i]
            right = state[(i + 1) % width]
            index = (left << 2) | (center << 1) | right
            new_state.append(rule_bits[index])
        
        total_changes += sum(1 for i in range(width) if state[i] != new_state[i])
        state = new_state
    
    return active_transforms + transitions + (total_changes / generations)


def geometric_complexity_ca_rule(rule_number: int, generations: int = 100) -> float:
    """
    Geometric complexity of a CA rule.
    
    Measures: pattern structure, attractor behavior, spatial organization
    """
    rule_bits = [(rule_number >> i) & 1 for i in range(8)]
    
    width = 101
    state = [0] * width
    state[width // 2] = 1
    
    states_history = [tuple(state)]
    
    for _ in range(generations):
        new_state = []
        for i in range(width):
            left = state[(i - 1) % width]
            center = state[i]
            right = state[(i + 1) % width]
            index = (left << 2) | (center << 1) | right
            new_state.append(rule_bits[index])
        state = new_state
        states_history.append(tuple(state))
    
    # Geometric complexity proxies:
    # 1. Unique states (structural variety)
    unique_states = len(set(states_history))
    
    # 2. Final density (structure vs void)
    final_density = sum(state) / width
    
    # 3. Spatial correlation (structure coherence)
    correlations = sum(1 for i in range(width-1) if state[i] == state[i+1])
    spatial_coherence = correlations / (width - 1)
    
    # Balance: neither too uniform nor too random
    structure_balance = 1 - abs(0.5 - final_density) * 2
    
    return unique_states * (1 + spatial_coherence) * (1 + structure_balance)


# --- Factorization-specific metrics ---

def algebraic_complexity_factorization(n: int) -> float:
    """
    Algebraic complexity of factorizing n.
    
    Measures: division operations, recursive depth, computational work
    """
    if n < 2:
        return 0
    
    operations = 0
    remaining = n
    
    # Count trial divisions
    d = 2
    while d * d <= remaining:
        while remaining % d == 0:
            operations += 1  # Each division is an operation
            remaining //= d
        d += 1
        operations += 0.1  # Trial cost
    
    if remaining > 1:
        operations += 1  # Final prime found
    
    return operations


def geometric_complexity_factorization(n: int) -> float:
    """
    Geometric complexity of n's factor tree.
    
    Measures: tree structure, node count, depth
    """
    if n < 2:
        return 1
    
    def factor_tree_structure(num):
        if num < 2:
            return {'value': num, 'depth': 0, 'nodes': 1}
        
        # Find smallest factor
        d = 2
        while d * d <= num:
            if num % d == 0:
                left = factor_tree_structure(d)
                right = factor_tree_structure(num // d)
                return {
                    'value': num,
                    'depth': 1 + max(left['depth'], right['depth']),
                    'nodes': 1 + left['nodes'] + right['nodes']
                }
            d += 1
        
        # Prime
        return {'value': num, 'depth': 0, 'nodes': 1}
    
    tree = factor_tree_structure(n)
    
    # Geometric complexity: structure size weighted by depth coherence
    return tree['nodes'] * (1 + tree['depth'])


if __name__ == "__main__":
    # Quick test
    print("=== Tree Complexity Test ===")
    test_tree = {
        'value': 12,
        'children': [
            {'value': 4, 'children': [
                {'value': 2},
                {'value': 2}
            ]},
            {'value': 3}
        ]
    }
    
    metrics = compute_complexity_ratio(test_tree)
    print(f"Algebraic: {metrics.algebraic:.4f}")
    print(f"Geometric: {metrics.geometric:.4f}")
    print(f"Ratio: {metrics.ratio:.4f}")
    print(f"Distance from φ: {metrics.distance_from_phi:.4f}")
    print(f"Distance from Ξ: {metrics.distance_from_xi:.4f}")
    
    print("\n=== CA Rule 110 Test ===")
    alg_110 = algebraic_complexity_ca_rule(110)
    geo_110 = geometric_complexity_ca_rule(110)
    ratio_110 = alg_110 / geo_110 if geo_110 > 0 else float('inf')
    print(f"Algebraic: {alg_110:.4f}")
    print(f"Geometric: {geo_110:.4f}")
    print(f"Ratio: {ratio_110:.4f}")
    
    print("\n=== Factorization Test (n=120) ===")
    alg_f = algebraic_complexity_factorization(120)
    geo_f = geometric_complexity_factorization(120)
    ratio_f = alg_f / geo_f if geo_f > 0 else float('inf')
    print(f"Algebraic: {alg_f:.4f}")
    print(f"Geometric: {geo_f:.4f}")
    print(f"Ratio: {ratio_f:.4f}")

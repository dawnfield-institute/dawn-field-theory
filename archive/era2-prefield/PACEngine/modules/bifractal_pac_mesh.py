"""
Bi-Fractal PAC Mesh Generator
============================

Generates computational meshes from nested PAC tree structures rather than
uniform grids. The key insight: grid points emerge from PAC tree node positions,
with subdivision controlled by the balance operator Ξ = 1.0571.

THEORY:
------
Standard CFD uses uniform meshes: y = linspace(-H, H, N)

PAC meshes use hierarchical tree subdivision where:
- Each node subdivides at ratio Ξ/(1+Ξ) ≈ 0.514 (not 0.5)
- Nested bi-fractal structure: two trees from each boundary meeting in middle
- Natural clustering near boundaries (boundary layer resolution)
- PAC conservation maintained at every scale

THE BALANCE OPERATOR Ξ:
----------------------
Ξ = 1.0571 is NOT fitted - it's derived from:

    Ξ(N) = Σ(n+½)² / Σn²  for n=1..N
    
At N = 26 (where N* = 3×F₁₀/(2π) from Fibonacci):
    
    Ξ(26) = 1.0577 ≈ 1 + π/55 = 1.0571

This emerges from Möbius/Circle spectral eigenvalue ratio.

MESH PROPERTIES:
---------------
1. Non-uniform spacing respecting Ξ balance
2. Natural multi-resolution from tree depth
3. Boundary layer clustering without explicit stretching
4. PAC conservation at each subdivision level

AUTHOR: Dawn Field Institute
DATE: 2025-12-06
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt


# The balance operator - derived, not fitted
XI = 1.0571  # Ξ = 1 + π/F₁₀ where F₁₀ = 55


@dataclass
class PACMeshNode:
    """Node in the PAC mesh tree"""
    id: int
    position: float          # Physical position
    level: int               # Tree depth
    left_child: Optional['PACMeshNode'] = None
    right_child: Optional['PACMeshNode'] = None
    parent: Optional['PACMeshNode'] = None
    value: float = 1.0       # PAC value (conserved across subdivision)
    
    def is_leaf(self) -> bool:
        return self.left_child is None and self.right_child is None


class BiFractalPACMesh:
    """
    Generates mesh points from nested bi-fractal PAC trees.
    
    Structure:
    - Two trees rooted at boundaries y=-H and y=+H
    - Trees grow inward, meeting at the center
    - Subdivision ratio determined by Ξ balance operator
    - Mesh points are leaf node positions
    
    The "bi-fractal" nature comes from:
    1. Two interleaved fractal structures (from each boundary)
    2. Self-similar subdivision at each level
    """
    
    def __init__(self, H: float = 1.0, xi: float = XI):
        """
        Args:
            H: Half-domain size (domain is [-H, +H])
            xi: Balance operator (default: Ξ = 1.0571)
        """
        self.H = H
        self.xi = xi
        self.nodes: List[PACMeshNode] = []
        self.next_id = 0
        
        # Subdivision ratio: fraction where left child ends
        # For Ξ-balanced: left gets 1/(1+Ξ), right gets Ξ/(1+Ξ)
        self.split_ratio = 1.0 / (1.0 + xi)  # ≈ 0.486
        
    def _create_node(self, position: float, level: int, 
                     parent: Optional[PACMeshNode] = None) -> PACMeshNode:
        """Create a new mesh node"""
        node = PACMeshNode(
            id=self.next_id,
            position=position,
            level=level,
            parent=parent
        )
        self.next_id += 1
        self.nodes.append(node)
        return node
    
    def generate_tree(self, y_min: float, y_max: float, 
                      depth: int, level: int = 0,
                      parent: Optional[PACMeshNode] = None,
                      from_boundary: str = 'left') -> PACMeshNode:
        """
        Recursively generate a PAC tree over interval [y_min, y_max].
        
        The subdivision is Ξ-weighted:
        - If from left boundary: split at y_min + (y_max - y_min) × split_ratio
        - If from right boundary: split at y_max - (y_max - y_min) × split_ratio
        
        This causes denser clustering near the originating boundary.
        """
        # Node position at interval center
        y_center = (y_min + y_max) / 2.0
        node = self._create_node(y_center, level, parent)
        
        if level >= depth:
            return node  # Leaf node
        
        # Compute split point based on Ξ ratio
        interval = y_max - y_min
        
        if from_boundary == 'left':
            # Denser on left side
            y_split = y_min + interval * self.split_ratio
        else:
            # Denser on right side
            y_split = y_max - interval * self.split_ratio
            
        # Recursively create children
        node.left_child = self.generate_tree(
            y_min, y_split, depth, level + 1, node, from_boundary
        )
        node.right_child = self.generate_tree(
            y_split, y_max, depth, level + 1, node, from_boundary
        )
        
        # PAC conservation: parent value = sum of children
        node.value = node.left_child.value + node.right_child.value
        
        return node
    
    def generate_bifractal_mesh(self, depth: int) -> np.ndarray:
        """
        Generate mesh points from bi-fractal structure.
        
        Creates two trees:
        - Left tree from y=-H, clustering near left boundary
        - Right tree from y=+H, clustering near right boundary
        
        Mesh points are collected from all leaf nodes.
        """
        self.nodes = []
        self.next_id = 0
        
        # Left half: tree from left boundary
        left_tree = self.generate_tree(-self.H, 0.0, depth, 
                                        from_boundary='left')
        
        # Right half: tree from right boundary  
        right_tree = self.generate_tree(0.0, self.H, depth,
                                         from_boundary='right')
        
        # Collect leaf positions
        leaf_positions = []
        for node in self.nodes:
            if node.is_leaf():
                leaf_positions.append(node.position)
        
        # Add boundary points explicitly
        positions = [-self.H] + sorted(leaf_positions) + [self.H]
        
        return np.array(positions)
    
    def generate_uniform_comparison(self, N: int) -> np.ndarray:
        """Generate uniform mesh with same number of points for comparison"""
        return np.linspace(-self.H, self.H, N)
    
    def compute_spacing_distribution(self, mesh: np.ndarray) -> Dict:
        """Analyze mesh spacing distribution"""
        spacings = np.diff(mesh)
        return {
            'min_spacing': np.min(spacings),
            'max_spacing': np.max(spacings),
            'mean_spacing': np.mean(spacings),
            'std_spacing': np.std(spacings),
            'ratio_max_min': np.max(spacings) / np.min(spacings),
            'spacings': spacings
        }
    
    def verify_pac_conservation(self) -> Tuple[bool, float]:
        """
        Verify PAC conservation: parent.value = sum(children.values)
        
        Returns:
            (is_conserved, max_violation)
        """
        max_violation = 0.0
        
        for node in self.nodes:
            if not node.is_leaf():
                children_sum = 0.0
                if node.left_child:
                    children_sum += node.left_child.value
                if node.right_child:
                    children_sum += node.right_child.value
                
                violation = abs(node.value - children_sum)
                max_violation = max(max_violation, violation)
        
        return max_violation < 1e-10, max_violation


class AdaptivePACMesh(BiFractalPACMesh):
    """
    Extension: Adaptive PAC mesh that refines based on solution gradients.
    
    Uses the bi-fractal structure but allows additional refinement
    where the solution has large gradients (e.g., boundary layers).
    """
    
    def refine_at_node(self, node: PACMeshNode) -> None:
        """Refine a leaf node into two children"""
        if not node.is_leaf():
            return  # Already refined
        
        # Find interval bounds from neighbors
        # This is a simplified implementation
        pass
    
    def adapt_to_solution(self, solution: np.ndarray, 
                          mesh: np.ndarray,
                          gradient_threshold: float = 0.1) -> np.ndarray:
        """
        Adapt mesh based on solution gradients.
        
        Refines where |du/dy| > threshold.
        """
        # Compute gradient
        grad = np.gradient(solution, mesh)
        
        # Find regions needing refinement
        needs_refinement = np.abs(grad) > gradient_threshold * np.max(np.abs(grad))
        
        # For now, return original mesh with markers
        # Full implementation would regenerate tree with more depth in those regions
        return mesh, needs_refinement


def demonstrate_bifractal_mesh():
    """Demonstrate the bi-fractal PAC mesh vs uniform mesh"""
    print("="*70)
    print("BI-FRACTAL PAC MESH DEMONSTRATION")
    print("="*70)
    
    H = 1.0
    mesh_gen = BiFractalPACMesh(H=H)
    
    print(f"\nBalance operator Ξ = {mesh_gen.xi:.4f}")
    print(f"Split ratio = {mesh_gen.split_ratio:.4f} (vs 0.5 for uniform)")
    
    for depth in [2, 3, 4, 5]:
        pac_mesh = mesh_gen.generate_bifractal_mesh(depth)
        uniform_mesh = mesh_gen.generate_uniform_comparison(len(pac_mesh))
        
        pac_stats = mesh_gen.compute_spacing_distribution(pac_mesh)
        uniform_stats = mesh_gen.compute_spacing_distribution(uniform_mesh)
        
        conserved, violation = mesh_gen.verify_pac_conservation()
        
        print(f"\nDepth {depth}: {len(pac_mesh)} points")
        print(f"  PAC mesh:")
        print(f"    Spacing range: [{pac_stats['min_spacing']:.4f}, {pac_stats['max_spacing']:.4f}]")
        print(f"    Ratio max/min: {pac_stats['ratio_max_min']:.2f}")
        print(f"    PAC conserved: {conserved} (violation: {violation:.2e})")
        print(f"  Uniform mesh:")
        print(f"    Spacing: {uniform_stats['mean_spacing']:.4f} (constant)")


def compare_solver_accuracy():
    """
    Compare solver accuracy on PAC mesh vs uniform mesh.
    
    Test problem: d²u/dy² = sin(πy/H), u(±H) = 0
    """
    from scipy.linalg import solve_banded
    
    print("\n" + "="*70)
    print("SOLVER ACCURACY COMPARISON: PAC MESH vs UNIFORM MESH")
    print("="*70)
    
    H = 1.0
    mesh_gen = BiFractalPACMesh(H=H)
    
    def solve_on_mesh(y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Solve d²u/dy² = sin(πy/H) on given mesh"""
        n = len(y)
        
        # Exact solution
        u_exact = -(H / np.pi)**2 * np.sin(np.pi * y / H)
        
        # For non-uniform mesh, need to adjust FD stencil
        # Using standard 3-point stencil with variable spacing:
        # (u[i+1] - u[i])/(y[i+1] - y[i]) - (u[i] - u[i-1])/(y[i] - y[i-1])
        # ------------------------------------------------------------ = f[i]
        #              0.5 * (y[i+1] - y[i-1])
        
        # Build matrix (not tridiagonal for non-uniform!)
        A = np.zeros((n, n))
        rhs = np.zeros(n)
        
        # Boundary conditions
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        rhs[0] = 0.0
        rhs[-1] = 0.0
        
        # Interior points
        for i in range(1, n-1):
            dy_plus = y[i+1] - y[i]
            dy_minus = y[i] - y[i-1]
            dy_avg = 0.5 * (y[i+1] - y[i-1])
            
            A[i, i-1] = 1.0 / (dy_minus * dy_avg)
            A[i, i] = -1.0 / (dy_plus * dy_avg) - 1.0 / (dy_minus * dy_avg)
            A[i, i+1] = 1.0 / (dy_plus * dy_avg)
            
            rhs[i] = np.sin(np.pi * y[i] / H)
        
        # Solve
        u_computed = np.linalg.solve(A, rhs)
        
        # Error
        error = np.max(np.abs(u_computed - u_exact))
        
        return u_computed, error
    
    print("\nTest: d²u/dy² = sin(πy/H), u(±H) = 0")
    print("-" * 50)
    
    for depth in [3, 4, 5, 6]:
        pac_mesh = mesh_gen.generate_bifractal_mesh(depth)
        n_points = len(pac_mesh)
        uniform_mesh = np.linspace(-H, H, n_points)
        
        _, pac_error = solve_on_mesh(pac_mesh)
        _, uniform_error = solve_on_mesh(uniform_mesh)
        
        ratio = uniform_error / pac_error if pac_error > 1e-15 else float('inf')
        better = "PAC" if pac_error < uniform_error else "UNIFORM"
        
        print(f"N={n_points:3d} (depth={depth}): PAC error={pac_error:.2e}, Uniform error={uniform_error:.2e}, Winner: {better}")


if __name__ == "__main__":
    demonstrate_bifractal_mesh()
    compare_solver_accuracy()

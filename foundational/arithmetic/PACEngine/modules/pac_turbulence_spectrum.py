"""
PAC Tree Turbulence Spectrum Analysis
=====================================

Exploring how to modify PAC tree structure to achieve different turbulence spectra:
- k^(-2): Binary tree (what we have)
- k^(-5/3): Kolmogorov 3D turbulence  
- k^(-3): 2D enstrophy cascade

Also extending to 3D PAC octrees to see if dimensionality changes the spectrum.

KEY QUESTION: Can we tune the tree branching to match any desired spectral law?
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import itertools


# Balance operator
XI = 1.0571


@dataclass
class PACNode3D:
    """3D PAC tree node (octree)"""
    id: int
    position: Tuple[float, float, float]  # (x, y, z)
    level: int
    children: List['PACNode3D'] = field(default_factory=list)
    parent: Optional['PACNode3D'] = None
    value: float = 1.0


class GeneralizedPACTree:
    """
    Generalized PAC tree with variable branching factor.
    
    To achieve E(k) ~ k^(-α):
    - Binary tree (b=2): gives k^(-2)
    - Ternary tree (b=3): gives k^(-log₃(9)) = k^(-2)... hmm
    
    The key is the FRACTAL DIMENSION of the tree!
    """
    
    def __init__(self, branching_factor: int = 2, xi: float = XI):
        self.b = branching_factor
        self.xi = xi
        self.nodes = []
        self.next_id = 0
    
    def generate_1d(self, x_min: float, x_max: float, depth: int, 
                    level: int = 0) -> List[float]:
        """Generate 1D mesh with variable branching"""
        if level >= depth:
            return [(x_min + x_max) / 2]
        
        positions = []
        interval = x_max - x_min
        
        # Divide interval into b parts with XI-weighting
        # For b=2: [0, 1/(1+XI), 1] 
        # For b=3: [0, 1/(1+2*XI), (1+XI)/(1+2*XI), 1]
        
        split_points = [x_min]
        for i in range(1, self.b):
            # XI-weighted split
            frac = i / self.b
            # Adjust by XI
            if i <= self.b // 2:
                frac = frac * (1 / (1 + self.xi * 0.1))
            else:
                frac = frac * (1 + self.xi * 0.1) / (1 + self.xi * 0.1)
            split_points.append(x_min + interval * i / self.b)
        split_points.append(x_max)
        
        # Recurse into each sub-interval
        for i in range(self.b):
            positions.extend(
                self.generate_1d(split_points[i], split_points[i+1], 
                                depth, level + 1)
            )
        
        return positions
    
    def compute_spectrum(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute energy spectrum from mesh positions"""
        spacings = np.diff(np.sort(positions))
        
        # Wavenumber k = 1/spacing
        k = 1.0 / spacings
        
        # Energy E ~ spacing^2
        E = spacings**2
        
        # Sort by k
        sort_idx = np.argsort(k)
        k_sorted = k[sort_idx]
        E_sorted = E[sort_idx]
        
        # Fit power law in inertial range
        log_k = np.log10(k_sorted)
        log_E = np.log10(E_sorted)
        
        n = len(log_k)
        inertial = slice(n//4, 3*n//4)
        
        coeffs = np.polyfit(log_k[inertial], log_E[inertial], 1)
        slope = coeffs[0]
        
        return k_sorted, E_sorted, slope


class PACOctree3D:
    """
    3D PAC octree for turbulence analysis.
    
    Each node splits into 8 children (octree).
    Question: Does 3D change the spectral exponent?
    """
    
    def __init__(self, xi: float = XI):
        self.xi = xi
        self.nodes: List[PACNode3D] = []
        self.next_id = 0
    
    def _create_node(self, position: Tuple[float, float, float], 
                     level: int, parent: Optional[PACNode3D] = None) -> PACNode3D:
        node = PACNode3D(
            id=self.next_id,
            position=position,
            level=level,
            parent=parent
        )
        self.next_id += 1
        self.nodes.append(node)
        return node
    
    def generate_octree(self, bounds: Tuple[Tuple[float, float], ...], 
                        depth: int, level: int = 0,
                        parent: Optional[PACNode3D] = None) -> PACNode3D:
        """
        Generate 3D octree over cubic region.
        
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        """
        # Center of region
        center = tuple((b[0] + b[1]) / 2 for b in bounds)
        node = self._create_node(center, level, parent)
        
        if level >= depth:
            return node
        
        # Split into 8 octants with XI-weighting
        split_ratio = 1.0 / (1.0 + self.xi)  # ~0.486
        
        # Compute split points for each dimension
        splits = []
        for b in bounds:
            mid = b[0] + (b[1] - b[0]) * 0.5  # Could use split_ratio here
            splits.append((b[0], mid, b[1]))
        
        # Generate 8 children (all combinations of low/high in each dim)
        for ix, iy, iz in itertools.product([0, 1], repeat=3):
            child_bounds = (
                (splits[0][ix], splits[0][ix+1]),
                (splits[1][iy], splits[1][iy+1]),
                (splits[2][iz], splits[2][iz+1])
            )
            child = self.generate_octree(child_bounds, depth, level + 1, node)
            node.children.append(child)
        
        # PAC conservation
        node.value = sum(c.value for c in node.children)
        
        return node
    
    def get_leaf_positions(self) -> np.ndarray:
        """Get positions of all leaf nodes"""
        leaves = [n for n in self.nodes if len(n.children) == 0]
        return np.array([n.position for n in leaves])
    
    def compute_3d_spectrum(self) -> Dict:
        """
        Compute 3D energy spectrum.
        
        In 3D, we look at distance from each point to its neighbors
        and compute shell-averaged spectrum.
        """
        positions = self.get_leaf_positions()
        n_points = len(positions)
        
        # Compute all pairwise distances
        distances = []
        for i in range(n_points):
            for j in range(i+1, n_points):
                d = np.linalg.norm(positions[i] - positions[j])
                distances.append(d)
        
        distances = np.array(distances)
        
        # Convert to wavenumbers
        k = 1.0 / distances
        
        # "Energy" at each scale
        E = distances**2
        
        # Bin by wavenumber shells
        k_bins = np.logspace(np.log10(k.min()), np.log10(k.max()), 20)
        E_spectrum = []
        k_centers = []
        
        for i in range(len(k_bins) - 1):
            mask = (k >= k_bins[i]) & (k < k_bins[i+1])
            if np.sum(mask) > 0:
                E_spectrum.append(np.mean(E[mask]))
                k_centers.append(np.sqrt(k_bins[i] * k_bins[i+1]))
        
        k_centers = np.array(k_centers)
        E_spectrum = np.array(E_spectrum)
        
        # Fit power law
        if len(k_centers) > 3:
            log_k = np.log10(k_centers)
            log_E = np.log10(E_spectrum)
            coeffs = np.polyfit(log_k, log_E, 1)
            slope = coeffs[0]
        else:
            slope = 0
        
        return {
            'k': k_centers,
            'E': E_spectrum,
            'slope': slope,
            'n_points': n_points
        }


class KolmogorovPACTree:
    """
    Modified PAC tree designed to achieve k^(-5/3) Kolmogorov spectrum.
    
    Strategy: Use non-binary branching that gives fractal dimension D
    such that E(k) ~ k^(-5/3).
    
    For binary tree: E ~ k^(-2) because spacing ~ 2^(-level)
    For k^(-5/3): we need spacing ~ 2^(-level * 5/6)
    
    This means each level should have 2^(6/5) ≈ 2.3 branches on average!
    We can achieve this with probabilistic branching or non-uniform trees.
    """
    
    def __init__(self, target_exponent: float = -5/3):
        self.target = target_exponent
        self.xi = XI
        
        # For E ~ k^α, spacing ~ k^(α/2)
        # If we double k, spacing changes by 2^(α/2)
        # For α = -5/3, spacing ratio = 2^(-5/6) ≈ 0.56
        self.spacing_ratio = 2**(target_exponent / 2)
        print(f"Target exponent: {target_exponent:.3f}")
        print(f"Required spacing ratio per level: {self.spacing_ratio:.4f}")
    
    def generate_kolmogorov_mesh(self, x_min: float, x_max: float, 
                                  n_levels: int) -> np.ndarray:
        """
        Generate mesh with Kolmogorov-like spacing distribution.
        
        Instead of binary splitting, we use variable splitting
        that achieves the target spectrum.
        """
        positions = [x_min, x_max]
        
        current_spacing = x_max - x_min
        
        for level in range(n_levels):
            # Target spacing at this level
            target_spacing = current_spacing * self.spacing_ratio
            
            # Add points to achieve this spacing
            new_positions = []
            sorted_pos = sorted(positions)
            
            for i in range(len(sorted_pos) - 1):
                a, b = sorted_pos[i], sorted_pos[i+1]
                spacing = b - a
                
                # How many subdivisions needed?
                n_divs = max(1, int(spacing / target_spacing))
                
                # Add subdivision points
                for j in range(1, n_divs):
                    new_positions.append(a + j * spacing / n_divs)
            
            positions.extend(new_positions)
            current_spacing = target_spacing
        
        return np.array(sorted(set(positions)))


def analyze_branching_vs_spectrum():
    """Test different branching factors and their spectra"""
    print("="*70)
    print("BRANCHING FACTOR vs SPECTRAL EXPONENT")
    print("="*70)
    
    depth = 7
    
    for b in [2, 3, 4, 5]:
        tree = GeneralizedPACTree(branching_factor=b)
        positions = tree.generate_1d(-1, 1, depth)
        positions = np.array(sorted(set(positions)))
        
        k, E, slope = tree.compute_spectrum(positions)
        
        print(f"\nBranching factor b={b}:")
        print(f"  Points: {len(positions)}")
        print(f"  Spectral slope: {slope:.3f}")
        print(f"  (Kolmogorov = -1.67, Binary tree = -2.00)")


def analyze_3d_octree():
    """Test 3D octree spectrum"""
    print("\n" + "="*70)
    print("3D OCTREE SPECTRUM ANALYSIS")
    print("="*70)
    
    for depth in [2, 3, 4]:
        octree = PACOctree3D(xi=XI)
        bounds = ((-1, 1), (-1, 1), (-1, 1))
        octree.generate_octree(bounds, depth)
        
        result = octree.compute_3d_spectrum()
        
        print(f"\nDepth {depth}:")
        print(f"  Leaf nodes: {result['n_points']}")
        print(f"  3D Spectral slope: {result['slope']:.3f}")
        print(f"  (3D Kolmogorov = -5/3 = -1.67)")


def analyze_kolmogorov_tree():
    """Test Kolmogorov-targeted tree"""
    print("\n" + "="*70)
    print("KOLMOGOROV-TARGETED PAC TREE")
    print("="*70)
    
    ktree = KolmogorovPACTree(target_exponent=-5/3)
    
    for n_levels in [5, 7, 9]:
        mesh = ktree.generate_kolmogorov_mesh(-1, 1, n_levels)
        
        # Compute spectrum
        spacings = np.diff(mesh)
        k = 1.0 / spacings
        E = spacings**2
        
        sort_idx = np.argsort(k)
        k_sorted = k[sort_idx]
        E_sorted = E[sort_idx]
        
        log_k = np.log10(k_sorted)
        log_E = np.log10(E_sorted)
        
        n = len(log_k)
        coeffs = np.polyfit(log_k[n//4:3*n//4], log_E[n//4:3*n//4], 1)
        slope = coeffs[0]
        
        print(f"\nLevels {n_levels}:")
        print(f"  Points: {len(mesh)}")
        print(f"  Achieved slope: {slope:.3f}")
        print(f"  Target: -1.67 (Kolmogorov)")
        print(f"  Error: {abs(slope + 5/3):.3f}")


def compare_xi_variations():
    """How does XI affect the spectrum?"""
    print("\n" + "="*70)
    print("XI VARIATIONS AND SPECTRUM")
    print("="*70)
    
    from bifractal_pac_mesh import BiFractalPACMesh
    
    for xi in [0.5, 0.75, 1.0, XI, 1.25, 1.5, 2.0]:
        mesh_gen = BiFractalPACMesh(H=1.0, xi=xi)
        pac_mesh = mesh_gen.generate_bifractal_mesh(depth=7)
        
        spacings = np.diff(pac_mesh)
        k = 1.0 / spacings
        E = spacings**2
        
        sort_idx = np.argsort(k)
        log_k = np.log10(k[sort_idx])
        log_E = np.log10(E[sort_idx])
        
        n = len(log_k)
        coeffs = np.polyfit(log_k[n//4:3*n//4], log_E[n//4:3*n//4], 1)
        slope = coeffs[0]
        
        marker = " <-- actual XI" if abs(xi - XI) < 0.01 else ""
        kolm_marker = " <-- Kolmogorov target" if abs(xi - 0.56) < 0.1 else ""
        print(f"XI = {xi:.4f}: slope = {slope:.3f}{marker}{kolm_marker}")


def deep_3d_analysis():
    """Detailed 3D analysis with varying XI"""
    print("\n" + "="*70)
    print("DEEP 3D ANALYSIS: DOES DIMENSIONALITY CHANGE THE LAW?")
    print("="*70)
    
    print("\nComparing 1D, 2D, 3D PAC structures...")
    
    depth = 5
    
    # 1D: Binary tree
    from bifractal_pac_mesh import BiFractalPACMesh
    mesh_1d = BiFractalPACMesh(H=1.0).generate_bifractal_mesh(depth)
    spacings_1d = np.diff(mesh_1d)
    k_1d = 1.0 / spacings_1d
    E_1d = spacings_1d**2
    slope_1d = np.polyfit(np.log10(np.sort(k_1d)), 
                          np.log10(E_1d[np.argsort(k_1d)]), 1)[0]
    
    # 2D: Quadtree
    class Quadtree:
        def __init__(self, xi=XI):
            self.xi = xi
            self.positions = []
            
        def generate(self, bounds, depth, level=0):
            if level >= depth:
                cx = (bounds[0] + bounds[1]) / 2
                cy = (bounds[2] + bounds[3]) / 2
                self.positions.append((cx, cy))
                return
            
            mx = (bounds[0] + bounds[1]) / 2
            my = (bounds[2] + bounds[3]) / 2
            
            # Four quadrants
            self.generate((bounds[0], mx, bounds[2], my), depth, level+1)
            self.generate((mx, bounds[1], bounds[2], my), depth, level+1)
            self.generate((bounds[0], mx, my, bounds[3]), depth, level+1)
            self.generate((mx, bounds[1], my, bounds[3]), depth, level+1)
    
    qt = Quadtree()
    qt.generate((-1, 1, -1, 1), depth)
    pos_2d = np.array(qt.positions)
    
    # Compute 2D distances
    dists_2d = []
    for i in range(len(pos_2d)):
        for j in range(i+1, len(pos_2d)):
            dists_2d.append(np.linalg.norm(pos_2d[i] - pos_2d[j]))
    dists_2d = np.array(dists_2d)
    k_2d = 1.0 / dists_2d
    E_2d = dists_2d**2
    slope_2d = np.polyfit(np.log10(np.sort(k_2d)), 
                          np.log10(E_2d[np.argsort(k_2d)]), 1)[0]
    
    # 3D: Octree
    octree = PACOctree3D()
    octree.generate_octree(((-1, 1), (-1, 1), (-1, 1)), depth)
    result_3d = octree.compute_3d_spectrum()
    slope_3d = result_3d['slope']
    
    print(f"\n1D Binary Tree (depth={depth}):")
    print(f"  Points: {len(mesh_1d)}")
    print(f"  Spectral slope: {slope_1d:.3f}")
    
    print(f"\n2D Quadtree (depth={depth}):")
    print(f"  Points: {len(pos_2d)}")
    print(f"  Spectral slope: {slope_2d:.3f}")
    
    print(f"\n3D Octree (depth={depth}):")
    print(f"  Points: {result_3d['n_points']}")
    print(f"  Spectral slope: {slope_3d:.3f}")
    
    print("\n" + "-"*50)
    print("THEORETICAL PREDICTIONS:")
    print(f"  Binary tree (any D):  k^(-2)")
    print(f"  3D Kolmogorov:        k^(-5/3) = k^(-1.67)")
    print(f"  2D enstrophy:         k^(-3)")
    print("-"*50)
    
    print("\nKEY FINDING:")
    if abs(slope_1d - slope_2d) < 0.2 and abs(slope_2d - slope_3d) < 0.2:
        print("  All dimensions give similar slopes!")
        print("  The k^(-2) law is TOPOLOGICAL, not dimensional.")
        print("  It comes from binary branching, not spatial dimension.")
    else:
        print(f"  Slopes vary: 1D={slope_1d:.2f}, 2D={slope_2d:.2f}, 3D={slope_3d:.2f}")
        print("  Dimensionality DOES affect the spectrum!")


if __name__ == "__main__":
    analyze_branching_vs_spectrum()
    analyze_3d_octree()
    analyze_kolmogorov_tree()
    compare_xi_variations()
    deep_3d_analysis()

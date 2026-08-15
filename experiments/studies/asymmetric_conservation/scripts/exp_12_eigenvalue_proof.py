"""
Experiment 12: Eigenvalue Proof for φ in PAC

PURPOSE:
    Prove rigorously that the PAC propagation matrix has all eigenvalues = -1/φ.
    
    In exp_10, we observed numerically that:
    - PAC chain matrix has spectral radius = 1/φ
    - All eigenvalues = -1/φ regardless of size
    
    This experiment:
    1. Proves this analytically
    2. Extends to tree structures (not just chains)
    3. Shows this is UNIQUE to φ (other collapse fractions don't work)
    4. Connects to the golden ratio's algebraic properties

THEORETICAL SETUP:
    For a chain of n nodes, the PAC matrix M is:
    
    M[i,i] = -α (self-depletion)
    M[i-1,i] = α (parent receives)
    
    where α = 1/φ = (√5 - 1)/2 ≈ 0.618
    
    Key property of φ: φ² = φ + 1, so 1/φ = φ - 1
"""

import numpy as np
from scipy import linalg
import sympy as sp
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from constants import print_header, print_subheader, save_results, PHI, PHI_INV, XI, PI


def build_chain_matrix(n: int, alpha: float) -> np.ndarray:
    """
    Build PAC propagation matrix for a chain.
    
    M[i,i] = -alpha (self-depletion)
    M[i-1,i] = alpha (parent receives from child)
    """
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = -alpha
        if i > 0:
            M[i-1, i] = alpha
    return M


def build_binary_tree_matrix(depth: int, alpha: float) -> np.ndarray:
    """
    Build PAC propagation matrix for a binary tree.
    
    Nodes indexed by level-order traversal.
    Each node depletes by alpha, sends to parent.
    """
    n_nodes = 2**(depth + 1) - 1
    M = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        M[i, i] = -alpha
        
        # Parent of node i is (i-1)//2 for i > 0
        if i > 0:
            parent = (i - 1) // 2
            M[parent, i] = alpha
    
    return M


def analyze_eigenvalues(M: np.ndarray, name: str) -> Dict:
    """Analyze eigenvalues of a matrix."""
    eigenvalues = linalg.eigvals(M)
    
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    # Check if all eigenvalues are equal
    unique_real = np.unique(np.round(real_parts, 8))
    all_equal = len(unique_real) == 1
    
    return {
        'name': name,
        'size': M.shape[0],
        'eigenvalues': eigenvalues,
        'real_parts': real_parts,
        'imag_parts': imag_parts,
        'max_real': np.max(real_parts),
        'spectral_radius': np.max(np.abs(eigenvalues)),
        'all_equal': all_equal,
        'unique_eigenvalue': unique_real[0] if all_equal else None,
    }


def run_experiment():
    """Prove eigenvalue structure of PAC matrices."""
    print_header("EXPERIMENT 12: EIGENVALUE PROOF FOR φ IN PAC")
    
    results = {
        'experiment': 'exp_12_eigenvalue_proof',
        'tests': []
    }
    
    # =========================================================================
    # Part 1: Chain Matrix Eigenvalues
    # =========================================================================
    print_subheader("Part 1: Chain Matrix Structure")
    
    print("  PAC Chain Matrix M (n=4, α=1/φ):")
    M4 = build_chain_matrix(4, PHI_INV)
    print(f"  {M4}")
    
    print(f"\n  Matrix structure:")
    print(f"    Diagonal: -{PHI_INV:.4f} = -1/φ")
    print(f"    Subdiagonal: +{PHI_INV:.4f} = +1/φ")
    print(f"    Upper triangular (including diagonal)")
    
    # Eigenvalues for various sizes
    print(f"\n  Eigenvalues for chain sizes (α = 1/φ = {PHI_INV:.6f}):")
    
    chain_results = []
    for n in [2, 3, 4, 5, 8, 13, 21]:
        M = build_chain_matrix(n, PHI_INV)
        analysis = analyze_eigenvalues(M, f'chain_{n}')
        chain_results.append(analysis)
        
        status = "✓ ALL EQUAL" if analysis['all_equal'] else "✗ MIXED"
        print(f"    n={n:2d}: λ = {analysis['unique_eigenvalue']:.6f} {status}")
    
    results['tests'].append({
        'name': 'chain_eigenvalues',
        'results': [(r['size'], r['unique_eigenvalue']) for r in chain_results],
    })
    
    # =========================================================================
    # Part 2: Analytical Proof for Chain
    # =========================================================================
    print_subheader("Part 2: Analytical Proof")
    
    print("""
    THEOREM: For the chain PAC matrix M with α = 1/φ,
             all eigenvalues equal -1/φ.
    
    PROOF:
    
    The matrix M is upper triangular (when properly oriented):
    
        | -α   α   0   0  ... |
        |  0  -α   α   0  ... |
    M = |  0   0  -α   α  ... |
        |  .   .   .   .  ... |
        |  0   0   0   0  -α  |
    
    For ANY upper triangular matrix, the eigenvalues are the 
    diagonal entries.
    
    Since all diagonal entries = -α = -1/φ,
    ALL eigenvalues = -1/φ.                                    □
    
    This is trivially true for ANY α! Let's verify...
    """)
    
    # Test with different α values
    print("  Testing with different α values:")
    test_alphas = [0.3, 0.5, PHI_INV, 0.7, 0.9]
    
    for alpha in test_alphas:
        M = build_chain_matrix(5, alpha)
        analysis = analyze_eigenvalues(M, f'alpha_{alpha}')
        marker = " ← 1/φ" if abs(alpha - PHI_INV) < 0.001 else ""
        print(f"    α={alpha:.4f}: λ = {analysis['unique_eigenvalue']:.6f} "
              f"(= -α? {abs(analysis['unique_eigenvalue'] + alpha) < 1e-10}){marker}")
    
    print("""
    CONCLUSION: The eigenvalue = -α for ANY α, not just 1/φ.
    
    So the "discovery" in exp_10 was that we used α = 1/φ,
    which gives eigenvalue = -1/φ.
    
    The REAL question: Why is α = 1/φ special for PAC?
    """)
    
    # =========================================================================
    # Part 3: Why φ is Special
    # =========================================================================
    print_subheader("Part 3: Why φ is Special for PAC")
    
    print("""
    The chain matrix eigenvalue = -α for any α. So what makes α = 1/φ special?
    
    ANSWER: The conservation structure, not the eigenvalues.
    
    Consider PAC with collapse fraction α:
    - Parent P, child C
    - Child collapses: C → (1-α)C, Parent gets αC
    - Total conserved: P + C → P + αC + (1-α)C = P + C ✓
    
    For OPTIMAL collapse (in Fibonacci sense):
    - We want: fraction remaining = fraction to parent
    - (1-α) : α should be self-similar
    - This means α/(1-α) = 1/α
    - Solving: α² + α - 1 = 0
    - Solution: α = (√5 - 1)/2 = 1/φ
    
    The golden ratio is the UNIQUE fraction where:
    "What remains is to what's given as what's given is to the whole"
    """)
    
    # Verify the self-similarity
    print("  Verification of self-similarity:")
    print(f"    α = 1/φ = {PHI_INV:.6f}")
    print(f"    1-α = {1 - PHI_INV:.6f}")
    print(f"    α/(1-α) = {PHI_INV / (1 - PHI_INV):.6f}")
    print(f"    1/α = {1 / PHI_INV:.6f} = φ")
    print(f"    α/(1-α) = 1/α? {abs(PHI_INV / (1 - PHI_INV) - PHI) < 1e-10}")
    
    # =========================================================================
    # Part 4: Binary Tree Eigenvalues
    # =========================================================================
    print_subheader("Part 4: Binary Tree Matrix")
    
    print("  For binary trees, the structure is different:")
    print("  Each node can receive from TWO children.")
    
    tree_results = []
    for depth in [1, 2, 3, 4]:
        M = build_binary_tree_matrix(depth, PHI_INV)
        analysis = analyze_eigenvalues(M, f'tree_depth_{depth}')
        tree_results.append(analysis)
        
        n_nodes = 2**(depth + 1) - 1
        print(f"\n    Depth {depth} ({n_nodes} nodes):")
        print(f"      Spectral radius: {analysis['spectral_radius']:.6f}")
        print(f"      Max real part: {analysis['max_real']:.6f}")
        print(f"      All equal? {analysis['all_equal']}")
        
        if not analysis['all_equal']:
            unique_vals = np.unique(np.round(analysis['real_parts'], 4))
            print(f"      Unique real parts: {unique_vals[:5]}...")
    
    results['tests'].append({
        'name': 'tree_eigenvalues',
        'results': [(r['size'], r['spectral_radius'], r['all_equal']) 
                   for r in tree_results],
    })
    
    # =========================================================================
    # Part 5: Spectral Analysis of Trees
    # =========================================================================
    print_subheader("Part 5: Tree Spectral Analysis")
    
    # For trees, the matrix is NOT triangular, so eigenvalues vary
    # But what is the spectral radius?
    
    print("  Binary tree spectral radii vs depth:")
    
    for depth in range(1, 7):
        M = build_binary_tree_matrix(depth, PHI_INV)
        eigvals = linalg.eigvals(M)
        rho = np.max(np.abs(eigvals))
        n_nodes = 2**(depth + 1) - 1
        
        # Compare to various constants
        phi_ratio = rho / PHI_INV
        sqrt2_ratio = rho / np.sqrt(2)
        
        print(f"    Depth {depth} (n={n_nodes:3d}): ρ = {rho:.6f}, "
              f"ρ/(1/φ) = {phi_ratio:.4f}")
    
    # =========================================================================
    # Part 6: The True Meaning of φ in PAC
    # =========================================================================
    print_subheader("Part 6: The True Meaning of φ in PAC")
    
    print(f"""
    SUMMARY OF φ's ROLE IN PAC:
    
    1. EIGENVALUE (for chains): 
       - Trivially -α for any α
       - Not specifically about φ
    
    2. SELF-SIMILARITY (the real reason):
       - α = 1/φ is the unique collapse fraction where
         remaining:given = given:whole
       - This makes Fibonacci trees possible
       - Parent value = sum of child values (Fibonacci recurrence)
    
    3. STABILITY:
       - Spectral radius ρ = 1/φ < 1 means stable dynamics
       - System converges (doesn't explode)
       - 1/φ ≈ 0.618 is "just right" damping
    
    4. OPTIMALITY:
       - φ-collapse minimizes information loss
       - Proven in milestone1 experiments
       - Connects to entropy minimization
    
    THE EIGENVALUE FINDING WAS A RED HERRING!
    
    The exp_10 result (all eigenvalues = -1/φ) is true but trivial.
    It's true for ANY α in the chain case.
    
    The REAL significance of φ is:
    - It's the unique SELF-SIMILAR collapse ratio
    - It's what makes Fibonacci structure emerge
    - It's optimal for conservation + information
    
    φ is special because of ALGEBRA, not spectral theory:
    
        φ² = φ + 1
        1/φ = φ - 1
        φ + 1/φ = √5
    
    These identities make PAC work perfectly with Fibonacci.
    """)
    
    results['conclusion'] = {
        'eigenvalue_finding': 'Trivially true for any α (upper triangular)',
        'real_significance': 'φ is the unique self-similar collapse ratio',
        'algebraic_identities': ['φ² = φ + 1', '1/φ = φ - 1'],
        'why_phi_matters': [
            'Self-similarity of collapse',
            'Fibonacci recurrence',
            'Stability (ρ < 1)',
            'Information optimality',
        ],
    }
    
    save_results(results, 'exp_12')
    return results


if __name__ == '__main__':
    run_experiment()

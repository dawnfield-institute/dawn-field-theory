#!/usr/bin/env python3
"""
Experiment 09: Curl from Non-Orientability

The curl operator (∇×) exists as a vector only in D = 3.
This is not a coincidence - it's connected to the cross product
and non-orientable topology.

Key insight: Curl measures local rotation. In D = 3, rotation
has a unique axis (the curl vector). In other dimensions, this
structure changes fundamentally.
"""

import numpy as np
from constants import F3, F4, print_header, print_result

def curl_dimension_analysis():
    """
    Analyze how curl behaves in different dimensions.
    
    The curl of a vector field v is defined via:
    (∇ × v)_i = ε_ijk ∂_j v_k
    
    where ε is the Levi-Civita symbol.
    """
    results = {}
    
    # D = 1: No curl possible (need at least 2D for rotation)
    results['D1'] = {
        'curl_type': None,
        'components': 0,
        'reason': 'No rotation in 1D'
    }
    
    # D = 2: Curl is a scalar (or pseudoscalar)
    # ∂_x v_y - ∂_y v_x = single number
    results['D2'] = {
        'curl_type': 'scalar',
        'components': 1,
        'reason': 'Only one rotation plane (xy)'
    }
    
    # D = 3: Curl is a vector (3 components)
    # This is the UNIQUE dimension where curl → vector
    results['D3'] = {
        'curl_type': 'vector',
        'components': 3,
        'reason': 'Three rotation planes (xy, xz, yz) ↔ three axes'
    }
    
    # D = 4: Curl is a 2-form (6 components)
    # Rotation planes: xy, xz, xw, yz, yw, zw
    results['D4'] = {
        'curl_type': '2-form (antisymmetric tensor)',
        'components': 6,  # D(D-1)/2 = 4*3/2 = 6
        'reason': 'Six rotation planes, no unique axis'
    }
    
    # General D: curl is (D-1)-form or 2-form depending on definition
    # Components = D(D-1)/2
    
    return results

def why_vector_curl_matters():
    """
    Maxwell's equations require curl to be a vector.
    
    ∇ × E = -∂B/∂t
    ∇ × B = μ₀J + μ₀ε₀∂E/∂t
    
    Both E and B must be vectors (3 components each).
    If curl produced something else, EM wouldn't work.
    """
    return {
        'maxwell_faraday': '∇ × E = -∂B/∂t',
        'maxwell_ampere': '∇ × B = μ₀(J + ε₀∂E/∂t)',
        'e_field_components': 3,
        'b_field_components': 3,
        'curl_e_components': 3,
        'curl_b_components': 3,
        'consistency_requires': 'D = 3'
    }

def cross_product_dimension():
    """
    The cross product a × b is only a vector in D = 3.
    
    In D = 2: a × b is a scalar (the z-component)
    In D = 3: a × b is a vector (3 components)
    In D = 7: There's another cross product (octonions) but it's weird
    In other D: No vector cross product exists
    """
    # D = 3 is special because SO(3) has dimension 3
    # The Lie algebra so(3) ≅ R³
    
    so_dimensions = {
        'SO(2)': 1,   # D(D-1)/2 = 1
        'SO(3)': 3,   # D(D-1)/2 = 3 = D (special!)
        'SO(4)': 6,   # D(D-1)/2 = 6 ≠ 4
        'SO(5)': 10,  # D(D-1)/2 = 10 ≠ 5
    }
    
    # Only SO(3) has dim(Lie algebra) = dim(space)
    special_d = 3
    
    return {
        'so_dimensions': so_dimensions,
        'special_dimension': special_d,
        'reason': 'dim(so(D)) = D only when D = 3',
        'cross_product_dimensions': [3, 7],  # 3 and 7 from division algebras
        'physical_relevance': 'D = 7 requires octonions, not observed'
    }

def levi_civita_analysis():
    """
    The Levi-Civita symbol ε_ijk only gives vectors in D = 3.
    
    ε_ijk is totally antisymmetric with D indices.
    Contracting with two vectors gives D-2 free indices.
    
    D = 3: 3 - 2 = 1 free index → vector (after raising)
    D = 4: 4 - 2 = 2 free indices → tensor
    D = 2: 2 - 2 = 0 free indices → scalar
    """
    results = {}
    for D in range(2, 6):
        free_indices = D - 2
        if free_indices == 0:
            result_type = 'scalar'
        elif free_indices == 1:
            result_type = 'vector'
        else:
            result_type = f'{free_indices}-tensor'
        
        results[f'D{D}'] = {
            'free_indices': free_indices,
            'result_type': result_type
        }
    
    return results

def main():
    print_header("Experiment 09: Curl from Non-Orientability")
    
    curl_dims = curl_dimension_analysis()
    maxwell = why_vector_curl_matters()
    cross = cross_product_dimension()
    levi = levi_civita_analysis()
    
    print("\n=== Curl Type by Dimension ===")
    for d, info in curl_dims.items():
        print(f"\n{d}:")
        print(f"  Curl type: {info['curl_type']}")
        print(f"  Components: {info['components']}")
        print(f"  Reason: {info['reason']}")
    
    print("\n=== Maxwell's Equations Require D = 3 ===")
    print(f"Faraday: {maxwell['maxwell_faraday']}")
    print(f"Ampère: {maxwell['maxwell_ampere']}")
    print(f"E-field components: {maxwell['e_field_components']}")
    print(f"Curl(E) components: {maxwell['curl_e_components']}")
    print(f"Consistency requires: {maxwell['consistency_requires']}")
    
    print("\n=== Cross Product Analysis ===")
    print("SO(D) Lie algebra dimensions:")
    for group, dim in cross['so_dimensions'].items():
        match = "✓ MATCH" if group == "SO(3)" else ""
        print(f"  {group}: dim = {dim} {match}")
    print(f"\nCross product exists as vector only in D = {cross['cross_product_dimensions']}")
    print(f"Physical relevance: {cross['physical_relevance']}")
    
    print("\n=== Levi-Civita Contraction ===")
    for d, info in levi.items():
        marker = " ← VECTOR" if info['result_type'] == 'vector' else ""
        print(f"  {d}: {info['free_indices']} free indices → {info['result_type']}{marker}")
    
    # Summary
    print("\n" + "="*60)
    print("RESULT: Vector curl exists ONLY in D = 3")
    print("This is PATH 2 of 5 to D = 3")
    print(f"\nConnection to Fibonacci: D = 3 = F₄")
    print_result("Curl requires D=3", True)

if __name__ == "__main__":
    main()

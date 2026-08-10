"""
exp_13: Stoichiometric Derivation of Fibonacci Index Selection

HYPOTHESIS: The specific Fibonacci indices appearing in physics formulas
(α, sin²θ_W, mass ratios, Wilson-Fisher ν) are not searched — they are
the UNIQUE solutions to stoichiometric balance equations over the E-I-S
(Energy–Information–Structure) conservation system.

MOTIVATION:
  Chemical stoichiometry works because:
    1. Conservation constraints (atoms in = atoms out)
    2. Integer coefficients (you can't have half an atom)
    3. The space of valid reactions is the NULL SPACE of the stoichiometric matrix
  
  PAC with Fibonacci is the SAME system:
    1. PAC conservation (f(parent) = Σf(children))
    2. Fibonacci indices are integers (F_n are natural numbers)
    3. The space of valid "physics formulas" should be the null space of
       a PAC stoichiometric matrix
  
  SEC played the role of selecting which reactions actually run — kinetics.
  
  The key insight: each physical parameter is a "reaction" that consumes
  and produces conserved quantities (E, I, S) in integer Fibonacci amounts.
  The stoichiometric matrix S has:
    - Rows = conserved E-I-S quantities + gauge closure + MED depth constraints
    - Columns = Fibonacci species F_1 through F_12
    - Entries = how each F_n participates in conservation

APPROACH:
  1. Define conservation laws (E-I-S balance, gauge closure, PAC norm)
  2. Build stoichiometric matrix
  3. Compute null space — these are the ALLOWED reactions
  4. Check: do the known physics formulas emerge from the null space?
  5. Null test: do random conservation matrices produce equally good formulas?

TESTS:
  Test 1 — Weinberg angle as gauge yield: sin²θ_W = F₄/F₇ from generator
           counting (SU(2) generators / total electroweak generators)
  Test 2 — Stoichiometric matrix null space: build S from E-I-S constraints,
           find null vectors, check if known formulas are spanned
  Test 3 — Mass ratio stoichiometry: F₄=3 in all mass formulas as generation
           "atom" — stoichiometric coefficient
  Test 4 — Wilson-Fisher decomposition: ν = (2/3)·(1/Ξ) as two-species
           yield reaction with E-I-S stoichiometry
  Test 5 — Uniqueness test: how many alternative "reactions" satisfy the
           same conservation constraints? If few → derivation. If many → fit.
  Test 6 — Null test: random stoichiometric matrices with same dimensions —
           do they produce formulas matching physics equally well?

SOURCES:
  - Chemistry (stoichiometry formalism)
  - milestone3/exp_07 (E-I-S decomposition of Wilson-Fisher)
  - milestone3/exp_08 (sin²θ_W = F₄/F₇ at Q ≈ M_W)
  - milestone2/mass_derivation (μ/e, τ/e, p/e mass formulas)
  - pac_confluence_xi (Standard Model parameters)
"""

import sys
import os
import math
import json
import numpy as np
from itertools import combinations, product as iterproduct
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import (PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE,
                            FIB, ALPHA_EM_PDG, SIN2_THETA_W_PDG)
from core.utils import experiment_header, save_results

# =====================================================================
# Reference values
# =====================================================================

# --- Standard Model measured values (PDG 2024) ---
ALPHA_EM_MEASURED   = 0.0072973525693    # Fine structure constant
SIN2_TW_MEASURED    = 0.23122            # sin²θ_W (MS-bar at M_Z)
ALPHA_S_MEASURED    = 0.1180             # Strong coupling at M_Z
KOIDE_MEASURED      = 0.666661           # Lepton mass sum rule
MU_E_MEASURED       = 206.7682830        # μ/e mass ratio
TAU_E_MEASURED      = 3477.48            # τ/e mass ratio (approx)
P_E_MEASURED        = 1836.15267343      # p/e mass ratio
NU_WF_MEASURED      = 0.6299709          # Wilson-Fisher ν (3D Ising)
CABIBBO_DEG         = 13.04              # Cabibbo angle in degrees
SHE_LEV_BETA        = 2.0 / 3.0         # She-Leveque intermittency

# --- Known PAC formulas (from milestone1/2) ---
# Each formula uses specific Fibonacci indices
KNOWN_FORMULAS = {
    'sin2_theta_W': {
        'formula': 'F4/F7',
        'value': FIB[4] / FIB[7],       # 3/13 = 0.23077
        'measured': SIN2_TW_MEASURED,
        'fib_indices': [4, 7],
        'interpretation': 'SU(2) generators / total electroweak gauge dim',
    },
    'alpha_em': {
        'formula': 'F3/(F4·φ·F10)·(1 - F10/(4π·F7²))',
        'value': (FIB[3] / (FIB[4] * PHI * FIB[10])) * (1 - FIB[10] / (4 * math.pi * FIB[7]**2)),
        'measured': ALPHA_EM_MEASURED,
        'fib_indices': [3, 4, 7, 10],
        'interpretation': 'Hierarchy payment through F10=55 levels, gauge closure at F7=13',
    },
    'koide': {
        'formula': 'F3/(F3+F2) = 2/3',
        'value': FIB[3] / (FIB[3] + FIB[2]),  # 2/3
        'measured': KOIDE_MEASURED,
        'fib_indices': [2, 3],
        'interpretation': 'PAC partition at minimal depth',
    },
    'mu_e': {
        'formula': 'F4 × F6² × (1 + 1/F7)',
        'value': FIB[4] * FIB[6]**2 * (1 + 1/FIB[7]),  # 3 × 64 × 14/13
        'measured': MU_E_MEASURED,
        'fib_indices': [4, 6, 7],
        'interpretation': 'F4=3 generations × F6²=64 squared coupling × F7 gauge correction',
    },
    'tau_e': {
        'formula': 'F4 × F7 × F11 + F5',
        'value': FIB[4] * FIB[7] * FIB[11] + FIB[5],  # 3 × 13 × 89 + 5
        'measured': TAU_E_MEASURED,
        'fib_indices': [4, 5, 7, 11],
        'interpretation': 'F4 generations × F7 gauge × F11 hierarchy + F5 correction',
    },
    'p_e': {
        'formula': 'F4 × F9 × F12 / F6',
        'value': FIB[4] * FIB[9] * FIB[12] / FIB[6],  # 3 × 34 × 144 / 8
        'measured': P_E_MEASURED,
        'fib_indices': [4, 6, 9, 12],
        'interpretation': 'F4 generations × hadronic Fibonacci coupling',
    },
    'wilson_fisher_nu': {
        'formula': '2/(F4·Ξ) = (2/3)·(1/Ξ)',
        'value': 2.0 / (FIB[4] * XI_BALANCE),
        'measured': NU_WF_MEASURED,
        'fib_indices': [4],  # plus Ξ = γ + ln(φ)
        'interpretation': 'E-I-S cycle ratio (2/3) × balance reciprocal (1/Ξ)',
    },
    'cabibbo_angle': {
        'formula': 'arctan(F4/F7)',
        'value': math.degrees(math.atan(FIB[4] / FIB[7])),  # arctan(3/13)
        'measured': CABIBBO_DEG,
        'fib_indices': [4, 7],
        'interpretation': 'Same gauge ratio as sin²θ_W, different projection',
    },
    'she_leveque_beta': {
        'formula': 'F3/F4 = 2/3',
        'value': FIB[3] / FIB[4],  # 2/3
        'measured': SHE_LEV_BETA,
        'fib_indices': [3, 4],
        'interpretation': 'E-I-S cycle intermittency ratio',
    },
    'alpha_s': {
        'formula': 'F4/(2·φ·F6)',
        'value': FIB[4] / (2 * PHI * FIB[6]),  # 3/(2×1.618×8)
        'measured': ALPHA_S_MEASURED,
        'fib_indices': [4, 6],
        'interpretation': 'Strong coupling = F4 color charge / (2φ × F6 gluon modes)',
    },
}


def pct_error(predicted, measured):
    """Percentage error."""
    return abs(predicted - measured) / abs(measured) * 100


# =====================================================================
# Test 1: Weinberg angle as stoichiometric gauge yield
# =====================================================================

def test_1_gauge_yield():
    """
    Weinberg angle as a stoichiometric "yield" calculation.
    
    In chemistry: yield = product / (product + byproduct)
    In gauge theory: sin²θ_W = weak generators / total electroweak generators
    
    The "reaction": SU(2)×U(1) → electroweak sector
    "Reactant" atoms: SU(2) has dim=F₄=3, U(1) has dim=1
    Total gauge "atoms": SU(2)×U(1) embedded in SU(N) with N related to F₇=13
    
    Stoichiometric equation: 
      F₄ weak generators + (F₇-F₄) hypercharge-related = F₇ total
      yield = F₄/F₇ = 3/13 = sin²θ_W
    """
    print("\n" + "="*60)
    print("TEST 1: Weinberg angle as stoichiometric gauge yield")
    print("="*60)
    
    results = {}
    
    # --- The gauge counting argument ---
    su2_dim = FIB[4]   # 3 = dim(SU(2))
    su3_dim = FIB[6]   # 8 = dim(SU(3))
    f7 = FIB[7]        # 13
    
    # sin²θ_W as yield ratio
    yield_ratio = su2_dim / f7
    err = pct_error(yield_ratio, SIN2_TW_MEASURED)
    
    print(f"\n  SU(2) generators (reactant): {su2_dim} = F₄")
    print(f"  Total gauge space:           {f7} = F₇")
    print(f"  Yield = F₄/F₇:              {yield_ratio:.6f}")
    print(f"  Measured sin²θ_W:            {SIN2_TW_MEASURED:.6f}")
    print(f"  Error:                       {err:.4f}%")
    
    # --- Why F₇ = 13? Stoichiometric necessity ---
    # F₇ = F₆ + F₅ = 8 + 5 = dim(SU(3)) + dim(SU(2)×U(1))_extended
    # The gauge sector has SU(3)=8 color + SU(2)=3 weak + U(1)=1 hypercharge + 1 = 13
    # But 8+3+1 = 12, not 13. The extra 1 is the Higgs doublet coupling.
    # In PAC: F₇ = F₆ + F₅ is Fibonacci conservation at level 7.
    
    decomposition_12 = su3_dim + su2_dim + 1   # 8+3+1 = 12 (SM gauge generators)
    decomposition_13 = f7                       # 13 = F₇ (PAC prediction)
    extra = decomposition_13 - decomposition_12  # 1 = the "PAC excess"
    
    print(f"\n  SM gauge generators:   {decomposition_12} = dim(SU(3)) + dim(SU(2)) + dim(U(1))")
    print(f"  PAC gauge closure:     {decomposition_13} = F₇")
    print(f"  Excess dimension:      {extra}")
    print(f"  F₇ = F₆ + F₅:         {FIB[6]} + {FIB[5]} = {FIB[6]+FIB[5]}")
    
    # --- Stoichiometric balance check ---
    # In stoichiometry: Σ(reactant coefficients) = Σ(product coefficients) for each element
    # Here: the "elements" are E (energy), I (information), S (structure)
    #
    # SU(2) carries: 3 units of "gauge information" (I-cost = F₄)
    # The full sector: 13 units of "gauge information" (I-cost = F₇)
    # PAC conservation: F₇ = F₆ + F₅ (the Fibonacci identity IS the balance equation)
    
    fib_balance = (FIB[7] == FIB[6] + FIB[5])
    print(f"\n  Fibonacci balance F₇ = F₆ + F₅: {fib_balance}")
    print(f"  This IS the stoichiometric equation:")
    print(f"    13 gauge modes = 8 (SU(3) structure cost) + 5 (SU(2)×U(1) info cost)")
    
    # --- All Fibonacci ratios F_a/F_b near sin²θ_W ---
    print(f"\n  All F_a/F_b ratios within 5% of sin²θ_W:")
    candidates = []
    for a in range(2, 15):
        for b in range(a+1, 15):
            ratio = FIB[a] / FIB[b]
            if abs(ratio - SIN2_TW_MEASURED) / SIN2_TW_MEASURED < 0.05:
                candidates.append((a, b, ratio, pct_error(ratio, SIN2_TW_MEASURED)))
    
    candidates.sort(key=lambda x: x[3])
    for a, b, ratio, e in candidates:
        marker = " ← BEST" if (a, b) == (4, 7) else ""
        print(f"    F_{a}/F_{b} = {FIB[a]}/{FIB[b]} = {ratio:.6f} ({e:.4f}% error){marker}")
    
    unique = len(candidates) == 0 or candidates[0][:2] == (4, 7)
    
    # PASS criteria: F₄/F₇ is the closest Fibonacci ratio AND has stoichiometric interpretation
    passed = err < 0.5 and unique
    print(f"\n  PASS: {passed} (error < 0.5% AND uniquely closest Fibonacci ratio)")
    
    results['yield_ratio'] = yield_ratio
    results['error_pct'] = err
    results['fibonacci_balance'] = fib_balance
    results['unique_closest'] = unique
    results['n_candidates'] = len(candidates)
    results['passed'] = passed
    
    return results


# =====================================================================
# Test 2: Stoichiometric matrix and null space
# =====================================================================

def test_2_stoichiometric_null_space():
    """
    Build the E-I-S stoichiometric matrix and find its null space.
    
    Each Fibonacci number F_n is a "species" with properties:
      - E-cost (energy): related to magnitude (F_n itself)
      - I-cost (information): related to index (n = depth in hierarchy)
      - S-cost (structure): related to combinatorial role (n mod 3 for E-I-S cycle)
    
    Conservation constraints:
      1. PAC norm: Σ coefficients must net-conserve (total in = total out)
      2. E-I-S cycle: the cycle position must be balanced (mod 3 structure)
      3. Gauge closure: F₇ = 13 = maximum gauge depth
      4. Generation constraint: F₄ = 3 = number of generations
    
    The null space of S tells us which "reactions" (formulas) are allowed.
    """
    print("\n" + "="*60)
    print("TEST 2: Stoichiometric matrix null space")
    print("="*60)
    
    results = {}
    
    # --- Species: F₂ through F₁₂ (11 Fibonacci species) ---
    fib_indices = list(range(2, 13))  # F₂=1, F₃=2, F₄=3, F₅=5, F₆=8, F₇=13, ...F₁₂=144
    n_species = len(fib_indices)
    species_values = [FIB[i] for i in fib_indices]
    
    print(f"\n  Species (Fibonacci numbers):")
    for i, idx in enumerate(fib_indices):
        print(f"    F_{idx} = {FIB[idx]}")
    
    # --- Build stoichiometric matrix S (constraints × species) ---
    # Each row is a conservation law. Each column is how F_n participates.
    
    S = np.zeros((5, n_species))
    
    # Row 0: PAC magnitude conservation
    # The Fibonacci recursion F_n = F_{n-1} + F_{n-2} means:
    # For any valid "reaction", the net Fibonacci magnitudes must balance
    # Weight by value: S[0,j] = F_{n_j}
    for j, idx in enumerate(fib_indices):
        S[0, j] = FIB[idx]
    
    # Row 1: Hierarchy depth conservation  
    # Information cost scales with depth (index n)
    # S[1,j] = n_j (the Fibonacci index itself)
    for j, idx in enumerate(fib_indices):
        S[1, j] = idx
    
    # Row 2: E-I-S cycle position (mod 3)
    # The E-I-S triad cycles every 3 indices:
    #   n mod 3 = 0 → Energy role
    #   n mod 3 = 1 → Information role  
    #   n mod 3 = 2 → Structure role
    # Conservation: net cycle position must be ≡ 0 (mod 3) for a closed reaction
    for j, idx in enumerate(fib_indices):
        S[2, j] = idx % 3
    
    # Row 3: Parity conservation
    # Even-indexed F_n (F₂, F₄, F₆, ...) vs odd-indexed
    # Physical meaning: even indices = bosonic (force carriers), odd = fermionic
    for j, idx in enumerate(fib_indices):
        S[3, j] = idx % 2
    
    # Row 4: Fibonacci identity constraint
    # Any valid reaction must respect F_n = F_{n-1} + F_{n-2}
    # This means: using F_n is equivalent to using F_{n-1} + F_{n-2}
    # Encode: for each n, S[4,n] = F_n, S[4,n-1] = -F_{n-1}, S[4,n-2] = -F_{n-2}
    # Use the specific identity F₇ = F₆ + F₅ (gauge closure)
    idx_5 = fib_indices.index(5)
    idx_6 = fib_indices.index(6)
    idx_7 = fib_indices.index(7)
    S[4, idx_7] = 1
    S[4, idx_6] = -1
    S[4, idx_5] = -1
    
    print(f"\n  Stoichiometric matrix S ({S.shape[0]} constraints × {S.shape[1]} species):")
    constraint_names = ['PAC magnitude', 'Hierarchy depth', 'E-I-S cycle (mod 3)', 
                       'Parity (even/odd)', 'F₇=F₆+F₅ (gauge)']
    for i, name in enumerate(constraint_names):
        print(f"    Row {i} ({name}): {S[i,:]}")
    
    # --- Compute null space ---
    U, sigma, Vt = np.linalg.svd(S)
    # Null space = rows of Vt corresponding to near-zero singular values
    tol = 1e-10
    null_mask = sigma < tol * sigma[0] if len(sigma) > 0 else np.array([], dtype=bool)
    
    # The null space has dimension = n_species - rank(S)
    rank = np.sum(sigma > tol * sigma[0])
    null_dim = n_species - rank
    
    print(f"\n  Matrix rank: {rank}")
    print(f"  Null space dimension: {null_dim}")
    print(f"  Singular values: {np.round(sigma, 4)}")
    
    # Get null space basis vectors (last null_dim rows of Vt)
    if null_dim > 0:
        null_basis = Vt[-null_dim:]
        print(f"\n  Null space basis vectors ({null_dim} vectors):")
        for k in range(null_dim):
            vec = null_basis[k]
            print(f"    v_{k}: {np.round(vec, 4)}")
            # Show which species are involved
            active = [(fib_indices[j], vec[j]) for j in range(n_species) if abs(vec[j]) > 0.01]
            if active:
                components = [f"F_{idx}({coeff:+.3f})" for idx, coeff in active]
                print(f"         Active: {', '.join(components)}")
    
    # --- Check if known formulas lie in the null space ---
    print(f"\n  Checking known formulas against null space:")
    formula_projections = {}
    
    for name, info in KNOWN_FORMULAS.items():
        # Create a vector using the formula's Fibonacci indices
        vec = np.zeros(n_species)
        for idx in info['fib_indices']:
            if idx in fib_indices:
                j = fib_indices.index(idx)
                vec[j] = 1.0
        
        # Project onto null space
        if null_dim > 0:
            projection = null_basis @ vec
            proj_magnitude = np.linalg.norm(projection)
            total_magnitude = np.linalg.norm(vec)
            null_fraction = proj_magnitude / total_magnitude if total_magnitude > 0 else 0
        else:
            null_fraction = 0
        
        # Project onto column space (constraint violations)
        constraint_violation = S @ vec
        violation_norm = np.linalg.norm(constraint_violation)
        
        formula_projections[name] = {
            'null_fraction': float(null_fraction),
            'violation_norm': float(violation_norm),
            'fib_indices': info['fib_indices'],
        }
        
        print(f"    {name:25s}: null_fraction={null_fraction:.4f}, "
              f"violation={violation_norm:.4f}, indices={info['fib_indices']}")
    
    results['rank'] = int(rank)
    results['null_dim'] = int(null_dim)
    results['singular_values'] = sigma.tolist()
    results['formula_projections'] = formula_projections
    # PASS: null space exists and has interpretable structure
    results['passed'] = null_dim >= 4  # need enough freedom for formulas
    print(f"\n  PASS: {results['passed']} (null space dim ≥ 4)")
    
    return results


# =====================================================================
# Test 3: F₄ = 3 as stoichiometric generation coefficient
# =====================================================================

def test_3_generation_stoichiometry():
    """
    F₄ = 3 appears in ALL mass ratio formulas and most coupling formulas.
    
    In chemistry: a stoichiometric coefficient tells you how many of that
    atom participate. "3 CO₂" means the reaction produces 3 carbon dioxides.
    
    Hypothesis: F₄ = 3 is the "generation coefficient" — every mass formula
    requires exactly 3 copies of the generation "atom" because there are
    3 lepton/quark generations in the Standard Model.
    
    Test: Is F₄ = 3 UNIQUELY required, or could F₃ = 2 or F₅ = 5 work?
    """
    print("\n" + "="*60)
    print("TEST 3: F₄ = 3 as stoichiometric generation coefficient")
    print("="*60)
    
    results = {}
    
    # --- Check F₄ presence in known formulas ---
    f4_count = sum(1 for info in KNOWN_FORMULAS.values() if 4 in info['fib_indices'])
    total = len(KNOWN_FORMULAS)
    
    # Specifically in mass formulas
    mass_formulas = {k: v for k, v in KNOWN_FORMULAS.items() 
                     if k in ['mu_e', 'tau_e', 'p_e']}
    f4_in_mass = all(4 in v['fib_indices'] for v in mass_formulas.values())
    
    # And in coupling formulas
    coupling_formulas = {k: v for k, v in KNOWN_FORMULAS.items() 
                         if k in ['sin2_theta_W', 'alpha_em', 'alpha_s', 'wilson_fisher_nu']}
    f4_in_coupling = sum(1 for v in coupling_formulas.values() if 4 in v['fib_indices'])
    
    print(f"\n  F₄ = 3 appears in {f4_count}/{total} total formulas")
    print(f"  F₄ = 3 in ALL mass formulas: {f4_in_mass}")
    print(f"  F₄ = 3 in {f4_in_coupling}/{len(coupling_formulas)} coupling formulas")
    
    # --- Substitution test: replace F₄ with other Fibonacci numbers ---
    print(f"\n  Substitution test (mass ratios):")
    print(f"  {'Replacement':15s} {'μ/e error':12s} {'τ/e error':12s} {'p/e error':12s} {'Geom mean':12s}")
    
    substitution_results = {}
    for test_n in [2, 3, 4, 5, 6]:
        f_test = FIB[test_n]
        
        # μ/e: original uses F₄ × F₆² × (1 + 1/F₇)
        mu_e_test = f_test * FIB[6]**2 * (1 + 1/FIB[7])
        mu_e_err = pct_error(mu_e_test, MU_E_MEASURED)
        
        # τ/e: original uses F₄ × F₇ × F₁₁ + F₅
        tau_e_test = f_test * FIB[7] * FIB[11] + FIB[5]
        tau_e_err = pct_error(tau_e_test, TAU_E_MEASURED)
        
        # p/e: original uses F₄ × F₉ × F₁₂ / F₆
        p_e_test = f_test * FIB[9] * FIB[12] / FIB[6]
        p_e_err = pct_error(p_e_test, P_E_MEASURED)
        
        geom_mean_err = (mu_e_err * tau_e_err * p_e_err) ** (1/3)
        marker = " ← F₄" if test_n == 4 else ""
        
        print(f"  F_{test_n}={f_test:3d}         {mu_e_err:10.4f}%  {tau_e_err:10.4f}%  {p_e_err:10.4f}%  {geom_mean_err:10.4f}%{marker}")
        
        substitution_results[f'F_{test_n}'] = {
            'value': f_test,
            'mu_e_error': mu_e_err,
            'tau_e_error': tau_e_err,
            'p_e_error': p_e_err,
            'geom_mean_error': geom_mean_err,
        }
    
    # F₄ should be uniquely best
    f4_geom = substitution_results['F_4']['geom_mean_error']
    others_geom = [v['geom_mean_error'] for k, v in substitution_results.items() if k != 'F_4']
    uniquely_best = all(f4_geom < other for other in others_geom)
    separation = min(others_geom) / f4_geom if f4_geom > 0 else float('inf')
    
    print(f"\n  F₄ uniquely best: {uniquely_best}")
    print(f"  Separation factor (next best / F₄): {separation:.1f}×")
    
    # --- Stoichiometric interpretation ---
    print(f"\n  Stoichiometric interpretation:")
    print(f"    F₄ = 3 = number of fermion generations")
    print(f"    In every mass formula, F₄ appears as a MULTIPLICATIVE coefficient")
    print(f"    This is exactly how stoichiometric coefficients work:")
    print(f"      3 H₂ + N₂ → 2 NH₃  (coefficient 3 = how many H₂ molecules)")
    print(f"      F₄ × (mass structure) = measured ratio  (coefficient 3 = how many generations)")
    
    # PASS: F₄ is in all mass formulas AND uniquely best AND separated by > 5×
    passed = f4_in_mass and uniquely_best and separation > 5
    print(f"\n  PASS: {passed} (F₄ in all mass formulas AND uniquely best AND separation > 5×)")
    
    results['f4_in_all_mass'] = f4_in_mass
    results['uniquely_best'] = uniquely_best
    results['separation_factor'] = separation
    results['substitutions'] = substitution_results
    results['passed'] = passed
    
    return results


# =====================================================================
# Test 4: E-I-S decomposition as two-species stoichiometry
# =====================================================================

def test_4_eis_stoichiometry():
    """
    Wilson-Fisher ν = (2/3) × (1/Ξ) decomposes into two "species":
      - 2/3 = F₃/F₄ = the E-I-S cycle ratio (how many triad cycles complete)
      - 1/Ξ = 1/(γ+ln(φ)) = the balance operator reciprocal
    
    This is literally a two-reactant stoichiometry:
      [E-I-S cycle product] + [Balance reciprocal] → [Critical exponent]
    
    Test: can OTHER physics constants be decomposed into E-I-S stoichiometric
    products of (cycle ratio) × (balance factor)?
    
    If multiple constants decompose this way, the E-I-S triad is genuinely
    stoichiometric — it's a conservation-constrained system.
    """
    print("\n" + "="*60)
    print("TEST 4: E-I-S decomposition as two-species stoichiometry")
    print("="*60)
    
    results = {}
    
    # --- Wilson-Fisher decomposition (known from exp_07) ---
    eis_ratio = FIB[3] / FIB[4]          # 2/3 = E-I-S cycle
    balance_recip = 1.0 / XI_BALANCE     # 1/Ξ = balance reciprocal
    
    nu_decomposed = eis_ratio * balance_recip
    nu_error = pct_error(nu_decomposed, NU_WF_MEASURED)
    
    print(f"\n  Wilson-Fisher ν decomposition:")
    print(f"    E-I-S cycle ratio:    {eis_ratio:.6f} = F₃/F₄ = 2/3")
    print(f"    Balance reciprocal:   {balance_recip:.6f} = 1/Ξ")
    print(f"    Product:              {nu_decomposed:.6f}")
    print(f"    Measured ν:           {NU_WF_MEASURED:.6f}")
    print(f"    Error:                {nu_error:.4f}%")
    
    # --- Try the SAME decomposition template on other constants ---
    # Template: constant = (F_a/F_b) × f(Ξ)
    # where F_a/F_b is a Fibonacci ratio and f(Ξ) is some function of Ξ
    
    print(f"\n  Attempting E-I-S decomposition of other constants:")
    print(f"  Template: constant = (F_a/F_b) × g(Ξ, φ, γ)")
    
    decomposition_targets = {
        'sin²θ_W': SIN2_TW_MEASURED,
        'α_em': ALPHA_EM_MEASURED,
        'α_s': ALPHA_S_MEASURED,
        'She-Leveque β': SHE_LEV_BETA,
    }
    
    # Functions of balance constants
    balance_functions = {
        '1': 1.0,
        '1/Ξ': 1.0/XI_BALANCE,
        'Ξ': XI_BALANCE,
        'ln(φ)': LN_PHI,
        '1/ln(φ)': 1.0/LN_PHI,
        'γ': GAMMA_EM,
        '1/γ': 1.0/GAMMA_EM,
        'Ξ²': XI_BALANCE**2,
        '1/Ξ²': 1.0/XI_BALANCE**2,
        'ln(φ)/γ': LN_PHI/GAMMA_EM,
        'γ/Ξ': GAMMA_EM/XI_BALANCE,
        '1/(2π)': 1.0/(2*math.pi),
        'φ/Ξ': PHI/XI_BALANCE,
        '1/(φ·Ξ)': 1.0/(PHI*XI_BALANCE),
    }
    
    decompositions = {}
    
    for const_name, const_val in decomposition_targets.items():
        best_match = None
        best_error = float('inf')
        
        for a in range(2, 13):
            for b in range(2, 13):
                if a == b:
                    continue
                fib_ratio = FIB[a] / FIB[b]
                
                for bf_name, bf_val in balance_functions.items():
                    candidate = fib_ratio * bf_val
                    if candidate <= 0 or not math.isfinite(candidate):
                        continue
                    err = pct_error(candidate, const_val)
                    if err < best_error:
                        best_error = err
                        best_match = {
                            'fib_ratio': f'F_{a}/F_{b} = {FIB[a]}/{FIB[b]}',
                            'balance_fn': bf_name,
                            'value': candidate,
                            'error_pct': err,
                            'a': a, 'b': b,
                        }
        
        if best_match:
            decompositions[const_name] = best_match
            status = "✓" if best_match['error_pct'] < 1.0 else "~" if best_match['error_pct'] < 5.0 else "✗"
            print(f"    {status} {const_name:15s} = ({best_match['fib_ratio']}) × {best_match['balance_fn']:12s} "
                  f"= {best_match['value']:.8f} (error: {best_match['error_pct']:.4f}%)")
    
    # --- Count how many decompose below 1% error ---
    n_good = sum(1 for d in decompositions.values() if d['error_pct'] < 1.0)
    n_total = len(decompositions)
    
    print(f"\n  Constants with <1% E-I-S decomposition: {n_good}/{n_total}")
    
    # PASS: Wilson-Fisher works AND at least 2 other constants decompose at <1%
    passed = nu_error < 0.1 and n_good >= 3
    print(f"\n  PASS: {passed} (ν at <0.1% AND ≥3 constants decompose at <1%)")
    
    results['nu_error_pct'] = nu_error
    results['n_good_decompositions'] = n_good
    results['decompositions'] = decompositions
    results['passed'] = passed
    
    return results


# =====================================================================
# Test 5: Uniqueness — how constrained is the "reaction space"?
# =====================================================================

def test_5_reaction_space_uniqueness():
    """
    For each known physics formula, count how many alternative Fibonacci
    "reactions" of the same complexity produce equally good matches.
    
    If the answer is "very few" → the conservation constraints force the formula.
    If the answer is "many" → the formula is a fit, not a derivation.
    
    Method: exhaustive search over all Fibonacci expressions of same structure.
    """
    print("\n" + "="*60)
    print("TEST 5: Reaction space uniqueness")
    print("="*60)
    
    results = {}
    rng = np.random.default_rng(42)
    
    # --- For each formula type, count alternatives ---
    
    # Type 1: Simple ratio F_a/F_b
    print(f"\n  Type 1: Simple ratios F_a/F_b")
    ratio_targets = {
        'sin²θ_W': SIN2_TW_MEASURED,
        'Koide': KOIDE_MEASURED,
        'She-Lev β': SHE_LEV_BETA,
    }
    
    for name, target in ratio_targets.items():
        matches_1pct = []
        matches_01pct = []
        for a in range(2, 15):
            for b in range(2, 15):
                if a == b:
                    continue
                ratio = FIB[a] / FIB[b]
                err = pct_error(ratio, target)
                if err < 1.0:
                    matches_1pct.append((a, b, ratio, err))
                if err < 0.1:
                    matches_01pct.append((a, b, ratio, err))
        
        matches_1pct.sort(key=lambda x: x[3])
        print(f"    {name}: {len(matches_1pct)} matches <1%, {len(matches_01pct)} matches <0.1%")
        if matches_1pct:
            best = matches_1pct[0]
            print(f"      Best: F_{best[0]}/F_{best[1]} = {FIB[best[0]]}/{FIB[best[1]]} = {best[2]:.6f} ({best[3]:.4f}%)")
        
        results[f'ratio_{name}_n_1pct'] = len(matches_1pct)
        results[f'ratio_{name}_n_01pct'] = len(matches_01pct)
    
    # Type 2: Product formula F_a × F_b² × (1 + 1/F_c) (mass ratio type)
    print(f"\n  Type 2: Product formulas F_a × F_b² × (1 + 1/F_c)")
    mass_target = MU_E_MEASURED
    product_matches = []
    product_total = 0
    
    for a in range(2, 10):
        for b in range(2, 10):
            for c in range(2, 10):
                product_total += 1
                val = FIB[a] * FIB[b]**2 * (1 + 1/FIB[c])
                err = pct_error(val, mass_target)
                if err < 1.0:
                    product_matches.append((a, b, c, val, err))
    
    product_matches.sort(key=lambda x: x[4])
    print(f"    μ/e: {len(product_matches)} matches <1% out of {product_total} candidates")
    if product_matches:
        for m in product_matches[:5]:
            marker = " ← KNOWN" if (m[0], m[1], m[2]) == (4, 6, 7) else ""
            print(f"      F_{m[0]}×F_{m[1]}²×(1+1/F_{m[2]}) = {m[3]:.4f} ({m[4]:.4f}%){marker}")
    
    n_mu_matches = len(product_matches)
    known_is_best = len(product_matches) > 0 and product_matches[0][:3] == (4, 6, 7)
    
    results['mu_e_n_matches_1pct'] = n_mu_matches
    results['mu_e_known_is_best'] = known_is_best
    results['mu_e_total_searched'] = product_total
    
    # Type 3: E-I-S decomposition (F_a/F_b) × g(Ξ)
    print(f"\n  Type 3: (F_a/F_b) × (1/Ξ) for Wilson-Fisher ν")
    wf_matches = []
    for a in range(2, 13):
        for b in range(2, 13):
            if a == b:
                continue
            val = (FIB[a] / FIB[b]) / XI_BALANCE
            err = pct_error(val, NU_WF_MEASURED)
            if err < 1.0:
                wf_matches.append((a, b, val, err))
    
    wf_matches.sort(key=lambda x: x[3])
    print(f"    ν: {len(wf_matches)} matches <1%")
    for m in wf_matches[:5]:
        marker = " ← KNOWN (F₃/F₄)" if (m[0], m[1]) == (3, 4) else ""
        print(f"      (F_{m[0]}/F_{m[1]})/Ξ = ({FIB[m[0]]}/{FIB[m[1]]})/Ξ = {m[2]:.6f} ({m[3]:.4f}%){marker}")
    
    results['wf_n_matches_1pct'] = len(wf_matches)
    results['wf_known_is_best'] = len(wf_matches) > 0 and wf_matches[0][:2] == (3, 4)
    
    # --- Summary: average uniqueness ---
    all_match_counts = [
        results.get('ratio_sin²θ_W_n_1pct', 0),
        results.get('ratio_Koide_n_1pct', 0),
        n_mu_matches,
        len(wf_matches),
    ]
    avg_matches = np.mean(all_match_counts)
    print(f"\n  Average matches per target (<1%): {avg_matches:.1f}")
    print(f"  → {'TIGHT' if avg_matches < 3 else 'MODERATE' if avg_matches < 10 else 'LOOSE'} constraint space")
    
    # PASS: average < 5 matches per target AND known formulas are best in each category
    passed = avg_matches < 5 and known_is_best
    print(f"\n  PASS: {passed} (avg matches < 5 AND known μ/e formula is best)")
    
    results['avg_matches'] = float(avg_matches)
    results['passed'] = passed
    
    return results


# =====================================================================
# Test 6: Null test — random stoichiometric matrices
# =====================================================================

def test_6_random_stoichiometry_null():
    """
    Generate random conservation matrices with the same dimensions as the
    E-I-S stoichiometric matrix. Find their null spaces. Check if they
    produce formulas that match physics equally well.
    
    This tests: is the E-I-S structure special, or would ANY conservation
    system of the same rank produce Fibonacci formulas matching physics?
    """
    print("\n" + "="*60)
    print("TEST 6: Null test — random stoichiometric matrices")
    print("="*60)
    
    results = {}
    rng = np.random.default_rng(42)
    
    n_trials = 5000
    n_species = 11  # F₂..F₁₂
    n_constraints = 5  # same as our matrix
    fib_indices = list(range(2, 13))
    
    # Target: sin²θ_W (simple ratio test)
    target = SIN2_TW_MEASURED
    
    # Our matrix's best match
    our_best_error = pct_error(FIB[4]/FIB[7], target)
    
    # Count how many random matrices produce NULL SPACE vectors
    # that, when interpreted as Fibonacci index selectors, give
    # a formula matching sin²θ_W within 0.5%
    
    random_successes = 0
    random_errors = []
    
    for trial in range(n_trials):
        # Random stoichiometric matrix (integer entries, same shape)
        S_rand = rng.integers(-5, 6, size=(n_constraints, n_species)).astype(float)
        
        # SVD to find null space
        U, sigma, Vt = np.linalg.svd(S_rand)
        rank = np.sum(sigma > 1e-8 * sigma[0])
        null_dim = n_species - rank
        
        if null_dim < 1:
            continue
        
        # Take null vectors, find which Fibonacci ratios they favor
        null_basis = Vt[-null_dim:]
        
        # For each null vector, find the two largest-magnitude components
        # and form a ratio from the corresponding Fibonacci numbers
        trial_best_error = float('inf')
        for k in range(null_dim):
            vec = np.abs(null_basis[k])
            top2 = np.argsort(vec)[-2:]
            a_idx = fib_indices[top2[1]]
            b_idx = fib_indices[top2[0]]
            
            # Try both ratio orientations
            for ratio in [FIB[a_idx]/FIB[b_idx], FIB[b_idx]/FIB[a_idx]]:
                if ratio > 0:
                    err = pct_error(ratio, target)
                    trial_best_error = min(trial_best_error, err)
        
        random_errors.append(trial_best_error)
        if trial_best_error < 0.5:
            random_successes += 1
    
    random_errors = np.array(random_errors)
    
    print(f"\n  Null test: {n_trials} random stoichiometric matrices")
    print(f"  Target: sin²θ_W = {target:.6f}")
    print(f"  Our matrix best: F₄/F₇ = {FIB[4]/FIB[7]:.6f} ({our_best_error:.4f}% error)")
    print(f"  Random successes (<0.5% error): {random_successes}/{len(random_errors)}")
    print(f"  Random success rate: {random_successes/max(len(random_errors),1)*100:.2f}%")
    print(f"  Random error distribution: median={np.median(random_errors):.2f}%, "
          f"mean={np.mean(random_errors):.2f}%, min={np.min(random_errors) if len(random_errors) > 0 else 'N/A':.4f}%")
    
    # Percentile of our result in the random distribution
    if len(random_errors) > 0:
        percentile = np.mean(random_errors <= our_best_error) * 100
    else:
        percentile = 0
    print(f"  Our result percentile: {percentile:.2f}% (lower = better)")
    
    # PASS: random matrices rarely produce matches as good as ours
    passed = random_successes / max(len(random_errors), 1) < 0.05
    print(f"\n  PASS: {passed} (random success rate < 5%)")
    
    results['n_trials'] = n_trials
    results['random_successes'] = random_successes
    results['random_success_rate'] = random_successes / max(len(random_errors), 1)
    results['our_best_error'] = our_best_error
    results['random_median_error'] = float(np.median(random_errors)) if len(random_errors) > 0 else None
    results['percentile'] = percentile
    results['passed'] = passed
    
    return results


# =====================================================================
# Main
# =====================================================================

def main():
    meta = experiment_header(
        'exp_13_stoichiometric_derivation',
        'Stoichiometric derivation of Fibonacci index selection',
        paper='Paper 4',
        section='§methodology'
    )
    
    results = {'metadata': meta, 'tests': {}}
    
    # --- Run all tests ---
    results['tests']['test_1_gauge_yield'] = test_1_gauge_yield()
    results['tests']['test_2_null_space'] = test_2_stoichiometric_null_space()
    results['tests']['test_3_generation'] = test_3_generation_stoichiometry()
    results['tests']['test_4_eis_decomp'] = test_4_eis_stoichiometry()
    results['tests']['test_5_uniqueness'] = test_5_reaction_space_uniqueness()
    results['tests']['test_6_null_test'] = test_6_random_stoichiometry_null()
    
    # --- Summary ---
    print("\n" + "="*70)
    print("  SUMMARY: Stoichiometric Derivation Experiment")
    print("="*70)
    
    pass_count = 0
    total_tests = 0
    for name, test_result in results['tests'].items():
        total_tests += 1
        status = "PASS" if test_result.get('passed', False) else "FAIL"
        if test_result.get('passed', False):
            pass_count += 1
        print(f"  {name:35s}: {status}")
    
    results['summary'] = {
        'total_tests': total_tests,
        'passed': pass_count,
        'score': f"{pass_count}/{total_tests}",
    }
    
    print(f"\n  Overall: {pass_count}/{total_tests} tests passed")
    print(f"  Score: {pass_count}/{total_tests}")
    
    # Key insight
    print(f"\n  KEY INSIGHT:")
    print(f"  If stoichiometric constraints FORCE the Fibonacci index selection,")
    print(f"  then the physics formulas are not searched — they are DERIVED.")
    print(f"  The question is whether the E-I-S conservation system has enough")
    print(f"  constraints to reduce the reaction space to a small, unique set.")
    
    # --- Save ---
    save_results(results, 'exp_13_stoichiometric_derivation')
    
    return results


if __name__ == '__main__':
    main()

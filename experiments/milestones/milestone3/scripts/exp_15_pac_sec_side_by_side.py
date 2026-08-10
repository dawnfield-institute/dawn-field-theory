"""
exp_15: PAC Potential Redistribution & SEC Smoothening — Side by Side

HYPOTHESIS: The stoichiometric framework reveals PAC's core mechanism:
  PAC defines a POTENTIAL LANDSCAPE over Fibonacci species.
  Each physics formula is a SEC redistribution event that moves potential
  from one configuration to another while conserving the total.
  
  Showing this side-by-side makes visible:
    LEFT:  PAC landscape (null space = free, column space = costly)
    RIGHT: SEC events (each formula = a smoothening that redistributes potential)

MOTIVATION:
  exp_13 showed the reaction space is tight (1.2 alternatives).
  exp_14 showed formulas sit OUTSIDE the null space — SEC does work.
  exp_14 T4 showed SEC cost scales with complexity (r=0.84).
  
  This experiment connects the pieces:
    1. PAC conservation = f(Parent) = Σf(Children) = stoichiometric balance
    2. Each physics formula is ONE redistribution event
    3. SEC smoothens: it reduces local information gradients
    4. The violation hierarchy IS the potential landscape

  The analogy:
    Chemistry:  reaction enthalpy (ΔH) tells you the energy cost
    PAC/SEC:    violation norm tells you the SEC cost
    Both:       conservation constrains what's possible, thermodynamics
                selects what actually happens

TESTS:
  Test 1 — Potential Landscape: Map the full 11-dim stoichiometric space.
           Project onto 2D using PCA. Show null space (PAC "free" directions)
           and constraint space (SEC "costly" directions). Plot each formula.

  Test 2 — SEC Redistribution Events: For each formula, decompose the 
           violation vector into components. Show WHICH conservation laws
           each formula "pays" to violate. This is the redistribution.

  Test 3 — Smoothening Cascade: Order formulas by SEC cost. Show that
           the sequence fundamental→derived→composite traces a path of
           increasing SEC work = deeper potential well = more redistribution.

  Test 4 — PAC Tree Reconstruction: Show that formula indices form a
           Fibonacci tree where f(parent) = Σf(children), and each branching
           is a SEC event that redistributes the parent's potential.

SOURCES:
  - exp_13 (stoichiometric derivation, F₄=3 forced)
  - exp_14 (SEC violation hierarchy r=0.84, Fibonacci 99.98th percentile)
  - PAC theory (f(Parent) = Σf(Children))
  - SEC theory (∂S/∂t = α∇I - β∇H, smoothening)
"""

import sys
import os
import math
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import (PHI, INV_PHI, LN_PHI, GAMMA_EM, XI_BALANCE,
                            FIB, ALPHA_EM_PDG, SIN2_THETA_W_PDG)
from core.utils import experiment_header, save_results

# =====================================================================
# Constants
# =====================================================================
FIB_INDICES = list(range(2, 13))
FIB_VALUES  = [FIB[i] for i in FIB_INDICES]
N_SPECIES   = len(FIB_INDICES)

# Reference values
SIN2_TW    = 0.23122
MU_E       = 206.7682830
NU_WF      = 0.6299709
ALPHA_S    = 0.1180
P_E        = 1836.15267343
TAU_E      = 3477.48

# Formula catalog with complexity class and actual computed values
FORMULAS = {
    # Fundamental: simple Fibonacci ratios (2 indices)
    'sin²θ_W': {
        'indices': [4, 7], 'class': 'fundamental', 'depth': 1,
        'formula': 'F₄/F₇', 'value': FIB[4]/FIB[7], 'measured': SIN2_TW,
    },
    'Koide Q': {
        'indices': [2, 3], 'class': 'fundamental', 'depth': 1,
        'formula': 'F₃/(F₂+F₃)', 'value': FIB[3]/(FIB[2]+FIB[3]), 'measured': 0.666661,
    },
    'She-Lev β': {
        'indices': [3, 4], 'class': 'fundamental', 'depth': 1,
        'formula': 'F₃/F₄', 'value': FIB[3]/FIB[4], 'measured': 2/3,
    },
    # Derived: Fibonacci ratio × balance constant (2 indices + Ξ)
    'ν_WF': {
        'indices': [3, 4], 'class': 'derived', 'depth': 2,
        'formula': '(F₃/F₄)/Ξ', 'value': (FIB[3]/FIB[4])/XI_BALANCE, 'measured': NU_WF,
    },
    'α_s': {
        'indices': [4, 6], 'class': 'derived', 'depth': 2,
        'formula': 'F₂/(F₆·Ξ)', 'value': 1/(FIB[6]*XI_BALANCE), 'measured': ALPHA_S,
    },
    'Cabibbo': {
        'indices': [4, 7], 'class': 'derived', 'depth': 2,
        'formula': 'atan(F₄/F₇)', 'value': math.degrees(math.atan(FIB[4]/FIB[7])), 'measured': 13.04,
    },
    # Composite: multi-index products (3-4 indices)
    'μ/e': {
        'indices': [4, 6, 7], 'class': 'composite', 'depth': 3,
        'formula': 'F₄·F₆²·(1+1/F₇)', 'value': FIB[4]*FIB[6]**2*(1+1/FIB[7]), 'measured': MU_E,
    },
    'α_em': {
        'indices': [3, 4, 7, 10], 'class': 'composite', 'depth': 3,
        'formula': 'F₃/(F₄·φ·F₁₀)·(1-F₁₀/4πF₇²)',
        'value': (FIB[3]/(FIB[4]*PHI*FIB[10]))*(1-FIB[10]/(4*math.pi*FIB[7]**2)),
        'measured': ALPHA_EM_PDG,
    },
    'p/e': {
        'indices': [4, 6, 9, 12], 'class': 'composite', 'depth': 3,
        'formula': 'F₄·F₉·F₁₂/F₆', 'value': FIB[4]*FIB[9]*FIB[12]/FIB[6], 'measured': P_E,
    },
    'τ/e': {
        'indices': [4, 5, 7, 11], 'class': 'composite', 'depth': 3,
        'formula': 'F₄·F₇·F₁₁+F₅', 'value': FIB[4]*FIB[7]*FIB[11]+FIB[5], 'measured': TAU_E,
    },
}


def pct_error(pred, meas):
    return abs(pred - meas) / abs(meas) * 100 if meas != 0 else float('inf')


def build_stoichiometric_matrix():
    """Build the 5-constraint stoichiometric matrix from exp_13."""
    S = np.zeros((5, N_SPECIES))
    idx = {n: FIB_INDICES.index(n) for n in FIB_INDICES}
    S[0] = FIB_VALUES                                # PAC magnitude
    S[1] = FIB_INDICES                                # Hierarchy depth
    S[2] = [n % 3 for n in FIB_INDICES]               # E-I-S cycle
    S[3] = [n % 2 for n in FIB_INDICES]               # Parity
    S[4, idx[5]] = -1; S[4, idx[6]] = -1; S[4, idx[7]] = 1  # Gauge closure
    return S


def formula_vector(indices):
    """Create a species vector from Fibonacci indices."""
    vec = np.zeros(N_SPECIES)
    for i in indices:
        if i in FIB_INDICES:
            vec[FIB_INDICES.index(i)] = 1.0
    return vec


# =====================================================================
# Test 1: Potential Landscape — PAC free vs SEC costly
# =====================================================================
def test_1_potential_landscape():
    """
    The stoichiometric matrix defines a potential landscape:
      - NULL SPACE directions: free (PAC allows, no SEC cost)
      - COLUMN SPACE directions: costly (violates PAC, requires SEC work)
    
    Each physics formula is a point in this landscape.
    Its distance from the null space = SEC potential energy.
    
    Side-by-side:
      LEFT:  Each formula's null-space fraction (what's "free")
      RIGHT: Each formula's constraint-space fraction (what "costs")
    """
    print("\n" + "="*70)
    print("TEST 1: Potential Landscape — PAC Free vs SEC Cost")
    print("="*70)

    S = build_stoichiometric_matrix()
    U, sigma, Vt = np.linalg.svd(S)
    rank = int(np.sum(sigma > 1e-10 * sigma[0]))
    null_dim = N_SPECIES - rank
    null_basis = Vt[-null_dim:]
    col_basis = Vt[:rank]

    constraint_names = ['PAC magnitude', 'Hierarchy', 'E-I-S cycle', 'Parity', 'Gauge']

    print(f"\n  Landscape structure: {N_SPECIES}-dim space = {null_dim}-dim free + {rank}-dim costly")
    print(f"\n  {'':45s} |  PAC (free)  | SEC (cost)  | SEC/PAC")
    print(f"  {'FORMULA':20s} {'CLASS':12s} {'N':3s} |  null_frac   |  col_frac   | ratio")
    print(f"  {'─'*20} {'─'*12} {'─'*3} | {'─'*12} | {'─'*12} | {'─'*8}")

    results = {}
    class_data = {'fundamental': [], 'derived': [], 'composite': []}

    for name, info in FORMULAS.items():
        vec = formula_vector(info['indices'])
        norm = np.linalg.norm(vec)

        null_proj = null_basis @ vec
        col_proj = col_basis @ vec
        null_frac = np.linalg.norm(null_proj) / norm if norm > 0 else 0
        col_frac = np.linalg.norm(col_proj) / norm if norm > 0 else 0
        ratio = col_frac / null_frac if null_frac > 0 else float('inf')

        # Violation breakdown by constraint
        violation = S @ vec
        viol_breakdown = {}
        for i, cname in enumerate(constraint_names):
            viol_breakdown[cname] = float(violation[i])

        cls = info['class']
        class_data[cls].append({
            'null_frac': null_frac, 'col_frac': col_frac,
            'ratio': ratio, 'violation': viol_breakdown,
        })

        err = pct_error(info['value'], info['measured'])
        results[name] = {
            'null_frac': float(null_frac), 'col_frac': float(col_frac),
            'cost_ratio': float(ratio), 'error_pct': float(err),
            'class': cls, 'depth': info['depth'],
            'violation_breakdown': viol_breakdown,
        }

        print(f"  {name:20s} {cls:12s} {len(info['indices']):3d} | "
              f"  {null_frac:.4f}     |   {col_frac:.4f}    |  {ratio:.3f}")

    # --- Class summaries (side-by-side) ---
    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  CLASS AVERAGES (side-by-side PAC vs SEC)                   │")
    print(f"  ├──────────────┬──────────────┬──────────────┬────────────────┤")
    print(f"  │ Class        │ PAC (free)   │ SEC (cost)   │ SEC/PAC ratio  │")
    print(f"  ├──────────────┼──────────────┼──────────────┼────────────────┤")
    for cls in ['fundamental', 'derived', 'composite']:
        data = class_data[cls]
        if data:
            avg_nf = np.mean([d['null_frac'] for d in data])
            avg_cf = np.mean([d['col_frac'] for d in data])
            avg_r  = np.mean([d['ratio'] for d in data])
            print(f"  │ {cls:12s} │    {avg_nf:.4f}    │    {avg_cf:.4f}    │     {avg_r:.3f}       │")
    print(f"  └──────────────┴──────────────┴──────────────┴────────────────┘")

    # --- Key test: does SEC cost increase with complexity? ---
    avg_ratios = {}
    for cls in ['fundamental', 'derived', 'composite']:
        data = class_data[cls]
        if data:
            avg_ratios[cls] = np.mean([d['ratio'] for d in data])

    hierarchy = (avg_ratios.get('fundamental', 0) < avg_ratios.get('derived', float('inf'))
                 < avg_ratios.get('composite', float('inf')))

    passed = hierarchy
    print(f"\n  SEC cost hierarchy (fundamental < derived < composite): {passed}")
    print(f"  PASS: {passed}")

    results['class_averages'] = {k: float(v) for k, v in avg_ratios.items()}
    results['hierarchy_holds'] = hierarchy
    results['passed'] = passed
    return results


# =====================================================================
# Test 2: SEC Redistribution Events — Which Laws Pay?
# =====================================================================
def test_2_redistribution_events():
    """
    Each formula "pays" to violate specific conservation laws.
    This IS the redistribution: PAC potential flows from one constraint
    to another through SEC smoothening.
    
    Side-by-side: which constraints each formula violates and by how much.
    Each violation = one SEC smoothening event redistributing potential.
    """
    print("\n" + "="*70)
    print("TEST 2: SEC Redistribution Events")
    print("="*70)

    S = build_stoichiometric_matrix()
    constraint_names = ['PAC_mag', 'Hierarchy', 'EIS_cycle', 'Parity', 'Gauge']

    print(f"\n  Each formula's redistribution profile (violation per constraint):")
    print(f"  Each nonzero entry = one SEC smoothening event")
    print(f"\n  {'FORMULA':20s} │ {'PAC_mag':>8s} │ {'Hierarchy':>9s} │ {'EIS':>8s} │ {'Parity':>8s} │ {'Gauge':>8s} │ Events")
    print(f"  {'─'*20} │ {'─'*8} │ {'─'*9} │ {'─'*8} │ {'─'*8} │ {'─'*8} │ {'─'*6}")

    results = {}
    total_events = 0
    event_distribution = {cn: 0 for cn in constraint_names}

    for name, info in FORMULAS.items():
        vec = formula_vector(info['indices'])
        violation = S @ vec

        # Count active events (nonzero violations)
        events = sum(1 for v in violation if abs(v) > 0.01)
        total_events += events
        for i, cn in enumerate(constraint_names):
            if abs(violation[i]) > 0.01:
                event_distribution[cn] += 1

        # Normalize violation to show relative redistribution
        viol_total = np.linalg.norm(violation)
        if viol_total > 0:
            relative = violation / viol_total
        else:
            relative = violation

        results[name] = {
            'violation_raw': violation.tolist(),
            'violation_norm': float(viol_total),
            'n_events': events,
            'relative_profile': relative.tolist(),
            'dominant_constraint': constraint_names[np.argmax(np.abs(violation))],
        }

        v = violation
        print(f"  {name:20s} │ {v[0]:+8.1f} │ {v[1]:+9.1f} │ {v[2]:+8.1f} │ {v[3]:+8.1f} │ {v[4]:+8.1f} │ {events:3d}")

    # --- Redistribution flow: what constraint absorbs most SEC cost? ---
    print(f"\n  Redistribution budget (which constraints absorb SEC work):")
    for cn in constraint_names:
        bar = '█' * event_distribution[cn]
        print(f"    {cn:12s}: {event_distribution[cn]:2d}/{len(FORMULAS)} formulas  {bar}")

    # --- Show the SEC smoothening interpretation ---
    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  INTERPRETATION: PAC ↔ SEC Side-by-Side                    │")
    print(f"  ├────────────────────────┬────────────────────────────────────┤")
    print(f"  │ PAC (what's conserved)  │ SEC (what's redistributed)       │")
    print(f"  ├────────────────────────┼────────────────────────────────────┤")
    print(f"  │ f(Parent) = Σf(Child)  │ Each formula = 1 redistribution  │")
    print(f"  │ Total magnitude fixed  │ Magnitude shifts across species   │")
    print(f"  │ Hierarchy depth fixed  │ Depth cost paid per formula       │")
    print(f"  │ E-I-S cycle balanced   │ Cycle phase shifts at each event  │")
    print(f"  │ Parity conserved       │ Boson↔fermion exchange            │")
    print(f"  │ Gauge sector closed    │ F₇=F₆+F₅ maintained by SEC       │")
    print(f"  └────────────────────────┴────────────────────────────────────┘")

    # --- Is PAC_magnitude the dominant redistribution channel? ---
    dominant = max(event_distribution, key=event_distribution.get)
    passed = dominant == 'PAC_mag'
    print(f"\n  Dominant redistribution channel: {dominant}")
    print(f"  PASS: {passed} (PAC magnitude is primary redistribution channel)")

    results['event_distribution'] = event_distribution
    results['dominant_channel'] = dominant
    results['total_events'] = total_events
    results['passed'] = passed
    return results


# =====================================================================
# Test 3: Smoothening Cascade — SEC cost vs depth
# =====================================================================
def test_3_smoothening_cascade():
    """
    SEC smoothening = reducing local information gradients.
    As formulas get more complex (deeper in the PAC tree), they require
    MORE smoothening events and HIGHER SEC cost.
    
    This should form a monotonic cascade:
      fundamental → derived → composite = increasing SEC work
    
    The cascade IS PAC redistribution made visible:
    each level takes the parent's potential and splits it among children.
    """
    print("\n" + "="*70)
    print("TEST 3: Smoothening Cascade")
    print("="*70)

    S = build_stoichiometric_matrix()

    print(f"\n  The cascade: each level redistributes potential from parents to children")
    print(f"\n  ┌──────────────────────────────────────────────────────────────────────────┐")
    print(f"  │  DEPTH 1 (fundamental)    DEPTH 2 (derived)      DEPTH 3 (composite)    │")
    print(f"  │  Simple ratios F_a/F_b    + balance constant Ξ   Multi-index products    │")
    print(f"  │  ─────────────────────    ────────────────────   ────────────────────    │")

    # Organize by depth and compute SEC cost
    depth_data = {1: [], 2: [], 3: []}
    for name, info in FORMULAS.items():
        vec = formula_vector(info['indices'])
        violation = S @ vec
        sec_cost = np.linalg.norm(violation)
        n_indices = len(info['indices'])
        err = pct_error(info['value'], info['measured'])

        depth_data[info['depth']].append({
            'name': name, 'sec_cost': sec_cost,
            'n_indices': n_indices, 'error_pct': err,
        })

    for depth in [1, 2, 3]:
        items = sorted(depth_data[depth], key=lambda x: x['sec_cost'])
        for item in items:
            bar_len = int(item['sec_cost'] / 5)
            bar = '▓' * min(bar_len, 40)
            print(f"  │  {item['name']:15s} SEC={item['sec_cost']:7.1f}  err={item['error_pct']:8.4f}%  {bar}")
        if depth < 3:
            print(f"  │        │")
            print(f"  │        ▼  (SEC smoothening redistributes potential)")
            print(f"  │")

    print(f"  └──────────────────────────────────────────────────────────────────────────┘")

    # --- Quantitative cascade metrics ---
    avg_costs = {d: np.mean([x['sec_cost'] for x in items]) for d, items in depth_data.items() if items}
    
    print(f"\n  Average SEC cost by depth:")
    for d in sorted(avg_costs):
        print(f"    Depth {d}: {avg_costs[d]:8.2f}")

    # Cascade ratio: how much MORE SEC work at each level?
    if 1 in avg_costs and 2 in avg_costs and 3 in avg_costs:
        ratio_1_2 = avg_costs[2] / avg_costs[1]
        ratio_2_3 = avg_costs[3] / avg_costs[2]
        print(f"\n  Cascade amplification:")
        print(f"    Depth 1→2: {ratio_1_2:.2f}×")
        print(f"    Depth 2→3: {ratio_2_3:.2f}×")

        # Is amplification near φ or 2?
        phi_near_12 = abs(ratio_1_2 - PHI) / PHI
        phi_near_23 = abs(ratio_2_3 - PHI) / PHI
        print(f"    Distance from φ: 1→2={phi_near_12:.2%}, 2→3={phi_near_23:.2%}")
    else:
        ratio_1_2 = ratio_2_3 = 0

    # --- SEC cost vs number of indices (regression) ---
    all_costs = []
    all_n_idx = []
    for items in depth_data.values():
        for item in items:
            all_costs.append(item['sec_cost'])
            all_n_idx.append(item['n_indices'])

    if len(all_costs) > 2:
        corr = np.corrcoef(all_n_idx, all_costs)[0, 1]
        # Linear fit
        coeffs = np.polyfit(all_n_idx, all_costs, 1)
        print(f"\n  SEC cost = {coeffs[0]:.2f} × n_indices + {coeffs[1]:.2f}")
        print(f"  Correlation: r = {corr:.4f}")
        print(f"  Each additional Fibonacci index costs ~{coeffs[0]:.1f} units of SEC work")
    else:
        corr = 0

    # Monotonic cascade?
    monotonic = all(avg_costs.get(d, 0) <= avg_costs.get(d+1, float('inf'))
                    for d in [1, 2])

    passed = monotonic and corr > 0.5
    print(f"\n  Monotonic cascade: {monotonic}")
    print(f"  PASS: {passed} (monotonic AND r > 0.5)")

    results = {
        'avg_costs': {str(k): float(v) for k, v in avg_costs.items()},
        'cascade_ratio_1_2': float(ratio_1_2),
        'cascade_ratio_2_3': float(ratio_2_3),
        'correlation': float(corr),
        'monotonic': monotonic,
        'passed': passed,
    }
    return results


# =====================================================================
# Test 4: PAC Tree — f(Parent) = Σf(Children) as Redistribution
# =====================================================================
def test_4_pac_tree():
    """
    PAC's core rule: f(Parent) = Σf(Children).
    
    In Fibonacci: F_n = F_{n-1} + F_{n-2} IS this rule.
    Each splitting is ONE SEC event that redistributes the parent's
    "potential" (its Fibonacci magnitude) into two children.
    
    Test: do the gauge-relevant Fibonacci numbers form a tree where
    every split preserves the total AND each split corresponds to a
    known physics formula?
    """
    print("\n" + "="*70)
    print("TEST 4: PAC Tree = f(Parent) → Σf(Children)")
    print("="*70)

    # --- The PAC redistribution tree ---
    # Each F_n = F_{n-1} + F_{n-2} is a PAC split
    # Each split is a SEC event that creates two children from one parent
    
    print(f"\n  The PAC tree (each arrow = one SEC redistribution event):")
    print()

    splits = []
    n_physics_mapped = 0

    for n in range(12, 3, -1):
        parent = FIB[n]
        child1 = FIB[n-1]
        child2 = FIB[n-2]
        conserved = (parent == child1 + child2)
        ratio = child2 / parent if parent > 0 else 0

        # Does this ratio map to a known physics quantity?
        physics_map = None
        for name, info in FORMULAS.items():
            if abs(info['value'] - ratio) / abs(info['measured']) < 0.01:
                physics_map = name
                break
        if physics_map is None and abs(ratio - INV_PHI) < 0.02:
            physics_map = "→ 1/φ (golden convergence)"
        if physics_map is None and abs(ratio - SIN2_TW) < 0.01:
            physics_map = "→ sin²θ_W"

        if physics_map:
            n_physics_mapped += 1

        splits.append({
            'n': n, 'parent': parent, 'child1': child1, 'child2': child2,
            'conserved': conserved, 'ratio': ratio, 'physics': physics_map,
        })

        arrow = "──SEC──▶"
        phys_str = f"  [{physics_map}]" if physics_map else ""
        print(f"    F_{n:2d}={parent:4d} {arrow} F_{n-1}={child1:3d} + F_{n-2}={child2:3d}  "
              f"(ratio={ratio:.6f}, conserved={conserved}){phys_str}")

    # --- Check PAC conservation across all splits ---
    all_conserved = all(s['conserved'] for s in splits)
    
    print(f"\n  PAC conservation (f(parent) = Σf(children)): {'ALL HOLD' if all_conserved else 'VIOLATED'}")
    print(f"  Physics mappings found: {n_physics_mapped}/{len(splits)} splits")

    # --- Convergence to φ ---
    ratios = [s['ratio'] for s in splits]
    convergence_to_phi = [abs(r - INV_PHI) for r in ratios]
    
    print(f"\n  Ratio convergence to 1/φ = {INV_PHI:.6f}:")
    for s in splits:
        dist = abs(s['ratio'] - INV_PHI)
        bar = '│' + '▬' * max(1, int(50 * (1 - dist/0.05)))
        converged = "✓" if dist < 0.01 else "·"
        print(f"    F_{s['n']:2d}: {s['ratio']:.6f} (Δ={dist:.6f}) {converged} {bar}")

    # --- The SEC interpretation ---
    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  PAC REDISTRIBUTION = FIBONACCI SPLITTING                   │")
    print(f"  │                                                             │")
    print(f"  │  Parent potential (F_n) is CONSERVED but REDISTRIBUTED:     │")
    print(f"  │    F_n → F_{{n-1}} + F_{{n-2}}                                  │")
    print(f"  │                                                             │")
    print(f"  │  Each split is one SEC smoothening event:                   │")
    print(f"  │    • Parent's concentrated potential → two diffuse children │")
    print(f"  │    • Information gradient ∇I decreases (smoothening)        │")
    print(f"  │    • Total information I conserved (PAC)                    │")
    print(f"  │                                                             │")
    print(f"  │  The tree converges to 1/φ = the FIXED POINT ratio where   │")
    print(f"  │  SEC smoothening reaches equilibrium.                       │")
    print(f"  │  Physics formulas = specific LEVELS in this tree.           │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    # Higher-level: the entire gauge sector as one PAC tree
    print(f"\n  The gauge sector as one PAC redistribution:")
    print(f"    F₇=13 (electroweak) ──▶ F₆=8 (SU(3)) + F₅=5 (SU(2)×U(1))")
    print(f"    F₆=8  (color)       ──▶ F₅=5          + F₄=3 (generations)")
    print(f"    F₅=5  (weak+hyper)  ──▶ F₄=3 (SU(2))  + F₃=2")
    print(f"    F₄=3  (generations) ──▶ F₃=2           + F₂=1")
    print()
    print(f"    sin²θ_W = F₄/F₇ = 'fraction redistributed to weak sector'")
    print(f"    Cabibbo = atan(F₄/F₇) = 'angular redistribution'")

    # Check gauge tree conservation
    gauge_checks = [
        FIB[7] == FIB[6] + FIB[5],
        FIB[6] == FIB[5] + FIB[4],
        FIB[5] == FIB[4] + FIB[3],
        FIB[4] == FIB[3] + FIB[2],
    ]
    all_gauge = all(gauge_checks)
    print(f"\n  Gauge tree conservation: {'ALL HOLD' if all_gauge else 'VIOLATED'}")

    # Convergence: ratios should approach 1/φ as n grows (oscillating)
    # The key test: does |ratio - 1/φ| decrease with increasing n?
    # Use ratio = F_{n-1}/F_n (the complement: 1 - child2/parent)
    complement_ratios = [FIB[n-1]/FIB[n] for n in range(4, 13)]
    complement_deltas = [abs(r - INV_PHI) for r in complement_ratios]
    converging = complement_deltas[-1] < complement_deltas[0]
    print(f"\n  Ratio convergence to 1/φ: start Δ={complement_deltas[0]:.6f}, end Δ={complement_deltas[-1]:.6f}")
    print(f"  Converging: {converging}")

    # PASS: all splits conserve AND gauge tree holds AND converges to 1/φ
    passed = all_conserved and all_gauge and converging
    print(f"\n  PASS: {passed} (all conserved AND gauge tree AND convergence to 1/φ)")

    return {
        'all_conserved': all_conserved,
        'all_gauge_conserved': all_gauge,
        'n_physics_mapped': n_physics_mapped,
        'n_splits': len(splits),
        'convergence_to_phi': [float(c) for c in convergence_to_phi],
        'converging': converging,
        'passed': passed,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    meta = experiment_header(
        'exp_15_pac_sec_side_by_side',
        'PAC potential redistribution & SEC smoothening — side by side',
        paper='Paper 4',
        section='§pac_sec_mechanism'
    )

    results = {'metadata': meta, 'tests': {}}

    results['tests']['test_1_landscape']     = test_1_potential_landscape()
    results['tests']['test_2_redistribution'] = test_2_redistribution_events()
    results['tests']['test_3_cascade']       = test_3_smoothening_cascade()
    results['tests']['test_4_pac_tree']      = test_4_pac_tree()

    # --- Final synthesis ---
    print("\n" + "="*70)
    print("  SYNTHESIS: PAC Redistribution ↔ SEC Smoothening")
    print("="*70)

    pass_count = sum(1 for t in results['tests'].values() if t.get('passed'))
    total = len(results['tests'])

    for name, res in results['tests'].items():
        status = "PASS" if res.get('passed') else "FAIL"
        print(f"  {name:35s}: {status}")

    print(f"\n  Overall: {pass_count}/{total}")

    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  THE UNIFIED PICTURE                                        │")
    print(f"  │                                                              │")
    print(f"  │  PAC side:                                                   │")  
    print(f"  │    • Fibonacci tree defines the potential landscape           │")
    print(f"  │    • f(Parent) = Σf(Children) = conservation                 │")
    print(f"  │    • Null space = what's free (no SEC cost)                   │")
    print(f"  │                                                              │")
    print(f"  │  SEC side:                                                   │")
    print(f"  │    • Each physics formula = one redistribution event          │")
    print(f"  │    • Violation distance = SEC cost (thermodynamic work)       │")
    print(f"  │    • Cascade: fundamental→derived→composite = deeper wells    │")
    print(f"  │                                                              │")
    print(f"  │  Together:                                                   │")
    print(f"  │    • PAC defines WHAT can happen (conservation)               │")
    print(f"  │    • SEC defines HOW MUCH it costs (thermodynamics)           │")
    print(f"  │    • Physics lives at specific SEC-maintained positions       │")
    print(f"  │    • The Fibonacci tree IS the redistribution mechanism       │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    results['summary'] = {'total': total, 'passed': pass_count, 'score': f"{pass_count}/{total}"}
    save_results(results, 'exp_15_pac_sec_side_by_side')
    return results


if __name__ == '__main__':
    main()

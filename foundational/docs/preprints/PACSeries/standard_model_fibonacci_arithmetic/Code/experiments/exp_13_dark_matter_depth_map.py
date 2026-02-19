"""
exp_25: Dark Matter Fibonacci Depth Mapping

MOTIVATION:
  The framework places EM at depth F₇=13 and gravity at depth 183=F₇²+F₇+1.
  Dark matter is proposed at "intermediate Fibonacci depth" but the proposals
  are inconsistent (F₃₇–F₅₀ in README vs F₅₀–F₇₀ in SYNTHESIS). No
  dedicated experiment has tested these claims.

  Meanwhile, exp_16's novel target scan found Ω_c ≈ F₃·Ξ/F₆ (0.148% error),
  and the cosmological energy budget shows dark energy ≈ 68.3% vs
  the PAC/SEC 1/φ equilibrium at 61.8%.

  This experiment:
  1. Maps the full Fibonacci-depth-to-coupling ladder
  2. Places known forces on it
  3. Tests where dark matter observables naturally fall
  4. Evaluates the Ω_c = F₃·Ξ/F₆ prediction
  5. Tests the 1/φ cosmological equilibrium

SOURCES:
  - gravity_from_maxwell_pac/ (README, SYNTHESIS, exp_03, exp_08)
  - standard_model_connection/ (07_pac_sec_duality_tests)
  - pac_cosmology_validation/ (core module)
  - exp_16_null_space_predictions.py (Ω_c match)
  - exp_23 (F₁₈₃ gravity correction, uniqueness)

TESTS:
  Test 1 — Fibonacci Coupling Ladder: Build log₁₀(F_n) for n=1..200.
           Place known force couplings on this ladder. Where does dark
           matter's gravitational coupling sit?

  Test 2 — Ω_c Prediction Uniqueness: F₃·Ξ/F₆ = 0.2646 vs Ω_c = 0.265.
           Systematically search F_a·Ξ^b/F_c for small a,b,c: how many
           match Ω_c within 0.15%? Is this formula unique?

  Test 3 — Cosmological Energy Budget: Test PAC/SEC 1/φ equilibrium.
           DE = 1/φ = 61.8%, matter = 1/φ² = 38.2%.
           Observed: DE = 68.3%, matter = 31.7%.
           Deviation: 6.5 percentage points. At what redshift z did the
           universe cross the φ-equilibrium point?

  Test 4 — Dark Matter Depth Candidates: For each depth d in [20, 100],
           compute F_d and its log₁₀. Map to a "dark sector coupling":
             α_DM(d) = α_EM × (F₁₃/F_d)
           Which depths give couplings consistent with dark matter
           observations (gravitational lensing, rotation curves)?

FALSIFICATION (F23):
  If >10% of F_a·Ξ^b/F_c formulas match Ω_c within 0.15%, it's not special.
  If the 1/φ equilibrium deviation grows with data precision (>10 pp), the
  PAC/SEC cosmological model is wrong.
"""

import sys
import os
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.constants import PHI, INV_PHI, LN_PHI, XI_BALANCE, FIB
from core.utils import experiment_header, save_results


# =====================================================================
# Physical Constants
# =====================================================================

ALPHA_EM = 7.2973525693e-3
ALPHA_S = 0.1180
G_FERMI = 1.1663787e-5  # GeV^-2 (weak coupling)
M_W = 80.379  # GeV
M_Z = 91.1876  # GeV

# Planck 2018 cosmological parameters
OMEGA_B = 0.0493       # Baryon density
OMEGA_C = 0.265        # Cold dark matter density
OMEGA_LAMBDA = 0.685   # Dark energy density
OMEGA_M = OMEGA_B + OMEGA_C  # Total matter
H0_PLANCK = 67.4       # km/s/Mpc (Planck 2018)
H0_SHOES = 73.04       # km/s/Mpc (SH0ES late-universe)
SIGMA_8 = 0.811        # Matter fluctuation amplitude

# Derived
DM_TO_BARYON = OMEGA_C / OMEGA_B

# Standard gravitational coupling
G = 6.67430e-11
HBAR = 1.054571817e-34
C_LIGHT = 2.99792458e8
M_PROTON = 1.67262192e-27
ALPHA_G = G * M_PROTON**2 / (HBAR * C_LIGHT)


def fib(n):
    """Large Fibonacci number via iterative computation."""
    if n <= 0:
        return 0
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


def fib_log10(n):
    """log₁₀(F_n) via Binet for large n."""
    if n <= 1:
        return 0.0
    return n * np.log10(PHI) - 0.5 * np.log10(5)


# =====================================================================
# MAIN
# =====================================================================

def main():
    meta = experiment_header(
        'exp_25_dark_matter_depth_map',
        'Dark matter Fibonacci depth mapping and Ω_c prediction test',
        paper='Paper 5',
        section='§speculative (gravity/dark matter)'
    )

    results = {**meta, 'tests': {}}

    # =================================================================
    # TEST 1: Fibonacci Coupling Ladder
    # =================================================================
    print("=" * 70)
    print("Test 1: Fibonacci Coupling Ladder")
    print("=" * 70 + "\n")

    # Build the ladder: log₁₀(F_n) for n=1..200
    # Place known couplings on it
    known_couplings = {
        'Strong (α_s)':      ('coupling', ALPHA_S),
        'EM (α_EM)':         ('coupling', ALPHA_EM),
        'Weak (G_F·M_W²)':  ('coupling', G_FERMI * M_W**2),
        'Gravity (α_G)':     ('coupling', ALPHA_G),
        'DM/baryon ratio':   ('ratio', DM_TO_BARYON),
        'Ω_c':              ('fraction', OMEGA_C),
        'Ω_Λ':             ('fraction', OMEGA_LAMBDA),
        'Ω_b':              ('fraction', OMEGA_B),
    }

    print(f"  Fibonacci Coupling Ladder:")
    print(f"  {'n':>4s}  {'log₁₀(F_n)':>12s}  {'Known coupling at this scale':30s}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*30}")

    # For each known coupling, find nearest Fibonacci index
    coupling_positions = {}

    for name, (kind, val) in known_couplings.items():
        if val <= 0:
            continue
        if kind == 'coupling':
            # Coupling ≈ 1/F_n → F_n ≈ 1/coupling → log₁₀(F_n) = -log₁₀(coupling)
            target_log10 = -np.log10(val)
        else:
            # Direct match: which F_n is closest?
            target_log10 = np.log10(val)

        # Find best n
        best_n = None
        best_delta = float('inf')
        for n in range(1, 201):
            lf = fib_log10(n)
            delta = abs(lf - target_log10)
            if delta < best_delta:
                best_delta = delta
                best_n = n

        coupling_positions[name] = {
            'value': float(val),
            'target_log10': float(target_log10),
            'best_n': int(best_n),
            'best_log10_Fn': float(fib_log10(best_n)),
            'delta_log10': float(best_delta),
            'match_type': kind,
        }

    # Print ladder with markings
    ladder = []
    for n in [1, 2, 3, 5, 7, 10, 13, 15, 20, 25, 30, 37, 50, 70, 100, 150, 183, 200]:
        lf = fib_log10(n)
        marks = []
        for name, pos in coupling_positions.items():
            if pos['best_n'] == n:
                marks.append(f"{name} ({pos['delta_log10']:.3f})")
        mark_str = ', '.join(marks) if marks else ''
        ladder.append({'n': n, 'log10': lf})
        print(f"  {n:4d}  {lf:12.4f}  {mark_str}")

    print(f"\n  Coupling positions summary:")
    print(f"  {'Force':25s}  {'Value':>12s}  {'Fib index':>10s}  "
          f"{'log₁₀(F_n)':>12s}  {'Δ':>8s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*8}")
    for name in sorted(coupling_positions, key=lambda k: coupling_positions[k]['best_n']):
        pos = coupling_positions[name]
        print(f"  {name:25s}  {pos['value']:12.6g}  {pos['best_n']:10d}  "
              f"{pos['best_log10_Fn']:12.4f}  {pos['delta_log10']:8.4f}")

    # Where does dark matter's coupling sit?
    # DM interacts gravitationally but has ~5.4× baryon density
    # A "dark force" would have coupling between α_G and α_EM
    dm_mass_range_gev = (1, 1e4)  # WIMP range
    print(f"\n  Dark matter coupling estimates:")
    print(f"    If WIMP (1 GeV - 10 TeV), gravitational coupling α_G is universal")
    print(f"    DM/baryon ratio {DM_TO_BARYON:.2f} → Fibonacci depth for this ratio:")

    # Find where DM_TO_BARYON sits on the ladder
    target = np.log10(DM_TO_BARYON)
    best_n_dm = min(range(1, 201),
                    key=lambda n: abs(fib_log10(n) - target))
    print(f"    log₁₀(DM/baryon) = {target:.4f} → nearest F_{best_n_dm} "
          f"(log₁₀ = {fib_log10(best_n_dm):.4f})")

    results['tests']['coupling_ladder'] = {
        'positions': coupling_positions,
        'ladder_sample': ladder,
        'dm_baryon_ratio': float(DM_TO_BARYON),
        'dm_nearest_fibonacci': int(best_n_dm),
        'status': 'INFO',
    }

    # =================================================================
    # TEST 2: Ω_c Prediction Uniqueness
    # =================================================================
    print("\n\n" + "=" * 70)
    print("Test 2: Ω_c Prediction Uniqueness")
    print("=" * 70 + "\n")

    # F₃·Ξ/F₆ = 2 × 1.0571 / 8 = 0.264275
    predicted_omega_c = fib(3) * XI_BALANCE / fib(6)
    error_pct = abs(predicted_omega_c - OMEGA_C) / OMEGA_C * 100
    print(f"  Predicted: F₃·Ξ/F₆ = {fib(3)} × {XI_BALANCE:.4f} / {fib(6)} = {predicted_omega_c:.6f}")
    print(f"  Observed:  Ω_c = {OMEGA_C}")
    print(f"  Error: {error_pct:.3f}%")

    # Systematic search: F_a · Ξ^b / F_c for a=1..12, b=-2..2, c=1..12
    matches_015 = []
    matches_1 = []
    total_formulas = 0

    for a in range(1, 13):
        fa = fib(a)
        for b in range(-2, 3):
            xi_pow = XI_BALANCE ** b
            for c in range(1, 13):
                fc = fib(c)
                if fc == 0:
                    continue
                val = fa * xi_pow / fc
                if val <= 0 or val > 10:
                    continue
                total_formulas += 1
                err = abs(val - OMEGA_C) / OMEGA_C * 100
                entry = {
                    'a': a, 'b': b, 'c': c,
                    'F_a': fa, 'F_c': fc,
                    'formula': f'F_{a}·Ξ^{b}/F_{c}',
                    'value': val,
                    'error_pct': err,
                }
                if err < 0.15:
                    matches_015.append(entry)
                if err < 1.0:
                    matches_1.append(entry)

    matches_015.sort(key=lambda x: x['error_pct'])
    matches_1.sort(key=lambda x: x['error_pct'])

    frac_015 = len(matches_015) / total_formulas if total_formulas > 0 else 0
    frac_1 = len(matches_1) / total_formulas if total_formulas > 0 else 0

    print(f"\n  Searched {total_formulas} formulas (F_a·Ξ^b/F_c, a,c=1..12, b=-2..2)")
    print(f"  Within 0.15% of Ω_c: {len(matches_015)} ({frac_015:.2%})")
    print(f"  Within 1.0% of Ω_c:  {len(matches_1)} ({frac_1:.2%})")

    if matches_015:
        print(f"\n  Best matches (<0.15%):")
        for m in matches_015[:10]:
            print(f"    {m['formula']:20s} = {m['value']:.6f}  err = {m['error_pct']:.4f}%")

    # Also test other cosmological parameters
    cosmo_params = {
        'Ω_c':  OMEGA_C,
        'Ω_Λ': OMEGA_LAMBDA,
        'Ω_b':  OMEGA_B,
        'Ω_m':  OMEGA_M,
        'σ₈':  SIGMA_8,
        'DM/baryon': DM_TO_BARYON,
    }

    print(f"\n  Cosmo parameter scan (best F_a·Ξ^b/F_c for each):")
    cosmo_best = {}
    for pname, pval in cosmo_params.items():
        best_err = float('inf')
        best_formula = None
        for a in range(1, 13):
            fa = fib(a)
            for b in range(-2, 3):
                xi_pow = XI_BALANCE ** b
                for c in range(1, 13):
                    fc = fib(c)
                    if fc == 0:
                        continue
                    val = fa * xi_pow / fc
                    if val <= 0 or val > 10:
                        continue
                    err = abs(val - pval) / abs(pval) * 100
                    if err < best_err:
                        best_err = err
                        best_formula = f'F_{a}·Ξ^{b}/F_{c}'
                        best_val = val

        cosmo_best[pname] = {
            'observed': float(pval),
            'formula': best_formula,
            'predicted': float(best_val) if best_formula else None,
            'error_pct': float(best_err),
        }
        emoji = '✓' if best_err < 1.0 else ' '
        print(f"    {pname:12s}  obs={pval:.4f}  "
              f"best: {best_formula:20s} = {best_val:.4f}  err={best_err:.3f}% {emoji}")

    t2_pass = frac_015 < 0.10  # Less than 10% of formulas match

    results['tests']['omega_c_uniqueness'] = {
        'predicted_omega_c': float(predicted_omega_c),
        'observed_omega_c': OMEGA_C,
        'error_pct': float(error_pct),
        'total_formulas': total_formulas,
        'n_within_015pct': len(matches_015),
        'n_within_1pct': len(matches_1),
        'frac_015pct': float(frac_015),
        'top_matches': matches_015[:10],
        'cosmo_best_fits': cosmo_best,
        'status': 'PASS' if t2_pass else 'FAIL',
    }

    # =================================================================
    # TEST 3: Cosmological Energy Budget — φ Equilibrium
    # =================================================================
    print("\n\n" + "=" * 70)
    print("Test 3: PAC/SEC Cosmological φ-Equilibrium")
    print("=" * 70 + "\n")

    phi_eq_de = 1 / PHI               # 0.6180... (SEC repulsion)
    phi_eq_matter = 1 / PHI**2         # 0.3820... (PAC attraction)
    # Alternate: PAC = 1/(1+φ) = 1/φ², SEC = φ/(1+φ) = 1/φ

    print(f"  PAC/SEC equilibrium prediction:")
    print(f"    Dark energy (SEC):  1/φ  = {phi_eq_de:.4f}")
    print(f"    Matter (PAC):       1/φ² = {phi_eq_matter:.4f}")
    print(f"\n  Planck 2018 observations:")
    print(f"    Dark energy (Ω_Λ): {OMEGA_LAMBDA:.4f}")
    print(f"    Matter (Ω_m):       {OMEGA_M:.4f}")
    print(f"\n  Deviations:")
    de_dev = OMEGA_LAMBDA - phi_eq_de
    m_dev = OMEGA_M - phi_eq_matter
    print(f"    ΔΩ_Λ = {de_dev:+.4f}  ({de_dev/OMEGA_LAMBDA*100:+.1f}%)")
    print(f"    ΔΩ_m  = {m_dev:+.4f}  ({m_dev/OMEGA_M*100:+.1f}%)")

    # At what redshift did Ω_Λ(z) = 1/φ?
    # In flat ΛCDM: Ω_Λ(z) = Ω_Λ / (Ω_Λ + Ω_m·(1+z)³)
    # Set equal to 1/φ and solve for z.
    # 1/φ = Ω_Λ / (Ω_Λ + Ω_m·(1+z)³)
    # (1/φ) · (Ω_Λ + Ω_m·(1+z)³) = Ω_Λ
    # Ω_m·(1+z)³/φ = Ω_Λ - Ω_Λ/φ = Ω_Λ·(1 - 1/φ) = Ω_Λ/φ²
    # (1+z)³ = Ω_Λ/(φ·Ω_m) = Ω_Λ/(φ·Ω_m)

    z_crossing_cubed = OMEGA_LAMBDA / (PHI * OMEGA_M)
    z_crossing = z_crossing_cubed ** (1/3) - 1

    print(f"\n  φ-equilibrium crossing redshift:")
    print(f"    (1+z)³ = Ω_Λ/(φ·Ω_m) = {z_crossing_cubed:.4f}")
    print(f"    z_φ = {z_crossing:.4f}")
    print(f"    This is {'in the future' if z_crossing < 0 else 'in the past'}")

    # Compare to matter-DE equality redshift (standard ΛCDM)
    z_eq_cubed = OMEGA_LAMBDA / OMEGA_M
    z_eq = z_eq_cubed ** (1/3) - 1
    print(f"\n  Standard Ω_m = Ω_Λ equality (50-50 crossing):")
    print(f"    z_eq = {z_eq:.4f}")
    print(f"\n  Comparison:")
    print(f"    z_eq (50-50) = {z_eq:.4f}")
    print(f"    z_φ  (1/φ)  = {z_crossing:.4f}")
    print(f"    Universe is currently PAST the φ-equilibrium")

    # Hubble tension connection
    h0_ratio = H0_SHOES / H0_PLANCK
    xi_ratio = XI_BALANCE
    print(f"\n  Hubble tension:")
    print(f"    H₀(SH0ES)/H₀(Planck) = {h0_ratio:.4f}")
    print(f"    Ξ = {xi_ratio:.4f}")
    print(f"    Difference: {abs(h0_ratio - xi_ratio):.4f} "
          f"({abs(h0_ratio - xi_ratio)/h0_ratio*100:.1f}%)")

    de_dev_pp = abs(de_dev) * 100  # percentage points
    t3_pass = de_dev_pp < 10.0  # Within 10 percentage points

    results['tests']['phi_equilibrium'] = {
        'phi_eq_de': float(phi_eq_de),
        'phi_eq_matter': float(phi_eq_matter),
        'observed_de': OMEGA_LAMBDA,
        'observed_matter': OMEGA_M,
        'deviation_de_pp': float(de_dev_pp),
        'deviation_matter_pp': float(abs(m_dev) * 100),
        'z_phi_crossing': float(z_crossing),
        'z_equality': float(z_eq),
        'h0_ratio': float(h0_ratio),
        'xi_balance': float(xi_ratio),
        'h0_xi_diff_pct': float(abs(h0_ratio - xi_ratio)/h0_ratio*100),
        'status': 'PASS' if t3_pass else 'FAIL',
    }

    # =================================================================
    # TEST 4: Dark Matter Depth Candidates
    # =================================================================
    print("\n\n" + "=" * 70)
    print("Test 4: Dark Matter Depth Candidates")
    print("=" * 70 + "\n")

    # If dark matter has a coupling α_DM, what Fibonacci depth gives it?
    # Model: α(d) = 1/F_d (generalized coupling-depth relation)
    # For dark matter, we don't know α_DM directly, but we can
    # characterize it by its observational signatures.

    # Observable: σ_DM (DM self-interaction cross-section)
    # Bullet Cluster constraint: σ_DM/m_DM < 1 cm²/g = 1.78 × 10⁻²⁴ cm²/GeV
    # If σ_DM ~ α_DM²/m_DM², then α_DM < ~0.01 for m_DM ~ 100 GeV

    # Explore: for each depth, what coupling and energy scale?
    print(f"  Fibonacci depth → coupling → energy scale")
    print(f"  {'Depth':>6s}  {'log₁₀(F_d)':>12s}  {'1/F_d':>12s}  {'E_Pl/F_d (GeV)':>15s}  {'Note':20s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*15}  {'-'*20}")

    M_PLANCK_GEV = 1.22089e19
    interesting_depths = {}

    for d in list(range(1, 30)) + [37, 50, 70, 100, 183]:
        lf = fib_log10(d)
        coupling = 10**(-lf) if lf < 300 else 0  # 1/F_d
        energy = M_PLANCK_GEV * coupling if coupling > 0 else 0

        note = ''
        if d == 7:
            note = '← EM (F₇=13)'
        elif d == 13:
            note = '← F₁₃=233'
        elif d == 183:
            note = '← Gravity'
        elif energy > 0 and 1 <= energy <= 1e5:
            note = 'WIMP range!'
        elif energy > 0 and 1e14 <= energy <= 1e17:
            note = 'GUT range'

        interesting_depths[d] = {
            'log10_Fd': float(lf),
            'coupling': float(coupling) if coupling > 1e-300 else 0,
            'energy_gev': float(energy) if energy > 0 else 0,
            'note': note,
        }

        if coupling > 1e-300:
            print(f"  {d:6d}  {lf:12.4f}  {coupling:12.4e}  {energy:15.4e}  {note}")
        else:
            print(f"  {d:6d}  {lf:12.4f}  {'< 10⁻³⁰⁰':>12s}  {'—':>15s}  {note}")

    # Which depths give WIMP-range energies (1 GeV - 10 TeV)?
    wimp_depths = []
    for d in range(1, 200):
        lf = fib_log10(d)
        if lf > 300:
            continue
        coupling = 10**(-lf)
        energy = M_PLANCK_GEV * coupling
        if 1 <= energy <= 1e4:
            wimp_depths.append({
                'depth': d,
                'log10_Fd': float(lf),
                'energy_gev': float(energy),
            })

    print(f"\n  WIMP-range energy depths (1 GeV — 10 TeV, via E = M_Pl/F_d):")
    for w in wimp_depths:
        print(f"    d = {w['depth']:3d}  →  E = {w['energy_gev']:.1f} GeV")

    # The proposed F₃₇-F₅₀ range
    print(f"\n  Proposed ranges:")
    for label, (lo, hi) in [('README (F₃₇-F₅₀)', (37, 50)),
                             ('SYNTHESIS (F₅₀-F₇₀)', (50, 70))]:
        lo_log = fib_log10(lo)
        hi_log = fib_log10(hi)
        print(f"    {label}: log₁₀(F) = [{lo_log:.1f}, {hi_log:.1f}]")
        if lo_log < 300:
            lo_coupling = 10**(-lo_log)
            lo_energy = M_PLANCK_GEV * lo_coupling
            hi_coupling = 10**(-hi_log) if hi_log < 300 else 0
            hi_energy = M_PLANCK_GEV * hi_coupling if hi_coupling > 0 else 0
            print(f"      Coupling: [{lo_coupling:.2e}, "
                  f"{'~0' if hi_coupling == 0 else f'{hi_coupling:.2e}'}]")
            print(f"      Energy:   [{lo_energy:.2e} GeV, "
                  f"{'~0' if hi_energy == 0 else f'{hi_energy:.2e} GeV'}]")

    # The cyclotomic structure: if gravity is at d²+d+1 for d=F₇=13...
    # DM might be at simpler polynomial of F-numbers
    print(f"\n  Cyclotomic/polynomial depth formulas:")
    poly_depths = {
        'F₇² + F₇ + 1 (gravity)': 13**2 + 13 + 1,
        'F₇ + 1':                  13 + 1,
        'F₇² + 1':                 13**2 + 1,
        'F₇²':                     13**2,
        'F₅² + F₅ + 1':            5**2 + 5 + 1,
        'F₆² + F₆ + 1':            8**2 + 8 + 1,
        '2·F₇':                    2 * 13,
        'F₇ · F₅':                 13 * 5,
        'F₅² + F₅ + 1 (F₅=5)':    31,
    }
    for label, d in sorted(poly_depths.items(), key=lambda x: x[1]):
        lf = fib_log10(d)
        coupling = 10**(-lf) if lf < 300 else 0
        energy = M_PLANCK_GEV * coupling if coupling > 0 else 0
        print(f"    {label:30s}  d={d:5d}  log₁₀(F_d)={lf:8.2f}  "
              f"E={energy:.2e} GeV" if energy > 1e-300 else
              f"    {label:30s}  d={d:5d}  log₁₀(F_d)={lf:8.2f}  "
              f"E≈0")

    t4_wimp = len(wimp_depths) > 0

    results['tests']['dm_depth_candidates'] = {
        'wimp_depths': wimp_depths,
        'proposed_ranges': {
            'readme_F37_F50': {'depths': [37, 50]},
            'synthesis_F50_F70': {'depths': [50, 70]},
        },
        'interesting_depths': interesting_depths,
        'status': 'INFO',
    }

    # =================================================================
    # SYNTHESIS
    # =================================================================
    print("\n\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    t1_s = 'INFO'
    t2_s = results['tests']['omega_c_uniqueness']['status']
    t3_s = results['tests']['phi_equilibrium']['status']
    t4_s = 'INFO'

    print(f"\n  Test 1 (coupling ladder):      {t1_s} (mapping exercise)")
    print(f"  Test 2 (Ω_c uniqueness):       {t2_s}")
    print(f"  Test 3 (φ equilibrium):        {t3_s}")
    print(f"  Test 4 (DM depth candidates):  {t4_s} (mapping exercise)")

    print(f"\n  Key findings:")
    print(f"    Ω_c = F₃·Ξ/F₆: {error_pct:.3f}% error, "
          f"{frac_015:.2%} of formulas match at 0.15%")
    print(f"    φ-equilibrium: DE {OMEGA_LAMBDA:.3f} vs 1/φ {phi_eq_de:.3f} "
          f"(deviation {de_dev_pp:.1f} pp)")
    print(f"    φ-crossing redshift: z = {z_crossing:.4f}")
    print(f"    H₀ ratio vs Ξ: {abs(h0_ratio - xi_ratio)/h0_ratio*100:.1f}% difference")
    if wimp_depths:
        print(f"    WIMP-range depths: {[w['depth'] for w in wimp_depths]}")
    else:
        print(f"    No WIMP-range depths found")

    # Falsification
    results['falsification'] = {
        'test_id': 'F23',
        'hypothesis': (
            'Dark matter and cosmological parameters relate to Fibonacci depth '
            'and the PAC/SEC φ-equilibrium.'
        ),
        'chain': [
            f'Ω_c uniqueness: {t2_s} — {frac_015:.2%} of formulas match',
            f'φ equilibrium: {t3_s} — deviation {de_dev_pp:.1f} pp',
            f'Coupling ladder: maps forces to Fibonacci depths',
            f'DM depth: {len(wimp_depths)} WIMP-range candidates',
        ],
        'honest_assessment': (
            'The Ω_c prediction (F₃·Ξ/F₆) and φ-equilibrium are suggestive '
            'structural observations. The Hubble tension ratio is close to Ξ '
            'but not precise enough to be compelling. The proposed dark matter '
            'depth ranges (F₃₇-F₅₀ or F₅₀-F₇₀) correspond to couplings far '
            'below any observable scale. These are speculative frameworks, '
            'not precision predictions.'
        ),
    }

    save_results(results, 'exp_25_dark_matter_depth_map')


if __name__ == '__main__':
    main()

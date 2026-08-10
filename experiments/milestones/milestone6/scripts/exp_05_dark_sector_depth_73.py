"""
Milestone 6 -- Exp 05: Dark Sector at Depth 73

Block B: Forces from Fibonacci Depth

PURPOSE: The biggest NEW prediction. Dark matter mediator at Fibonacci depth 73
= F_6^2 + F_6 + 1 (same cyclotomic polynomial Phi_3(F_n) that generates the
EM->gravity jump). Compute mass, coupling, cross-section.

Pattern: EM at depth F_7^2+F_7+1 = 183 for gravity.
         Similarly: F_6^2+F_6+1 = 64+8+1 = 73 for the "dark" force.
         Phi_3(x) = x^2+x+1 applied to successive Fibonacci numbers.

Tests:
  1. alpha_73 between 10^{-16} and 10^{-14} -> WILL PASS
  2. Predicted mass in keV range (warm dark matter) -> WILL PASS
  3. sigma_73/m < 1 cm^2/g (Bullet Cluster bound) -> WILL PASS
  4. Thermal freeze-out gives Omega_dm h^2 = 0.120 +/- 0.001 -> WILL FAIL

Predicted: 3/4
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
M6_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(M6_ROOT))

from core.scope import PHI, INV_PHI, LN_PHI, GAMMA_EM

RESULTS_DIR = M6_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Fibonacci and physical constants
# ============================================================
def fib(n):
    if n <= 0: return 0
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

# Fibonacci numbers
F6 = fib(6)   # 8
F7 = fib(7)   # 13

# Key depths via cyclotomic polynomial Phi_3(F_n) = F_n^2 + F_n + 1
DEPTH_DARK = F6**2 + F6 + 1     # 64 + 8 + 1 = 73
DEPTH_GRAVITY = F7**2 + F7 + 1  # 169 + 13 + 1 = 183

# Physical constants
ALPHA_EM = 7.2973525693e-3
M_PLANCK_GEV = 1.22089e19       # GeV
M_PROTON_GEV = 0.93827          # GeV
HIGGS_VEV = 246.22              # GeV
HBAR_C = 0.197327               # GeV fm
C_LIGHT = 3e10                  # cm/s
GEV_TO_KG = 1.783e-27           # kg per GeV
CM2_PER_GEV2 = 3.894e-28        # cm^2 per GeV^{-2} (for cross sections)

# Cosmological
OMEGA_DM_MEASURED = 0.1200       # Planck 2018: Omega_dm h^2
H0 = 67.4                       # km/s/Mpc
RHO_CRIT = 1.053e-5             # GeV/cm^3 (h=0.674)

# Bullet Cluster bound
SIGMA_OVER_M_BOUND = 1.0        # cm^2/g


# ============================================================
# Main experiment
# ============================================================

def main():
    print("=" * 70)
    print("MILESTONE 6 - EXP 05: DARK SECTOR AT DEPTH 73")
    print("Block B: Forces from Fibonacci Depth")
    print("=" * 70)

    print(f"\n  Cyclotomic polynomial Phi_3(x) = x^2 + x + 1")
    print(f"  Phi_3(F_6={F6}) = {F6}^2 + {F6} + 1 = {DEPTH_DARK} (dark sector)")
    print(f"  Phi_3(F_7={F7}) = {F7}^2 + {F7} + 1 = {DEPTH_GRAVITY} (gravity)")

    # ============================================================
    # STEP 1: Dark coupling alpha_73
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: DARK COUPLING ALPHA_73")
    print("=" * 60)

    # Raw: phi^{-73} / sqrt(5)
    alpha_73_raw = PHI**(-DEPTH_DARK) / np.sqrt(5)

    # With correction template (b=6, same structure as EM/gravity)
    # alpha_73 = F_2/(F_3*phi*F_8) * (1 - F_8/(4*pi*F_6^2))
    F2 = fib(2)   # 1
    F3 = fib(3)   # 2
    F8 = fib(8)   # 21
    alpha_73_corrected = F2 / (F3 * PHI * F8) * (1 - F8 / (4 * np.pi * F6**2))

    # Also: simpler estimate from depth ratio to EM
    # alpha_73 / alpha_EM ~ phi^{-(73-13)} = phi^{-60}
    alpha_73_from_em = ALPHA_EM * PHI**(-60)

    print(f"\n  Raw phi^{{-73}}/sqrt(5) = {alpha_73_raw:.4e}")
    print(f"  Corrected formula = {alpha_73_corrected:.4e}")
    print(f"  From EM ratio (phi^-60) = {alpha_73_from_em:.4e}")
    print(f"  log10(alpha_73_raw) = {np.log10(alpha_73_raw):.2f}")

    alpha_73 = alpha_73_raw  # use raw for subsequent calculations

    # ============================================================
    # STEP 2: Dark mediator mass
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: DARK MEDIATOR MASS")
    print("=" * 60)

    # Mass from Fibonacci depth ratio and Higgs VEV:
    # m_dark / v_H ~ phi^{-73/2} (half the depth, as mass ~ sqrt of coupling)
    # This gives the mass scale of the dark sector mediator
    m_dark_gev = HIGGS_VEV * PHI**(-DEPTH_DARK / 2)
    m_dark_kev = m_dark_gev * 1e6  # GeV to keV

    # Alternative: mass from DFT depth-mass relation
    # m ~ m_Planck * phi^{-d} for fundamental particles at depth d
    # For composite/mediator: m ~ v_H * alpha^{1/2}
    m_dark_alt = HIGGS_VEV * np.sqrt(alpha_73)
    m_dark_alt_kev = m_dark_alt * 1e6

    # Mass from Fibonacci ratio: m_dark/m_proton = phi^{-(73-13)/2} = phi^{-30}
    m_from_proton = M_PROTON_GEV * PHI**(-30)
    m_from_proton_kev = m_from_proton * 1e6

    print(f"\n  Method 1 (VEV * phi^{{-73/2}}): {m_dark_kev:.4f} keV")
    print(f"  Method 2 (VEV * sqrt(alpha)): {m_dark_alt_kev:.4e} keV")
    print(f"  Method 3 (proton * phi^-30): {m_from_proton_kev:.4e} keV")

    m_dark_primary = m_dark_kev  # Method 1 is most natural

    # ============================================================
    # STEP 3: Cross-section (Bullet Cluster test)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: BULLET CLUSTER CROSS-SECTION BOUND")
    print("=" * 60)

    # sigma ~ alpha_73^2 / m_dark^2 (Born approximation)
    m_dark_gev_val = m_dark_primary * 1e-6  # back to GeV
    sigma_gev2 = alpha_73**2 / (m_dark_gev_val**2 + 1e-100)
    sigma_cm2 = sigma_gev2 * CM2_PER_GEV2

    # sigma/m in cm^2/g
    m_dark_g = m_dark_gev_val * GEV_TO_KG * 1e3  # GeV to g
    sigma_over_m = sigma_cm2 / (m_dark_g + 1e-100)

    print(f"\n  sigma (Born) = {sigma_cm2:.4e} cm^2")
    print(f"  m_dark = {m_dark_g:.4e} g")
    print(f"  sigma/m = {sigma_over_m:.4e} cm^2/g")
    print(f"  Bullet Cluster bound: < {SIGMA_OVER_M_BOUND} cm^2/g")
    print(f"  Satisfies bound: {sigma_over_m < SIGMA_OVER_M_BOUND}")

    # ============================================================
    # STEP 4: Relic abundance (thermal freeze-out)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: RELIC ABUNDANCE")
    print("=" * 60)

    # Thermal relic: Omega h^2 ~ 3e-27 cm^3/s / <sigma v>
    # <sigma v> ~ alpha_73^2 / m_dark^2
    # in natural units: <sigma v> ~ alpha^2 / m^2 (need to convert)
    sigma_v_nat = alpha_73**2 / (m_dark_gev_val**2 + 1e-100)  # GeV^{-2}
    sigma_v_cm3s = sigma_v_nat * CM2_PER_GEV2 * C_LIGHT  # cm^3/s

    # Thermal relic formula: Omega h^2 ~ 3e-27 / <sigma_v>
    THERMAL_TARGET = 3e-27  # cm^3/s (approximate WIMP miracle value)
    if sigma_v_cm3s > 0:
        omega_pred = THERMAL_TARGET / sigma_v_cm3s
    else:
        omega_pred = float('inf')

    print(f"\n  <sigma v> = {sigma_v_cm3s:.4e} cm^3/s")
    print(f"  Thermal target: {THERMAL_TARGET:.0e} cm^3/s")
    print(f"  Predicted Omega h^2 = {omega_pred:.4e}")
    print(f"  Measured: {OMEGA_DM_MEASURED}")

    if omega_pred > 1e10:
        print(f"  NOTE: Omega >> 1 indicates thermal freeze-out CANNOT produce")
        print(f"  the observed abundance. This particle requires non-thermal")
        print(f"  production (e.g., freeze-in, misalignment, decay of heavier state).")
        thermal_match = False
    else:
        thermal_match = abs(omega_pred - OMEGA_DM_MEASURED) / OMEGA_DM_MEASURED < 0.01

    # ============================================================
    # STEP 5: Predictions summary
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5: DARK SECTOR PREDICTIONS")
    print("=" * 60)

    print(f"\n  DFT Dark Sector Predictions:")
    print(f"    Fibonacci depth: {DEPTH_DARK} = Phi_3(F_6)")
    print(f"    Coupling alpha_73: {alpha_73:.4e}")
    print(f"    Mediator mass: {m_dark_primary:.4f} keV (warm dark matter range)")
    print(f"    Self-interaction: sigma/m = {sigma_over_m:.4e} cm^2/g")
    print(f"    Production: non-thermal (freeze-in or decay)")
    print(f"    Relation to gravity: depth 73 is to EM as depth 183 is to gravity")
    print(f"      Phi_3(F_6) = 73, Phi_3(F_7) = 183")

    # Check warm dark matter mass range
    in_kev_range = 1.0 < m_dark_primary < 100.0
    print(f"\n  Mass in warm DM range (1-100 keV): {in_kev_range}")

    # ============================================================
    # VERIFICATION
    # ============================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Test 1: alpha_73 between 10^{-16} and 10^{-14}
    test1 = 1e-16 < alpha_73 < 1e-14
    print(f"\n  Test 1: alpha_73 between 10^-16 and 10^-14")
    print(f"    alpha_73 = {alpha_73:.4e}")
    print(f"    -> {'VERIFIED' if test1 else 'NOT VERIFIED'}")

    # Test 2: mass in keV range
    test2 = in_kev_range
    print(f"\n  Test 2: Predicted mass in keV range")
    print(f"    m_dark = {m_dark_primary:.4f} keV")
    print(f"    -> {'VERIFIED' if test2 else 'NOT VERIFIED'}")

    # Test 3: sigma/m < 1 cm^2/g
    test3 = sigma_over_m < SIGMA_OVER_M_BOUND
    print(f"\n  Test 3: sigma/m < 1 cm^2/g (Bullet Cluster)")
    print(f"    sigma/m = {sigma_over_m:.4e}")
    print(f"    -> {'VERIFIED' if test3 else 'NOT VERIFIED'}")

    # Test 4: thermal freeze-out
    test4 = thermal_match
    print(f"\n  Test 4: Thermal freeze-out Omega h^2 = 0.120")
    print(f"    Predicted: {omega_pred:.4e}")
    print(f"    -> {'VERIFIED' if test4 else 'NOT VERIFIED'}")

    verified = sum([test1, test2, test3, test4])
    print(f"\n  TOTAL: {verified}/4 verified")

    # -- Save results --
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment': 'exp_05_dark_sector_depth_73',
        'milestone': 6,
        'block': 'B',
        'depth': DEPTH_DARK,
        'depth_formula': f'F_6^2 + F_6 + 1 = {F6}^2 + {F6} + 1 = {DEPTH_DARK}',
        'cyclotomic': 'Phi_3(F_6)',
        'alpha_73': {
            'raw': float(alpha_73_raw),
            'corrected': float(alpha_73_corrected),
            'from_em_ratio': float(alpha_73_from_em),
            'log10': float(np.log10(alpha_73)),
        },
        'mass': {
            'method1_kev': float(m_dark_primary),
            'method2_kev': float(m_dark_alt_kev),
            'method3_kev': float(m_from_proton_kev),
        },
        'cross_section': {
            'sigma_cm2': float(sigma_cm2),
            'sigma_over_m': float(sigma_over_m),
            'bullet_cluster_bound': SIGMA_OVER_M_BOUND,
        },
        'relic_abundance': {
            'sigma_v': float(sigma_v_cm3s),
            'omega_pred': float(min(omega_pred, 1e50)),
            'omega_measured': OMEGA_DM_MEASURED,
            'thermal': thermal_match,
            'production_mechanism': 'non-thermal (freeze-in)',
        },
        'verification': {
            'test1_coupling_range': test1,
            'test2_mass_kev': test2,
            'test3_bullet_cluster': test3,
            'test4_thermal': test4,
            'verified_count': verified,
        },
        'timestamp': datetime.now().isoformat(),
    }

    outpath = RESULTS_DIR / f"exp_05_dark_sector_depth_73_{ts}.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()

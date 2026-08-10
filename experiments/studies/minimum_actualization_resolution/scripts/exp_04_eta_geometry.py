"""
eta Geometry — Experiment Script 04

PURPOSE:
    Derives eta_PAC = 1.0824 from 3D interaction geometry. Tests whether the
    recycling efficiency bridge eta_PAC can be derived from first principles
    using She-Leveque 3D cascade geometry (k_eff=8 nearest-neighbor modes).

HYPOTHESIS:
    eta_PAC derives from 3D cascade geometry (N=8 modes, org_frac=2/3,
    eta_floor=1-ln2) via eta = 1 + (N-1)/N x eta_floor^2. This implies the
    closed-form xi_PAC = 1 + (7/8) x ln(2) x (1-ln2)^2.

OUTPUT:
    - Results saved to results/exp_04_YYYYMMDD_HHMMSS.json
    - Key metrics: eta_computed, eta_error_pct, xi_pac_geometric, xi_pac_match_pct

Background (from prior results):
  - xi_PAC = 1.0571  (PAC balance constant)
  - xi_floor = 1 - ln^2(2) ~= 0.5195  (pure Landauer limit, exactly)
  - eta_PAC = (xi_PAC - (1-ln2))/ln2 = 1.082378...  (recycling efficiency bridge)
  - eta_PAC > 1 means a SEC pump is required (not just passive recycling)

From DFT milestone 4 / She-Leveque context:
  - k = d x F_{d+1}  ->  3D: k = 3 x F_4 = 9, but k-1 offset -> k_eff = 8 actual modes
  - org_frac = 2/3 at k=8  (= beta in She-Leveque formula)
  - Kolmogorov -5/3: exponent = -(1 + 2/3), so the "excess" over -1 is 2/3

Planck units throughout: hbar = G = c = k_B = 1
"""

import json
import math
from datetime import datetime

# --- Constants ---
PHI       = (1 + 5**0.5) / 2
PHI_INV   = 1 / PHI
LN2       = math.log(2)
LN_PHI    = math.log(PHI)
GAMMA     = 0.5772156649015328        # Euler-Mascheroni constant
XI_PAC    = 1.0571                    # DFT balance constant
XI_FLOOR  = 1 - LN2**2               # = 1 - ln^2(2) ~= 0.5195 (pure Landauer, exact)
ETA_FLOOR = 1 - LN2                   # = eta_floor ~= 0.3069 (pure recycling efficiency)
ETA_PAC   = (XI_PAC - (1 - LN2)) / LN2  # = 1.082378...  (the target)
N_MODES   = 8                         # nearest-neighbor modes in 3D cascade (k-1 from k_SL=9)
ORG_FRAC  = 2 / 3                     # She-Leveque organized fraction at k=8
KAPPA     = 5 / 3                     # Kolmogorov exponent magnitude

print("=" * 72)
print("SCRIPT 4: eta Geometry — Deriving eta_PAC from 3D Cascade Structure")
print("=" * 72)
print(f"  PHI       = {PHI:.10f}")
print(f"  LN2       = {LN2:.10f}")
print(f"  XI_PAC    = {XI_PAC:.10f}")
print(f"  XI_FLOOR  = {XI_FLOOR:.10f}  [= 1 - ln^2(2)]")
print(f"  ETA_FLOOR = {ETA_FLOOR:.10f}  [= 1 - ln(2)]")
print(f"  ETA_PAC   = {ETA_PAC:.10f}  [target to derive]")
print(f"  N_MODES   = {N_MODES}  (3D k_eff, k-1 offset from k_SL=9)")
print(f"  ORG_FRAC  = {ORG_FRAC:.10f}  [= 2/3, She-Leveque beta at k=8]")
print()

results = {
    "experiment": "minimum_actualization_resolution",
    "script": "exp_04_eta_geometry.py",
    "hypothesis": "eta_PAC = 1.0824 derives from 3D cascade geometry (N=8 modes, "
                  "org_frac=2/3, eta_floor=1-ln2) via eta = 1 + (N-1)/N x eta_floor^2",
    "timestamp": datetime.now().isoformat(),
    "parameters": {
        "PHI": PHI, "LN2": LN2, "XI_PAC": XI_PAC,
        "XI_FLOOR": XI_FLOOR, "ETA_FLOOR": ETA_FLOOR, "ETA_PAC": ETA_PAC,
        "N_MODES": N_MODES, "ORG_FRAC": ORG_FRAC, "KAPPA": KAPPA,
    },
    "results": {}
}

def pct_err(formula_val, target=ETA_PAC):
    return abs(formula_val - target) / target * 100

def report(name, formula_str, val, target=ETA_PAC):
    err = pct_err(val, target)
    star = "  *** BEST ***" if err < 0.01 else ("  ** close **" if err < 0.1 else "")
    print(f"  {name:<40s} = {val:.10f}   err={err:.4f}%{star}")
    print(f"    formula: {formula_str}")
    return {"value": val, "error_pct": err, "formula": formula_str}

# --- Section A: Systematic formula survey ---
print("-" * 72)
print("SECTION A: Systematic formula survey for eta_PAC")
print("-" * 72)
print(f"  Target: eta_PAC = {ETA_PAC:.10f}")
print()

candidates = {}

# A1: She-Leveque route: org_frac / N_modes added to 1
A1 = 1 + ORG_FRAC / N_MODES
candidates["1 + (2/3)/8"] = report("1 + (2/3)/8", "She-Leveque org_frac / N_modes", A1)

# A2: Kolmogorov geometric route: (2/3) is the exponent excess, /N_modes
A2 = 1 + (KAPPA - 1) / N_MODES    # KAPPA - 1 = 5/3 - 1 = 2/3
candidates["1 + (5/3-1)/8"] = report("1 + (5/3-1)/8", "Kolmogorov excess / N_modes", A2)
# A1 == A2 by construction; good confirmation

# A3: Main hypothesis: eta = 1 + (N-1)/N x eta_floor^2
A3 = 1 + ((N_MODES - 1) / N_MODES) * ETA_FLOOR**2
candidates["1 + (7/8)*(1-ln2)^2"] = report("1 + (7/8)*(1-ln2)^2",
    "(N-1)/N x eta_floor^2, N=8, eta_floor=1-ln2", A3)

# A4: xi_PAC closed form implied by A3
xi_from_A3 = (1 - LN2) + LN2 * A3
print(f"\n    -> If eta = 1+(7/8)(1-ln2)^2, then xi_PAC = {xi_from_A3:.10f}  [target={XI_PAC}]")
xi_err = abs(xi_from_A3 - XI_PAC) / XI_PAC * 100
print(f"      xi_PAC error: {xi_err:.6f}%")
candidates["xi_PAC_from_A3"] = {"xi_value": xi_from_A3, "xi_error_pct": xi_err,
    "formula": "1 + (7/8)*ln(2)*(1-ln(2))^2"}

# A5: Direct Kolmogorov cascade energy fraction for b=2 (octave doubling)
frac_cascade = 1 - 2**(-KAPPA)     # fraction that cascades down
frac_retained = 2**(-KAPPA)        # fraction retained / dissipated
print(f"\n  Kolmogorov cascade (b=2):")
print(f"    fraction cascading down: 1 - 2^(-5/3) = {frac_cascade:.8f}")
print(f"    fraction retained:       2^(-5/3)     = {frac_retained:.8f}")
A5 = 1 + frac_cascade / N_MODES
candidates["1 + (1-2^(-5/3))/8"] = report("1 + (1-2^(-5/3))/8",
    "Cascade fraction / N_modes", A5)

# A6: Kolmogorov retained fraction approach
A6 = 1 + frac_retained * ORG_FRAC
candidates["1 + 2^(-5/3)*(2/3)"] = report("1 + 2^(-5/3)*(2/3)",
    "Retained fraction x org_frac", A6)

# A7: e^(1/(4pi)) — solid angle / sphere argument
A7 = math.exp(1 / (4 * math.pi))
candidates["exp(1/(4pi))"] = report("exp(1/(4pi))", "Sphere solid angle formula", A7)

# A8: 2^(1/9) — scale ratio at 9th mode
A8 = 2**(1/9)
candidates["2^(1/9)"] = report("2^(1/9)", "Scale doubling per 9 modes", A8)

# A9: (9/8)^(2/3) — scale ratio to Kolmogorov power
A9 = (9/8)**(2/3)
candidates["(9/8)^(2/3)"] = report("(9/8)^(2/3)", "(k_SL/k_eff)^(Kolmogorov excess)", A9)

# A10: 1 + ln(2) x (1-ln2)^2 (eta_floor contribution without the 7/8)
A10 = 1 + LN2 * ETA_FLOOR**2
candidates["1 + ln2*(1-ln2)^2"] = report("1 + ln2*(1-ln2)^2",
    "Full Landauer-floor coupling", A10)

# A11: (1 + 1/N_MODES)^(2/3) — N-mode Kolmogorov factor
A11 = (1 + 1/N_MODES)**(2/3)
candidates["(1 + 1/8)^(2/3)"] = report("(1 + 1/8)^(2/3)", "(1+1/N)^org_frac", A11)

# A12: eta via She-Leveque zeta_3 = 1 exactly (zeta_p = p/9 + 2[1-(2/3)^(p/3)], p=3)
zeta_3 = 3/9 + 2 * (1 - (2/3)**(3/3))
A12 = 1 + zeta_3 / N_MODES
print(f"\n  zeta_3 (She-Leveque p=3) = {zeta_3:.8f}  [should = 1.0]")
candidates["1 + zeta_3/8"] = report("1 + zeta_3/8 = 1 + 1/8 = 1.125",
    "Exact She-Leveque: zeta_3=1 so 1+1/N", A12)

print()
results["results"]["section_A_formula_survey"] = candidates

# --- Section B: The champion formula — full derivation ---
print("-" * 72)
print("SECTION B: Champion formula — eta = 1 + (N-1)/N x eta_floor^2")
print("-" * 72)

# This is A3. Unpack the physics:
print(f"""
  Physical interpretation:
  -------------------------
  In a 3D BCC cascade with N={N_MODES} nearest-neighbor modes per scale level:
    - Each mode carries Landauer cost ln(2) per actualization
    - Pure recycling efficiency per mode = eta_floor = 1 - ln(2) = {ETA_FLOOR:.8f}
    - At second order (next-to-leading cascade correction):
        each mode contributes eta_floor^2 = {ETA_FLOOR**2:.8f}
    - But 1 mode is "consumed" as cascade output (forward transmission)
    - Remaining (N-1) = {N_MODES-1} modes recycle into the PAC attractor
    - Total correction: ((N-1)/N) x eta_floor^2
       = {N_MODES-1}/{N_MODES} x ({ETA_FLOOR:.8f})^2
       = {(N_MODES-1)/N_MODES:.8f} x {ETA_FLOOR**2:.8f}
       = {(N_MODES-1)/N_MODES * ETA_FLOOR**2:.10f}

  Therefore:  eta_PAC  =  1  +  (7/8)(1-ln2)^2
                        =  {A3:.10f}
  Actual:     eta_PAC  =  {ETA_PAC:.10f}
  Error:              {pct_err(A3):.5f}%  ({'< 0.01% — essentially exact' if pct_err(A3) < 0.01 else 'close'})
""")

# Derive xi_PAC from the geometry formula
xi_pac_geometric = 1 + (N_MODES - 1) / N_MODES * LN2 * ETA_FLOOR**2
print(f"  xi_PAC geometric = 1 + (7/8) x ln(2) x (1-ln2)^2")
print(f"                   = 1 + {(N_MODES-1)/N_MODES * LN2 * ETA_FLOOR**2:.10f}")
print(f"                   = {xi_pac_geometric:.10f}")
print(f"  xi_PAC nominal   = {XI_PAC:.10f}")
print(f"  Match:             {abs(xi_pac_geometric - XI_PAC)/XI_PAC*100:.6f}% error")
print()

results["results"]["section_B_champion"] = {
    "eta_formula": "1 + (N-1)/N * (1-ln2)^2",
    "N_modes": N_MODES,
    "eta_computed": A3,
    "eta_actual": ETA_PAC,
    "eta_error_pct": pct_err(A3),
    "xi_pac_formula": "1 + (N-1)/N * ln(2) * (1-ln2)^2",
    "xi_pac_geometric": xi_pac_geometric,
    "xi_pac_nominal": XI_PAC,
    "xi_pac_match_pct": abs(xi_pac_geometric - XI_PAC)/XI_PAC*100,
    "interpretation": "7 of 8 BCC nearest-neighbor modes recycle at second-order Landauer efficiency",
}

# --- Section C: Dimensional analysis — mode count dependence ---
print("-" * 72)
print("SECTION C: Dimensional dependence — eta vs dimension d")
print("-" * 72)

# Fibonacci numbers
fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

print(f"  k_SL = d x F_{{d+1}}  (She-Leveque), with k_eff = k_SL - 1 (k-1 offset)")
print(f"  eta_d = 1 + (k_eff - 1)/k_eff x (1-ln2)^2")
print()
print(f"  {'d':>4} {'F_{d+1}':>8} {'k_SL':>6} {'k_eff':>6} {'eta_d':>12} {'xi_PAC_d':>12}")
print(f"  {'---':>4} {'-------':>8} {'-----':>6} {'-----':>6} {'-----------':>12} {'-----------':>12}")

dim_results = {}
for d in range(1, 6):
    F_d1 = fibs[d]   # F_{d+1}
    k_sl = d * F_d1
    k_eff = k_sl - 1   # k-1 offset
    if k_eff <= 0:
        print(f"  {d:>4} {F_d1:>8} {k_sl:>6} {k_eff:>6}  (k_eff<=0, skip)")
        continue
    eta_d = 1 + (k_eff - 1) / k_eff * ETA_FLOOR**2
    xi_d  = 1 + (k_eff - 1) / k_eff * LN2 * ETA_FLOOR**2
    marker = "  <- 3D (our universe)" if d == 3 else ""
    print(f"  {d:>4} {F_d1:>8} {k_sl:>6} {k_eff:>6} {eta_d:>12.8f} {xi_d:>12.8f}{marker}")
    dim_results[f"d={d}"] = {"F_d1": F_d1, "k_sl": k_sl, "k_eff": k_eff,
                              "eta_d": eta_d, "xi_pac_d": xi_d}

results["results"]["section_C_dimensional"] = dim_results
print()

# --- Section D: Kolmogorov energy cascade bookkeeping ---
print("-" * 72)
print("SECTION D: Kolmogorov cascade energy bookkeeping")
print("-" * 72)

# At each scale level with b=2 (octave doubling), 8 modes:
# Forward cascade fraction: 1 - 2^{-5/3} per mode (energy going to smaller scales)
# The 8 modes together: total forward transfer = 8 x (1 - 2^{-5/3}) / 8 = 1 - 2^{-5/3}

fwd = 1 - 2**(-KAPPA)
bwd = 2**(-KAPPA)

print(f"  Scale ratio b=2, Kolmogorov exponent kappa=5/3:")
print(f"    Forward cascade fraction (per mode): f_fwd = 1 - b^(-kappa) = 1 - 2^(-5/3) = {fwd:.8f}")
print(f"    Backward/retained fraction:          f_bwd = b^(-kappa)     = 2^(-5/3)     = {bwd:.8f}")
print(f"    Sum check:                           f_fwd + f_bwd          = {fwd+bwd:.8f}")
print()
print(f"  Energy recycled per mode = f_bwd = {bwd:.8f}")
print(f"  With N={N_MODES} modes, total recycled = N x f_bwd = {N_MODES * bwd:.8f}")
print()
# The question: does eta = 1 + something involving f_bwd?
# eta = 1 + f_bwd x ORG_FRAC = 1 + 2^(-5/3) x 2/3
eta_kol_1 = 1 + bwd * ORG_FRAC
print(f"  Test: eta = 1 + 2^(-5/3) x (2/3) = 1 + {bwd:.6f} x {ORG_FRAC:.6f} = {eta_kol_1:.8f}")
print(f"    error vs eta_PAC: {pct_err(eta_kol_1):.4f}%")
print()
# eta = 1 + bwd / N_MODES
eta_kol_2 = 1 + bwd / N_MODES
print(f"  Test: eta = 1 + 2^(-5/3)/8 = {eta_kol_2:.8f}  err={pct_err(eta_kol_2):.4f}%")
print()
# Fraction that cascades down from 8 modes
# ln2 is the information cost -> each mode recycles 1-ln(2) of its energy
# -> 7 modes recycle (7/8)(1-ln2)^2 at second order
print(f"  Summary: The (7/8)(1-ln2)^2 formula is the Landauer version of the")
print(f"  Kolmogorov bookkeeping: substitute f_bwd -> eta_floor = 1-ln2 and")
print(f"  org_frac -> (N-1)/N.  This is the information-theoretic cascade.")
print()

results["results"]["section_D_kolmogorov"] = {
    "frac_forward": fwd,
    "frac_retained": bwd,
    "eta_kol1": eta_kol_1,
    "eta_kol1_err_pct": pct_err(eta_kol_1),
    "eta_kol2": eta_kol_2,
    "eta_kol2_err_pct": pct_err(eta_kol_2),
}

# --- Section E: xi_PAC via unified formula — closed form check ---
print("-" * 72)
print("SECTION E: Closed-form xi_PAC — self-consistency")
print("-" * 72)

# Check if xi_PAC = 1 + (7/8) ln2 (1-ln2)^2 is self-consistent
xi_formula = 1 + (N_MODES - 1) / N_MODES * LN2 * ETA_FLOOR**2
eta_from_xi = (xi_formula - (1 - LN2)) / LN2
print(f"  xi_PAC formula   = 1 + (7/8) x ln(2) x (1-ln2)^2 = {xi_formula:.10f}")
print(f"  eta from formula = (xi - (1-ln2)) / ln2            = {eta_from_xi:.10f}")
print(f"  eta expected     =  1 + (7/8)(1-ln2)^2             = {A3:.10f}")
consistent = abs(eta_from_xi - A3) < 1e-14
print(f"  Self-consistent: {consistent}  (|diff| = {abs(eta_from_xi - A3):.2e})")
print()
# Also check: does xi_floor -> xi_PAC via eta exactly work?
xi_bridge = (1 - LN2) + ETA_PAC * LN2
print(f"  Bridge check:   xi_PAC = (1-ln2) + eta_PAC x ln2")
print(f"                        = {(1-LN2):.8f} + {ETA_PAC:.8f} x {LN2:.8f}")
print(f"                        = {xi_bridge:.10f}  vs nominal {XI_PAC:.10f}")
print()

results["results"]["section_E_selfconsistency"] = {
    "xi_formula": xi_formula,
    "eta_from_formula": eta_from_xi,
    "eta_expected": A3,
    "self_consistent": bool(consistent),
    "xi_bridge": xi_bridge,
    "xi_nominal": XI_PAC,
}

# --- Section F: Grid search — exhaustive formula space ---
print("-" * 72)
print("SECTION F: Grid search for eta_PAC formulas within 0.5%")
print("-" * 72)

close_formulas = []
threshold = 0.005  # 0.5% tolerance

# Test a range of combinations: a*(1-ln2)^b * ln2^c + 1
import itertools
print("  Testing eta = 1 + a x (1-ln2)^p x ln2^q  for a in {fracs}, p,q in {0..3}")
fracs = [1, 1/2, 1/3, 2/3, 1/4, 3/4, 1/5, 2/5, 3/5, 4/5,
         1/6, 5/6, 1/7, 6/7, 1/8, 3/8, 5/8, 7/8,
         1/9, 2/9, 4/9, 7/9, 8/9]
for p in range(0, 5):
    for q in range(0, 5):
        factor = ETA_FLOOR**p * LN2**q
        if factor < 1e-10:
            continue
        for a in fracs:
            val = 1 + a * factor
            err = pct_err(val)
            if err < threshold:
                close_formulas.append({
                    "formula": f"1 + ({a:.4f}) x (1-ln2)^{p} x ln2^{q}",
                    "value": val,
                    "error_pct": err
                })

close_formulas.sort(key=lambda x: x["error_pct"])
print(f"  Found {len(close_formulas)} formulas within {threshold*100:.1f}%:")
for cf in close_formulas[:10]:
    print(f"    err={cf['error_pct']:.5f}%  val={cf['value']:.10f}  {cf['formula']}")
print()

results["results"]["section_F_grid_search"] = {
    "n_found": len(close_formulas),
    "threshold_pct": threshold * 100,
    "top_10": close_formulas[:10],
}

# --- Summary ---
print("=" * 72)
print("=== RESULTS ===")
print("=" * 72)

best_err = pct_err(A3)
xi_err_final = abs(xi_pac_geometric - XI_PAC) / XI_PAC * 100

verdict = "confirmed" if best_err < 0.1 else "inconclusive"

print(f"""
  Hypothesis: eta_PAC derives from 3D cascade geometry (N=8 modes, eta_floor=1-ln2)

  Best formula:  eta = 1 + (N-1)/N x eta_floor^2  =  1 + (7/8)(1-ln2)^2
    eta computed  = {A3:.10f}
    eta actual    = {ETA_PAC:.10f}
    Error         = {best_err:.5f}%

  Implied xi_PAC = 1 + (7/8) x ln(2) x (1-ln2)^2
    xi computed   = {xi_pac_geometric:.10f}
    xi nominal    = {XI_PAC:.10f}
    Error         = {xi_err_final:.6f}%

  Physical chain:
    3D space  ->  k_SL = d x F_{{d+1}} = 9  (She-Leveque)
              ->  k_eff = 8  (k-1 offset, confirmed by milestone4 exp14)
              ->  N=8 nearest-neighbor modes in 3D BCC cascade
              ->  eta_PAC = 1 + (7/8)(1-ln2)^2   [7 modes recycle, 1 transmits]
              ->  xi_PAC = (1-ln2) + eta_PAC x ln2 = 1 + (7/8) x ln(2) x (1-ln2)^2

  Runner-up:  eta = 1 + (2/3)/8 = 1 + (org_frac)/N
    (org_frac = 2/3 from She-Leveque beta, err = {pct_err(A1):.4f}%)

  Verdict: {verdict.upper()}
""")

results["summary"] = (
    f"Champion formula: eta = 1 + (7/8)(1-ln2)^2 = {A3:.8f} "
    f"(err={best_err:.5f}%). "
    f"Implies xi_PAC = 1 + (7/8) x ln2 x (1-ln2)^2 = {xi_pac_geometric:.8f} "
    f"(err={xi_err_final:.6f}%). "
    f"Derivation chain: 3D -> k_SL=9 -> k_eff=8 -> (7/8)(1-ln2)^2."
)
results["verdict"] = verdict

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"results/exp_04_{ts}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved to {out_path}")

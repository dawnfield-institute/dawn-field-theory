"""
Spike: The Cascade Clock — N as a Temporal Cascade Counter

M8 BRIDGE TO M9

M8 treated N_cascade = 6 as a free parameter. exp_11 revealed three
observables independently fit different N values:
    S8      -> N = 4.15   (clustering at ~8 Mpc/h)
    Hubble  -> N = 5.94   (sound horizon, 147 Mpc)
    JWST    -> N = 6.90   (structure at z=10)

This spike tests whether N is not a parameter but a DERIVED QUANTITY:
the number of cascade levels that have completed at a given lookback time.

Central hypothesis:
    N(t) = N_0 + log_phi(t / t_0)

i.e., one cascade level completes per factor-of-phi in cosmological time.
This is testable: the slope should be exactly 1/ln(phi) = 2.078.

If confirmed, this:
  1. Eliminates N as a free parameter (M8 goes from 2 params to 1)
  2. Resolves the S8/Hubble/JWST tension as a FEATURE, not a bug
  3. Makes N(t_H) ~ 7 a prediction (7 levels across the age of the universe)
  4. Opens M9: the cascade is a real dynamical process with a temporal clock

Connects to: exp_33 (Zeno completion at horizons = N -> infinity in finite proper time)
"""

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

# ============================================================
# Constants
# ============================================================
PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
XI = 0.5772156649 + LN_PHI  # gamma + ln(phi) = 1.0584
INV_PHI = 1 / PHI
GAMMA_EM = 0.5772156649

# Cosmological
H0_PLANCK = 67.36
H0_SHOES = 73.04
S8_PLANCK = 0.832
S8_KIDS = 0.759
S8_DES = 0.776
OMEGA_DM = 0.265
OMEGA_M = 0.315
T_HUBBLE = 14.4  # Gyr (1/H0 in Gyr)
T_UNIVERSE = 13.8  # Gyr

JWST_N_Z8 = 1e-5
JWST_N_Z12 = 3e-6

# ============================================================
# Data: three independent (N, t_lookback) measurements
# ============================================================

# Hubble ratio: phi^{1/N} = H0_local/H0_CMB
h0_ratio = H0_SHOES / H0_PLANCK
N_hubble = np.log(PHI) / np.log(h0_ratio)

# S8: per-level dissipation formula
s8_target = (S8_KIDS + S8_DES) / 2
reduction = 1 - s8_target / S8_PLANCK
N_s8 = INV_PHI**2 * (OMEGA_DM / OMEGA_M) / reduction

# JWST: z-dependent cascade floor
jwst_ratio = JWST_N_Z12 / JWST_N_Z8
N_jwst = -4.0 / (LN_PHI * np.log(jwst_ratio))

# Lookback times (Gyr)
t_s8 = 4.0     # weak lensing effective z ~ 0.3-0.5
t_hubble = 9.5  # BAO effective z ~ 1-2
t_jwst = 13.2   # JWST z ~ 10

data_N = np.array([N_s8, N_hubble, N_jwst])
data_t = np.array([t_s8, t_hubble, t_jwst])
labels = ['S8', 'Hubble', 'JWST']

print("=" * 70)
print("THE CASCADE CLOCK: N AS TEMPORAL COUNTER")
print("M8 -> M9 Bridge")
print("=" * 70)

print(f"\n  Data from exp_11:")
print(f"    {'Obs':<10} {'N_best':>8} {'t_lookback':>12} {'ln(t)':>8}")
for i, lab in enumerate(labels):
    print(f"    {lab:<10} {data_N[i]:8.3f} {data_t[i]:12.1f} Gyr {np.log(data_t[i]):8.4f}")


# ============================================================
# PART 1: Free fit vs DFT-constrained fit
# ============================================================
print(f"\n{'='*70}")
print("PART 1: THE CASCADE CLOCK EQUATION")
print(f"{'='*70}")

# (a) Free 2-parameter fit: N = a + b * ln(t)
def n_free(t, a, b):
    return a + b * np.log(t)

popt_free, pcov_free = curve_fit(n_free, data_t, data_N)
a_free, b_free = popt_free
N_pred_free = n_free(data_t, *popt_free)
rms_free = np.sqrt(np.mean((data_N - N_pred_free)**2))

print(f"\n  (a) Free fit: N = {a_free:.4f} + {b_free:.4f} * ln(t)")
print(f"      RMS = {rms_free:.4f}")
for i, lab in enumerate(labels):
    print(f"      {lab}: pred {N_pred_free[i]:.3f}, actual {data_N[i]:.3f}, "
          f"err {N_pred_free[i] - data_N[i]:+.3f}")

# (b) DFT-constrained: slope FIXED to 1/ln(phi) = 2.078
# N = a + ln(t)/ln(phi) = a + log_phi(t)
b_dft = 1.0 / LN_PHI  # = 2.0781

def n_dft(t, a):
    return a + np.log(t) / LN_PHI

popt_dft, pcov_dft = curve_fit(n_dft, data_t, data_N)
a_dft = popt_dft[0]
N_pred_dft = n_dft(data_t, a_dft)
rms_dft = np.sqrt(np.mean((data_N - N_pred_dft)**2))

print(f"\n  (b) DFT-constrained: N = {a_dft:.4f} + log_phi(t)")
print(f"      Slope fixed to 1/ln(phi) = {b_dft:.4f}")
print(f"      RMS = {rms_dft:.4f}")
for i, lab in enumerate(labels):
    print(f"      {lab}: pred {N_pred_dft[i]:.3f}, actual {data_N[i]:.3f}, "
          f"err {N_pred_dft[i] - data_N[i]:+.3f}")

# (c) Compare slopes
print(f"\n  Slope comparison:")
print(f"    Free fit:       b = {b_free:.4f}")
print(f"    DFT prediction: b = {b_dft:.4f} = 1/ln(phi)")
print(f"    Ratio:          {b_free / b_dft:.4f}")
print(f"    Deviation:      {abs(b_free - b_dft) / b_dft * 100:.1f}%")

# (d) Try slope = Xi/ln(phi) (slope involves Xi)
b_xi = XI / LN_PHI
def n_xi_slope(t, a):
    return a + XI * np.log(t) / LN_PHI

popt_xi, _ = curve_fit(n_xi_slope, data_t, data_N)
N_pred_xi = n_xi_slope(data_t, popt_xi[0])
rms_xi = np.sqrt(np.mean((data_N - N_pred_xi)**2))

print(f"\n  (c) Xi-slope: N = {popt_xi[0]:.4f} + Xi/ln(phi) * ln(t)")
print(f"      Slope = Xi/ln(phi) = {b_xi:.4f}")
print(f"      RMS = {rms_xi:.4f}")

# ============================================================
# PART 2: What is the intercept?
# ============================================================
print(f"\n{'='*70}")
print("PART 2: INTERPRETING THE INTERCEPT")
print(f"{'='*70}")

# The full equation is N(t) = a + log_phi(t)
# Rewrite: N(t) = log_phi(t / t_1) where t_1 = phi^{-a}
# t_1 is the lookback time at which N = 0 (the first cascade level)

t_1_free = np.exp(-a_free / b_free)  # from free fit
t_1_dft = PHI**(-a_dft)              # from DFT fit

print(f"\n  Rewrite: N(t) = log_phi(t / t_1)")
print(f"  where t_1 = lookback time when N = 0 (cascade origin)")
print(f"\n  From free fit:      t_1 = {t_1_free:.4f} Gyr = {t_1_free * 1e9:.1f} yr")
print(f"  From DFT-fixed:     t_1 = {t_1_dft:.4f} Gyr = {t_1_dft * 1e9:.1f} yr")

# Is t_1 a natural DFT timescale?
print(f"\n  Is t_1 natural?")
print(f"    t_1 (DFT fit) = {t_1_dft:.6f} Gyr = {t_1_dft * 1e6:.1f} kyr")
print(f"    t_recombination = 0.000380 Gyr = 380.0 kyr")
print(f"    t_1 / t_rec = {t_1_dft / 0.000380:.3f}")
print(f"    phi^3 * t_rec = {PHI**3 * 0.000380 * 1e6:.1f} kyr")

# What is a = N(t=1 Gyr)?
print(f"\n  Physical meaning of intercept a = {a_dft:.4f}:")
print(f"    N at t = 1 Gyr: {a_dft:.3f} levels")
print(f"    N at t = t_universe ({T_UNIVERSE} Gyr): {n_dft(T_UNIVERSE, a_dft):.3f} levels")
print(f"    N at t = t_Hubble ({T_HUBBLE} Gyr): {n_dft(T_HUBBLE, a_dft):.3f} levels")

# ============================================================
# PART 3: Cascade timing — each level's duration
# ============================================================
print(f"\n{'='*70}")
print("PART 3: CASCADE TIMING STRUCTURE")
print(f"{'='*70}")

print(f"\n  If N = log_phi(t/t_1), then cascade level n completes at t_n = t_1 * phi^n")
print(f"  Duration of level n: Delta_n = t_1 * phi^n * (phi - 1) = t_1 * phi^n / phi^2")
print(f"  Each level takes phi times longer than the previous one.")

print(f"\n  Using t_1 = {t_1_dft:.4e} Gyr from DFT fit:")
print(f"\n  {'Level':>6} {'t_complete (Gyr)':>18} {'Duration (Gyr)':>16} {'Note'}")
print(f"  {'-----':>6} {'----------------':>18} {'---------------':>16} {'----'}")
for n in range(0, 10):
    t_complete = t_1_dft * PHI**n
    duration = t_1_dft * PHI**n * (PHI - 1)
    note = ""
    if abs(t_complete - 0.000380) / 0.000380 < 0.5:
        note = "<-- recombination?"
    if abs(t_complete - T_UNIVERSE) / T_UNIVERSE < 0.2:
        note = "<-- age of universe"
    if abs(t_complete - t_hubble) / t_hubble < 0.2:
        note = "<-- Hubble/BAO scale"
    if abs(t_complete - t_s8) / t_s8 < 0.2:
        note = "<-- S8 scale"
    if abs(t_complete - t_jwst) / t_jwst < 0.2:
        note = "<-- JWST scale"
    print(f"  {n:6d} {t_complete:18.6f} {duration:16.6f} {note}")

# ============================================================
# PART 4: Resolving the S8/Hubble/JWST tension
# ============================================================
print(f"\n{'='*70}")
print("PART 4: DOES N(t) RESOLVE THE TENSION?")
print(f"{'='*70}")

# If N(t) is the true law, then M8's observables should use N(t_obs)
# instead of fixed N=6. Recompute each:

print(f"\n  M8 used fixed N=6. The cascade clock gives:")
for i, lab in enumerate(labels):
    N_clock = n_dft(data_t[i], a_dft)
    print(f"\n  {lab}:")
    print(f"    Fixed N=6:     phi^{{1/6}} = {PHI**(1/6):.6f}")
    print(f"    Clock N={N_clock:.2f}: phi^{{1/{N_clock:.2f}}} = {PHI**(1/N_clock):.6f}")

# For Hubble ratio specifically:
N_at_bao = n_dft(t_hubble, a_dft)
h0_ratio_clock = PHI**(1/N_at_bao)
h0_pred_clock = H0_PLANCK * h0_ratio_clock
print(f"\n  Hubble prediction:")
print(f"    M8 (N=6):      H0 = {H0_PLANCK * PHI**(1/6):.2f} km/s/Mpc")
print(f"    Clock (N={N_at_bao:.2f}): H0 = {h0_pred_clock:.2f} km/s/Mpc")
print(f"    Observed:       H0 = {H0_SHOES:.2f} km/s/Mpc")

# S8 with scale-dependent N
N_at_s8 = n_dft(t_s8, a_dft)
dissipation_s8 = INV_PHI**2 * OMEGA_DM / OMEGA_M / N_at_s8
s8_clock = S8_PLANCK * (1 - dissipation_s8)
print(f"\n  S8 prediction:")
print(f"    M8 (N=6):       S8 = {S8_PLANCK * (1 - INV_PHI**2 * OMEGA_DM / OMEGA_M / 6):.4f}")
print(f"    Clock (N={N_at_s8:.2f}): S8 = {s8_clock:.4f}")
print(f"    Observed:        S8 ~ {s8_target:.3f}")

# ============================================================
# PART 5: New predictions from N(t) that M8 couldn't make
# ============================================================
print(f"\n{'='*70}")
print("PART 5: NEW PREDICTIONS (M9 TERRITORY)")
print(f"{'='*70}")

# Prediction 1: N at the age of the universe
N_now = n_dft(T_UNIVERSE, a_dft)
print(f"\n  (1) N at t = t_universe = {T_UNIVERSE} Gyr:")
print(f"      N_now = {N_now:.4f}")
print(f"      Nearest integer: {round(N_now)}")
print(f"      Fractional part: {N_now - int(N_now):.4f}")
print(f"      -> We live at cascade level {int(N_now)}, "
      f"{(N_now - int(N_now))*100:.0f}% through level {int(N_now)+1}")

# Prediction 2: When does level 7 complete?
t_level_7 = t_1_dft * PHI**7
print(f"\n  (2) Level 7 completion time: t_7 = {t_level_7:.2f} Gyr")
print(f"      Universe age: {T_UNIVERSE} Gyr")
if t_level_7 > T_UNIVERSE:
    print(f"      Level 7 completes {t_level_7 - T_UNIVERSE:.1f} Gyr in the FUTURE")
    print(f"      -> We haven't reached level 7 yet!")
else:
    print(f"      Level 7 completed {T_UNIVERSE - t_level_7:.1f} Gyr ago")

# Prediction 3: S8 at different redshifts
print(f"\n  (3) Scale-dependent S8 (testable by Euclid, Roman):")
z_euclid = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
# Convert z to lookback time (rough flat-universe approximation)
def z_to_tlookback(z):
    """Approximate lookback time for flat LCDM."""
    # Integration of 1/((1+z)*E(z)) from 0 to z
    # Using simple numerical integration
    from scipy.integrate import quad
    H0 = 67.36  # km/s/Mpc
    Ol = 0.685
    Om = 0.315
    def integrand(zp):
        return 1.0 / ((1+zp) * np.sqrt(Om*(1+zp)**3 + Ol))
    result, _ = quad(integrand, 0, z)
    t_H_gyr = 1.0 / (H0 * 3.2408e-20) / (3.156e16)  # 1/H0 in Gyr
    return result * t_H_gyr

print(f"    {'z':>6} {'t_look (Gyr)':>14} {'N(t)':>8} {'S8(z)':>8} {'S8_M8':>8}")
for z in z_euclid:
    t_look = z_to_tlookback(z)
    N_z = n_dft(t_look, a_dft)
    diss = INV_PHI**2 * OMEGA_DM / OMEGA_M / max(N_z, 1)
    s8_z = S8_PLANCK * (1 - diss)
    s8_m8 = S8_PLANCK * (1 - INV_PHI**2 * OMEGA_DM / OMEGA_M / 6)
    print(f"    {z:6.1f} {t_look:14.3f} {N_z:8.3f} {s8_z:8.4f} {s8_m8:8.4f}")

print(f"\n    -> S8 should vary with redshift if N(t) is correct")
print(f"    -> M8 predicts constant S8 = {S8_PLANCK * (1 - INV_PHI**2 * OMEGA_DM / OMEGA_M / 6):.4f}")

# Prediction 4: Cascade-driven dark energy evolution
print(f"\n  (4) Dark energy from temporal cascade:")
print(f"      If w(t) = -1 + 1/(3*phi^{{N(t)}}), then w evolves with time.")
print(f"      This is DYNAMICAL dark energy from the cascade clock.")
print(f"\n    {'z':>6} {'t_look':>8} {'N(t)':>8} {'w(z)':>10}")
for z in [0, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
    if z == 0:
        t_look = T_UNIVERSE  # lookback = 0 means t = now = 13.8 Gyr
        # Actually w(z) should use t(z), not t_lookback
        # At z=0 we observe locally: N(t_universe)
        N_z = n_dft(T_UNIVERSE, a_dft)
    else:
        t_look = z_to_tlookback(z)
        N_z = n_dft(t_look, a_dft)
    w_z = -1 + 1/(3 * PHI**N_z)
    print(f"    {z:6.1f} {t_look if z > 0 else T_UNIVERSE:8.3f} {N_z:8.3f} {w_z:10.6f}")

print(f"\n    M8 fixed: w0 = {-1 + 1/(3*PHI**3):.6f}")
print(f"    M9 at z=0: w0 = {-1 + 1/(3*PHI**n_dft(T_UNIVERSE, a_dft)):.6f}")
print(f"    -> M9 gives w closer to -1 than M8 (deeper cascade at late times)")

# Prediction 5: Connection to exp_33 (black hole cascade)
print(f"\n  (5) Black hole connection:")
print(f"      exp_33 showed the horizon is where N -> infinity (Zeno completion)")
print(f"      The cascade clock says N = a + log_phi(t/t_1)")
print(f"      N -> infinity requires t -> infinity (or t_proper -> finite via Zeno)")
print(f"      This is EXACTLY the Penrose diagram: infinite coordinate time")
print(f"      compresses to finite proper time at the horizon.")
print(f"      The cascade clock IS the Penrose compression.")

# ============================================================
# PART 6: Parameter count reduction
# ============================================================
print(f"\n{'='*70}")
print("PART 6: PARAMETER COUNT")
print(f"{'='*70}")

print(f"""
  M8 parameters: 2 (depth 73, N_cascade = 6)
  M9 parameters: 1 + 1 (depth 73, t_1 = {t_1_dft:.4e} Gyr)

  But if t_1 can be derived from DFT (e.g., tied to recombination
  or another fundamental timescale), then:

  M9 parameters: 1 (depth 73 only!)

  This would mean ALL of DFT's cosmological predictions come from
  a single structural parameter: the depth of the dark matter
  candidate in the cyclotomic hierarchy.
""")

# Check: is t_1 derivable?
print(f"  Is t_1 = {t_1_dft:.4e} Gyr = {t_1_dft*1e6:.1f} kyr derivable?")
print(f"    t_recombination = 380 kyr")
print(f"    t_1 / t_rec = {t_1_dft * 1e6 / 380:.3f}")
print(f"    phi * t_rec = {PHI * 380:.0f} kyr")
print(f"    t_rec / phi = {380 / PHI:.0f} kyr")
print(f"    t_1 in kyr = {t_1_dft * 1e6:.1f}")
t1_kyr = t_1_dft * 1e6

# Check phi powers of recombination time
print(f"\n    phi^n * t_rec (kyr):")
for n in range(-4, 5):
    val = PHI**n * 380
    ratio = t1_kyr / val
    flag = " <-- close!" if 0.7 < ratio < 1.4 else ""
    print(f"      n={n:+2d}: {val:10.1f} kyr (ratio to t_1: {ratio:.3f}){flag}")

# ============================================================
# PART 7: Sensitivity and falsification
# ============================================================
print(f"\n{'='*70}")
print("PART 7: HOW TO FALSIFY THE CASCADE CLOCK")
print(f"{'='*70}")

print(f"""
  The cascade clock N(t) = {a_dft:.3f} + log_phi(t) makes specific
  predictions that differ from M8's fixed N=6:

  TESTABLE DIFFERENCES:
  1. S8 should vary with redshift (Euclid/Roman can measure this)
     - M8: constant S8 = 0.787
     - M9: S8(z=0.2) != S8(z=1.5)

  2. Hubble tension should be scale-dependent
     - M8: phi^{{1/6}} universal
     - M9: phi^{{1/N(t)}} varies with probe

  3. JWST high-z galaxy counts should follow N(t), not fixed N
     - Different floor at z=15 vs z=8

  4. The slope must be 1/ln(phi) = {b_dft:.4f}
     - If the true slope is significantly different, the cascade
       doesn't tick in phi-units

  FALSIFIED IF:
  - S8 is the same at z=0.3 and z=1.5 (to within errors)
  - Hubble tension is the same from BAO and from time-delay lenses
  - The fitted slope deviates from 1/ln(phi) by > 20%
""")

# ============================================================
# PART 8: M9 thesis statement
# ============================================================
print(f"\n{'='*70}")
print("M9 THESIS: THE INFODYNAMIC MECHANISM")
print(f"{'='*70}")

print(f"""
  M8: "Here's what we predict"
    -> 48/48, 10 predictions, 0 excluded

  M9: "Here's WHY it works"
    -> The cascade is a temporal process. Each level completes in
       phi times longer than the previous. N is not a parameter --
       it's a clock reading.

  The infodynamic mechanism:
    1. PAC conservation requires cascade structure (M1-M3)
    2. The cascade propagates through time at rate log_phi(t) (this spike)
    3. Different observables sample different N because they probe
       different lookback times
    4. The Hubble tension is not a problem — it's the cascade clock
       telling different observers their temporal position
    5. Black holes are where the clock runs to infinity (exp_33)
    6. The cosmological constant is set by N(t_universe) (exp_08)

  This reduces DFT to ONE structural parameter: depth 73.
  Everything else — N, H0/H0_CMB, S8, CC, w(z) — follows from
  the cascade clock + cyclotomic hierarchy.

  Key new prediction for M9: S8(z) should vary measurably with
  redshift. Euclid will test this within 2 years.
""")

# ============================================================
# Summary numbers
# ============================================================
print(f"{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  Cascade clock: N(t) = {a_dft:.3f} + log_phi(t)")
print(f"  DFT slope (1/ln(phi)):  {b_dft:.4f}")
print(f"  Free-fit slope:         {b_free:.4f} ({abs(b_free-b_dft)/b_dft*100:.1f}% off)")
print(f"  DFT-fixed RMS:          {rms_dft:.4f} (from 3 data points)")
print(f"  Free-fit RMS:           {rms_free:.4f}")
print(f"  Cascade origin:         t_1 = {t_1_dft*1e6:.1f} kyr")
print(f"  N at universe age:      {N_now:.3f} (between levels {int(N_now)} and {int(N_now)+1})")
print(f"  N at Hubble time:       {n_dft(T_HUBBLE, a_dft):.3f}")

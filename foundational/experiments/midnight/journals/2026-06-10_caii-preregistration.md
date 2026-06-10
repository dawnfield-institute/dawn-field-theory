# Pre-Registration: CaII Low-z Kill Test (exp_18)

**Date:** 2026-06-10
**Status:** REGISTERED BEFORE DATA — no CaII measurements examined as of this commit
**Experiment:** `scripts/exp_18_caii_low_z.py`

---

## Data state at registration

The only CaII file on disk, `data/sdss_mgii/CaII_Sardane_2014.tsv` (fetched 2026-06-08), is a
747-byte VizieR **error response** ("Table 'table3' does not exist in catalog: J/MNRAS/444/1747").
No CaII absorber measurements have been seen. The target catalog is Sardane, Rao & Turnshek 2014
(MNRAS 444, 1747), SDSS CaII absorbers (~435 systems, z ≲ 0.7), to be fetched from VizieR
**after** this registration is committed.

Phase A of exp_18 (CIV model comparison, z = 1.5–4.5) uses already-analyzed data and is
calibration, not registration. Its outputs are locked in
`results/exp_18_caii_low_z_20260610_100906.json`:
clock R² = 0.853 / BIC = 325.2 (2 params), z² R² = 0.864 / BIC = 322.2 (3 params),
z³ R² = 0.877 / BIC = 317.1 (4 params), halo virial R² = 0.780 / BIC = 368.8 (α = 0.0003).

## Locked model

Cascade clock, **no free parameters beyond a per-observable affine map** (amplitude + offset):

- N(z) = 1.360 + (1/ln φ) · ln(t_lookback[Gyr]), slope 1/ln φ = 2.0781
- Floor: N ≥ 1 (M9 boundary handling), crossed at **z\* = 0.061**
- Cosmology: H₀ = 67.36, Ωm = 0.3153, ΩΛ = 0.6847 (identical to exp_12/13)

## Registered predictions

1. **Steepening.** The clock's gradient grows sharply toward low z:
   dN/dz = 12.43 at z = 0.15, 5.58 at z = 0.30, 2.92 at z = 0.50, vs 0.32 at z = 2.0.
   That is a **4.3× steepening within the CaII range** (z = 0.15 vs z = 0.5) and **17.7×**
   relative to the CIV range (z = 0.3 vs z = 2.0). Polynomial mimics trained on the CIV range
   (z², z³ from Phase A) have gentle, near-constant low-z gradients. If absorber kinematics/EW
   track N, the CaII binned medians must curve the way the log does, not the way polynomials do.

2. **Floor flattening.** Below z\* = 0.061 the clock predicts *constant* N = 1 — evolution stops.
   No polynomial does this. (Likely untestable with ~435 systems; registered as qualitative B3.)

3. **Observable & method (locked).** Primary observable: rest EW of CaII K λ3934 (W3934),
   binned medians in z, ≥ 15 systems/bin, 6–15 bins over the catalog's z range.
   Secondary (if available): CaII doublet ratio W3934/W3969.

## Registered tests and decision rule

- **B1 (shape extrapolation, equal parameter count):** CIV-trained z² and z³ shapes vs the clock
  shape N(z), each given only an affine rescale (2 params) onto the CaII bins. Compare R²/BIC.
- **B2 (direct family fits):** each family fit directly to CaII bins with its own parameters
  (z: 2, z²: 3, z³: 4, clock: 2). Compare BIC.
- **B3 (floor):** qualitative check below z = 0.1.

**Decision rule (BIC, lower = better):**
- ΔBIC(clock − best polynomial) ≤ −6 in B1 **or** B2 → **discriminated FOR the clock**
- ΔBIC ≥ +6 in **both** B1 and B2 → **clock killed in this channel**
- otherwise → **inconclusive**

## Registered threats to validity

- CaII absorbers select dusty, high-column sightlines — a different population than CIV.
  A clock failure here scopes the clock's reach across populations; it does not falsify PAC.
- ~435 systems is small; bins will be noisy. Inconclusive is the likely outcome and will be
  reported as such.
- The Sardane catalog provides EW, not Doppler b — the cross-observable amplitude freedom
  (already conceded in Paper 12 §4.2) applies; this is a *shape* test only.

## Outcome commitment

The result — for, against, or inconclusive — goes in Paper 12 §6.1 and this journal,
whichever way it lands.

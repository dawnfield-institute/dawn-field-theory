# CaII Low-z Kill Test: Outcome (exp_18 Phase B)

**Date:** 2026-06-10
**Pre-registration:** commit `fbad01d1` (same day, before data download)
**Result:** `results/exp_18_caii_low_z_20260610_142903.json`
**Verdict (registered rule): INCONCLUSIVE**

---

## What happened

The Sardane, Rao & Turnshek 2014 catalog (VizieR J/MNRAS/444/1747/table1) was fetched
after the registration commit: 435 CaII systems, z = 0.028–1.343, observable W0a
(λ3934 rest EW). Binned to 12 medians per the registered rule (≥15 systems/bin).

One loader adaptation was made post-registration: the EW column in the VizieR export is
named `W0a`, not `W3934`; the column-candidate list was extended. No registered prediction,
binning rule, or decision threshold was touched.

## The numbers

| Test | Model | R² | BIC |
|------|-------|-----|-----|
| B1 (CIV-trained shapes, k=2 each) | z² shape | 0.051 | −55.4 |
| | z³ shape | 0.049 | −55.4 |
| | clock shape N(z) | 0.027 | −55.1 |
| B2 (direct fits) | z linear (k=2) | 0.052 | −55.4 |
| | z² (k=3) | 0.054 | −52.9 |
| | z³ (k=4) | 0.164 | −51.9 |
| | clock (k=2) | 0.027 | −55.1 |

ΔBIC(clock − best polynomial) = **+0.3** in both B1 and B2. The registered thresholds
were ±6. B3 (floor): only 1 bin below z = 0.1 — untestable, as anticipated.

## The honest reading

The verdict is inconclusive, but the *reason* matters: **every model fails**. The best fit
anywhere in the table is the 4-parameter cubic at R² = 0.16; everything else sits at
R² ≈ 0.03–0.05. CaII λ3934 median EW shows essentially no redshift evolution in this
sample. There is no trend for the clock and its mimics to fight over.

This is consistent with the registered threat to validity: CaII absorbers select dusty,
high-column sightlines — a population whose EW statistics are dominated by local
conditions, not cosmic epoch. (Weak/no EW evolution in CaII samples is also what Sardane+
themselves report.) The low-z edge cannot discriminate the clock with this observable
and this sample size.

## What this changes

- Paper 12 §6.1: the CaII test moves from "forward program" to "performed, inconclusive —
  observable carries no usable z-trend." The discriminating burden now rests on the
  z > 5 edge (JWST NIRSpec) and the proposed ionization-crossover derivation.
- The clock fitting slightly *worse* than linear z here (0.027 vs 0.052, ΔBIC 0.3) is
  noise, not signal — but it is reported, not hidden.
- A future low-z discrimination attempt needs a kinematic observable (Doppler b or
  velocity spread) at z < 0.5, not EW — e.g., high-resolution follow-up of the CaII
  sample or low-z MgII kinematics.

## Score

Phase B executed exactly as registered: 1 outcome reported / 1 outcome registered.
No post-hoc model additions, no threshold adjustments, no observable swaps.

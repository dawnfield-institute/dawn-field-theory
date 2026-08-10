# Outcomes: Ionization Coupling Law (exp_19) and Severance as Regulator (exp_20)

**Date:** 2026-06-11
**Pre-registration:** commit `d9c77a81` (2026-06-10, before execution)
**Results:** `exp_19_ionization_coupling_20260611_082108.json`,
`exp_20_severance_regulator_20260611_081943.json`

Both experiments executed exactly as registered. Neither prediction survived its registered
test cleanly; both produced something sharper than what they replaced.

---

## exp_19: INCONCLUSIVE (registered rule: non-monotone primary curve)

| Test | Result |
|------|--------|
| T1 monotonicity (Spearman β vs IP) | FAIL — ρ = +0.54, p = 0.13 |
| T2 zero crossing at 27.2 eV | FAIL — E₀ CI95 = [6.11, 6.43] eV (artifact, see below) |
| T3 CI width < bracket | PASS (0.32 eV) |
| T4 CaII weakest coupling | **PASS** |

### The diagnosis: the metric is not survey-portable

The registered threat ("normalization cannot remove population differences") dominated.
XQR-30 β values are 1–2 orders larger than SDSS values for the same ions (tiny high-z EWs,
compressed N range): MgII is +0.215 in DR16 but −0.43 in XQR-30; SiII's β = −21.6 has
CI95 = [−69, +1.8]. Mixing surveys in one curve made the isotonic fit collapse to the
bottom edge — T2's "crossing at 6.11 eV" is the pooling artifact, not a measurement.
The Hartree prediction is therefore **untested, not killed**.

### What survives, within-survey (homogeneous data)

1. **CaII coupling is consistent with ZERO**: β = +0.030, CI95 = [−0.058, +0.101] — the only
   ion whose CI95 contains 0. The settled-end anchor of the coupling law is confirmed with an
   error bar, upgrading the exp_18 "flatness" from description to measurement (T4 PASS).
2. **SDSS-only ordering is monotone-consistent**: CaII +0.03 → FeII +0.13 ≈ MgII +0.22 →
   CIV +0.59 (all CIs exclude zero except CaII). Four ions is too few for significance,
   but the ordering matches the law.
3. **XQR-30-only reproduces the sign flip**: FeII −1.25, SiII −21.6 (wide CI) below;
   SiIV +15.1, CIV +10.4 (CIV CI95 = [+7.1, +17.9], solidly positive) above. The crossover
   exists in homogeneous high-z data, between 8.15 and 33.49 eV — consistent with the
   bracket and with 27.2 eV, but the SiII/SiIV error bars cannot localize it.

### What would test the Hartree prediction

The pin requires intermediate-IP statistics (CII 11.3, AlIII 18.8) that current catalogs
lack (46 and 30 valid XQR-30 rows — excluded by the locked occupancy floor, as registered;
AlII excluded with 80 rows / <4 bins). Forward path: a single-survey design — extended
XQR-30 / KODIAQ-class samples for CII and AlIII, or a within-survey rank-based coupling
metric. The registered prediction (27.2 eV) stands for that retest.

---

## exp_20: KILLED_no_clamp (registered rule)

| Prediction | Result |
|------------|--------|
| P1 baseline violates bound | PASS — Gini_A = 0.930 > 0.618 |
| P2 stress severance clamps ≤ 1/φ | **FAIL** — Gini_B = 0.941 (θ sweep 0.940–0.945) |
| P3 clamps AT 1/φ ± 0.02 | FAIL (off by 0.32) |
| P4 random-trigger null distinct | PASS |

Severance, as implemented with the validated M-R components, does not regulate the cascade —
at the loosest threshold it slightly *increased* concentration. The hypothesis as registered
is dead in this harness.

### What the corpse teaches

The all-d-edges-overstressed trigger (M-R exp_15) requires *collective* stress: a node severs
only when its neighbors are also overstressed. But rich-get-richer concentration produces an
**isolated peak surrounded by poor neighbors** — the trigger structurally cannot fire at the
one node that needs severing. Severance regulates *extended* overstress (the regime M-R
validated it in); it cannot regulate *isolated* concentration.

This is a clean division of labor, and it is consistent with what M11 already found:
singularity resolution there comes from **cascade saturation** (MVAE density ceiling), not
radiative bleed. Concentration has two distinct regulators in the framework — severance for
extended disequilibrium, saturation for isolated peaks — and exp_20 demonstrates the gap
between their domains is real: a peak that cannot radiate keeps growing until saturation.
That is qualitatively a black-hole statement, derived from a failed regulator test.

### Harness caveats (reported, not excuses)

- Near-deterministic dynamics: argmax-entry + φ-split collapses 50 trials to one trajectory
  (σ = 0.0000). The result is robust to the θ sweep but rests on one dynamical regime.
- The rate-matched control under-delivered (6 vs 46 events/trial — random picks usually hit
  below-mean nodes with no excess to sever). P4 passes trivially; a better control would
  rate-match on *severed mass*, not trigger probability.

### Forward path

Registered-next candidate (exp_21): test the **saturation** regulator on the same harness —
add the MVAE density ceiling (M11) instead of severance and ask whether Gini clamps, and at
what value. If saturation clamps where severance could not, the division of labor becomes a
quantitative prediction rather than an interpretation.

---

## Score and registry

- exp_19: 2/4, INCONCLUSIVE — Hartree prediction untested, CaII zero-coupling measured,
  sign flip confirmed in-survey.
- exp_20: 2/4, KILLED_no_clamp — severance cannot regulate isolated peaks; saturation/
  severance division of labor identified.
- Pre-registration discipline held: no metric changes, no threshold adjustments, no
  post-hoc model swaps. Outcomes reported as committed at `d9c77a81`.

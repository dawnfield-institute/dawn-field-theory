# Pre-Registration: Ionization Coupling Law (exp_19) and Severance as Regulator (exp_20)

**Date:** 2026-06-10
**Status:** REGISTERED BEFORE EXECUTION — neither experiment's registered quantities have been
computed as of this commit. Scripts contain the locked analysis; only loader/harness smoke
tests have been run.
**Experiments:** `scripts/exp_19_ionization_coupling.py`, `scripts/exp_20_severance_regulator.py`

---

## exp_19: The Ionization Coupling Law

### Origin

exp_18's pre-registered CaII test returned inconclusive because CaII λ3934 EW carries no
redshift trend at all (435 systems, every model R² ≤ 0.16). Set against the strongly evolving
CIV kinematics (R² = 0.853 vs N(z)), the barely-drifting MgII (ρ = +0.06), and the A-E plane
(low-IP species shrink with N, high-IP grow, crossover bracketed 18.8–33.5 eV), the failures
arrange themselves into one hypothesis:

**Coupling to the cascade clock is a monotonic function of ionization energy. Gas below a
threshold energy is PAC-settled (locally equilibrated; statistics frozen); gas above it still
actualizes and carries the clock.**

### Registered derivation of the crossover (committed BEFORE measuring the curve)

Milestone R exp_24 (4/4) resolved the DFT energy-scale problem: EM-scale energies are
α(d)² × m_mediator, validated by deriving the Rydberg as α²mₑc²/2 at 11.4 ppm. The Rydberg
is the *binding* energy of the settled state (virial-halved). Severing a PAC ledger costs the
**full Coulomb energy** e²/a₀ = α²mₑc² = **one Hartree = 27.20 eV** — the energy to remove
the partner from the interaction entirely, not to sit in the bound state.

**Registered prediction: the coupling zero crossing is at E_cross = α²mₑc² = 27.2 eV.**

The factor-of-2 choice (Hartree, not Rydberg) is committed here, before measurement, with the
severance argument above. The Rydberg (13.6 eV) falls below the known bracket; if the measured
zero lands at 13.6 rather than 27.2, the severance argument is wrong and that is reportable.

### Honest scoping

The bracket [18.8, 33.5] eV was already known from the A-E plane (exp_13 / XQR-30), and every
catalog used below is on disk and has been analyzed in other experiments. What is registered
is: the point value 27.2 eV, its derivation, the monotone functional form, the exact coupling
metric, and the decision rule — all stated before the coupling curve is computed. Under a
uniform prior on the known bracket, hitting any given ±2 eV window has ~27% probability; the
registered content is the derivation chain, the localization, and the kill condition.

### Locked metric

- Clock: N(z) = 1.360 + (1/ln φ)·ln(t_lookback), floor N ≥ 1; Planck-2018 cosmology
  (H₀ = 67.36, Ωm = 0.3153) — identical to exp_12/13/18.
- Per ion: bin systems in z (≥ 15 systems/bin, 6–20 bins over the ion's z range), take the
  median rest EW per bin, fit median = A + β·N(z); **coupling = β normalized by the ion's
  overall median EW** (fractional change per cascade level). Sign: positive = grows with N
  (strengthens toward early universe).
- Uncertainty: bootstrap over systems (1000 resamples), CI95 per ion.
- Data sources (all on disk): XQR-30 merged catalog (per-species W; z = 2–6.5) for
  FeII, SiII, CII, AlIII, SiIV, CIV, NV, MgII as available; SDSS DR16 MgII + FeII catalogs;
  SDSS DR12 CIV; Sardane 2014 CaII (W0a). Where an ion appears in two sources, both β values
  are reported; the XQR-30 value is primary for ions above z = 2 (uniform methodology),
  the SDSS value primary below.
- IP convention (locked, matches the published A-E plane — creation energy of the ion):
  AlII 5.99, CaII 6.11, MgII 7.65, FeII 7.87, SiII 8.15, CII 11.26, AlIII 18.83,
  SiIV 33.49, CIV 47.89, NV 77.47 eV. Ions below the locked bin-occupancy floor
  (in XQR-30: CII 46, AlIII 30, NV 26 valid rows) are excluded and listed,
  not silently dropped.
- Ions with < 4 usable bins are excluded and listed (no silent drops).

### Registered decision rule

1. **Monotonicity:** Spearman(β, IP) > 0 with p < 0.05 (one-sided) across ≥ 6 ions.
2. **Zero crossing:** isotonic (monotone) fit of β vs ln(IP); zero crossing E₀ with bootstrap CI95.

- **SUPPORTED:** monotone AND 27.2 eV ∈ CI95(E₀) AND CI95 width < 14.7 eV (the known bracket).
- **KILLED:** monotone AND 27.2 eV ∉ CI95(E₀).
- **INCONCLUSIVE:** non-monotone, or CI95 spans the full bracket.

Registered threats: different surveys have different selection (CaII dusty; XQR-30 high-z);
the normalization makes β dimensionless but cannot remove population differences. EW is a
column-density observable; the law may hold only for kinematic observables — if β(EW) is
non-monotone but β(b) is monotone (CIV b is the only kinematic column available across a
wide range), that is reported as a scoped result, not silently swapped.

---

## exp_20: Severance as the Regularity Enforcer

### Origin

midnight exp_17 (1/4): the φ-split clamps the **maximum node share** at exactly 1/φ = 0.618
(structural, per-node), but distribution-wide Gini concentration reaches 0.72 — the cascade
does not globally self-regulate. exp_16 (3/4): topology sets plateaus but fails φ-scaling.
Milestone R has every component of a regulator — stress-triggered severance (exp_15, 4/4,
universal k = 1.16 ± 0.02), equilibrium-shift ejection (exp_07, 10.6× efficient), exact PAC
conservation under ledger splitting (exp_01) — but the direct test (concentration with vs
without a severance channel) has never been run.

**Hypothesis: radiation (ledger severance) is the mechanism that propagates the local φ-bound
to the global distribution. A cascade with a stress-triggered severance channel clamps its
steady-state Gini at 1/φ; without it, concentration exceeds the bound.**

If confirmed, severance is not incidental to PAC dynamics — it exists *because* the regularity
bound must be enforced. This would supply Milestone R's missing "why."

### Locked design

- Dynamic accumulating cascade (NOT exp_17's single-shot deterministic split, whose Gini 0.72
  is simply the Gini of the geometric φ-sequence): N = 32 nodes on a chain; each generation an
  erasure event deposits a unit of potential φ-split along the chain starting from the current
  highest-value node (erasure occurs where structure concentrates — the rich-get-richer regime
  that produces the violation); nodes accumulate across generations; 200 generations × 50
  trials per condition.
- Stress: node share relative to the current mean. A node severs when stress > θ on the node
  AND both its edges connect to neighbors also above θ (the all-d-edges-overstressed trigger,
  M-R exp_15). Threshold sweep θ ∈ {1.2, 1.4, 1.6, 1.8, 2.0} registered; primary readout θ = 1.6.
- Severance: the node's excess above the mean leaves the system as an independent ledger
  (equilibrium-shift, M-R exp_07). Conservation invariant: retained + severed = injected,
  to machine precision, every generation, all conditions.
- Conditions: **A** no severance; **B** stress-triggered; **C** random-trigger severance
  rate-matched to B's realized severance rate (separates "stress enforces φ" from "any
  dissipation lowers Gini").

### Registered predictions

- **P1:** Condition A steady-state Gini > 1/φ = 0.618 (the violation reproduces dynamically).
  If A does NOT violate the bound, the harness is void — stop and report.
- **P2:** Condition B steady-state Gini ≤ 1/φ.
- **P3:** Condition B clamps AT the bound: |Gini_B − 1/φ| < 0.02 (not merely below it).
- **P4 (null):** Condition C does not land in 1/φ ± 0.02.

**Kill conditions:** Gini_B stays > 0.65; or B clamps at a value unrelated to 1/φ with C
behaving identically (then severance is generic dissipation, not a φ-bound enforcer).

---

## Outcome commitment

Both outcomes — supported, killed, or inconclusive — will be journaled and integrated
(Paper 12 §6 for exp_19; Milestone R notes for exp_20), whichever way they land.

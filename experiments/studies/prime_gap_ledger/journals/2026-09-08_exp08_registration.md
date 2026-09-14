# exp_08 registration — is the coherence transition just the depth moving?

**Date:** 2026-09-08 · **Layer:** arithmetic · every "φ" is Euler's totient.
**Status: SEALED by the commit that carries this file and the gate results** (`results/exp_08_gates_20260908_113619.json`).
Run after; scored to this text.
**Target script:** `scripts/exp_08_position_is_depth.py` · **Gates (passed first):** `scripts/exp_08_gates.py`.

## §0 The question, and what has been seen

Round 2 found, **in exploring mode and never registered**, that the residue bias climbs from the primes' value
at the origin to the uniform loop's value "within half a unit of depth" and then plateaus out to 10⁵⁰. That is
a coherence transition: at the origin every residue is the integer's own name and all moduli agree; far from it
they decorrelate.

This round asks whether that transition is **an independent phenomenon at all**, or is entirely the effective
depth moving. If position acts only through y_eff, then

> **δ_q(u) = F_c( φ(q) / ḡ(y_eff(u)) )**, with F_c = tanh(a·x) and a = 1.2998 **sealed in exp_07**,
> and y_eff(u) solved from the measured density at each position.

**Zero free parameters.** Nothing is fitted in this round. If it holds, round 2's transition width is a
consequence of Buchstab's ratio relaxing to 1, not a constant of its own, and rounds 2, 3, 5 and 7 become one
statement.

**Seen before this seal.** All of rounds 1–7. The gate quantities at every position — counts, densities,
Buchstab ratios, y_eff — which contain no deficit (exp_03/06/07 precedent). **And, disclosed because it is
direct overlap:** round 2's exploring run measured **δ₃ at y = 4473 across positions**, the same modulus and
depth used here, on a coarser u-grid. q = 3 is therefore not fresh. It is one of 44 moduli; R1 is additionally
**recorded with q = 3 excluded**, and if the two disagree materially that is reported.

**Not seen:** δ_q at any position in this round, for any modulus, at this u-grid.

## §1 Objects and instruments

- **Depth** y = 4473 (round 2's), window L = 2·10⁶, loop period vastly exceeding the window.
- **Positions:** u ∈ {2.0, 2.1, 2.2, 2.35, 2.5, 2.65, 2.8, 3.0, 3.5, 4.5, 6.0}, with N = round(y^u).
  **The grid samples u directly** — see §2's note on the revision.
- **The floor at u = 2 is principled, not chosen:** below u = 2 every y-rough number in the window *is* prime,
  since a composite with all factors > y would exceed y² > N. Those positions are a different object; the gates
  show Buchstab missing by 15.7 % there against ≤ 0.4 % above. Not scored, at any u < 2.
- **y_eff(u):** solved from the measured density at each position (`y_eff_from_density`, 2·10⁶ grid — y_eff
  *exceeds* y near the origin, reaching 12,149 at u = 2).
- **Modulus pool:** q ∈ [3, 60] excluding q = 2·odd (δ_{2q} = δ_q, exp_04's theorem, gate G5). Saturated
  moduli recorded, never scored.
- **F_c:** loaded from the sealed exp_07 gate file. Not refitted, not rescaled.

## §2 Gates (all 8 PASS; `results/exp_08_gates_20260908_113619.json`)

| gate | claim |
|---|---|
| G1 | F_c and a load from the sealed exp_07 gate file |
| G2 | the density ratio tracks Buchstab e^γω(u) within 5 % at every scored position (observed: ≤ 0.4 %) |
| G3 | the transition exists: the shift is +0.119 at u = 2 and decays to ~0 by u ≈ 2.5 |
| G4 | the y_eff solver resolves at every position |
| G5 | δ_{2q} = δ_q for odd q |
| G6 | **recorded, not gated** — see below |
| G7 | **POWER**: the fixed-depth control separates from the tracking model at ≥ 3σ by simulation over the actual cell count |
| G8 | all thresholds fixed here |

**Two things went wrong in the gates and are recorded rather than quietly fixed.**

**(a) G6 originally gated on the per-cell prediction span exceeding the per-cell residual, and FAILED at 0.65.**
That is the comparison shown to be invalid in exp_06's correction (`aace9b3d`) — the test aggregates over
cells, so the relevant spread is the sampling spread of the rms difference, which G7 simulates. I wrote the
discredited criterion into a fresh gate hours after correcting it. G6 is demoted to a **recorded quantity**,
with the superseded criterion named in the script. **It carries a real limit that §7 inherits: the effect is
sub-residual per cell and resolves only in aggregate, so no per-cell reading of this round is supported.**

**(b) The position grid was resampled before sealing.** The first run used powers of ten in N, which put
exactly one scored position inside u ∈ [2, 2.7] — the band where essentially the whole transition happens. The
grid now samples u directly. **Thresholds were not changed; only the sampling was.**

## §3 Registered relations (M = 3)

**R1 — position is depth.** δ_q(u) = F_c(φ(q)/ḡ(y_eff(u))) across all scored positions and moduli, zero free
parameters. **CONFIRM if** rms ≤ **1.75 × 0.03043 = 0.05325**. **KILL if** > 3.0 × = **0.09129**.
Recorded beside it: the same rms with q = 3 excluded (§0).

**R2 — the transition is universal across moduli.** With C_q(u) = [δ_q(u) − δ_q(6.0)] / [δ_q(2.0) − δ_q(6.0)],
the coherence curves collapse: **CONFIRM if** the standard deviation of C_q(u) across moduli, averaged over
scored u, is ≤ **0.15**. **KILL if** > **0.35**. Only moduli whose endpoint difference |δ_q(2.0) − δ_q(6.0)|
exceeds 3× the cell SE are included — otherwise C is a ratio of noise.

**R3 — the positive control.** Predicting every position at the **fixed** depth y, ignoring position, must be
**worse**. **CONFIRM if** rms(tracking) < rms(fixed) at ≥ 3σ by paired bootstrap (10,000 resamples, seed
20260912). G7 established this comparison has 3σ+ power before any δ was formed.

Precedence: **KILL first**, then CONFIRM, then CONVERGED, else INCONCLUSIVE.

## §4 What would count as vacuous

- **R1 vacuous if F_c is flat over the range these positions occupy.** Guard: F_c's span across scored cells
  > 10 × 0.03043.
- **R2 vacuous if the endpoints do not separate.** Guard above: moduli failing the 3×SE endpoint test are
  excluded and counted; if fewer than 10 moduli survive, R2 is INCONCLUSIVE.
- **R3 vacuous if y_eff ≈ y at every scored position.** Guard: at least three scored positions must have
  |log y_eff/log y − 1| > 0.01, else the control cannot discriminate.
- Saturated moduli recorded, never scored.

## §5 Kill scope

- **R1 KILL:** position does *not* act through the effective depth alone. Round 2's transition is then a
  separate phenomenon. Does not touch exp_05/07's collapse on the loop or at the primes.
- **R2 KILL:** the transition is modulus-dependent — there is no single coherence curve, and Era-1's
  phase-coherence intuition does not survive in this form.
- **R3 KILL:** tracking y_eff is not better than a fixed depth, which would undercut R1's interpretation even
  if R1's rms passed.

## §6 Counting basis and outputs

Up to 44 moduli × 11 positions = 484 cells, less saturated and R2-excluded ones. Scored relations: 3. Outputs
append-only to `results/exp_08_position_is_depth_<ts>.json`.

## §7 Registered threats

1. **Sub-residual per cell.** From G6: the prediction moves less across the whole u-range than the per-cell
   residual. Everything here rests on aggregation over cells. **No per-cell or per-modulus claim is supported**,
   and none is made.
2. **q = 3 is not fresh** (§0). Recorded with and without it.
3. **One depth.** y = 4473 only. Nothing is claimed about the transition at other depths.
4. **F_c is empirical**, carrying exp_07's ~0.030 floor into this round's residual.
5. **Buchstab is used as a gate, not an input.** y_eff comes from the *measured* density, not from ω(u); G2's
   agreement is a check on the object, not an assumption of the model.

## Outcome commitment

Outcomes filed the same day, in `journals/2026-09-08_exp08_outcomes.md`, citing this seal's commit hash.
Failures reported as failures.

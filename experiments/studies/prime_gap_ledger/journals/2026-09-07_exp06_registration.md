# exp_06 registration — do the primes inherit the collapse?

**Date:** 2026-09-07 (night) · **Layer:** arithmetic (a study; no physics; every "φ" here is Euler's totient).
**Status: SEALED by the commit that carries this file and the gate results** (`results/exp_06_gates_20260907_190634.json`).
Run after; scored to this text. Kills have the scopes in §5.
**Target script:** `scripts/exp_06_primes_inherit.py` · **Gates (passed first):** `scripts/exp_06_gates.py`.

## §0 This one is a prediction, not a postdiction — and exactly what was seen

Rounds 2–5 were postdictions tested out of sample. This is not. Every ingredient is already fixed:

- **F is sealed** (exp_05, seal `5d6acfc6`). It was fitted on **loop cells only** — CRT-uniform sampled loops
  at y ∈ {360 … 92,160}. No prime was involved in producing it, at any stage.
- **y_eff is solved, not fitted** — from the arc's measured density by `y_eff_from_density`, the round-3
  method, whose consistency is gate G5.
- Therefore **δ_q(primes) = F(φ(q)/ḡ(y_eff)) has zero free parameters.** Nothing is fitted in this round.

**Seen before this seal:** all of exp_04 and exp_05, including F's bin centres and means; round 3's *script*
(read for mechanics — `chunked_read`, `y_eff_from_density`); this round's gate quantities, which are **counts,
densities, y_eff values, mean gaps and transition counts only**. exp_03's gates set that precedent explicitly
("read the 10¹⁰ decade for its count and transition count only — no δ at 10¹⁰ was formed"). Density is a gate
quantity here; the deficit is not.

**Also seen, and disclosed because it is adjacent:** the study README quotes a handful of round-1 and round-3
derived numbers at the primes — ε ≈ 0.0093–0.0099, and one r_q pair at u = 3 for q = 3. Those are *derived*
(ratios against the loop at y) for **7 moduli at 2 decades**; R1 scores **δ_q itself** for up to 44 moduli at
3 decades, and δ_q(primes) for this pool has not been formed or read by me at any point.

**Not seen:** any δ_q at any decade in this round; `results/exp_03_*.json` (deliberately not opened).

## §1 Objects and instruments (closed at this seal; counting basis §6)

- **The object:** the decade [10^m, 2·10^m) sieved to y = ⌈√(2·10^m)⌉ — which *is* the set of primes of that
  decade (gate G2 checks the count against sympy's `primepi` difference, independently of our sieve).
- **Decades:** m ∈ {7, 8, 9}. Chunked with the residue carried across boundaries; transitions = n − 1 exactly (G3).
- **y_eff:** solved from the arc's measured density on the prime grid (`y_eff_from_density`), both bracketing
  primes recorded, mismatch in log M reported. **ḡ(y_eff) = e^γ·log y_eff**, whose agreement with the measured
  mean gap is G7.
- **F:** the sealed exp_05 bins, loaded from `results/exp_05_collapse_20260907_185245.json`, evaluated by the
  same linear interpolation between bin centres, clipped at the ends. **Not refitted, not rescaled, not shifted.**
- **Modulus pool:** q ∈ [3, 60] excluding q = 2·odd (δ_{2q} = δ_q exactly for odd q — exp_04's theorem, G4).
  A modulus is **recorded but never scored** if its φ(q)/ḡ(y_eff) falls outside F's fitted domain (G6), or if
  its δ is within 1 SE of 1.0 at every decade (saturated, carries no information).
- **Errors:** per cell, the de-trended chunk-scatter SE (`detrended_se_log`), as round 3.
- **Reference for tolerances:** exp_05's held-out rms, **0.03459**, recorded in the gate file.

## §2 Gates (all PASS before this seal; `results/exp_06_gates_20260907_190634.json`)

| gate | claim |
|---|---|
| G1 | F loads from the sealed exp_05 result and is non-degenerate (≥10 bins, range > 0.5, monotone) |
| G2 | the read at u = 2 **is** the primes: count == `primepi(2N) − primepi(N)` at every decade |
| G3 | the chunk carry is exact: transitions == n − 1 at every decade |
| G4 | δ_{2q} = δ_q for odd q — justifies the pool's exclusion |
| G5 | the y_eff solver is consistent: `mertens_product(y_eff)` reproduces the measured density to < 1e-3 |
| G6 | F's domain covers ≥ 20 moduli per decade; those outside are recorded, never scored |
| G7 | ḡ(y_eff) is the measured mean gap within 2 % at every decade |
| G8 | the comparison, tolerances, pool and controls are fixed here, before any δ_q is formed |

## §3 Registered relations (M = 3)

**R1 — the primes inherit F.** For every scored (q, m), predict δ_q = F(φ(q)/ḡ(y_eff)) with **no fitting**.
**CONFIRM if** rms(measured − predicted) ≤ **2.0 × 0.03459 = 0.0692**. **KILL if** > 3.0 × = 0.1038.
The factor 2 is registered because the primes are a different object from the loop F was built on; it is not
tuned to any observed value, and no prime δ has been formed.

**R2 — the depth shift is doing the work (positive control).** The identical prediction using **ḡ(y)** in place
of **ḡ(y_eff)** must be **worse**. **CONFIRM if** rms(y_eff) < rms(y), resolved at ≥ 3σ by paired bootstrap over
scored cells (10,000 resamples, seed 20260910). Without this, R1 could pass merely because δ is a smooth
function of φ and any nearby depth would serve.

**R3 — the ω(q) structure persists at the primes.** The residual (measured − predicted) correlates with ω(q),
the number of distinct prime factors, with the **same sign registered in exp_05: positive**, at ≥ 3σ.

Precedence: **KILL first** (itself resolved at ≥ 3σ), then CONFIRM, then CONVERGED, else INCONCLUSIVE.

## §4 What would count as vacuous

- **R1 vacuous if F is flat over the range the primes occupy.** Guard: F's range across the scored cells' x
  values must exceed 10 × 0.03459. Recorded either way.
- **R2 vacuous if y_eff ≈ y.** Guard: |log y_eff / log y − 1| must exceed 0.02 at every scored decade, else the
  control cannot distinguish the two and R2 is INCONCLUSIVE. Round 3 found the shift is Buchstab's ratio, so
  this is expected to hold — but it is checked, not assumed.
- **R3 vacuous on noise.** Guard: the residual rms must exceed the median cell SE.
- A modulus outside F's domain (G6) or saturated at 1 is recorded, never scored.

## §5 Kill scope

- **R1 KILL** (rms > 0.1038): the primes do **not** inherit the loop's collapse under the round-3 depth shift.
  Scope: this claim only. It does not touch exp_05's 3/3 on the loop, exp_04's theorem, or round 3's ρ_q = 0 —
  round 3 measured a ratio against the loop at y and is independent of F.
- **R2 KILL** (rms(y) ≤ rms(y_eff) at ≥ 3σ): the depth shift is not what carries the primes onto F. The
  inheritance could still hold in some other variable.
- **R3 KILL** (correlation negative at ≥ 3σ): the ω structure does not transfer from the loop to the primes.
  Scope: the singular-series reading only.

No threshold in this file is relaxed after the run.

## §6 Counting basis and outputs

Up to 44 moduli × 3 decades = 132 cells, less those recorded-not-scored under §1/§4. Scored relations: 3.
Outputs append-only to `results/exp_06_primes_inherit_<ts>.json`: per cell the measured δ and SE, y_eff and its
bracket, the predicted δ under both ḡ(y_eff) and ḡ(y), the residuals, the three verdicts with their numbers,
and every §4 guard.

## §7 Registered threats to validity

1. **F is a binned curve, not a closed form.** Interpolation error near a bin edge is part of the residual and
   is not separable. Recorded: the residual against distance to the nearest bin centre.
2. **The primes are one arc, not a sample.** Unlike the loop there is no window ensemble at fixed N; the SE is
   the de-trended chunk scatter, as round 3. Chunk count per decade is recorded.
3. **Three decades is a short lever.** If R1 confirms, it confirms at these decades only; no extrapolation to
   10¹⁰ or beyond is claimed.
4. **ω(q) and q may not separate at the primes** even though they did on the loop (exp_05: +0.686 vs −0.027).
   The residual's correlation with q is recorded beside ω's; if they do not separate, R3 is reported as
   consistent-with-both and said so plainly.
5. **The prediction inherits exp_05's own limits.** F carries exp_05's ~0.035 held-out rms as an irreducible
   floor; R1's tolerance is set as a multiple of it for that reason.

## Outcome commitment

Outcomes filed the same day as the run, in `journals/2026-09-07_exp06_outcomes.md`, citing this seal's commit
hash. Failures reported as failures. Guards and floors reported per cell.

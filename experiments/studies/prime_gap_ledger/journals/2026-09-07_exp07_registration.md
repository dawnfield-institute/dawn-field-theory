# exp_07 registration — the closed form, and where the primes actually read

**Date:** 2026-09-07 (night) · **Layer:** arithmetic · every "φ" is Euler's totient.
**Status: SEALED by the commit that carries this file and the gate results** (`results/exp_07_gates_20260907_195941.json`).
Run after; scored to this text.
**Target script:** `scripts/exp_07_closed_form.py` · **Gates (passed first):** `scripts/exp_07_gates.py`.

## §0 Disclosure — two of my own errors, and what that implies for R3

**(a) The 18-bin F used in exp_05 and exp_06 was a poor estimator.** Its bins were equal-width in x while the
cells concentrate at small x, so it was coarse where F is steep and noisy where cells were sparse. A
one-parameter closed form beats it on the loop's held-out cells (gate G2). Both sealed rounds' verdicts stand —
each was scored against tolerances fixed in advance and passed honestly — but part of why those tolerances
were comfortable is that I had handicapped the predictor. Recorded, not repaired retroactively.

**(b) exp_06's R2 diagnosis was wrong and has been corrected** (commit `aace9b3d`). I claimed the control was
underpowered by comparing per-cell signal to per-cell residual; the test aggregates over 117 cells and had
~4σ of power. The correction concluded that both y and y_eff were excluded and the depth sat between them.

**(c) And that conclusion is itself estimator-dependent, which is the reason R3 exists.** Redone with the
closed form, the depth position λ (0 = y, 1 = y_eff) moves from ≈ 0.40 under the binned F to **0.799, 0.750,
0.701** at m = 7, 8, 9 — much nearer y_eff, which is round 3's prediction. **My reading of the depth has now
changed twice under two estimators.** It is therefore registered here as a forecast on a fresh decade rather
than asserted: the three λ values fall on a line with steps of −0.049, giving **λ(m = 10) = 0.652**.

**Seen before this seal:** all of exp_04–exp_06 including the primes' δ at m = 7, 8, 9; the λ values above;
the closed form and its fitted a. **Not seen:** any δ_q at m = 10. The gates read m = 10 for its count,
density, y_eff and transition count only, as exp_03 and exp_06 did before it.

## §1 Objects and instruments

- **The closed form:** F_c(x) = **tanh(a · x)**, x = φ(q)/ḡ. Constrained by the asymptotics — F(0) = 0, linear
  at the origin, → 1 as x → ∞. `a` is fitted **on the loop's training moduli only** (exp_05's 30), never on
  primes and never on m = 10, and is frozen at this seal by gate G1.
- **The test object:** the decade [10¹⁰, 2·10¹⁰) sieved to y = ⌈√(2·10¹⁰)⌉ — the primes of that decade,
  count verified against sympy's `primepi` (G3), chunked in 100 chunks of 10⁸ with the residue carried (G4).
- **y_eff:** solved from the measured density (G5); ḡ(y_eff) = e^γ log y_eff, checked against the measured
  mean gap (G6).
- **Modulus pool:** q ∈ [3, 60] excluding q = 2·odd. **The closed form has no fitted domain**, so unlike
  exp_06 no modulus is dropped for falling outside it; saturated moduli are still recorded, never scored.
- **λ:** the position of the rms-minimising depth between y (0) and y_eff (1) in log, at **zero offset** —
  depth and a constant offset are degenerate (exp_06 correction), so λ is only defined with the offset fixed
  at zero, and that is how it is registered.

## §2 Gates (all PASS before this seal; `results/exp_07_gates_20260907_195941.json`)

| gate | claim |
|---|---|
| G1 | `a` fitted on loop training moduli only, generalising to the loop's held-out moduli |
| G2 | the closed form beats the 18-bin F on the loop's held-out cells, established before any prime is touched |
| G3 | the m = 10 read **is** the primes: count == `primepi(2N) − primepi(N)` |
| G4 | chunk carry exact at m = 10: transitions == n − 1 |
| G5 | the y_eff solver is consistent at m = 10 |
| G6 | ḡ(y_eff) is the measured mean gap within 2 % |
| G7 | **POWER** — the depth comparison resolves at ≥ 3σ, by simulating the sampling spread of the rms difference over the actual cell count. This is exp_06's error corrected at the gate rather than in the outcomes |
| G8 | every threshold, the λ forecast and its tolerance fixed here |

## §3 Registered relations (M = 3)

**R1 — the closed form predicts the primes at a fresh decade, zero further freedom.** `a` comes from the loop;
y_eff is solved from density; nothing is fitted at m = 10. **CONFIRM if** rms(measured − F_c) ≤ **1.5 ×** the
loop's held-out rms of 0.03043 = **0.04565**. **KILL if** > 2.5 × = **0.07608**.

**R2 — the closed form beats the bins on the primes.** Same cells, both predictors. **CONFIRM if**
rms(F_c) < rms(F_bins) at ≥ 3σ by paired bootstrap (10,000 resamples, seed 20260911). This is what makes §0(a)
a finding rather than an apology.

**R3 — the depth forecast.** λ(m = 10) is predicted at **0.652** from the linear trend at m = 7, 8, 9.
**CONFIRM if** |λ(10) − 0.652| ≤ **0.15**. **KILL if** λ(10) falls outside **[0.35, 0.95]** — that range spans
both the binned-F reading (≈0.40) and y_eff itself (1.00), so a kill means the depth is somewhere neither
estimator suggested.

Precedence: **KILL first** (resolved at ≥ 3σ where a σ applies), then CONFIRM, then CONVERGED, else INCONCLUSIVE.

## §4 What would count as vacuous

- **R1 vacuous if F_c is flat over the primes' x range.** Guard: F_c's span across the scored cells > 10 × the
  loop's held-out rms.
- **R2 vacuous if the two predictors agree everywhere.** Guard: the mean |F_c − F_bins| over scored cells must
  exceed the sampling spread of the rms difference; recorded either way.
- **R3 vacuous if the rms-vs-depth curve has no resolved minimum.** Guard: **G7's power test must have
  passed**, and in the run the rms at the minimum must be below the rms at both endpoints by more than the
  bootstrap spread. If not, λ is not measurable and R3 is INCONCLUSIVE regardless of where the argmin sits.
- Saturated moduli recorded, never scored.

## §5 Kill scope

- **R1 KILL:** the closed form does not carry to the primes at 10¹⁰. Does not touch exp_05/06's verdicts,
  which used the binned F and were scored on their own terms.
- **R2 KILL:** the bins are not worse; §0(a) is then wrong and should be struck.
- **R3 KILL:** the λ trend does not extrapolate. Scope: the depth's location only. It leaves R1 untouched —
  the primes could lie on F_c without λ following a line.

## §6 Counting basis and outputs

Up to 44 moduli at m = 10, less saturated ones. Scored relations: 3. Outputs append-only to
`results/exp_07_closed_form_<ts>.json`: per cell δ and SE, both predictions, the depth scan, the three
verdicts with their numbers, and every §4 guard.

## §7 Registered threats

1. **One decade.** R1 and R3 rest on m = 10 alone; m = 7–9 are already seen and cannot serve as tests.
2. **tanh is empirical.** The asymptotics justify a sigmoid, not this sigmoid. `x/(a+x)` was tested and fails
   badly (held-out 0.099 against 0.030), so it is not "any sigmoid fits" — but no derivation is claimed.
3. **λ's trend is three points.** A line through three points is weak evidence; that is precisely why it is
   registered as a forecast on a fourth rather than reported as a law.
4. **The offset remains degenerate with depth** and is fixed at zero by construction (§1). A round that
   breaks that degeneracy is not this one.
5. **`a` is frozen from the loop.** If the primes prefer a different a, R1 will show it as inflated rms; the
   best-fit a at m = 10 is recorded, never scored.

## Outcome commitment

Outcomes filed the same day, in `journals/2026-09-07_exp07_outcomes.md`, citing this seal's commit hash.
Failures reported as failures.

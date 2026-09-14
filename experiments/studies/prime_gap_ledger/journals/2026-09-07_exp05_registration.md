# exp_05 registration — the collapse: δ_q(y) = F(φ(q)/ḡ)

**Date:** 2026-09-07 (night) · **Layer:** arithmetic (a study; no physics; no φ_golden, Ξ or Fibonacci enters —
"φ" throughout this file is Euler's totient and nothing else).
**Status: SEALED by the commit that carries this file and the gate results** (`results/exp_05_gates_20260907_184901.json`).
Run after; scored to this text. Kills have the scopes in §5.
**Target script:** `scripts/exp_05_collapse.py` · **Gates (passed first):** `scripts/exp_05_gates.py`.

## §0 Postdiction disclosure — this round is a postdiction turned prediction, and says so

The collapse was **found by looking at exp_04's exploring data**, not predicted in advance. exp_04 set out to
find a closed form for β_q, the exponent of δ_q(y) ~ (log y)^{−β_q}. It failed, and the failure was diagnostic:
β is a local log-log slope on a curve approaching a bound, so its value is set by where on the curve the cell
sits, and it is therefore inseparable from the δ-level (exp_04 §5). β was the wrong object.

Fitting δ itself over 44 moduli × 6 depths (264 cells, ḡ spanning a factor 2.90), a single smooth curve in
**φ(q)/ḡ** gave R² = 0.9272 against 0.8354 for q/ḡ, 0.8339 for φ alone and 0.7862 for q alone; residuals
carried no remaining correlation with φ (t = +0.67) or log y (t = +0.60), and a correlation with ω(q), the
count of distinct prime factors, of +0.359 (t = +6.21).

**Seen before this seal:** all of exp_04, including that grid and those numbers; the gates of §2.
**Not seen:** any δ at the six depths of §1 — they are fresh values never read in exp_04 — under the fresh
seed 20260909; any held-out cell at any depth; any fitted F.

## §1 Objects and instruments (closed at this seal; counting basis §6)

- **The object:** the loop of units mod P(y), sampled CRT-uniformly (`loop_sample`), W = 40 windows of
  L = 2·10⁶, seed **20260909** (fresh; exp_04's exploring used 20260908).
- **Depths:** y ∈ {90, 360, 1440, 5760, 23040, 92160} — **none read in exp_04**. ḡ = e^γ·log y, whose identity
  with the loop's mean gap is gate G7.
- **Modulus pool:** q ∈ [3, 60] excluding q = 2·odd, which are not independent cells (δ_{2q} = δ_q exactly for
  odd q — proved in exp_04 §2, re-gated here as G2). 44 moduli.
- **Split (mechanical, fixed at this seal, G5):** held-out = every third modulus in sorted order —
  {5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57}; training = the other 30. No cherry-picking, and the
  held-out φ/ḡ range lies inside the training range at every depth (G6 — no extrapolation).
- **The fit F (no free choices remain, G8):** 18 bins, edges `linspace(min, max)` over the **training** x only,
  bin statistic = mean δ, linear interpolation between bin centres, clipped at the end bins.
  Primary x = **φ(q)/ḡ**. Control x = **q/ḡ**, identical procedure.
- **Errors:** per cell, the window-scatter SE (≥ 8 usable windows or the cell is dropped, loudly).

## §2 Gates (all PASS before this seal; `results/exp_05_gates_20260907_184901.json`)

| gate | claim |
|---|---|
| G1 | round 1's exact q = 3 deficits reproduce (5/12, 223/552, 2860783/8291520) |
| G2 | δ_{2q} = δ_q exactly for odd q — justifies the pool's exclusion |
| G3 | the sampler is unbiased against the enumerated exact value |
| G4 | the declared seed reproduces δ bit for bit |
| G5 | the split is mechanical, disjoint, and covers the pool |
| G6 | no extrapolation: held-out φ/ḡ inside the training range at every depth |
| G7 | loop mean gap = 1/mertens_product — ḡ is the measured quantity |
| G8 | the fit is fully specified here: bins, edges, statistic, interpolation, tolerance |

## §3 Registered relations (M = 3)

**R1 — the collapse generalises.** F fitted on the 30 training moduli predicts the 14 held-out moduli at all
six depths. **CONFIRM if** held-out rms residual ≤ 1.5 × training rms residual. This is a generalisation test,
not a goodness-of-fit: F never sees a held-out modulus.

**R2 — the variable is φ/ḡ, not q/ḡ (the positive control).** The identical procedure run with x = q/ḡ must do
**worse** on the held-out cells. **CONFIRM if** rms(φ/ḡ) < rms(q/ḡ) on held-out, with the difference resolved
at ≥ 3σ by a paired bootstrap over held-out cells (10,000 resamples, seed 20260909). Without this, R1 could
pass on any variable that merely sorts the data.

**R3 — the residual is structured by ω(q).** On held-out cells only, the residual δ − F(φ/ḡ) correlates with
ω(q), the number of distinct prime factors, **positively**, at ≥ 3σ. Sign registered in advance: **positive**.

Precedence, as round 3: **KILL first** (and a KILL must itself be resolved at ≥ 3σ), then CONFIRM, then
CONVERGED, else INCONCLUSIVE.

## §4 What would count as vacuous

- If R1 passed because F is nearly flat — a constant predicts as well. **Guard:** R1 counts only if F's range
  across the training bins exceeds 10 × the training rms residual. Recorded either way.
- If R2 passed because q/ḡ and φ/ḡ are near-identical orderings. **Guard:** Spearman correlation between the
  two orderings over all scored cells is recorded; above 0.99 the control is declared uninformative and R2 is
  INCONCLUSIVE regardless of the rms comparison.
- If R3 passed on a residual that is pure noise. **Guard:** the residual's rms must exceed the median cell SE,
  else the "structure" is measurement error and R3 is INCONCLUSIVE.
- A modulus whose δ is within 1 SE of 1.0 at every depth carries no information; recorded, never scored.

## §5 Kill scope

- **R1 KILL** if held-out rms > 2.0 × training rms. Scope: the claim that F is universal across moduli at
  these depths. It does not touch exp_04's theorem, nor rounds 1–3.
- **R2 KILL** if rms(q/ḡ) ≤ rms(φ/ḡ) on held-out, resolved at ≥ 3σ. Scope: the identification of φ as the
  collapse variable. The collapse itself could still stand in another variable.
- **R3 KILL** if the correlation with ω(q) is **negative** at ≥ 3σ. Scope: the singular-series reading of the
  residual only.

A KILL on any relation is reported as a KILL. No threshold in this file is relaxed after the run.

## §6 Counting basis and outputs

44 moduli × 6 depths = 264 cells; 30 × 6 = 180 training, 14 × 6 = 84 held-out. Scored: R1, R2, R3 = 3.
Outputs append-only to `results/exp_05_collapse_<ts>.json`: every cell's δ and SE, the fitted bin centres and
means for both x, the per-cell residuals, the three verdicts with their numbers, and the §4 guards.

## §7 Registered threats to validity

1. **The collapse is a postdiction** (§0). R1's held-out design and R2's control exist for exactly this reason.
2. **ḡ = e^γ log y is asymptotic**; at y = 90 the Mertens error is largest. Recorded per depth: the measured
   mean gap against e^γ log y. If they differ by > 2 % at any depth, that depth's cells are recorded and not
   scored, and the fact is reported.
3. **Binning is a choice.** 18 bins is fixed at this seal. The run also records rms at 12 and 24 bins as a
   sensitivity, **recorded, never scored** — the verdict is 18 bins as sealed.
4. **The residual test shares cells with the fit's tails.** R3 is evaluated on held-out cells only.
5. **ω(q) correlates with q over this pool.** The residual's correlation with q is recorded beside it; if
   ω and q cannot be separated here, R3 is reported as consistent-with-both and said so plainly.

## Outcome commitment

Outcomes are filed the same day as the run, in `journals/2026-09-07_exp05_outcomes.md`, citing this seal's
commit hash. Failures are reported as failures. Floors and guards are reported per cell.

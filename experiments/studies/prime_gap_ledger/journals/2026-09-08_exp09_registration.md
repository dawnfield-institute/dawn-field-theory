# exp_09 registration — is F's residual real structure, or was it our sampling?

**Date:** 2026-09-08 · **Layer:** arithmetic · every "φ" is Euler's totient.
**Status: SEALED by the commit that carries this file and the gate results**
(`results/exp_09_gates_20260908_132250.json`, 8/8 PASS). Run after; scored to this text.
**Target script:** `scripts/exp_09_exact_residual.py` · **Gates:** `scripts/exp_09_gates.py` ·
**New instrument:** `core/exact_gaps.py`.

## §0 Why this round exists, and what is new

Every residual this study has reported was measured against **sampled** δ_q: the ~0.030 rms under F_c, and
the ω(q) correlation at t ≈ 8 in exp_05 and exp_06. Those numbers cannot distinguish *"F is incomplete"* from
*"our sampling is noisy."* The question has been confounded since exp_05 and no amount of extra sampling
resolves it — more windows shrink the noise but never tell you how much of the residual was noise.

**What is new: δ_q is now derivable exactly.** `core/exact_gaps.py` computes the loop's gap distribution by
inclusion–exclusion over the interior positions —

```
N(gap = g) = sum over S subset of {2,4,...,g-2} of (-1)^|S| * T(S union {0,g})
T(A)       = prod_{p <= y} (p - |A mod p|)          a k-tuple count, exact
```

— which is the **Boolean-lattice Möbius inversion** identified as this study's obstruction on 2026-09-08
(`note_the_sieve_recursion_and_where_it_fails`), computed rather than avoided. Gate G1 verifies it against the
enumerated loop at k = 9 to **4.6 × 10⁻¹⁵**, and G2 verifies the resulting δ_q to −0.0 at printed precision.

This does **not** derive F. It derives δ_q, which lets F be tested against a derivation instead of against a
measurement. tanh remains an empirical form with one fitted constant, frozen since exp_07.

**Seen before this seal:** all of rounds 1–8 and today's notes; the gate quantities. **Not seen:** any residual
of exact δ against F_c, at any cell.

## §1 Objects and instruments

- **δ_q, derived.** Hybrid by construction: exact by inclusion–exclusion for gaps ≤ G = 40, plus a **measured**
  tail above it (24 windows of 2·10⁶). Truncation is therefore a small reported quantity, not a bias — without
  the tail, any q whose multiples all exceed G would return δ_q = 1 trivially. Exact mass per depth: 1.000000,
  0.999960, 0.999415, 0.996831, 0.992051 (G3).
- **Depths:** y ∈ {23, 47, 97, 199, 401}; ḡ = 1/mertens_product ∈ [6.11, 10.77]. **ḡ is the exact mean gap
  here, not the asymptotic e^γ log y** — the derivation gives it exactly, so nothing needs approximating.
- **Moduli:** q ∈ [3, 30] excluding q = 2·odd (δ_{2q} = δ_q, exp_04's theorem, G5). 21 moduli × 5 depths = 105
  cells, x = φ(q)/ḡ ∈ [0.186, 4.581].
- **F_c:** tanh(a·x) with **a = 1.2998 loaded from the sealed exp_07 gate file** (G4). Not refitted, rescaled
  or shifted.
- **The extrapolation wall:** G6 records that **zero** cells fall below x ≈ 0.085, so nothing here relies on
  the unmeasurable region flagged in `note_F_below_x_0085_is_extrapolation`.

## §2 Gates (8/8 PASS; `results/exp_09_gates_20260908_132250.json`)

| gate | claim |
|---|---|
| G1 | the derived gap distribution reproduces the enumerated loop, max diff 4.6e-15 |
| G2 | derived δ_q matches enumerated δ_q at y = 23 (diff −0.0 at printed precision) |
| G3 | truncation controlled: exact mass > 0.99 at every depth, remainder measured |
| G4 | F_c and a = 1.2998 load from the sealed exp_07 gate file |
| G5 | δ_{2q} = δ_q for odd q, 42/42 |
| G6 | x ∈ [0.186, 4.581]; **0 cells** below the extrapolation wall |
| G7 | **POWER on the actual grid**: simulating the actual t-test on the actual ω vector at the reported effect size, median t = 7.71, **100 %** of replicates reach 3σ |
| G8 | all thresholds fixed before any residual is formed |

G7 is written against the procedure as built, not an idealisation of it — the correction earned across three
rounds of botched power estimates (exp_06's per-cell/aggregate error, and exp_08's gates twice).

## §3 Registered relations (M = 3)

**R1 — the residual is real structure, not sampling noise.** rms(exact δ_q − F_c) over the 105 cells.
**CONFIRM if ≥ 0.60 × 0.03043 = 0.01826** — at least 60 % of the measured residual survives on noise-free
values, so F is genuinely incomplete. **KILL if ≤ 0.30 × = 0.00913** — the residual mostly vanishes, meaning it
was our sampling and F is near-exact. Both outcomes are informative; the KILL is the more interesting one.

**R2 — the ω(q) structure is real.** On exact values, the residual correlates with ω(q), the number of distinct
prime factors, **positively**, at ≥ 3σ. Sign registered in advance, as in exp_05 and exp_06. G7 establishes
this resolves if the effect is real at its reported size.

**R3 — truncation is not driving it.** The residual computed at G = 36 and at G = 40 differs by **< 20 %**.
Otherwise the result is an artifact of the cutoff and R1/R2 are INCONCLUSIVE regardless of their values.

Precedence: **KILL first**, then CONFIRM, then CONVERGED, else INCONCLUSIVE.

## §4 What would count as vacuous

- **R1 vacuous if F_c is flat over this x range.** Guard: F_c's span across the scored cells > 10 × 0.03043.
- **R2 vacuous if ω is constant or near-constant on the pool.** Guard: ω takes ≥ 3 distinct values with ≥ 10
  cells each; recorded.
- **R2 confounded if ω and q cannot be separated here.** The residual's correlation with q is recorded beside
  ω's, as in exp_06 §7.4. If they do not separate, R2 is reported as consistent-with-both.
- **R3 vacuous if the tail correction is itself negligible at both G.** Recorded either way.

## §5 Kill scope

- **R1 KILL:** the residual was sampling noise; F_c is near-exact on the loop at these depths. This would
  retire the ω(q) residual as an artifact and make exp_05's and exp_06's R3 verdicts noise-driven — a
  substantial retraction, and the reason the threshold is registered in advance.
- **R2 KILL** (correlation negative at ≥ 3σ): the ω structure is real but opposite in sign to what exp_05 and
  exp_06 reported on sampled values. Scope: the ω reading only.
- **R3 KILL:** the cutoff drives the answer; the instrument is not yet trustworthy at these depths and both
  other relations are void.

## §6 Counting basis and outputs

105 cells (21 moduli × 5 depths). Scored relations: 3. Append-only to
`results/exp_09_exact_residual_<ts>.json`: per cell the exact and tail parts of P(q|g), δ_q, F_c's prediction,
the residual, and every §4 guard.

## §7 Registered threats

1. **The hybrid tail is measured, not derived.** Its contribution is 0.000000 to 0.007916 of the mass by depth.
   Recorded per cell; if the tail's share of P(q|g) exceeds 10 % for a cell, that cell is recorded, not scored.
2. **Depths are shallow** (y ≤ 401, ḡ ≤ 10.8) because inclusion–exclusion cost grows as 2^(g/2) and the mass
   must be captured. Nothing is claimed at the depths exp_05–exp_08 measured.
3. **F_c was fitted on sampled loop cells** at other depths. A residual against exact values partly measures
   that fit's own sampling error — which is the point, but it means R1 cannot separate "F's form is wrong"
   from "a was fitted on noisy data". Recorded: the best-fit a on exact values, never scored.
4. **This is the loop only.** Nothing about the primes.

## Outcome commitment

Outcomes filed the same day, in `journals/2026-09-08_exp09_outcomes.md`, citing this seal's commit hash.
Failures reported as failures.

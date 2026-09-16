# exp_32b outcomes — 4/4: the edge is a single crossing at κ_c = 1.168 ± 0.026, it does not move with gravity across a fourfold range, and it does not move with size. Only one of the three amendments did any work, and it was the instrument one.

**Registration:** `journals/2026-09-16_exp32b_registration.md`, sealed at dawn-field-theory **`3dbc0304`**
and pushed before any run. Instruments and runs: reality-engine `feat/v4-ledger-virial` at **`7be57cb`**.
Scorer `scripts/exp_32b_edge.py` (`--selftest` OK, 15 strings). Grids with per-run SHA256s in
`results/exp_32b_edge_grid_{fine,g,double}_20260916_132712.json`; scored JSON
`results/exp_32b_edge_20260916_132717.json`. **42 runs, 0 gate failures, no threshold moved.**

## The score: 4/4

| test | verdict | the numbers |
|---|---|---|
| **T1** the edge is a single crossing | **PASS** | one sign change per seed across κ ∈ [1.00, 1.30]; κ_c = 1.1975 / 1.1564 / 1.1491, mean **1.1677**, spread 0.0483, sd 0.0261 |
| **T2** the local edge coincides | **PASS** | median crosses in the same step (s17, s18) or the adjacent one (s16); fraction at the crossing 0.468 / 0.494 / 0.473 |
| **T3** the edge does not move with gravity | **PASS** | g = 0.75: −0.93 / −0.50 / −0.60 at κ = 1, +0.12 / +0.16 / +0.10 at 1.25. g = 3.0: −0.95 / −1.16 / −1.73 and +0.25 / +0.28 / +0.30. 12/12 |
| **T4** the edge does not move with size | **PASS** | n = 8000: −0.48 / −0.64 / −0.76 at κ = 1, +0.124 / +0.161 / +0.149 at 1.25; fraction 0.526 / 0.524 / 0.550 |

Neither kill fired. No vacuity clause triggered: the edge is resolved on the fine grid and the budget
binds in every arm.

## The result that matters: the edge reproduces on an independent seed triple

exp_32 read κ_c = 1.171 ± 0.038 unofficially on seeds 13–15. exp_32b scores **1.168 ± 0.026** on seeds
16–18. Two independent triples, three thousandths apart in the mean. That is what "the edge is a
number" was asking, and it is now a scored answer rather than an unscored reading.

The seed spread is real and is **not** small: 0.048 here, 0.072 there. SP5 predicted ≥ 0.05 and gets
0.0483 — **the one side prediction that misses**, by two thousandths, and it misses in the direction
that says the spread is a little tighter than the previous triple suggested. κ_c is a property of the
substrate to about ±0.03 in κ and of the seed's collapse geometry below that.

## What the three amendments actually did, tested rather than asserted

A 4/4 on a round whose registration loosened two bars is exactly where to expect tuning. It is not
what happened, and this is checkable from the grids:

| amendment | would the old threshold have changed the verdict? |
|---|---|
| identity gate 0.02 → **0.08** | **YES, decisively.** At 0.02 it fires on **32/42** — the round would have been UNSCORED a second time |
| T1's κ_c spread bar 0.05 → **struck** | **No.** Measured spread 0.0483; the old bar would have passed |
| T2's fraction band [0.45, 0.55] → **[0.40, 0.60]** | **No.** The three crossing fractions are 0.468, 0.494, 0.473 — all inside the *old* band |

So T1 and T2 passed on their merits under either threshold, and the only amendment that decided
anything is the one that is purely about what the integrator can resolve.

**And the rejected value was the right thing to reject.** The bearing from exp_32 proposed 0.06. At
0.06 the gate fires on exactly one run of 42 — **g = 3.0, κ = 1.00, seed 18, residual 0.0644** — and one
firing voids the run set. The 0.06 gate would have unscored this round too. It was rejected before the
seal on the argument that 1.12× an order statistic is not headroom; 42 runs later there is a
counterexample.

## A finding about the instrument, not the substrate: the truncation scales with gravity

The three worst residuals in the round are all **g = 3.0 at κ = 1.00** — 0.0644, 0.0555, 0.0514 —
while the whole fine sweep at g = 1.5 stays inside 0.0344 and the g = 0.75 arm inside 0.0194.

exp_32's §0.7 measurement pooled all 36 runs and reported 0.0154–0.0535. Pooling the arms hid the fact
that the tail belongs to one arm. The floor is not a constant of the integrator; it rises with the
gravitational coupling, which is what one would expect of a truncation. **For the next round the
resolution probe should report per arm, not pooled** — `scripts/resolution_probe.py` already separates
arms by (g, size), after its first draft got this wrong; the registration's §0.7 number is the pooled
one and should be read as a lower bound on the g = 3 tail.

## T4 passed here and was 2/3 unscored on the previous triple, with the threshold untouched

exp_32's seed 13 sat at −0.04 |U₀| at κ = 1.25, n = 8000. This round's three seeds are +0.124, +0.161,
+0.149 — no near miss. Nothing in T4 was changed between the rounds: same 3/3, same fraction band
[0.45, 0.60]. The difference is the seeds. The registered prior was 4 in 10 and it came in.

The κ = 1.30 point added to the size arm — registered as reported, not scored, so a failure could be
located — was not needed: W_p is +0.169 / +0.191 / +0.205 there, simply further past an edge already
crossed by 1.25. It cost 3 runs and would have been the difference between "T4 failed" and "T4 failed
and here is where the edge went".

## Side predictions (registered, unscored)

- **SP1 — the compression ratio crosses 1 within one step of W_p.** 3/3. Same step on seeds 17 and 18,
  adjacent on 16. The ledger identity's geometric reading holds on fresh seeds.
- **SP4 — `conn_q05` at κ = 1.25 below its κ = 1.00 value.** 12/12 across every arm and seed. The spine
  is gone past the edge whatever the couplings: 0.627 → 0.092, 0.653 → 0.081, 0.726 → 0.102 on the base
  geometry, and no recovery at 1.30 (0.089 / 0.077 / 0.079).
- **SP5 — κ_c's spread ≥ 0.05.** **Misses**, 0.0483.
- SP2 and SP3 are recorded in the scored JSON.

## What stands

The edge is a single crossing of the bounded pressure's net work, at κ_c = 1.168 ± 0.026 on this
substrate, reproducing to three thousandths across two independent seed triples. It survives a
fourfold change in gravity (12/12 signs) and a doubling of particle count (3/3). The local edge — the
median particle — crosses with the global one. The compression ratio crosses one with it. The web's
spine is gone on the far side in every arm.

**What the derivation still does not give** is κ_c's value from the kernels and the clock. The scaling
argument predicted the *invariances* and they hold; the number itself is measured, and its seed spread
says it is a property of the collapse geometry at the ~0.03 level.

## What was mine

Three registration errors of one class across exp_30–32 — a bar set without measuring the instrument.
This round set every bar from a measurement recorded in the seal, and the measurement is in the repo
(`scripts/resolution_probe.py`) so the seal is auditable. Two residual faults, both disclosed above:
the pooled §0.7 floor that hid the g = 3 tail, and the probe's own first draft, which merged arms and
disagreed with the sealed κ_c spread — the disagreement is what caught it.

## Bookkeeping, and one question for Peter

Milestone R's scorecard counts exp_32's four tests in the denominator while recording it UNSCORED, so
71/128 charges an unscored round as 0/4. With exp_32b at 4/4 the arithmetic is **75/132 (57 %)** under
that convention, or **75/128 (59 %)** if an unscored round is excluded from the denominator rather than
zeroed. The README is updated to 75/132; which convention Milestone R wants is Peter's call and the
rows for both exp_32 and exp_32b stay in the record either way.

## The bearing

**R2 is unchanged and unblocked.** Its substrate is κ = 0.5, four steps below a now-scored edge in
every seed and every arm. Its three open design questions — the ratio trigger beside `local_edge`,
κ = 1 as contrast or second claim arm, and t_arm — are Peter's and are still open
(`internal/dft/2026-09-15_exp33_r2_registration_DRAFT.md`, amended 2026-09-16 to make T3 an ordering
test rather than a factor with no prior).

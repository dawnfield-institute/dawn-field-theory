# exp_31 outcomes — 3/4: at matched occupancy, paired, the ledgered engine at κ = 0.5 adds connectivity to gravity's web on six fresh seeds; κ = 1's signature and the edge reproduce; the spine clause hit a ceiling

**Registration:** `journals/2026-09-14_exp31_registration.md`, sealed at dawn-field-theory
**`b8ff9125`** and pushed before any run. Instruments and runs: reality-engine `feat/v4-pac-ledger-r1c`
(stacked on PR #12) — the connectivity instrument at `5f5d690` (the commit every run records), the grid
and one aggregator addition at `6322387`. Scored by `scripts/exp_31_paired_ledger.py` against the
sealed thresholds (`--selftest` OK, 15 strings; refuses seeds outside 7–12, incomplete grids, runs
without the observable). Grid JSON (per-run SHA256s, `results/exp_31_paired_ledger_grid_full_20260914_195725.json`)
and the scored JSON (`results/exp_31_paired_ledger_20260914_195725.json`) are in `results/`.
**No threshold moved.**

## Score: 3/4, and the kill does not fire

| Test | n = 4000, seeds {7, …, 12}, paired on seed |
|---|---|
| T1 the claim: Δ₁ = conn_q10(κ = 0.5) − conn_q10(κ = 0); sign 6/6, mean ≥ 0.05, paired t ≥ 3 | **PASS.** Δ₁ = 0.042 / 0.042 / 0.079 / 0.043 / 0.052 / 0.133 — **6/6, mean +0.065 ± 0.015, t 4.43**. |
| T2 gravity's web, better connected: spine lift ≥ 5/6 at t ≥ 2; contrast ratio in [0.85, 1.15], 6/6 | **FAIL.** Spine Δ₂ = −0.005 / 0.000 / 0.069 / 0.000 / 0.130 / 0.082 — **3/6** positive (t 2.01). Contrast ratio 1.00 / 1.13 / 1.01 / 1.06 / 0.99 / 1.02 — 6/6 in band. |
| T3 κ = 1's signature: vs κ = 0.5, body up ≥ 5/6, spine down ≥ 5/6; cv(κ = 1)/cv(κ = 0) < 0.70, 6/6 | **PASS.** Body +0.081 / +0.078 / +0.459 / +0.009 / +0.255 / +0.126 (6/6); spine −0.341 / −0.179 / −0.026 / −0.158 / −0.390 / −0.342 (6/6); contrast ratio 0.57–0.61 (6/6). |
| T4 the edge: at κ = 1.25 net pressure work > 0 and spine below gravity, 6/6; at κ = 1 net work < 0, 6/6 | **PASS.** Work / P₀ at 1.25: +0.14 / +0.14 / +0.10 / +0.10 / +0.15 / +0.08; at 1: −0.85 / −0.96 / −0.87 / −1.18 / −0.95 / −0.79; spine at 1.25 − gravity: −0.27 to −0.89 (6/6). |

Gates: transfer residual ≤ 3.8 × 10⁻¹⁰, closure ≤ 0.0073, `at_cap` ≤ 0.011, thirty runs finite, 62–128 s
each. Floors (uniform positions, 20 draws): conn_q05 0.037 ± 0.010, conn_q10 0.038 ± 0.009,
conn_q20 0.084 ± 0.027. Vacuity: gravity alone sits at conn_q10 0.33–0.92, far above the floor — there
was a web to add to; the budget binds; the arms resolve. Scorecard **68/124 → 71/128.**

## What the arms actually did

**T1 — the claim reproduces, at half the size.** On the six seeds the arm and the occupancy were chosen
on, the lift was +0.138; on six seeds they had never seen it is +0.065 ± 0.015 — smaller, as a post-hoc
number should be expected to shrink, and in every seed, at t 4.4 against a bar of 3. The lift is
largest where gravity's own web is worst (seed 12: gravity 0.80 → 0.93; seed 9: 0.33 → 0.41) and
smallest where gravity is already at 0.90–0.92 (seeds 7, 8, 10: +0.04 each). Occupancy at κ = 0.5 is
within 0.006 of gravity's in every seed (SP2): the arms were matched by the physics as well as by
construction. The legacy `percolation` — the statistic exp_29 registered at this very budget — lifts
by +0.092 ± 0.015, t 6.1, 6/6 (SP1): **exp_29's claim at κ = 0.5 was right, and its unpaired d > 2 bar at
n = 3 could not see it.**

**T2 — a ceiling, recorded as the fail it is.** Gravity alone's spine (the densest 5 %) is already one
component on seeds 7, 8 and 10: 0.99 / 1.00 / 0.99. There is nothing to add, the paired lift there is
−0.005 / 0.000 / 0.000, and "sign ≥ 5/6" cannot be met. On the three seeds with headroom (gravity's
spine 0.39 / 0.72 / 0.90) the lift is +0.069 / +0.130 / +0.082, 3/3. The contrast clause — the half of
T2 that says *it is gravity's web* — holds in 6/6 (ratio 0.99–1.13; void fraction +0.04 above
gravity's in 6/6, SP3). I registered a sign count on a statistic that saturates at 1, and half the
seeds saturated. The reading that survives is narrower than the clause: the ledgered engine at
κ = 0.5 never makes the spine worse, makes it better wherever it is not already whole, and leaves the
contrast alone. The score says 3/4 and that is the score.

**T3 — the pattern predicted from six seeds held on six others, clause by clause.** κ = 1 against
κ = 0.5: the body more connected in 6/6 (+0.17 on average), the spine less connected in 6/6 (−0.24),
the contrast 0.57–0.61 of gravity's in 6/6, occupancy +0.045 to +0.063 above gravity's in 6/6 (SP2),
void fraction −0.05 below (SP3). This is the regime exp_30 scored and misread: a fatter, smoother web
whose densest cells are more broken than gravity's own.

**T4 — the edge is where the sweep put it.** The bounded pressure's net work over the run is negative
at κ = 1 in every seed (−0.79 to −1.18 P₀) and positive at κ = 1.25 in every seed (+0.08 to +0.15 P₀),
and on the positive side the spine is gone (conn_q05 0.09–0.13 against gravity's 0.39–1.00) while the
body still stands (conn_q20 0.71–0.84). One budget step above the fattening regime the repulsion
stops being paid for by collapse and the web's spine does not survive it. The edge is between κ = 1
and 1.25.

**Side predictions.** SP1 holds (6/6, t 6.1). SP2 holds in both clauses (6/6, 6/6). SP3 holds (6/6, 6/6).
SP4 holds: KE/|U| ordered κ = 1 < 0.5 < 0 and 1.25 > 1 in 6/6 (0.33 < 0.54 < 0.82; 0.37). SP5 holds
(6/6).

## The reading

Three rounds asked one question and the answer depended on how it was asked. exp_29 asked it at
κ = 0.5, unpaired, at n = 3, against d > 2: fail. exp_30 asked it at κ = 1 on the same terms: fail, and
the post-mortem found the terms. exp_31 asked it paired, at matched occupancy, on a count deposit,
on six fresh seeds, at the budget the plateau sits on: **the ledgered engine adds connectivity to
gravity's web**, in every seed, by a modest and consistent amount — six or seven hundredths of the
largest-component fraction at the registered occupancy, nine hundredths on the legacy statistic —
without changing what the web is (contrast, occupancy, void fraction all gravity's). The kill did not
fire and the adder question is answered for this mapping: yes, at κ on the plateau, by a small amount.

What the same grid says about the mapping's shape, now as scored tests rather than a description:
κ = 1 is a different regime (T3), and κ = 1.25 is past the edge (T4). The operating point for a bound
substrate that still holds gravity's web better than gravity does is κ ≈ 0.5, not κ = 1: at κ = 1 the
substrate is more bound (KE/|U| ⅓ against ½) but the web it holds is a different, fatter object with
a broken spine, and the edge is one step above it.

**Exploring ≠ predicting:** the ceiling reading of T2 and the "operating point" sentence are readings
of a scored grid, not scored claims. T2 failed.

## The bearing

- **R2, registrable now:** severance on the ledgered substrate at n = 4000, **κ = 0.5** (bound at
  KE/|U| ≈ ½, gravity's web better connected, the pressure doing −4 P₀ of work, three steps below the
  edge), fresh seeds, paired on seed, connectivity at fixed occupancy as the structure statistic, the
  box-to-range condition carried. κ = 1 is a second substrate, not a margin.
- **The adder question is closed, answered.** A further round would be to *size* the effect (more
  seeds, a confidence interval), not to establish it; it is not needed for R2.
- **T2's clause** returns, if it returns, as "spine ≥ gravity's in 6/6 and > in every seed with
  headroom" — a form that does not saturate. Registered here as the lesson, not as a claim.
- **M18's dynamics question** now has a candidate substrate: bound, conserving, holding a web that is
  gravity's and better connected. Whether that web is φ-coupled is R3's question and is not touched.

## Process notes

Nothing ran before the seal; the seal and the instrument were pushed to the remote before the first
fresh-seed run. Found at scoring: the aggregator summarised `percolation`, `xi_u` and `occupancy` over
the window but not `cv` and `void`, which T2/T3 read and the marks already carried; two window means
were added to `exp_04` (`6322387`), the thirty runs were untouched (their JSONs and SHA256s are
unchanged), and the grid was re-aggregated — the first grid JSON was deleted before it was committed
anywhere. The scorer's synthetic exercise had carried the missing keys and so did not catch this;
the next scorer exercise uses a grid produced by the real aggregator on real runs. T2's sign clause
was registered on a statistic that saturates; that is a design error of the registration, recorded
in §5 of the next one. No threshold moved after any result. This journal was written once, from the
scored JSON.

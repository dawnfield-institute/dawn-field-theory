# Milestone R exp_31 — R1c, paired at matched occupancy on fresh seeds: 3/4; the ledgered engine at κ = 0.5 adds connectivity to gravity's web; κ = 1's signature and the edge reproduce

**Date**: 2026-09-14 · **Branch**: `exp/mr-31-paired-ledger` (stacked on `exp/mr-30-ledger-r1b`, PR #190) · **Registration sealed**: `b8ff9125`

## What

exp_30's post-mortem re-specified the adder question: the design had thrown away its pairing (one
seed = one initial condition shared by every arm), compared arms at unmatched occupancy (κ = 1's
overdense set 60 % larger than gravity's), and read a mass-weighted threshold sitting on the
count-two boundary. New instrument (reality-engine `5f5d690`, seven tests): connectivity of the
densest q of cells on a cloud-in-cell count field — occupancy matched by construction, no mass draw.
Registered paired on seed, six fresh seeds 7–12, κ ∈ {0, 0.5 (claim), 1 (contrast), 1.25 (edge), ∞},
thirty runs; every prior post hoc on seeds 1–6 and a κ sweep {0.25, 0.75, 1.25} on seeds 1–3, all
disclosed in §0. Sealed and pushed before any run.

## Result

**3/4; the kill does not fire.** T1 PASS: Δ conn_q10(κ = 0.5 − 0) = +0.065 ± 0.015, 6/6, paired t 4.4
(bar: 6/6, ≥ 0.05, t ≥ 3); the legacy percolation lifts +0.092, t 6.1, 6/6 — exp_29's claim at this
budget was right and its unpaired d > 2 bar at n = 3 could not see it. T2 FAIL: the spine clause
(sign ≥ 5/6) hit a ceiling — gravity's densest 5 % is already one component (0.99–1.00) on three seeds,
lift exactly zero there, +0.07 to +0.13 on the three with headroom; the contrast clause held 6/6.
T3 PASS: κ = 1 vs 0.5 — body up 6/6, spine down 6/6, contrast 0.57–0.61 of gravity's 6/6. T4 PASS: net
pressure work positive at κ = 1.25 (+0.08 to +0.15 P₀) and negative at 1 (−0.79 to −1.18) in 6/6, the
spine collapsed at 1.25 in 6/6. Scorecard 68/124 → 71/128.

**Reading.** The ledgered engine adds connectivity to gravity's web, in every seed, by a modest and
consistent amount, without changing what the web is. κ = 1 is a different regime (fatter, smoother,
broken spine) and 1.25 is past the edge. R2's substrate is κ = 0.5. T2's saturating sign clause is a
registration design error, recorded.

## Files

- `journals/2026-09-14_exp31_registration.md` (sealed), `2026-09-14_exp31_outcomes.md`;
  `scripts/exp_31_paired_ledger.py`; `results/` grid copy and scored JSON
- README (row 31, finding 44, one honest-failure row, P41; 71/128), meta.yaml, THEORY_MAP, ROADMAP;
  generated indexes
- Companion: reality-engine `feat/v4-pac-ledger-r1c` — instrument `5f5d690`, results `6322387`
  (`results/full_r1c/`, positions in the sidecars — NOT masses, corrected same day in the outcomes
  journal); the exploratory sweep in
  `results/full_explore/` on `feat/v4-pac-ledger-r1b`

## Process

Found at scoring: the aggregator did not carry `cv`/`void` window means the scorer reads; added
(`6322387`), runs untouched, grid re-aggregated. No threshold moved after any result.

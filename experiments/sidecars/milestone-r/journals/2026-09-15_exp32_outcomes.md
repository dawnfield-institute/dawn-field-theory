# exp_32 outcomes — UNSCORED by its own instrument gate; read unofficially, the edge is a single sign change on every seed at κ_c ≈ 1.17, gravity leaves it in place 12/12, and two of my registered bars were set without a floor

**Registration:** `journals/2026-09-15_exp32_registration.md`, sealed at dawn-field-theory **`091ddab2`**
and pushed before any run. Instruments and runs: reality-engine `feat/v4-ledger-virial` (runs at
`2b8c1d0`; grids committed at `7be57cb`). Scorer `scripts/exp_32_edge.py` (`--selftest` OK, 13 strings).
Grids (per-run SHA256s) in `results/exp_32_edge_grid_{fine,g,double}_*.json`. **No threshold moved.**

## The verdict by the letter: UNSCORED

The registration's instrument gates include *|ΔE_SEC − (T − W_p)| ≤ 0.02 |U₀|* on every run, "the
identities the derivation stands on, re-checked from the recorded marks". On 29 of the 36 runs the
residual is 0.022–0.053 |U₀|, so the gate fires and by the sealed text the runs are invalid and no
test is scored. **The gate was my error.** The residual is the integrator's known truncation: the
scoping note the registration cites records it at ~0.027 |U₀| (a constant ~7 × 10³ on all 48 prior
runs), and exp_29 carried the same truncation as a 10 % allowance on the pressure work. I sealed
a tolerance below the instrument's documented floor. The scorer output is the record
(the sealed script exits 2 with the 29 gate lines); nothing was re-scored under the seal.

A second sealed-scorer defect was found only because the gate had to be relaxed to read the
numbers: the T2 code path indexed the fine grid with a floating-point bracket value
(1.2000000000000002) and crashed. Fixed as a bug with the criteria untouched (`min` over the grid
instead of `tuple.index`), disclosed here; the sealed run never reached it.

## The unofficial reading (a scratch copy of the sealed scorer with the gate relaxed; not a score)

| seed | W_p/|U₀| at κ = 1.00 … 1.25 (step 0.05) | crossing | κ_c (interp.) | median crossing | positive-work fraction at κ_c |
|---|---|---|---|---|---|
| 13 | −1.26 −0.78 −0.31 −0.09 −0.001 +0.12 | (1.20, 1.25) | 1.20 | (1.20, 1.25) | 0.48 |
| 14 | −0.70 −0.33 −0.10 +0.08 +0.13 +0.18 | (1.10, 1.15) | 1.13 | (1.15, 1.20) | 0.47 |
| 15 | −0.89 −0.54 −0.32 −0.08 +0.03 +0.10 | (1.15, 1.20) | 1.19 | none (−0.02 at 1.25) | 0.48 |

- **T1 (would FAIL).** A single sign change on every seed, negative to positive — the edge is a number.
  By interpolation κ_c = 1.20, 1.13, 1.19: **1.17 ± 0.04**. The registered bar was a bracket spread ≤ 0.05;
  the brackets are 1.225, 1.125, 1.175, spread 0.10. I registered a spread bar with no prior (the
  fine grid had never been run) at half the spread three seeds show.
- **T2 (would FAIL, 2/3).** The median particle's cumulative pressure work crosses in the same or the
  adjacent step on seeds 13 and 14 and sits at −0.02 P₀/n at κ = 1.25 on seed 15 — the local edge trails
  the global one by about a step there. The positive-work fraction at the crossing is 0.47–0.48 in 3/3.
- **T3 (would PASS, 12/12).** g = 0.75: W_p/|U₀| −1.02 / −0.61 / −0.73 at κ = 1 and +0.06 / +0.15 / +0.12
  at 1.25; g = 3.0: −1.56 / −0.75 / −1.10 and +0.23 / +0.26 / +0.25. The positive-work fraction 35–38 % at
  κ = 1 and 46–54 % at 1.25 across a fourfold range of g. The reservoir at κ = 1 is 1.4–1.9 (g = 0.75)
  and 1.6–2.5 (g = 3) |U₀| — the entropy clock's leak, monotone, sign unchanged. The spine is gone at
  1.25 in every arm (conn_q05 0.09–0.17); the body that survives scales with g (conn_q10 0.14 vs 0.34–0.50).
- **T4 (would FAIL, 2/3).** n = 8000: W_p/|U₀| −0.59 / −2.15 / −0.65 at κ = 1; at 1.25 **−0.04** / +0.07 / +0.14.
  Seed 13's edge sits at or just above 1.25 at the larger size; the positive-work fraction at 1.25 is
  0.49–0.53. The edge moves upward by about a step from n = 4000 to 8000 on one seed, which is within
  the seed spread the fine sweep shows and is not resolved by this arm.
- Side predictions: the compression ratio E_SEC(end)/T crosses 1 within one step of the work's sign
  change on every seed (same step on 14 and 15, adjacent on 13); `conn_q05` at 1.25 is below its κ = 1
  value in every arm and seed.

## What stands, what fell, and what was mine

**Stands (exploring, since nothing is scored):** the edge is a single sign change on fresh seeds at
κ_c ≈ 1.17 with a seed spread of ~0.1 at this resolution; it does not move with gravity across a
fourfold range (twelve of twelve signs); the local edge (the median particle) crosses with or one step
behind the total; the compression ratio crosses one with it; the pair coupling cancels exactly (the
registration's §0.5, the reason it was never an arm). **Fell:** "the edge is a number of the substrate
to ± 0.05" — the seeds spread twice that; and "the edge does not move with size" — one seed at n = 8000
is a step higher. **Mine:** the gate below the instrument's floor (this round unscored); a spread bar
with no prior; a scorer path never exercised. Three registration errors of one class in three rounds —
a bar set without first measuring what the instrument can resolve.

## The bearing

- **exp_32b**, if Peter re-seals: the same fine sweep and gravity arms on seeds {16, 17, 18}, the identity
  gate at the documented floor (0.06 |U₀|, twice the measured residual), T1 as "a single negative-to-
  positive sign change in every seed, κ_c reported by interpolation with its seed spread" (no spread
  bar), T2 with "same or adjacent step" carried and the fraction band carried, T3 carried, T4 carried
  with the honest prior that it fails on one seed in three; and the scorer exercised on the real
  aggregator's output on every code path before the seal.
- **The derivation's status.** The scaling argument survived its test on fresh seeds (T3) and the pair
  coupling's exact cancellation is now spec; what the derivation does not yet give is κ_c's value from
  the kernels and the clock, and its seed spread says the number is a property of the seed's collapse
  geometry to ~0.1 in κ.
- **R2** is unchanged by this round: its substrate is κ = 0.5, three steps below the edge in every seed
  and every arm. Its form is Peter's open decision (`internal/dft/2026-09-15_exp33_r2_registration_DRAFT.md`, not in the repo until sealed).

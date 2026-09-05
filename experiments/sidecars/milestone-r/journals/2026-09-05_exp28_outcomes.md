# exp_28 outcomes — the derived trigger fires at the extrema of the field, and here the extrema are the cores

**Registration:** `journals/2026-09-05_exp28_registration.md`, sealed at dawn-field-theory
**`bf833113`**. Instruments and runs: reality-engine `feat/v4-derived-sink` — gates and calibration
at `aef135e` (pre-seal), runner at `2c2dd17`, proxy grid at `befc79a`, full grid at
`623f573` (PR reality-engine #10). Scored by `scripts/exp_28_dynamical_severance.py` against the sealed thresholds; the grid
JSONs (with per-run SHA256s) and the scored JSONs are in `results/`. Thresholds unchanged.

## Score: 0/4 — and the zero is the finding

| Test | Proxy (n = 1000, 3 seeds, τ ∈ {5, 10, 20}) | n = 4000 at τ\* = 10 |
|---|---|---|
| T1 the sink is real (`e_int^S < e_int^B0` every mark in [5, 15]) | **FAIL.** τ = 5, 10: 3/3 but vacuous. τ = 20 (informative): 1/3 — in seeds 1 and 3 the retained per-particle energy is *above* B0 at **every** mark | 3/3 (informative: 48–54% severed, 1827–2062 retained). The pass needs ≥ 2 of 3 τ on the proxy: FAIL |
| T2 holding (`perc_S > perc_B0` at matched count, 3/3, > 2× pooled σ) | **FAIL, with sign at τ = 20:** S 0.091 / 0.084 / 0.091 vs B0-matched 0.123 / 0.113 / 0.099 — **below in 3/3**, margin −0.023, pooled σ 0.009 (−2.6 σ) | **FAIL, a null:** S 0.051 / 0.046 / 0.051 vs B0-matched 0.076 / 0.037 / 0.058 — margin −0.007, pooled σ 0.014 (−0.5 σ) |
| T3 vs matched-energy drag D | UNINFORMATIVE (T2 failed). Reported at τ\*: D-matched 0.100 / 0.137 / 0.092 ≥ S 0.092 / 0.106 / 0.087 in 3/3 | UNINFORMATIVE. Reported: D-matched 0.077 / 0.036 / 0.063 vs S, margin −0.009, σ 0.015 |
| T4 vs matched-count random severance R | UNINFORMATIVE (T2 failed). Reported at τ\*: R 0.090 / 0.106 / 0.099 ≈ S; at τ = 20 R 0.131 / 0.146 / 0.108 > S 3/3 | UNINFORMATIVE. Reported: R 0.055 / 0.046 / 0.052 vs S 0.051 / 0.046 / 0.051, margin −0.001, σ 0.004 — identical |

Kills: **K1 and K2 do not fire** — both are conditional on T2 passing. Vacuity: τ = 5 (74–83%
severed, 181–292 retained) and τ = 10 (seed 2 retains 498 < 512) are vacuous on the proxy by the
sealed rule; τ = 20 is the proxy's informative τ; at n = 4000, τ\* = 10 is informative. τ\* = 10 by
the pre-declared default (null at every proxy τ). Every closure residual ≤ 6 × 10⁻⁷; `at_cap` ≤ 0.018;
no tick at the dt floor; every run finite. Scored file: `results/exp_28_dynamical_severance_20260905_232916.json`
(proxy grid `…grid_proxy_20260905_230513.json` + full grid `…grid_full_20260905_232915.json`).
Scorecard: **62/112 → 62/116.**

## What the arms actually did

**B0 cannot hold — now measured, three seeds.** Percolation of the set peaks at 0.37 / 0.21 / 0.18
at t = 3–4 (the first collapse) and dissolves to 0.10 / 0.08 / 0.06 over t ∈ [10, 15]; KE/|U| 11–18;
`e_int` +578 / +1036 / +602. The retained set is unbound ten-to-one, not seven-hundred-to-one: the
design pass's numbers were the self-propelling force's, and are lineage.

**The sink is real in the ledger and wrong in the average.** Energy leaves: `Σ loss_severance_energy`
= 4.1–7.1 × 10⁴ at τ = 20 (5.9–25 × 10⁴ at τ = 5), all positive, `Σ u_out` negative in every run
(−7 × 10³ at τ = 20, −6 × 10⁴ at τ = 5: the severed particles were bound to what stayed). But at
τ = 20 the per-particle energy of what remains *rises* in two seeds of three, at every mark: what
leaves carries less than the mean. This is the direction the registration named as live, and it is
what happened.

**Severance removes structure; random removal at the same count removes less.** At the informative
τ the severed set is less connected than a random subset of B0 of the same size, seed by seed, and
less connected than the random-severance control R, seed by seed (0.131 / 0.146 / 0.108 vs 0.091 /
0.084 / 0.091). At τ\* = 10, S and R coincide (0.092 / 0.106 / 0.087 vs 0.090 / 0.106 / 0.099):
when 40–50% is removed, *which* 40–50% no longer matters. The matched drag D — the tuned sink this
round set out to replace — holds as much or more than severance in 3/3.

**Side predictions (registered, unscored).** SP2 holds: severed fraction at τ = 10 is ordered
0.19–0.30 (`memory_decay` 0.90) < 0.40–0.50 (0.95) < 0.61–0.78 (0.98), and T2's sign is the same
at all three. SP3 misses: `Σ loss_landauer / Σ work_pressure` = 1.24 / 0.94 / 1.26 × 10⁻², against
the registered < 10⁻² — Landauer erasure is one percent of the pressure work, not a sink; S+L is
indistinguishable from S (0.074 / 0.097 / 0.119 vs 0.092 / 0.106 / 0.087). SP1 was not evaluated:
the grid carries no no-removal counter, and the design pass's no-removal numbers are pre-fix. SP4
is a gate and passed.

## The reading: the barrier fires at extrema

The trigger `min_{j ∈ neighbours} |S_i − S_j| > τ` fires only where *every* bond carries a large
gradient — at a local extremum of the field. Milestone R exp_15 established it on a field of
noise (the random-walk stress), whose extrema are wherever the noise happens to peak: random with
respect to the graph, which is exactly what made the p^d degree barrier and exp_16's universal
statistics come out. On this substrate the SEC entropy is not noise. It grows where the density is
high (`SECUpdate`'s `dense` rule), so its extrema are the collapse cores, and the barrier is a
**core detector**. What it severs is the bound, dense, connected part — `u_out < 0` in every run,
fired/retained KE ratio ≈ 0.5 at first firing — and the retained set is left hotter per particle
(T1) and less connected (T2) than either a random subset or a random removal.

So the exp_15 trigger, imported literally, is not a radiation channel on this substrate. Nothing in
this touches exp_15/16 themselves (graph results on a noise field) or Milestone R's thesis that
radiation is severance. What it kills is narrower and useful: *the entropy-gradient barrier cannot
be the amount-free rule that turns severance into a dynamical sink*, because on a structured field
it selects the structure — at the onset. Where it removes half (τ = 10, both sizes) its selection is
indistinguishable from random, and nothing holds either way.

## The bearing

- **C4.1 option 3 closes.** A derived severance channel, as Milestone R licenses it, does not let
  the v4 substrate hold structure: negative at −2.6 σ where it removes the first firers (proxy,
  τ = 20), a null where it removes half (n = 4000, τ\* = 10), and never above a random subset.
- **Option 2 is what remains:** the impulsive pressure that detonates the collapse (accel_p99 ≫
  free fall; KE/|U| from 0.07 to 10 between t = 3 and t = 5) is the term to derive from the SEC
  functional rather than dissipate after the fact. A sink that acts *after* the detonation is
  removing the wrong energy from the wrong particles; the fix is upstream of the sink.
- **A sign question for Milestone R, not a claim:** exp_14 found the *relaxation* triggers had the
  wrong sign; here the *stress* trigger fires where structure is. Whether a barrier on
  *under*-stress — a particle all of whose bonds carry a small gradient, the flat interior of a
  core — is the right dynamical reading is a registrable question and not this round's.
- **The engine keeps its instruments.** The ledger, the severance bookkeeping and the third-law
  pressure are gated and stay; the tuned drag stays a control, not an equation of motion.

## n = 4000 at τ\* = 10: a null, and what it bounds

Twelve runs (B0, S, R, D × seeds {1, 2, 3}; 2.0–2.4 min each; closure ≤ 3 × 10⁻⁷; `at_cap` ≤ 0.011).
B0 dissolves as predicted: percolation 0.035 / 0.038 / 0.028 over the window on its own 4000,
KE/|U| 18–22. Severance at τ = 10 removes 48–54% and leaves 1827–2062 — informative by the sealed
rule — and the retained set is **a statistical null against a random subset of B0 at the same
count** (0.051 / 0.046 / 0.051 vs 0.076 / 0.037 / 0.058; −0.5 σ) and **identical to random removal**
(R 0.055 / 0.046 / 0.052; −0.001 ± 0.004). The drag control, calibrated first-shot to KE/N ratios
1.08 / 1.09 / 1.11, gives 0.077 / 0.036 / 0.063 at matched count. T1 holds 3/3 here, as it did at
proxy τ = 10.

So the negative sign belongs to τ = 20, where the trigger removes only the first 6–10% to fire, and
it washes out at τ = 10, where it removes half. That is what a core detector diluted by its own
cascade looks like: the first firers are the cores (`u_out < 0`, KE ratio ≈ 0.5), and once the
neighbourhood degree collapses the barrier fires on everything. The proxy decided the question; the
full size bounds the mechanism: the selection effect is real at the onset and gone by the time
severance is a bulk process.

## Process notes (what the round cost and what it caught)

The design pass matched the drag control on a ratio of *signed* per-particle energies; that ratio
crosses zero as the set unbinds (ρ = −5.4 on the smoke test) and the control had collapsed onto B0.
Caught by the smoke test before the seal, replaced by matching the retained kinetic energy per
particle — positive-definite — with one pre-declared refinement; all three proxy calibrations landed
inside ×1.25 on the first attempt. The same smoke test printed sink-arm structure numbers on seed 9;
they are in the registration's §0.4, and the expected-direction text moved on their account before
the seal (T2 "open" → "leaning null"), which is what §0 is for. The aggregator once globbed its own
output and crashed the second aggregation; the first two scored JSONs in `results/` are the τ\*
selection on the pre-D grid, the third carries D. None of this touched a threshold.

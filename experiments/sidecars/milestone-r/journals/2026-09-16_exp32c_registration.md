# exp_32c registration — κ_c is predicted, seed by seed, from the initial large-scale power. Six predictions are written below before any run (SEALED by the commit carrying this file)

**Layer: physics → `theory/`** (THEORY_MAP sidecar-R row; ROADMAP Milestone R). Instruments unchanged
from exp_32b: reality-engine `feat/v4-ledger-virial` at `59f38a6`. Scored to this text by
`scripts/exp_32c_predict.py`, thresholds byte-equal to §4 (`--selftest`). Frame §3. Kill scope §6.

**Branched from `main`, not from exp_32b.** Four Milestone R rounds are stacked and unmerged because
every round edits the shared scorecard. This round touches no shared file: no README, no meta.yaml.
Its scorecard entry is a follow-up after the stack lands.

## Why this round exists, and what it replaces

exp_32b established the edge at κ_c = 1.170 ± 0.012 over six seeds with a seed-to-seed sd of 0.029.
I proposed testing whether that spread shrinks with system size. **That experiment cannot work at any
affordable size.** Comparing spreads between two arms of m seeds is an F-test on (m−1, m−1) degrees of
freedom; detecting the 1/√2 ratio in question at 80 % power needs roughly 40 seeds per arm, which is
63 hours of compute. Six seeds per arm can only detect a ratio near 2.5. Registering it would have
been registering a test that cannot fail informatively.

This round asks a better question that the same data suggested: **is κ_c predictable from the initial
condition?** If it is, the spread is not noise to be averaged away but structure to be read, and the
size question answers itself — a larger box self-averages the predictor.

## §0 Postdiction disclosure — and the exploration that produced the hypothesis

**All of this was computed before this seal and none of it is scored.**

1. **κ_c on six seeds.** 13–15 from exp_32 (UNSCORED, read with the gate relaxed): 1.2003, 1.1288,
   1.1851. 16–18 from exp_32b (scored 4/4, seal `3dbc0304`): 1.1967, 1.1546, 1.1491 on a κ grid halved
   to 0.025 over the crossing region. Mean 1.1691, sd 0.0290.
2. **The grid is not the spread.** Halving the κ step moved κ_c by at most 0.0017 and left the
   three-seed sd at 0.026. An earlier claim of mine that the estimator contributed a third of the
   spread was wrong: it compared polynomial fits to each other, and those fits were the unreliable
   estimators, not the linear interpolation. **The 0.029 is real.**
3. **Bulk clustering does not predict it.** |U₀| varies 0.7 % across the six seeds while κ_c varies
   6 %; r = −0.49, not significant at n = 6.
4. **Large-scale power does look like it might.** Define, on the initial shaped field δ_k as
   `particles.py` generates it, **f = Σ|δ_k|² over |k| ≤ 3 k_fundamental, divided by Σ|δ_k|² over all
   k.** Across the six seeds, r(f, κ_c) = **+0.878**.
5. **The honest caveats on that number, stated in full.** I tested four statistics on six points
   without pre-registering them, which puts the family-wise p < 0.05 line near |r| = 0.88 rather than
   0.81. At n = 6 the 95 % interval on r = 0.88 runs from roughly 0.2 to 0.98. **And I chose the cutoff
   by hand.** Sensitivity: r = 0.752 (1.5 k_f), 0.692 (2.0), 0.865 (2.5), **0.878 (3.0)**, 0.803 (3.5),
   0.673 (4.0), 0.672 (5.0), 0.557 (6.0), 0.352 (8.0). A broad hump, not a spike, and declining as the
   cut admits smaller scales — which is what large-scale dependence should look like. **Cut = 3 k_f is
   also a wavelength of 20 = 2r₀, the interaction diameter, which is why it is defensible as a choice
   rather than only as a fit. It is also where r peaks. Both are disclosed; a reader may discount it.**
6. **This round exists to test 4 out of sample.** In-sample fitting proves nothing; six frozen
   predictions on unseen seeds can fail.

## §1 Objects (closed at the seal)

- **Substrate:** exp_29–32b's, unchanged. n = 4000, box 60, r₀ = 10, g = 1.5, sec = Ξ/φ, lattice start.
- **Seeds:** {19, 20, 21, 22, 23, 24} — fresh, never run at any κ or coupling in any POC-12 results
  directory (verified: seeds 1–18 only).
- **Arm:** the fine sweep κ ∈ {1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30} — 42 runs.
- **κ_c:** the sign change of W_p on the fine grid, by linear interpolation between the bracketing
  marks, exactly as exp_32b defined it.

### The predictor, fixed to the bit

f is computed from the seed alone with no simulation. `torch.manual_seed(seed + 991)`, then
`torch.randn(16, 16, 16)` in float32, FFT, amplitude `k^(index/2)` with index = −1.727, DC zeroed;
spacing 3.75, box 60, k_fundamental = 2π/60 = 0.1047198. **Recorded environment: torch 2.13.0,
numpy 2.5.2** — f depends on torch's RNG stream, so the f values below are the contract, not the code.

### THE FROZEN LINE

    kappa_c = 1.011924 + 0.874965 * f

Fitted on seeds 13–18 only. Training residual sd **0.01555**; sd about the training mean 0.02907;
training mean κ_c **1.169100**. Neither coefficient moves after this commit.

### THE SIX PREDICTIONS, made before any run

| seed | f | predicted κ_c | ±2 residual sd |
|---|---|---|---|
| 19 | 0.213757 | **1.1990** | [1.1679, 1.2301] |
| 20 | 0.142599 | **1.1367** | [1.1056, 1.1678] |
| 21 | 0.126312 | **1.1224** | [1.0913, 1.1535] |
| 22 | 0.220468 | **1.2048** | [1.1737, 1.2359] |
| 23 | 0.137870 | **1.1326** | [1.1015, 1.1637] |
| 24 | 0.190808 | **1.1789** | [1.1478, 1.2100] |

Predicted spread 0.0824 — wider than the training spread. Seeds 21 and 22 sit outside the training
range of κ_c entirely (1.1288–1.2003), so the line is being asked to extrapolate, not interpolate.

## §2 Pre-seal numbers

§0.1–0.6 and the table above.

## §3 Frame

**Sampled:** the whole retained set, pair-form quantities. **Expectation:** for each fresh seed, the
number in the table, computed from its initial condition before the run existed. **Statistic:**
out-of-sample prediction error against the error of the null that predicts the training mean.

## §4 Tests (M = 4; thresholds fixed)

**T1 — the line beats the mean, out of sample.** Over the six fresh seeds, the RMS error of the frozen
line is **strictly smaller** than the RMS error of predicting the frozen training mean 1.169100.

**T2 — the sign holds on fresh seeds alone.** Spearman ρ(f, κ_c) over the six fresh seeds is **> 0**.

**T3 — the predictions are calibrated.** At least **4 of 6** measured κ_c fall inside their ±2
residual-sd band as tabulated in §1.

**T4 — the edge is still a single crossing.** On all **6 of 6** fresh seeds, W_p changes sign exactly
once across κ ∈ [1.00, 1.30], negative to positive. (exp_32b's T1, carried to a larger seed set.)

**Instrument gates (invalidate the run, not the claim):** finite; `transfer_residual_max` ≤ 10⁻⁶;
`closure_pac_max` ≤ 0.05; `at_cap_max` ≤ 0.05; |ΔE_SEC − (T − W_p)| ≤ 0.08 |U₀|; and
|Σ_i W_{p,i} − W_p| ≤ 10⁻⁶ |U₀|. The identity tolerance is exp_32b's, which fired on 0 of 42 there.

**Kill.** *If T1 fails — the frozen line does not beat the training mean on unseen seeds — then f
carries no out-of-sample information about κ_c, the r = 0.878 was six points and four statistics, and
κ_c's spread is not predictable from the initial power spectrum. The seed spread then stands as an
irreducible property of the collapse at this resolution, and Milestone R stops trying to derive κ_c
and treats 1.170 ± 0.012 as the measurement it is.*

## §5 Operating characteristics, measured before the seal

Milestone R has set four bars without first measuring the instrument. The same discipline applies to a
test's own statistics, so T1 and T2 were Monte-Carloed (200 000 draws) before this commit.

**T1** — under the null that κ_c is independent of f and drawn from N(1.169100, 0.02907²):

| | |
|---|---|
| false positive rate | **0.079** |
| power, if the line is true with residual sd 0.01555 | **0.996** |
| median skill when true | +0.843 |
| skill at the 95 % null quantile | +0.175 |

So T1 is a real test: it fails 92 % of the time when f carries nothing and passes essentially always
when the line holds. A skill floor of 0.30 would give 0.034 / 0.987 instead; the simpler form is kept
because the gain is small and the threshold would be one more unmeasured number.

**T2** — false positive rate **0.499**. A coin flip, as §7 says. It is registered as the weakest of
the four and distinguishes "wrong sign" from "right sign, poor magnitude", nothing more.

### Expected direction, stated honestly

Given the above, my prior is about my belief in f rather than in the tests: **I put roughly 55 % on f
carrying real out-of-sample information.** That makes T1 pass with probability ≈ 0.55 × 0.996 +
0.45 × 0.079 ≈ **0.58**. T2 ≈ 0.7. **T3: 5 in 10** — the ±2σ band uses a residual sd estimated on four
degrees of freedom and is probably too narrow, and two of the six predictions extrapolate beyond the
training range. **T4: 9 in 10** — nine of nine seeds so far gave exactly one crossing.

## §6 Kill scope

A T1 kill retires the *predictability* of κ_c from the initial spectrum. It does not touch exp_32b's
4/4, the edge's existence, its value, or its invariance to g and n. The instrument gates invalidate
runs, not claims.

## §7 What would count as vacuous

- **T1 is not free.** The null is a fixed number, 1.169100, frozen from the training seeds; the line
  can lose to it and will if f carries nothing.
- **T2 with 6 points** has a 50 % chance under the null of a positive sign, so T2 alone proves little
  and is registered as the weakest of the four. It is here to distinguish "wrong sign" from "right
  sign, poor magnitude".
- W_p within ±0.02 |U₀| of zero at every κ on the fine grid — the edge unresolved — is UNSCORED.
- An arm whose budget never binds (`budget_bound_frac_max` < 0.01) is UNSCORED.

## §8 Side predictions (registered, unscored)

SP1 the twelve-seed pooled sd of κ_c stays within 0.020–0.040. SP2 the refitted line over all twelve
seeds has a slope within 2 training-sd of 0.874965. SP3 the compression ratio crosses 1 within one
step of W_p on 6/6. SP4 `conn_q05` at 1.25 is below its 1.00 value on 6/6.

## §9 Excluded

Any arm at n = 8000; seeds 1–18; any refit of the line before scoring; any cutoff other than 3 k_f;
any named constant for κ_c; severance (R2).

## §10 Outputs and the pre-seal scorer exercise

reality-engine results dir `full_r2c_fine/`; the grid copied here as
`results/exp_32c_predict_grid_fine_<ts>.json`; the scored JSON; and
`journals/2026-09-1X_exp32c_outcomes.md`. Statistics read from the grid's `_summary` only.

**Before this seal** `scripts/exp_32c_predict.py` was driven down every non-happy path — selftest
mismatch, unregistered seed, incomplete grid, instrument-gate failure, `--selftest` alone, and T1's
failure branch, which raises the kill — against a synthetic fixture built from exp_32b's grid with
seeds relabelled. The fixture is scratch only and never a result.

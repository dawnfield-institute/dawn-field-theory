# exp_30 registration — R1b: at full size, does the ledgered engine at κ = 1 hold more web than gravity alone? (SEALED by the commit carrying this file)

**Layer: physics → `theory/`** (THEORY_MAP sidecar-R row; ROADMAP Milestone R; the long-horizon
plan's R1b, `internal/dft/2026-09-06_long_horizon_plan.md`). Instruments unchanged from exp_29 and
gated in reality-engine (`.spec/v4-pac-ledger.spec.md`, `tests/v4/test_pac_ledger.py`, POC-12
exp_01); no ledger code changes in this round — κ is a declared, swept ratio (spec R4, R8) and
1.5 is a new value of it, not a new operator. Scored to this text by
`scripts/exp_30_ledger_r1b.py`, thresholds as module constants byte-equal to §4 (`--selftest`).
Frame declared in §3 (STANDARDS §2.7.6). Layer named (§2.7.7). Kill scope in §6.

**Thesis under test.** exp_29 (3/4, `43e4ebc9`) found that a PAC ledger makes the v4 substrate
*bound* at both sizes and lets a web *survive* (4.8–5.7 σ over the unbounded engine), but its
registered claim — that the ledgered engine at κ = 0.5 holds more web than **gravity alone** — failed
by the letter (2/3 on the proxy at 0.2 σ; 3/3 at n = 4000 at 1.1 σ against a 2 σ bar). Its outcomes
recorded, unscored and post hoc, that at n = 4000 the engine at **κ = 1** sits above gravity in every
seed by 2.8 σ, and that on the proxy the pressure's range 2r₀ = 20 exceeds half the box (37.8), so
the proxy could not decide a pressure-range question. This round asks the question exp_29 should
have asked, at the size that can answer it, on seeds it has never seen.

## §0 Postdiction disclosure

Everything below was computed before this seal, on the three registered seeds {1, 2, 3} of exp_29
at n = 4000 (reality-engine POC-12 `results/full/`, grid `7b06fea`). **None of it is scored here.**
Seeds 1–3 are excluded from this round's scoring by construction; the claim is made on seeds
{4, 5, 6}, which have not been run at any κ.

1. **The post-hoc number this round exists to test.** Whole-set percolation over t ∈ [10, 15] at
   n = 4000: κ = 1 **0.76 / 0.70 / 0.80** against gravity alone (κ = 0) **0.40 / 0.32 / 0.61** —
   above in every seed, margin 0.315, pooled σ 0.113, **2.8 σ**. κ = 0.5: 0.62 / 0.44 / 0.78 (margin
   0.174, 1.1 σ). κ = 2: 0.09–0.14, below gravity. Ordering at full size: 1 > 0.5 > 0 > 2 > ∞.
2. **Why the proxy is retired as the decider.** The pressure's range is 2r₀ = 20 at both sizes. On
   the proxy (box 37.8) the ratio box/2r₀ = **1.89**: the repulsion spans more than half the box and
   acts as a global restoring force (pressure work −1.2 P₀ at κ = 0.5); at n = 4000 (box 60) the
   ratio is **3.0** and the repulsion is local (−4.2 P₀ at κ = 0.5, −0.6 to −1.4 at κ = 1). exp_29
   named the proxy as the decider and did not declare this ratio. This registration declares it as
   a condition (§1) and runs only at the size that meets it.
3. **The virial arithmetic** (bound through κ = 1; U-shaped KE/|U| with its minimum at κ = 1;
   monotone above) held in all six exp_29 seeds. It is carried here as T2/T3, extended to κ = 1.5,
   which has never been run at either size: its position is **predicted**, not seen — between κ = 1
   and κ = 2 on every KE/|U| measure, and, on §0.1, holding less web than κ = 1 and more than κ = 2.
4. **No smoke test, no pre-seal run on seeds 4–6.** The pipeline is exp_29's, unchanged; its gates
   (transfer residual ≤ 10⁻⁶, closure ≤ 0.05, at-cap ≤ 0.02, finite) are re-verified from the grid
   JSON at scoring time. The scorer's code paths were exercised pre-seal on exp_29's own grid (which
   it must refuse — seeds 1–3 — and does) and on a synthetic grid of invented numbers in the
   scratchpad, never on the substrate.
   The only change on the reality-engine side is an output-directory override in `run_grid.sh` so
   this round's runs land in `results/full_r1b/` and cannot be confused with exp_29's.

## §1 Objects (closed at this seal)

- **Substrate:** as exp_29 §1, unchanged: `n = 4000, box = 60, r0 = 10, g = 1.5, dims = 3,
  ic = lattice`; `sec_balance = XI_ANALYTIC / PHI`; `memory_decay = 0.95` (inherited, untouched);
  `damping = 1.0` in every arm; `t_end = 15`; marks every 1.0 of simulated time. **Full size only.**
- **The declared condition:** box / 2r₀ ≥ 3. The full configuration gives 3.0 exactly and meets it;
  the proxy (1.89) does not and is not run. This is a *condition of the claim*, not a tuning: a
  pressure-range question is asked only on a box that is large compared with the range.
- **Seeds:** {4, 5, 6}. Fresh — never run at any κ, at either size.
- **Arms (κ):** **0** — gravity only (control) · **0.5** — exp_29's registered arm, carried for
  continuity · **1** — the arm the claim is made at · **1.5** — new, predicted intermediate ·
  **∞** — today's substrate, the unbounded engine (control). Fifteen runs.
- **Instruments:** as exp_29: `structure.web_metrics` of the whole set at `matched_res(4000)`;
  the ledger metrics; the random-field floor at the same count and resolution (20 uniform draws)
  beside every percolation.
- **Derived vs declared vs inherited:** unchanged from exp_29 §1. κ = 1.5 is declared and swept.
  Nothing fitted.

## §2 Pre-seal numbers

The exp_29 n = 4000 grid on seeds {1, 2, 3} (`results/exp_29_pac_ledger_grid_full_20260906_180850.json`),
window means t ∈ [10, 15]:

| κ | percolation s1 / s2 / s3 | KE/\|U_grav\| window max s1 / s2 / s3 |
|---|---|---|
| 0 | 0.395 / 0.315 / 0.608 | — |
| 0.5 | 0.622 / 0.441 / 0.778 | 0.53 / 0.54 / 0.53 |
| **1** | **0.76 / 0.70 / 0.80** | 0.33 / 0.34 / 0.36 |
| 2 | 0.09–0.14 | 0.58–0.59 |
| ∞ | 0.035 / 0.038 / 0.028 | 18.5 / 22.3 / 20.0 |

Random-field floor at n = 4000, matched resolution: 0.047 ± 0.013. These numbers set the expected
direction in §5 and nothing else.

## §3 Frame (STANDARDS §2.7.6)

**Sampled:** the density field of the whole set of each arm at `res = matched_res(4000)`,
occupancy reported beside every percolation. **Expectation:** the same instrument on the two
controls at the same count — the engine removed (κ = 0) and the engine unbounded (κ = ∞) — and the
random-field floor. **Same scope:** the whole box. **Statistic:** the mean over marks
t ∈ {10, …, 15}, never the t = 15 point; for T2, every mark of the window. **Seeds:** the three
fresh seeds only; seeds 1–3 appear nowhere in the score.

## §4 Tests (M = 4; thresholds fixed; invariants only)

**T1 — the claim.** At **κ = 1**, **3/3 seeds**: whole-set percolation over the window exceeds the
**κ = 0** control seed by seed, and the mean margin exceeds **2× the pooled σ of the two arms**
(exp_29's definition, unchanged: √((s²_{κ=1} + s²_{κ=0}) / 2) over the three seeds — the definition
that gives the 0.113 in §0.1). (The κ = ∞ comparison is not scored here — exp_29 established it at
4.8–5.7 σ at this size and it is reported, not counted.)

**T2 — bound, at the new arm too.** At κ = 1 **and** at κ = 1.5, 3/3 seeds: **KE/|U_grav| < 1 at
every mark** of the window.

**T3 — the predicted position of κ = 1.5.** Per seed, 3/3: the window mean of KE/|U_grav| is
ordered **κ = 1 < κ = 1.5 < κ = ∞**; and the window-mean percolation is ordered **κ = 1 > κ = 1.5**.
The first clause is the virial arithmetic (injected energy ≤ κ|U₀|); the second is §0.1's ordering
extended to a value never run. Both must hold.

**T4 — the ledger did the work.** On every ledgered run (κ ∈ {0.5, 1, 1.5}): net pair energy created
by entropy change ≤ P(0) exactly; pressure work ≤ P(0) within the integrator's **10% truncation
allowance**; and at κ = 1, 3/3 seeds, **budget_bound_frac_max = 1**.

**Kill.** *If T1 fails — the ledgered engine at κ = 1, on a box three times its range, on seeds it has
never seen, does not hold more web than gravity alone by 2 σ in 3/3 — then the 2.8 σ of exp_29 was
seed selection, and a bounded density-sourced repulsion adds no structure beyond gravity at any
budget tested. The mapping is retired as a structure-ADDER; it stays as what exp_29 showed it to
be, a bound-maker. The pressure's form (the functional's gradient-penalising term) becomes the
only lead, and R2 runs on κ = 1 anyway, because R2 needs a bound substrate, not a better-than-gravity
one.*

## §5 Expected direction, stated honestly

**T1 leaning pass** on §0.1 (2.8 σ on three seeds is the expected direction, disclosed as post hoc;
the honest prior is that a 2.8 σ post-hoc margin on three seeds reproduces at > 2 σ on three fresh
seeds perhaps two times in three). **T2 pass** (bound through κ = 1 held 6/6 in exp_29; κ = 1.5 is
below the κ = 2 threshold where exp_29's total first went positive, so predicted bound). **T3 pass**
(the virial ordering has never failed; the percolation ordering κ = 1 > 1.5 is the genuinely new
prediction and the one most likely to surprise — if κ = 1.5 holds *more* web than κ = 1, the optimum
is above 1 and §0.1's peak was a coarse-grid artifact; that is recorded as a T3 FAIL, not softened).
**T4 pass** (the ledger's exactness is an instrument property).

The risk stated: T1 fails on seed variance alone — the margin exists but the three fresh seeds spread
wider than 0.113. That outcome sends the next round to a six-seed registration at the same
configuration, not to a new mapping.

## §6 Kill scope

The kill retires "the ledgered engine adds web beyond gravity" as a claim about this mapping at
this size. It does not retire the ledger (gated, stays), exp_29's bound result (3/4, stands), or
Milestone R's thesis. The spec's instrument falsification invalidates the *run*, not the claim.

## §7 What would count as vacuous

- **The budget never binds at κ = 1** (`budget_bound_frac_max < 0.01`) — pre-seal it binds fully.
- **Gravity alone exceeds the random floor by > 2σ** — expected (§0, exp_29) and moves nothing: the
  frame is the difference over G0, already in T1.
- **κ = 1 and κ = 0 indistinguishable**: percolation at κ = 1 within 1 pooled σ of κ = 0 in every
  seed. That is not vacuity, it is a **T1 FAIL** and is recorded as one — written here so it cannot
  later be reframed as "the instrument could not resolve it".
- **Seed contamination**: any run with seed ∈ {1, 2, 3} present in `results/full_r1b/` voids the
  grid; the scorer refuses it.

## §8 Side predictions (registered, unscored)

SP1 pressure work at κ = 1 is negative in 3/3 (exp_29: −0.6 to −1.4 P₀). SP2 spent fraction falls
monotonically 0.5 → 1 → 1.5. SP3 the step never shrinks below `dt_ref` at κ ≤ 1.5. SP4 κ = 0.5 sits
between κ = 0 and κ = 1 in percolation in 3/3 (exp_29 seeds: yes). SP5 the random floor is below
every arm's window mean.

## §9 Excluded

The proxy size; seeds 1–3; any Ξ-based amount; the functional's `β∇²A` and `γT·A` terms; severance
and Landauer (R2); `memory_decay` as a variable; any κ not in {0, 0.5, 1, 1.5, ∞}.

## §10 Outputs

reality-engine POC-12 `results/full_r1b/` (one JSON + positions sidecar per run; grid JSON with
SHA256s and commit, from `exp_04_aggregate.py --results-dir results/full_r1b`). Here:
`results/exp_30_ledger_r1b_grid_full_<ts>.json` (copy), `results/exp_30_ledger_r1b_<ts>.json`
(scored), `journals/2026-09-1X_exp30_outcomes.md` citing this seal's hash and the reality-engine
hashes.

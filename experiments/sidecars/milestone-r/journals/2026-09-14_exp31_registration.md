# exp_31 registration — R1c: at matched occupancy, paired on the initial condition, does the ledgered engine add connectivity to gravity's web? (SEALED by the commit carrying this file)

**Layer: physics → `theory/`** (THEORY_MAP sidecar-R row; ROADMAP Milestone R; the long-horizon
plan's R1). Instruments: reality-engine `structure.connectivity_at_occupancy` on
`structure.cic_deposit` (new, `tests/v4/test_structure_connectivity.py`, 7 tests), recorded per mark
by POC-12 `exp_03` as `conn_q05 / conn_q10 / conn_q20`, floored on uniform positions by `exp_04`;
the PAC ledger unchanged since exp_29 (`.spec/v4-pac-ledger.spec.md`); no ledger code change (spec
R4, R8). Scored to this text by `scripts/exp_31_paired_ledger.py`, thresholds as module constants
byte-equal to §4 (`--selftest`). Frame in §3 (STANDARDS §2.7.6). Layer named (§2.7.7). Kill scope §6.

**Thesis under test.** exp_29 and exp_30 asked whether the ledgered engine holds more web than
gravity alone and scored it with `percolation` — the largest connected component of the cells above
twice the mean density — at n = 3, unpaired, against a bar of twice the pooled per-seed σ. Both
kills fired. exp_30's post-mortem (its outcomes journal, and the changelog of this round) found the
test mis-specified in three ways that this registration removes: the design discards the pairing
(each seed is one initial condition shared by every arm — the first mark is identical across arms);
the statistic compares arms at unmatched occupancy (κ = 1's overdense set is 60 % larger than
gravity's, and a fatter set percolates more easily for reasons that are not structure); and the
recorded instrument deposits a random mass draw at a threshold that sits exactly on the count-two
boundary, an instrument noise of 0.01–0.05 on the same positions. This round asks the question
paired, at matched occupancy, on a count deposit, on six fresh seeds, at the budget that the
six burned seeds say is the connective one — and predicts, on the same six seeds' pattern, what
κ = 1 does differently and where the mapping's edge is.

## §0 Postdiction disclosure

**Everything below was computed after the fact** on seeds {1, 2, 3} (exp_29, plus a κ sweep run
today for this design) and {4, 5, 6} (exp_30), with the instrument this registration introduces
applied to their saved positions. None of it is scored. The choice of κ = 0.5 as the claim arm, of
q = 0.10 as the registered occupancy, and of every pattern in T2–T4 was made looking at these
numbers; the honest prior for each test is stated in §5 with that in mind.

1. **Paired lifts over gravity alone, six seeds, connectivity at fixed occupancy** (window means,
   t ∈ [10, 15]): at q = 0.10, κ = 0.5 **+0.138 ± 0.032** (6/6, paired t 4.4); κ = 1 +0.181 ± 0.075
   (5/6, t 2.4). At q = 0.05 (the spine): κ = 0.5 **+0.112 ± 0.031** (6/6, t 3.6); κ = 1 −0.081 ± 0.104
   (3/6). At q = 0.20 (the body): κ = 0.5 +0.104 ± 0.022 (6/6); κ = 1 +0.271 ± 0.051 (6/6).
2. **κ = 1 minus κ = 0.5, paired:** q = 0.05 −0.193 ± 0.081 (1/6 positive); q = 0.20 +0.167 ± 0.041
   (6/6). Contrast (`cv`) at κ = 1 is 0.43–0.63 of gravity's in 6/6; at κ = 0.5 it is 0.90–1.05.
   Void fraction at κ = 1 is −0.06 below gravity's (0/6 positive); at κ = 0.5 +0.04 above (6/6).
   **The pattern:** κ = 0.5 is gravity's web, better connected at every cut; κ = 1 is a fatter,
   smoother web whose densest spine is more broken than gravity's own.
3. **The κ curve on seeds 1–3** (κ ∈ {0, 0.25, 0.5, 0.75, 1, 1.25, 2, ∞}; 0.25/0.75/1.25 run today,
   unregistered) — arm means of the window statistics, conn at q = 0.05 / 0.10 / 0.20, then occupancy,
   contrast, net pressure work over P(0):

   | κ | q05 | q10 | q20 | occ | cv | work / P₀ | KE/\|U\| |
   |---|---|---|---|---|---|---|---|
   | 0 | 0.61 | 0.51 | 0.51 | 0.091 | 5.34 | — | 0.83 |
   | 0.25 | 0.68 | 0.60 | 0.64 | 0.095 | 5.25 | −4.07 | 0.68 |
   | **0.5** | **0.76** | **0.69** | 0.63 | 0.095 | 5.22 | −4.24 | 0.53 |
   | 0.75 | 0.76 | 0.72 | 0.74 | 0.099 | 5.27 | −4.48 | 0.39 |
   | 1 | 0.67 | 0.82 | 0.87 | 0.153 | 2.84 | −0.93 | 0.33 |
   | 1.25 | 0.13 | 0.22 | 0.72 | 0.166 | 1.96 | +0.10 | 0.36 |
   | 2 | 0.08 | 0.09 | 0.36 | 0.190 | 1.26 | +0.29 | 0.58 |
   | ∞ | 0.04 | 0.05 | 0.11 | 0.189 | 1.02 | — | 20.25 |

   **Four regimes.** κ = 0.25: barely distinct from gravity (spine +0.07, 2/3). **κ = 0.5–0.75: the
   connective plateau** — occupancy and contrast unchanged from gravity's, the spine +0.15 in 3/3 at
   both, the pressure doing −4 P₀ of net work. **κ = 1: fattening** — occupancy +60 %, contrast halved,
   the body +0.36 but the spine only +0.06 (2/3), net work −0.9 P₀. **κ ≥ 1.25: the edge** — net work
   positive ([0.14, 0.08, 0.08] P₀ per seed at 1.25; [-0.58, -1.45, -0.76] at 1), the spine collapsed (−0.47, 0/3) while the body
   still stands at 1.25 (+0.20, 3/3) and is gone by 2. The edge is between κ = 1 and 1.25, not 1 and
   1.5 as exp_30 read it. κ = 0.5 is registered as the claim arm because it is exp_29's original
   budget and sits on the plateau; 0.75 is as good on these seeds and is NOT chosen for that reason.
4. **The instrument's own noise.** Redrawing the masses (1 ± 0.1) on the same positions moves the
   recorded `percolation` by 0.012 (κ = 0, 0.5), 0.027 (κ = 1), 0.054 (κ = 1.5). The count deposit
   has no such term. Masses are now saved in the sidecar.
5. **No run on seeds 7–12 at any κ.** The instrument was exercised on the saved positions of seeds
   1–6 (above), on synthetic webs and noise (its tests), and the scorer on synthetic grids in the
   scratchpad. The reality-engine instrument commit precedes this seal and is cited in §10.

## §1 Objects (closed at this seal)

- **Substrate:** as exp_29/30, unchanged: `n = 4000, box = 60, r0 = 10, g = 1.5, dims = 3,
  ic = lattice` (the uncorrelated 10 % jitter, zero velocity); `sec_balance = XI_ANALYTIC / PHI`;
  `memory_decay = 0.95` (inherited); `damping = 1.0`; `t_end = 15`; marks every 1.0. **Full size
  only; box / 2r₀ = 3.0 ≥ 3** (the condition declared in exp_30, carried).
- **Seeds:** {7, 8, 9, 10, 11, 12} — six, fresh, never run at any κ.
- **Arms:** κ = **0** (gravity alone; the paired control) · **0.5** (the claim arm) · **1** (the
  contrast arm, T3) · **1.25** (the edge arm, T4) · **∞** (the unbounded engine; reported). Thirty runs.
- **Design:** paired on seed. The seed fixes the jitter and the mass draw for every arm; the
  comparison in every test is the within-seed difference.
- **Statistic:** the mean over marks t ∈ {10, …, 15} of `conn_q10` (registered), `conn_q05`,
  `conn_q20`; `cv`, `void`, `work_pressure_cum / P(0)`; `percolation` and `occupancy` reported.
- **Instrument:** `conn_qXX` = largest face-connected component of the densest XX % of cells of the
  cloud-in-cell COUNT field of the whole alive set at `matched_res(4000) = 16`, as a fraction of
  those cells. Random floor at each q from 20 uniform draws at the same count and resolution.
- **Derived vs declared vs inherited:** unchanged from exp_29 §1. κ = 1.25 is declared and swept.
  q = 0.10 is declared (the registered occupancy); 0.05 and 0.20 are the spine and the body.

## §2 Pre-seal numbers

See §0.1–0.3. Random floor at n = 4000, res 16 (uniform positions, 20 draws): conn_q05 0.037 ± 0.010,
**conn_q10 0.038 ± 0.009**, conn_q20 0.084 ± 0.027; legacy percolation 0.047 ± 0.013 (exp_29/30's floor,
reproduced). Gravity alone sits at conn_q10 0.38–0.87 on seeds 1–6: there is a web to add to.

## §3 Frame (STANDARDS §2.7.6)

**Sampled:** the CIC count field of the whole alive set of each arm at res 16, ranked. **Expectation:**
the same statistic on the same seed's gravity-only run (the paired control) and on uniform positions
(the floor). **Same scope:** the whole box; the same occupied fraction in every arm by construction.
**Statistic:** window means, never the t = 15 point; within-seed differences, six seeds; the paired
t is the mean difference over its standard error, five degrees of freedom. **Seeds:** 7–12 only;
seeds 1–6 appear nowhere in the score.

## §4 Tests (M = 4; thresholds fixed; invariants only)

**T1 — the claim.** Δ₁ = conn_q10(κ = 0.5) − conn_q10(κ = 0), per seed. **Sign 6/6**, **mean Δ₁ ≥ 0.05**,
and **paired t ≥ 3.0**. (Prior: +0.138 ± 0.032, 6/6, t 4.4 on the seeds it was chosen on.)

**T2 — it is gravity's web, better connected.** Δ₂ = conn_q05(κ = 0.5) − conn_q05(κ = 0): **sign ≥ 5/6**
and **paired t ≥ 2.0**; and the contrast ratio **cv(κ = 0.5) / cv(κ = 0) ∈ [0.85, 1.15] in 6/6**.

**T3 — κ = 1's signature, predicted as a pattern.** Δ₃ = conn(κ = 1) − conn(κ = 0.5), per seed:
**at q = 0.20 positive in ≥ 5/6** and **at q = 0.05 negative in ≥ 5/6**; and **cv(κ = 1) / cv(κ = 0) < 0.70
in 6/6**.

**T4 — the edge.** At κ = 1.25, **6/6 seeds**: the pressure's net work over the run is **positive**
(`work_pressure_cum / P(0) > 0`) and conn_q05(κ = 1.25) − conn_q05(κ = 0) is **negative**; and at κ = 1,
6/6, the net work is **negative**. (Prior: one seed at κ = 1.25 — +0.14 P₀, spine −0.44; six at κ = 1,
all negative.)

**Instrument gates (not scored, invalidate the run):** transfer residual ≤ 10⁻⁶, closure ≤ 0.05,
`at_cap` ≤ 0.02, all runs finite, every arm's `conn_q10` recorded at every mark of the window.

**Kill.** *If T1 fails — at matched occupancy, paired on the initial condition, on six seeds it has
never seen, the ledgered engine at its most connective budget does not add connectivity to gravity's
web — then the adder question is closed for this mapping at every budget: κ = 0.5 stands for the
plateau (0.5–0.75) that the sweep shows, κ = 1 is the fattening regime (T3), and above it the mapping
destroys. "The ledger is a bound-maker" becomes the final reading, exp_30's post-mortem was a
description of six seeds and nothing more, and R2 proceeds on the bound substrate unchanged.*

## §5 Expected direction, stated honestly

**T1 leaning pass** — 6/6 and t 4.4 on the six seeds the arm and the occupancy were chosen on; a
post-hoc 6/6 reproduces as 6/6 at t ≥ 3 on six fresh seeds perhaps seven times in ten. **T2 pass**
(the same seeds, weaker: one seed at +0.00; t ≥ 2 and ≥ 5/6 leaves room for one). **T3 pass** with the
least confidence — a three-clause pattern from six seeds, each clause 5/6 or 6/6; call it six in ten.
**T4 pass** on the arithmetic (net pressure work is monotone in κ and crosses zero once) with one seed
of evidence at 1.25; if the crossing sits above 1.25 for some seed, T4 fails and the edge is between
1.25 and 1.5. The risk stated: T1 passes on the sign and the mean and misses t ≥ 3 on one wide seed;
that is recorded as a T1 FAIL and the kill fires — the bar is the bar.

## §6 Kill scope

The kill closes "the ledgered engine adds connectivity to gravity's web" for this mapping. It does
not touch the ledger (gated, stays), exp_29's bound result, exp_30's result, or Milestone R's thesis.
The instrument's own falsification (a synthetic web not scoring ~1, noise not scoring low — the
tests) invalidates the run, not the claim.

## §7 What would count as vacuous

- **Gravity alone has no web to add to:** conn_q10(κ = 0) within 2σ of the random floor in ≥ 3/6 seeds.
- **The instrument cannot resolve the arms:** |Δ₁| ≤ 0.02 in 6/6 (the resolution set by 20 uniform
  draws' σ at q = 0.10). That is recorded as a T1 FAIL, not as vacuity.
- **The budget never binds at κ = 0.5** (`budget_bound_frac_max < 0.01`).
- **Seed contamination:** any seed ∉ {7, …, 12} in the grid voids it; the scorer refuses.

## §8 Side predictions (registered, unscored)

SP1 the legacy `percolation` lift at κ = 0.5 is positive in ≥ 5/6 (it was 6/6 on seeds 1–6 — the
registered statistic of exp_29 would have passed a paired test). SP2 `occupancy` at κ = 0.5 is within
±0.01 of gravity's in 6/6 (matched by physics, not only by construction); at κ = 1 it is ≥ 0.04 above
in 6/6. SP3 void fraction: κ = 0.5 above gravity's in ≥ 5/6, κ = 1 below in ≥ 5/6. SP4 KE/|U| window
means ordered **κ = 1 < κ = 0.5 < κ = 0** and **κ = 1.25 > κ = 1**, 6/6 (the U-shape with its minimum at 1). SP5 the random floor sits below gravity alone at every q, 6/6.

## §9 Excluded

The proxy size; seeds 1–6; κ ∈ {0.25, 0.75, 1.5, 2} (swept or run before, reported, unscored);
the mass-weighted `percolation` as a scored quantity; any threshold other than rank at q ∈ {0.05,
0.10, 0.20}; the functional's `β∇²A` and `γT·A` terms; severance and Landauer (R2).

## §10 Outputs

reality-engine POC-12 `results/full_r1c/` (thirty JSONs + position/mass sidecars; grid JSON with
SHA256s and commit, `exp_04_aggregate.py --results-dir results/full_r1c`); the instrument commit
`5f5d690` on `feat/v4-pac-ledger-r1c` (stacked on PR #12). Here: `results/exp_31_paired_ledger_grid_full_<ts>.json` (copy),
`results/exp_31_paired_ledger_<ts>.json` (scored), `journals/2026-09-1X_exp31_outcomes.md`.

# ADE Cascade Round 1 — Pre-Registration

**Date:** 2026-07-17
**Status:** REGISTERED BEFORE DATA — no ADE-coupling cascade has ever been run as of
this commit. The only cascade data in existence are the legacy path-kernel runs in
`milestone4/results/` (exp_03, exp_14, exp_15), which constitute the A-family arm's
historical record and the source of all priors cited below.
**Target scripts:** `scripts/exp_01_diagram_selectivity.py`, `scripts/exp_02_affine_vertex.py`
**Gate script (must pass first):** `scripts/exp_00_baseline_gate.py`

## Data state at registration

`ade_cascade/results/` is empty. `core/coupling.py` and the experiment scripts do not
yet exist. The engine refactor (optional `coupling_matrix` argument on
`milestone4/core/utils.py::energy_cascade`) has not been written. No exploratory ADE
runs of any kind have been performed, informally or otherwise.

## Locked engine and parameters

- Engine: `milestone4/core/utils.py::energy_cascade` + `measure_exponent(trim=2)`,
  extended with an optional injected base coupling matrix. **Refactor contract:** with
  no matrix injected, behavior is bit-identical to the current code; with the A-family
  distance kernel injected, results are bit-identical to the legacy kernel at equal
  seed (the A path graph's distance metric is |i−j|). Gate: exp_00 must show exact
  equality AND reproduce exp_14's canonical baseline (−1.6083 at n_modes=8, cd=0.1,
  ns=0.3) within seed noise before any registered run executes.
- Canonical parameters (identical across every arm): injection_energy = 1.0,
  n_scales = 25, n_samples = 15000, coupling_decay cd = 0.1 (kernel scale),
  nonlinear_strength ns = 0.3, Landauer floor and 0.98 transfer factor as in the
  engine. The `means` vector depends only on rank and cd, so it is **identical across
  families at equal rank** — the coupling matrix is the sole varying quantity.
- Coupling map: C[i,j] = exp(−d_G(i,j)·cd), d_G = shortest-path distance on the
  diagram. Node order = the canonical order of the existing builders
  (`milestone12/core/connection_geometry.DynkinDiagram` for A/D/E;
  `milestone15/core/representative.build_cycle` for affine-A = Ã).
  The engine's existing PSD safeguard (eigenvalue shift when λ_min < 1e-10) is
  retained; its activation is recorded per arm (threat T1 below).
- Seeds: 100 per arm, seed_i = 42 + i·1000 (the exp_15 convention). Per-arm statistic:
  mean exponent across seeds with percentile-bootstrap 95% CI (10,000 resamples).

## Arms

| Arm group | Diagrams | Purpose |
|-----------|----------|---------|
| A-family | A_6, A_7, A_8, A_9 | R1 baseline + R3 (A_8 = legacy 3D configuration) |
| D-family | D_6, D_7, D_8 | R1 |
| E-family | E_6, E_7, E_8 | R1 (rank 6–8 is the only window where all three families coexist — and it brackets the physical 3D mode count 8) |
| Affine-A | Ã_6, Ã_7, Ã_8 (cycles on 7, 8, 9 nodes) | R2 |

## Registered predictions (relations only, per the invariant-registration rule)

**R1 — Diagram selectivity.** At equal rank, the A/D/E arms produce distinct spectral
exponents; the null (only mode count matters) predicts all three indistinguishable.
Registered relations: (a) existence of at least one family pair with non-overlapping
95% CIs, per rank; (b) consistency of the family *ordering* of exponents across ranks.
No direction is registered (we do not know which family cascades harder a priori —
registering a direction would be a coordinate guess).

**R2 — Affine vertex (the k−1 offset re-posed).** Define per rank r ∈ {6,7,8}:
shift_affine(r) = |E[exp(Ã_r)] − E[exp(A_r)]| (adding the affine node),
shift_path(r) = |E[exp(A_{r+1})] − E[exp(A_r)]| (adding an ordinary node),
ρ(r) = shift_affine(r) / shift_path(r).
Registered relation: the affine node acts as a reference, not an active mode —
ρ is small. (Prior context, stated for honesty: the legacy k−1 offset means the
engine at n modes behaves like the formula's k = n+1; if the affine vertex is that
+1, closing the cycle should barely move the exponent.)

**R3 — Bridge.** Within each family, the exponent is strictly monotone in rank
(the mode-count map survives the kernel swap). Tested on A (4 ranks), D (3), E (3).

## Decision rules (locked)

- **R1:** CONFIRM if at ≥ 2 of the 3 ranks, ≥ 1 family pair has non-overlapping 95%
  CIs AND the family ordering is identical at every rank where separation exists.
  KILL if all pairs' CIs overlap at all ranks. Otherwise INCONCLUSIVE.
  *Justification:* CI half-widths at these settings are ≈ 0.0013 (exp_15 B.2 measured
  std_error 0.00065 with 100 seeds); overlap therefore means any effect is below
  instrument precision, and one-rank-only separation is not a structure claim.
- **R2:** CONFIRM if median ρ over the three ranks < 0.25. KILL if median ρ > 0.75.
  Otherwise INCONCLUSIVE. *Justification:* ρ < 0.25 = the affine node contributes at
  most a quarter-mode of shift (reference-like); ρ > 0.75 = it is statistically an
  ordinary mode (the affine reading is wrong); the band between is declared
  inconclusive NOW so it cannot be adjudicated post hoc.
- **R3:** CONFIRM if strictly monotone in every family; any non-monotone family is
  reported as a registered discovery about the legacy map (it would mean the old
  engine's monotonicity was a kernel artifact), not smoothed over.

## Registered threats to validity

- **T1 (PSD shift):** the engine's eigenvalue shift may activate differently across
  topologies (cycles especially). Recorded per arm; any arm where the shift magnitude
  exceeds 1e-3 is flagged in outcomes and the arm's result treated as contaminated.
- **T2 (feedback wash-out):** the nonlinear feedback (ns = 0.3) adds a rank-one term
  that could mask topology. If R1 KILLs, a **secondary registered arm** at ns = 0
  runs under identical rules (declared here, now, not invented later).
- **T3 (node ordering):** D/E node order affects which nodes carry high `means`
  weight. Accepted as a design choice (canonical builder order); a future round may
  register order-permutation invariance as its own relation.

## Outcome commitment

Results — CONFIRM, KILL, or INCONCLUSIVE, in any mix — are recorded in an outcomes
journal citing this registration's commit hash, pushed to the same PR, and folded into
the lore FDOs (`cascade-turbulence-mode-count` and a new ade-cascade FDO) regardless
of direction. Thresholds and statistics above are final; any post-registration edit
to them voids the affected claim.

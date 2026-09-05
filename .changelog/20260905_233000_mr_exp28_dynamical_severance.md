# Milestone R exp_28 — dynamical severance on reality-engine's v4 substrate: 0/4, the trigger is a core detector

**Date**: 2026-09-05 · **Branch**: `exp/mr-28-dynamical-severance` · **Registration sealed**: `bf833113`

## What

First dynamical run of Milestone R's derived severance trigger (exp_15: a vertex severs when
*all* its bonds are overstressed, `min_j |S_i − S_j| > τ`, at the lattice spacing) and derived
form (exp_01: decoupling, not destruction) as whole-particle severance on reality-engine's v4
particle substrate — corrected third-law pressure, energy ledger closing to ≤ 6 × 10⁻⁷ per tick,
no speed cap, no drag in any derived arm. Registered before running (STANDARDS §2.7): M = 4,
invariants only, frame declared (retained set at matched count), two matched controls (drag
calibrated to the same retained kinetic energy per particle; random severance replaying the same
event log), two kill sentences, τ ∈ {5, 10, 20} fixed from firing rates alone.

## Result

**0/4.** T1 (the sink lowers the retained per-particle energy) FAIL: energy leaves in every run,
but what leaves carries less than the mean — the fired particles are the bound cores (`u_out < 0`
everywhere) — so the retained set ends hotter. **T2 FAIL with sign**: at the informative τ the
severed set is *less* connected than a random subset of the baseline at the same count, 3/3 seeds,
−2.6 pooled σ (proxy, τ = 20); at n = 4000, τ\* = 10 (half removed) it is a null (−0.5 σ) and identical
to random removal; random removal at the onset τ holds more; the tuned drag holds as much or more.
T3/T4 uninformative (conditional on T2); K1/K2 do not fire. SP2 (`memory_decay` ordering) holds;
SP3 (Landauer < 1% of pressure work) missed by a hair. Scorecard 62/112 → 62/116.

**Reading.** The all-edges barrier is true only at an extremum of the field. On exp_15's noise field
the extrema are random, which is what made the degree barrier and exp_16's universal statistics come
out; on a structured field the extrema are the cores, and the barrier selects the structure. exp_15/16
stand; what closes is the entropy-gradient barrier as an amount-free *dynamical* sink (reality-engine
C4.1 option 3). Option 2 — derive the detonating pressure term from the SEC functional — remains.
Registrable next: a barrier on *under*-stress, and the sign question beside exp_14.

## Files

- `experiments/sidecars/milestone-r/journals/2026-09-05_exp28_registration.md` (sealed),
  `2026-09-05_exp28_outcomes.md`
- `experiments/sidecars/milestone-r/scripts/exp_28_dynamical_severance.py` — scorer, thresholds
  byte-equal to the seal (`--selftest`); accepts the proxy and n = 4000 grids together
- `experiments/sidecars/milestone-r/results/exp_28_dynamical_severance_grid_*.json` (copies of the
  reality-engine grids with per-run SHA256s), `exp_28_dynamical_severance_*.json` (scored)
- README (row 28, finding 41, two honest-failure rows, P38), meta.yaml, THEORY_MAP sidecar row,
  ROADMAP Milestone R section (also corrects the "eight failures" claim to the 08-27 finding: two)

## Process

The drag control's design-pass definition (a ratio of signed energies) crossed zero as the set
unbound and had collapsed onto the baseline; caught by a pre-seal smoke test on an out-of-registry
seed, replaced by a positive-definite match, and the smoke test's own numbers disclosed in §0 of the
registration. No threshold changed after any result. Reality-engine side: POC-11
(`proof_of_concepts/v4/poc_11_derived_sink/`), branch `feat/v4-derived-sink`.

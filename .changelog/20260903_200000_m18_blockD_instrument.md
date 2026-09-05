# M18 Block D: the fold-certificate instrument built and gated; O2 replay harness; pre-seal script written

**Date:** 2026-09-03 (evening) · **Milestone:** 18 · **Layer:** instruments (→ `core/`) for the physics
layer's Block D · **Seal status:** Block D remains UNSEALED (gate is Peter's); no certificate has been
computed on any Block D object.

- `core/certificate.py` (draft §2): `grade` (census semantics, verbatim), `golden_pairs`, `rational_cores`,
  `halves`, **`sector_projector`** — the spectral projector onto ker q(C) as a polynomial in C from the
  Bézout identity of q against its cofactor p/q over ℚ(√5) (exact division; cores killed automatically;
  no nullspaces, no conic) — `certificate` (tr(R·D), off→off leak, total leak, vertex multiset),
  `class_sectors` (H₂/H₃/H₄ up to uniform bond scaling; handles the split of H₂ into linear factors
  over ℚ(√5)), guards (regular ⇒ blind; degenerate ⇒ declared), `evaluate`.
- **Gates (all PASS, `results/explore_d0_certificate_gates_*.json`):** projector recipe — identical to
  `bezout_proj` on 13/13 strict exp_13 folds and to the sealed exp_14 nullspace recipe on 80/80 core
  folds; KA1 A₄/E₈; KA2 D₆ (**corrected**: D₆ is not in exp_14's clean regime — core kernel not
  B-invariant, masses {2/5, 3/5} — so the off-core laws fail and the trace law holds; the draft's
  expectation was recalled from the n = 12 record and was wrong; forward note on the draft); KA3
  A₅/D₅/E₆/E₇ → none; KA4 the four asymmetric n = 20 trees reproduce the seven listed tr values, leak
  ≠ 2/5; KA5 det −80 carries H₃, det −620 carries H₂; KA6 C₅ blind; KA7 PAC d = 3, 4, 5 partial with one
  pair, H₂-type at scale² 2.
- `core/growth_harness.py` (O2): replays `evolve_pac_tree` recording topology; reproduces the recorded
  exp_07/exp_08 values to 1e-12 on all 31 seeds and the live engine bit-for-bit on integer metrics
  (`results/explore_d0b_growth_harness_gate_*.json`, PASS). O2 is the original object. Evaluable subset
  (n ≤ 100): the two-children trees, depth ≤ 4, a few at depth 5.
- `scripts/explore_d1_inventory_grades.py` written (grades/halves/nulls/|E| only), not yet run.
- Earlier gate runs that failed on my own bugs (cofactor division, star-import, classifier on the
  conjugate member and on split linear pairs, a key type) are kept as timestamped results.

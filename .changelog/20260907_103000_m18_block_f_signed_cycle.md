# M18 Block F — Orientation: the signed cycle and the first non-tree class (exp_20)

**Date:** 2026-09-07 · **Branch:** `m18/block-f-signed-cycle` (PR #187) · **Seal:** `872c3ad2` · **Layer:** mathematics → `formal/`

## What

Signed graphs enter the corpus. On a cycle a ±1 edge signing has exactly two switching classes:
**balanced** (a global gauge exists — the ordinary Cartan matrix, hence every tree result of M18) and
**twisted** (the Möbius class — an odd cycle of signs, no global gauge). The corpus had never graded
a signed graph or the affine cycle Ã_n (exp_12 tested trees only; M15's holonomy ran on unsigned
cycles only). The block opened from Peter's reading — local, global, and the delta between them;
local equals global only at the root; a recursive structure with no root is where the Möbius comes
in — recorded in `journals/2026-09-07_local_global_delta.md` (exploring, unscored).

## Theorems filed (three; proofs in the registration, indexed in `formal/theorems/README.md`)

1. **The signed cycle.** The twist selects the odd exponents (2cos((2j+1)π/n) against
   2cos(2πj/n)); charpoly(C₂ₙ^bal) = charpoly(Cₙ^bal)·charpoly(Cₙ^tw); det C_tw = 4; balanced pairs
   over ℚ(√d) iff cond(d) | n, twisted iff cond(d) | 2n; twisted strict iff d = 2 and 4 | n; the
   balanced constant null vector is the zero mode ("the root"), the twisted class has none.
2. **The Möbius holonomy relation.** On M15's instrument H_tw = ε·S·H_bal·S⁻¹: angles θ ↦ π − θ,
   det ↦ (−1)^k det, and M15's C₆ = −I becomes H = I under the twist. Forward note to M15 exp_05 K3:
   the recorded per-row reflection counts are eigenvector-sign-gauge dependent (21 of 38 rows differ
   across numpy builds); det H, angles and reflection parity reproduce exactly.
3. **The cover doubles the cyclomatic number** — a construction parent is a tree iff its diagram is;
   the twisted-diagram route to the third species is closed.

Theorems are gates, not tests (STANDARDS §2.8): KA-1..8 on the record before the seal
(`results/explore_f0_gates_20260907_100206.json`; the first run `_095646` compared a non-invariant
and is kept as the correction's evidence). 392 grade cells at the predicted grade, 74 holonomy
cells, M15's 38 K3 rows, 1,400 grader cells against exp_12's grade, A001429 counts n ≤ 11.

## Registered round (signed connected unicyclic graphs, 61,131 graphs × 2 classes, n ≤ 14): 1/3

- **T1 PASS** — the parity law holds on the first non-tree class: strict over √5 only at n = 8
  (3: 2 balanced, 1 twisted) and n = 12 (21: 12 and 9); 0 at 6, 10, 14 and every odd n.
- **T2 FAIL as sealed** — "the twist never loses a field" is a cycle theorem only: 112 of 1,040
  graphs at n ≤ 10 lose a field under the twist (every field lost somewhere; 54 pure losses).
- **T3 FAIL as sealed, informative** — at n ≡ 2 (mod 4) the parity obstruction is an
  odd-multiplicity rational factor of degree ≥ 2, not a rational root: 639 of 2,889 root-free
  signed graphs at n = 6, 10, 14 pair over √5, all at partial grade.

Milestone 55/74 → **56/77**; theorems 10 → 13. Outcomes `journals/2026-09-07_exp20_outcomes.md`.

## Files

- `experiments/milestones/milestone18/core/signed.py` (new): switching, signed Cartan channel,
  exact charpoly, per-factor grading (bypasses `certificate.grade`'s odd-degree screen), the
  unicyclic enumerator (A001429-asserted), M15's holonomy on a signed adjacency (imported).
- `scripts/explore_f0_gates.py`, `scripts/exp_20_signed_unicyclic.py`; results (append-only).
- Journals: `2026-09-07_blockF_registration.md`, `2026-09-07_exp20_outcomes.md`,
  `2026-09-07_local_global_delta.md`.
- README (Block F row, scorecard, theorems line), `meta.yaml`, `THEORY_MAP.md` (signed-cycle row;
  parity row), `ROADMAP.md` (M15 item (a) started here), `formal/theorems/README.md` (+3),
  `formal/conjectures/m18_open.md` (field-resonance and parity rows).

## Reported, not touched (the engine is Peter's lane)

reality-engine v3: the Möbius identification is applied only by the Reynolds projector; `twist()`
carries no sign; `laplacian()` and `projections.py` use replicate/periodic boundaries and never see
the seam; a v3 test docstring contradicts its assertion; "Poincaré activation" is defined nowhere.

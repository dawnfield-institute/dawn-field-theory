# exp_12 outcomes — resonance, signature, and fold-invariant laws at n ≤ 14

**Registration:** `5bc8faff` (Phase 3, `2026-09-01_phase3_registration.md`), scored to the sealed
text. **Scripts:** `exp_12_part1_resonance.py`, `exp_12_part2_fold_laws.py` (n ≤ 12 and n ≤ 14);
explorations `explore_r7_two_component_corollary.py`, `explore_l1b_quotient_folds_n14.py`,
`explore_r8_t6_half_choices.py`, `explore_r9_mixed_trees.py`. **Results:** `explore_l1b_quotient_folds_n14.json`,
`explore_r8_t6_half_choices_n14.json`, `exp_12_part1_resonance.json`,
`exp_12_part2_n12.json`, `exp_12_part2_n14.json`, `explore_r7_two_component_corollary.json`.
**Verdict: 3/6** (T1, T2, T4 pass; T3, T5, T6 fail as sealed — see each section for scope and
what replaces the sealed clause). **Inputs:** the exhaustive n ≤ 13 census (`explore_g1_census_20260901.json`) and exp_11b's
exhaustive n = 14 list (56 golden trees; 41 Galois folds, 15 without a one-5 partner).

## Scorecard

| Test | Result | Count |
|---|---|---|
| T1 resonance law | **PASS** | finite 4/4; affine trees never strict (recorded, not scored) |
| T2 det(parent) = N(q(0)) | **PASS** | 68/68 Galois folds at n ≤ 14 |
| T3 signature split | **FAIL** as sealed | copy = diagram signature 37/37; conjugate definite 34/37 |
| T4 strict-fold invariant | **PASS** | 7/7 (exp_11 T4: no strict fold at 14, so nothing new to test) |
| T5 core-fold invariant | **FAIL** as sealed | tr(RD) = 2/√5 on 12/12 of the registered domain; ‖(I−P)BP‖² = 2/5 on 10/12 |
| T6 quotient diagnostic | **FAIL** as sealed | (0, 1) on 8/11 verified quotient folds; two n = 14 golden trees fold by both mechanisms on separate automorphism sectors |

**3/6.** Kill relevance (per the seal): none for the milestone kill sentence. T1 and T2 pass, so the
one-5 conjecture is not retired at n = 14. Each failure has a scope, stated in its section; what
each replaces it with is stated there too.

## T1 — field-resonance law (finite trees with 13 ≤ n ≤ 14; affine trees)

Law: a tree pairs over ℚ(√d) iff √d ∈ ℚ(ζ_{2h}). Observed grades per d ∈ {2,3,5,7,13,15}:

| Tree | h | predicted d | observed non-trivial d (grade) | law holds |
|---|---|---|---|---|
| A13 | 14 | [7] | ['7'] (partial) | yes |
| A14 | 15 | [5] | ['5'] (partial) | yes |
| D13 | 24 | [2, 3] | ['2', '3'] (partial, partial) | yes |
| D14 | 26 | [13] | ['13'] (core) | yes |

Affine trees (h is not the invariant; the spectrum contains 0):

| Tree | non-trivial grades | strict? |
|---|---|---|
| D~5 | none | strict: no |
| D~6 | 2:partial | strict: no |
| D~7 | 5:partial | strict: no |
| D~8 | 3:partial | strict: no |
| E~6 | none | strict: no |
| E~7 | 2:partial | strict: no |
| E~8 | 5:partial | strict: no |

Every finite tree pairs exactly over the predicted fields and nowhere else; no affine tree is strict
over any of the six fields. (The affine "partial" grades are the finite sub-diagrams' factors showing
through; not scored.)

## T2 — determinant law

det(parent) = N(q(0)) = q(0)·σ(q(0)) on all 68 Galois folds at n ≤ 14 (27 at n ≤ 12, 41 at n = 14).
This is a corollary of the Ledger theorem (charpoly = q·σ(q)) and was expected to pass.

## T3 — signature split: what was sealed, what is true

Sealed: "(k,0)/(k,0) for definite diagrams and (k,0)/(k−1,1) for hyperbolic ones", i.e. the copy
definite and the conjugate carrying the negative direction.

With P defined as the seal defines it — the Bezout projector for the **diagram's own** polynomial q —
the copy is the eigenspace sum of the diagram's Gram eigenvalues, so **signature(copy) =
signature(diagram)** (37/37 evaluable folds), and the complementary copy is the eigenspace sum of
σ(q)'s roots, which are the Gram eigenvalues of the Galois-conjugate diagram (bond −φ ↦ 1/φ, a
bipartite sign gauge away from −1/φ), so **signature(conjugate) = signature(σ-diagram)**. Both
statements are theorems (charpoly(σM) = σ(charpoly M)); the seal's label assignment was written from
Panel G3's factor convention instead of from the seal's own definition of P, and is reversed.

The empirical content that remains is whether the σ-diagram is definite. It is not always:

- sign(det parent) = (−1)^(negatives in copy + negatives in conjugate), since det = Π over both.
- For a hyperbolic diagram (one negative in the copy) the conjugate is therefore definite **iff
  det(parent) < 0**.
- Every hyperbolic fold at n ≤ 12 has det < 0. At n = 14 three have det > 0 — dets 16, 176, 36 —
  and both copies are indefinite (6,1)/(6,1) on all three, as the sign rule requires.

Scored **FAIL** to the sealed text (§2.6: no relaxation after the run). What survives: the two
signature theorems above, plus the det-sign rule, which replaces the sealed clause with a criterion
that needs no small-n luck. The three positive-det folds are the first parents whose H-diagram's
Galois conjugate is itself hyperbolic.

## T4 — strict-fold invariant

7/7 strict folds (n = 4, 8, 8, 12, 12, 12, 12): tr(RD) = 2/√5 and ‖(I−P)BP‖²_F = 2/5 exactly.
exp_11 T4 found no strict tree at n = 14, so T4 had nothing new to meet there. PASS.

## T5 — the invariant on core-grade Galois folds

The sealed recipe (rational core removed from q; core resolved on the golden conic; gauge
dependence declared) reaches a core only when its eigenspace has dimension 2 and the conic
c·N(τ) + b·Tr(τ) + a = 0 has a rational point. Every fold's P was checked against the Ledger
identities (P² = P, P + σP = I, rank n/2) before evaluation.

**Registered domain** — conic-resolved cores whose invariants agree at two conic points
(12 folds, all Ledger projectors):

| n | det | tr(RD) | ‖(I−P)BP‖² | leak clause |
|---|---|---|---|---|
| 10 | -16 | 2*sqrt(5)/5 | 2/5 | ok |
| 10 | -80 | 2*sqrt(5)/5 | 2/5 | ok |
| 12 | -99 | 2*sqrt(5)/5 | 2/5 | ok |
| 14 | -16 | 2*sqrt(5)/5 | 2/5 | ok |
| 14 | -44 | 2*sqrt(5)/5 | 28/45 | **fails** |
| 14 | 16 | 2*sqrt(5)/5 | 2/5 | ok |
| 14 | -80 | 2*sqrt(5)/5 | 2/5 | ok |
| 14 | 176 | 2*sqrt(5)/5 | 2/5 | ok |
| 14 | -284 | 2*sqrt(5)/5 | 28/45 | **fails** |
| 14 | -80 | 2*sqrt(5)/5 | 2/5 | ok |
| 14 | -176 | 2*sqrt(5)/5 | 2/5 | ok |
| 14 | -464 | 2*sqrt(5)/5 | 2/5 | ok |

tr(RD) = 2/√5 on **12/12**. The leakage norm is 2/5 on 10/12 and 28/45 = 2/5 + 2/9 on two folds at
n = 14 (dets −44 and −284). Scored **FAIL** as sealed (the seal demanded both). Reading, not yet
decomposed: the extra 2/9 is B-leakage from the off-core copy into the conjugate core line — the
core line itself is B-constant on these folds (that is why they are gauge-independent), so it
cannot leak, but the off-core projector can leak into it. The trace law and the leakage law
separate here for the first time: tr(RD) = 2/√5 has now held on 19/19 evaluable Galois folds
(strict and core) with a genuine Ledger projector, while the leakage law is a strict-fold statement
plus the core-fold cases where the core line is decoupled.

**Declared, not scored:**
- 18 folds with gauge-dependent core traces (the core line touches vertices where B varies):
  dets [-356, -284, -171, -164, -144, -124, -116, -44, -36, -20, -4, 4, 36]. The invariant is not defined for these under the recipe.
- 31 folds outside the recipe: 28 with a (t−2)^m core of multiplicity > 2 (Grassmannian, not a
  conic — the seal did not anticipate m > 2; at n = 14 they are the majority), 3 with a quadratic
  rational core (t² − 4t + 2)² (eigenvalues 2 ± √2, outside ℚ(√5); the registered conic lives in
  ℚ(√5)). For these the script builds a declared stand-in (half the core projector), which is not a
  projector, and they are excluded from every scored count.

## Exploration (not registered): the two-component corollary — explore_r7

From T4's invariants alone. Let s_v = √5·R_vv = ±1 on a strict fold (|R_vv| = 1/√5 at every vertex,
Panel L2; re-verified 7/7 here on the exp_12 instrument).
- tr(R) = 0 ⇒ n/2 vertices on each side.
- tr(RD) = 2/√5 ⇒ Σ d_v s_v = 2 ⇒ degree-sum(copy) = n, degree-sum(conjugate) = n − 2 ⇒
  e_copy − e_conj = 1 ⇒ **components(conjugate) = components(copy) + 1**, and the cut has exactly
  2·components(copy) edges.
- Observed 7/7: the copy side is connected ⇒ cut = 2 edges, conjugate side = exactly two
  components.
- Observed 7/7: the sizes of the two conjugate components equal the sizes of the two halves of the
  H-diagram when its 5-bond is removed:

| n | det | conjugate component sizes | diagram halves at the 5-bond | cut edges |
|---|---|---|---|---|
| 4 | 5 | [1, 1] | [1, 1] | 2 |
| 8 | 1 | [1, 3] | [1, 3] | 2 |
| 8 | -11 | [2, 2] | [2, 2] | 2 |
| 12 | -11 | [1, 5] | [1, 5] | 2 |
| 12 | -31 | [1, 5] | [1, 5] | 2 |
| 12 | -71 | [2, 4] | [2, 4] | 2 |
| 12 | -95 | [3, 3] | [3, 3] | 2 |

**Post-scoring extension (explore_r7b_core_fold_structure.py):** on the 12 registered-domain
core-grade folds, |R_vv| is NOT constant (four values per fold: ±1/√5 plus a second, fold-dependent
pair), so the vertex law is strict-only; but under the sign split 11/12 show the full structure
(connected copy, 2-edge cut, two conjugate components with sizes equal to the diagram's halves at
the 5-bond), and the exception (n = 14, det −464: copy 3 components, cut 6, conjugate 4 components)
still satisfies components(conj) = components(copy) + 1 and cut = 2·components(copy) — the two
relations derivable without constancy of |R_vv|. Component relations: 19/19 across strict and
core-domain folds.

Reading (labelled as such): the conjugate-heavy half of the parent is the diagram with its golden
bond cut; the copy-heavy half is connected and carries the two cut edges, which sit over the
5-bond. The lemma is proved; the two 7/7 observations are candidates for a future registration
(a T7) and are not claimed beyond n ≤ 12.

## Instrument record

1. First-run scoring was stricter than the seal (demanded the T5 invariant on gauge-dependent and
   out-of-recipe rows). Corrected to the seal's text; the seal governs in both directions.
2. n = 14, run 1: crash on a multiplicity-2 core whose conic has no small rational point →
   declared path added (`stand-in:conic-unresolved`).
3. n = 14, run 2: quadratic rational cores had only one root removed (P + σP ≠ I on three folds,
   visible as a copy of dimension 8 on a 14-vertex tree) → all roots removed; the case declared.
4. The Ledger identities are now recorded per fold (`ledger_projector`); every scored row has them.
5. The r7 probe's first version took one irreducible piece of q instead of q — caught because it
   contradicted L2's |R_vv| = 1/√5; rebuilt on exp_12's instrument (q from the diagram itself).

## Registration lessons (added to the register)

- Define every operator named in a clause by construction inside the registration; do not write a
  clause from memory of an exploration's convention (T3).
- Enumerate the cases the recipe cannot reach before sealing; multiplicity > 2 and quadratic
  rational cores were both reachable by a five-minute count at n = 12 (T5).
- A script implements a seal; when they diverge the seal governs, whichever direction is stricter.
- A clause scoped over a class ("every quotient fold at n = 14") must ship with the instrument that
  decides class membership at that size; L1b existed only at n ≤ 12 when T6 was sealed.

## T6 — quotient-fold diagnostic

**Scope first.** The seal defines a quotient fold as a golden tree without a one-5 partner that
folds by its automorphism-orbit quotient — Panel L1b's criterion: the quotient matrix's
characteristic polynomial carries exactly the tree's golden factors. exp_11 did not check this at
n = 14; `explore_l1b_quotient_folds_n14.py` does (orbits from the AHU rooted canonical form, which
is exact for trees, in place of automorphism enumeration — the double stars at n = 14 have ~10⁶
automorphisms). Result: **11 of 13 characteristic-polynomial classes (15 trees) are quotient
folds. Two are not** — dets −620 and −80. Their quotients carry only part of the golden spectrum;
the rest lives outside the symmetric sector (for −620 the pair t − 3/2 ± √5/2, t − 5/2 ± √5/2 —
the A₄/H₂ factors; for −80 a conjugate quadratic pair). These trees are not quotient folds, so they are
outside T6's scope. What they are is settled by `explore_r9_mixed_trees.py` (run after scoring):
restrict the Cartan matrix to the symmetric sector S (span of the orbit indicators) and to S⊥. For
both trees the golden part of S⊥ is an exact Galois pair q·σ(q), and its eigenvectors are supported
on two isomorphic subtrees swapped by the automorphism — two A₄ paths for det −620 (spectrum of
S⊥ = (t−2)·charpoly(A₄)), two D₆ subtrees for det −80 (spectrum of S⊥ = (t−2)³ × D₆'s golden
quadratics). S carries the quotient's golden factors. **Both mechanisms act, on different
automorphism sectors:** the trivial sector folds as a quotient, the sign sector folds as a Galois
fold of the swapped subtree (A₄ → H₂, D₆ → H₃). Refined classification (exploration-grade,
n ≤ 14): the golden spectrum of a tree decomposes over its automorphism sectors, and each sector's
golden part is a quotient (trivial sector) or a Galois fold (non-trivial sectors); Panel L1b's
"Galois fold" and "quotient fold" are the pure cases. Exhaustiveness is restored in this form.

**Evaluation** (pure (t−2)^m cores; P = golden half + half core, as sealed). At n = 14, 8 verified
quotient-fold trees were evaluable: 5 give (tr(RD), tr(PB)) = (0, 1) — dets −400, −256 (three
cospectral trees), −1280; 3 give other values — det −16: −14√5/25; det −576: 824√5/465;
det −25600: 112√5/25. `explore_r8_t6_half_choices.py` evaluates every choice of golden half: for
det −16 (two σ-pairs) no choice gives 0; dets −576 and −25600 have a single pair and no freedom.
With n ≤ 12 (3/3): **8/11**. Not evaluated: 5 verified quotient folds with non-(t−2) cores
(the sealed construction resolves λ = 2 only). The two mixed trees are out of scope (they are the
other two "failures" the raw script reported; det −620 shows the Galois value 2/√5 under one
half-choice, consistent with the reading above).

Scored **FAIL** to the sealed text ("tr(RD) = 0 for every quotient fold"). What survives is
one-sided: no quotient fold shows the Galois value (0/8), and every Galois fold with a Ledger
projector shows it (19/19: 7 strict + 12 core-domain). **2/√5 certifies a Coxeter-diagram fold;
0 does not characterize a symmetry quotient.**

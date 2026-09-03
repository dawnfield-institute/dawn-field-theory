# exp_13 outcomes — Phase 4 at n = 16 (registration ade32b0a, SEALED)

**Verdict: 5/7** — T1, T3, T5, T5b, T6 pass; T2 and T4 fail as sealed, and each failure is a
discovery. Census: 19,320 trees — 14 strict, 103 core, 919 partial, 15,443 none. All instrument
self-tests passed before the run (grade known-answers; full known-answer runs of the T3/T4
evaluator on E₈ and of the T6 sector instrument on the n = 6 quotient fold).

| Test | Result | Count |
|---|---|---|
| T1 one-5 conjecture at k = 8 | **PASS** | zero orphans — every 8-node one-5 diagram has a 16-vertex parent |
| T2 strict ⇒ one-5 fold | **FAIL** | 13 of 14 strict trees are one-5 folds; one is not (det −775, below) |
| T3 strict invariant on fresh folds | **PASS** | **13/13**: tr(RD) = 2/√5 and leak 2/5 on every fresh strict fold |
| T4 vertex law + structure | **FAIL** as sealed | 12/13; the exception fails only the halves clause, resolved below |
| T5 trace law on core folds | **PASS** | 4/4 on the registered domain (76 declared out of recipe) |
| T5b component relations | **PASS** | 4/4 |
| T6 sector classification | **PASS** | 24/24 non-Galois golden trees: 21 pure quotient, 3 mixed, 0 unexplained |

T3 is the headline: the invariant, built entirely on seven folds at n ≤ 12, held on thirteen
strict folds at n = 16 that no law was fitted to. The first genuinely out-of-sample confirmation.

## T2 — the sector-strict tree (det −775)

One strict tree has no one-5 partner. Its characteristic polynomial over ℚ(√5) is
(A₄'s golden pair)² · q₁·σ(q₁) · q₂·σ(q₂) — no rational factor, every factor σ-paired, four
golden eigenvalues doubled. Its shape explains it: a degree-4 hub carrying **two A₄ branches
exchanged by an automorphism**, plus two asymmetric branches. This is the explore_r9 mixed
mechanism producing *strictness*: the antisymmetric sector holds an intact A₄ spectrum (the
squared pair), the symmetric sector's golden part pairs on its own, and no sector contributes a
rational factor. The first **sector-strict** tree — strict without being a one-5 fold. The
odd-k theorem explains why n ≤ 13 never showed one; at even k nothing forbids it, and 16 is the
first even-k tier with room for it. T6 — sharpened at the pre-seal review precisely to refuse
vacuous classifications — covered this tree correctly (its sectors pair; it is "mixed").
Scored FAIL to the sealed text; the corrected classification claim for a future seal is:
*every strict tree is a one-5 fold or a sector-strict tree, and the sector decomposition
certifies which.*

## T4 — cospectral 5-bond placements

The failing fold passes the vertex law, connected copy, 2-edge cut, and two conjugate
components — only the halves clause fails: observed sizes [3, 5] against "the diagram's"
halves [1, 7]. Diagnosis: the same 8-node tree carries its 5-bond at two inequivalent edges
giving the **same** q·σ(q) (halves [1, 7] and [3, 5]); the instrument's partner map keeps the
first. The observed structure matches the second placement exactly. So: (a) the halves law holds
under "some one-5 partner" semantics, 13/13; (b) the fold's vertex structure **selects the
placement** — the parent distinguishes cospectral diagrams, which no spectral datum can.
Scored FAIL to the sealed text (registration lesson #4: a clause that names "the" partner must
either prove uniqueness or quantify over partners). The selection phenomenon is new information
beyond the sealed claim.

## T5/T5b — the declared wall

At n = 16 the registered core recipe reaches almost nothing: 4 of 80 core folds (all four exact:
trace 2/√5, leak 2/5, relations hold). 76 declared: 73 multiplicity > 2 (some alongside
quadratic rational cores (t² − 4t + 2)², one with conics unresolved at 2 ± √2). The off-core
formulation (explore_r11–r14, journal `2026-09-01_r11_core_anatomy.md`) needs none of these
declarations and is the natural Phase 5 instrument. The dated r11 prediction that sealed T5b
might fail at 16 where the off-core form holds did **not** materialize on this small domain —
recorded as written.

## T6 — the sector classification generalizes

21 pure quotient folds, 3 mixed trees (the mechanism found at n = 14 recurs at 16), 0 unexplained,
0 unpaired — including the sector-strict tree. The per-sector picture (quotient in the trivial
sector, Galois folds in sign sectors) now stands at n ≤ 16 with zero residue.

## Consistency checks

det(parent) = N(q(0)) on all 13 partnered strict folds (recorded, theorems, not scored).

## Registration lessons

4. A clause that names "the" partner of an object must prove uniqueness or quantify — cospectral
   one-5 placements exist at k = 8 (T4).
5. A universally quantified classification ("every strict tree is a fold") sealed at odd-k sizes
   must anticipate the mechanisms even k re-admits (T2; the odd-k theorem was itself the warning).

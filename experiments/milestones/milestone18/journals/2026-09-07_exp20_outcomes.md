# exp_20 outcomes — Block F: the parity law holds on the first non-tree class; the twist is not a conductor rule; the obstruction is not a root

**Registration:** `872c3ad2` (`2026-09-07_blockF_registration.md`), scored to the sealed text.
**Gates:** `results/explore_f0_gates_20260907_100206.json` (KA-1..8 PASS; the first run, `_095646`,
compared M15's non-invariant reflection *count* and failed on 21 rows — kept as the correction's
evidence, see §2 of the registration). **Runs:** `results/exp_20_signed_unicyclic_n3to12_20260907_100350.json`
(n ∈ [3, 12], seven fields at n ≤ 10; 276 s) and `results/exp_20_signed_unicyclic_n13to14_20260907_100352.json`
(n ∈ {13, 14}, √5 only; 222 s); logs beside them. **Objects:** 61,131 signed connected unicyclic graphs
(A001429 asserted at every n), both switching classes: 122,262 signed objects.
**Verdict: 1/3** (T1 pass; T2, T3 fail as sealed — T3 informative, as the seal said it would be either way).

## Scorecard

| Test | Result | Count |
|---|---|---|
| T1 the parity law on the first non-tree class | **PASS** | strict over √5 only at n = 8 (3) and n = 12 (21); 0 at 6, 10, 14 and every odd n; both classes |
| T2 the twist never loses a field (n ≤ 10, seven fields) | **FAIL** as sealed | 112 exceptions of 1,040 graphs (748 equal, 180 proper supersets) |
| T3 the obstruction is a root (n ≡ 2 mod 4) | **FAIL** as sealed, informative | 639 of 2,889 root-free signed graphs pair over √5, all at partial grade |

Kill relevance (per the seal): none for the milestone. T2's kill fires — "the twist doubles the
conductor" dies beyond the pure cycle, where it is Theorem 1. T3's kill fires — the rational-root
mechanism dies as *the* explanation of the parity law; Panel G's odd-diagram theorem stands as a
diagram statement. T1 gives the parity law its first non-tree confirmation.

## T1 — the parity law holds, in both classes

| n | graphs | root-free (bal / tw) | screened (bal / tw) | strict (bal / tw) |
|---|---|---|---|---|
| 4 | 2 | 0 / 1 | 0 / 1 | 0 / 0 |
| 6 | 13 | 3 / 4 | 0 / 0 | 0 / 0 |
| 8 | 89 | 12 / 18 | 2 / 7 | **2 / 1** |
| 10 | 657 | 60 / 74 | 10 / 13 | 0 / 0 |
| 12 | 5,026 | 253 / 301 | 25 / 37 | **12 / 9** |
| 14 | 39,260 | 1,268 / 1,480 | 75 / 90 | 0 / 0 |
| odd n | 16,084 | (n = 13: 84 / 84) | 0 / 0 | 0 / 0 |

Strict signed unicyclic graphs exist (24 signed objects), so the clause is informative, and every
one sits at n ≡ 0 (mod 4). The parity law — observed on trees exhaustively to n = 24 and proved only
for construction parents — holds on the first class of graphs that are not trees.

**Recorded, not scored.** The twist both creates and destroys strictness: 14 balanced-strict and
10 twisted-strict objects, and **no graph is strict in both classes** (0 of 24). Strictness is
class-exclusive at n ≤ 14. The twisted square (n = 4) is the one root-free, norm-screened object
below n = 8 and is not strict over √5 — it is strict over √2 (Theorem 1), the field the seal did
not score.

## T2 — the twist moves fields both ways

Beyond the pure cycle, where conductor doubling is a theorem, the twisted class is *not* a superset
of the balanced class in the fields it pairs over. Exceptions per size: 0, 0, 0, 3, 5, 15, 19, 70 for
n = 3..10. Every field is lost somewhere:

| d | 2 | 3 | 5 | 6 | 7 | 13 | 15 |
|---|---|---|---|---|---|---|---|
| lost by the twist (graphs) | 36 | 27 | 28 | 24 | 18 | 5 | 5 |
| gained alongside a loss | 24 | 21 | 18 | 5 | 4 | 2 | 0 |

In 54 of the 112 exceptions the twist gains nothing in return. The balanced and twisted classes
have identical field sets on 748 of 1,040 graphs and the twisted class is a proper superset on 180.
Reading (labelled as such): on the pure cycle the twist acts on one Chebyshev family and can only
enlarge the conductor; on a unicyclic graph the tails break the Chebyshev structure and the sign
acts on the characteristic polynomial as a genuine deformation, in no fixed direction. "The twist
doubles the conductor" is a statement about the cycle, not about orientation.

## T3 — the obstruction at n ≡ 2 (mod 4) is an odd-multiplicity rational factor, not a root

Among signed unicyclic graphs with **no integer Cartan eigenvalue** at n = 6, 10, 14 (7 + 134 +
2,748 = 2,889 objects), 2 + 37 + 600 = **639 pair over √5, every one at partial grade**: golden
content beside a rational irreducible factor of odd multiplicity and degree ≥ 2. The degrees of
those factors: 2 (27), 3 (40), 4 (20), 5 (50), 6 (45), 7 (20), 8 (40), 9 (2), 10 (273), 12 (212).
The most frequent single factor is t² − 4t + 2 (19 times) — the Cartan form of adjacency ±√2, the
twisted square's field — but the bulk are large irreducible rational factors (degrees 10 and 12 at
n = 14). So the parity obstruction on graphs is generic rational irreducibility of odd multiplicity,
not the fixed point λ = 2 of the bipartite duality. Panel G's mechanism ("odd k forces a rational
root") is a theorem about diagrams and stays one; the mechanism of the parity law on trees and
graphs is **open**, and the tree proof should not look for a root.

## What was learned about the instruments (recorded)

1. **A test on a Chebyshev class is a theorem.** The design's pure-cycle tests were all provable
   before running and would have been tautological under §2.8; they became gates KA-3..6 and three
   filed theorems. The live round is on the first class where nothing forces the answer.
2. **Compare holonomy invariants only.** M15 exp_05 K3's per-row reflection counts are
   eigenvector-sign-gauge dependent (21 of 38 rows differ between the 2026-07-17 record and this
   numpy); det H, the angles and the reflection parity reproduce exactly. The anchor was corrected
   before the seal; both gate files are on the record.
3. **`certificate.grade`'s odd-degree screen is a tree shortcut.** As a general grade it returns
   `none` for a polynomial with golden content whenever any ℚ-factor has odd degree > 1 (balanced
   C₂₈ over √7). The round grades per ℚ-irreducible factor (= exp_12's grade, 1,400/1,400 cells).
4. **Enumerators are gated by count** (A001429 at every n). An enumerator that miscounts is not a
   census.

## What is not claimed

No physics; nothing here has been shown to a dynamics, and the standing kill-sentence is untouched.
The strict-exclusivity of the two classes (0 of 24) is recorded at n ≤ 14 and not claimed beyond it.
Bicyclic graphs — the first class where the holonomy relation of Theorem 2 is *not* forced — are the
named follow-on and were not run.

## Forward note

Layer: mathematics. The parity-law row of `formal/conjectures/m18_open.md` gains the unicyclic
result and loses the rational-root mechanism as its candidate explanation; the field-resonance row
gains the signed cycle as a theorem and the two-way deformation beyond it. Theorems 1–3 are indexed
in `formal/theorems/README.md`. ROADMAP M15 item (a) is marked started here.

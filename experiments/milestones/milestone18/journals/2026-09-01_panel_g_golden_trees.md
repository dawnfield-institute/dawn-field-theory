# 2026-09-01: Panel G — the geometry and number theory of golden trees

**Mode: EXPLORING, declared.** Exact arithmetic throughout (sympy); numerics only as
prefilters, always confirmed exactly. Nothing here is scored; the sealed consequences are in
the Phase 3 registration draft. Scripts: `explore_g1_census_exhaustive.py`,
`explore_g3_g4_g6_geometry.py`, and the inline probes recorded below.

## G4 — The fourth folding: cat8 → [3,5,3]

The census's hyperbolic caterpillar (8 vertices, spectral radius 2.0953, det −11) folds:

    charpoly(cat8) = q · σ(q),   q = charpoly of the Coxeter diagram [3,5,3]

verified exactly, with cross-checks ([3,3,5] against cat8, [3,5,3] against E₈) false.
[3,5,3] is the symmetry group of the icosahedral honeycomb {3,5,3} in hyperbolic 3-space
(Gram signature (3,1)). With the classical three the picture is:

| golden tree | vertices | det | folds to | Coxeter diagram | geometry |
|---|---|---|---|---|---|
| A₄ | 4 | 5 | H₂ | [5] | pentagon |
| D₆ | 6 | 4 | H₃ | [3,5] | icosahedron |
| E₈ | 8 | 1 | H₄ | [3,3,5] | 600-cell |
| **cat8** | 8 | **−11** | — | **[3,5,3]** | **icosahedral honeycomb, H³** |

The fold is a Galois projection, not an automorphism quotient: cat8 has a Z₂ flip, but the
orbit-averaged Cartan's spectrum (−0.095, 1.262, 2.477, 3.356) is not [3,5,3]'s (−0.095,
1.523, 2.477, 4.095) — only the flip-even pair coincides. E₈, with trivial Aut, already
showed foldings need no automorphism; cat8 shows that even when one exists it is not the map.

## The one-5 conjecture (verified k = 2..6, i.e. parents on 4..12 vertices)

Probing every linear Coxeter diagram of k ≤ 6 nodes against all trees on 2k vertices:

| diagram | one 5? | Gram sig | tree parent (2k vertices) | parent det |
|---|---|---|---|---|
| [5], [3,5], [3,3,5] | yes | definite | A₄, D₆, E₈ | 5, 4, 1 |
| [3,5,3] | yes | (3,1) | cat8 | −11 |
| [3,5,3,3] | yes | (4,1) | 10-vertex tree, 2 branch points | −36 |
| [5,3,3,3] = H₅ | yes | (4,1) | 10-vertex tree, 1 branch point | −4 |
| [3,3,3,3,5] | yes | (5,1) | 12-vertex tree | −11 |
| [3,3,3,5,3] | yes | (5,1) | 12-vertex tree | −71 |
| [3,3,5,3,3] | yes | (5,1) | 12-vertex tree | −95 |
| [5,3,5], [5,3,3,5], [3,5,3,5], [5,3,3,3,5], [3,5,3,3,5] | two 5s | — | **none** | — |
| [3,4,3,5] | contains 4 | — | **none** | — |

**Conjecture (exploring; sealed at n = 14 in the Phase 3 draft):** a linear Coxeter diagram
of 3s and 5s has a golden tree parent iff it contains exactly one 5. Every hyperbolic
parent has negative determinant — the diagram's signature (k−1,1) doubles to (2k−1,1) —
and det(parent) = N(q(0)), the field norm of the diagram polynomial's constant term.

Trivial companion theorem: strict complete pairing forces even n (q·σ(q) has even degree);
odd-n golden content can only be of ledger-with-rational-core grade.

Correction owed to exp_08: the Prüfer-sampled census (86 of 106 classes at n = 10) reported
"n = 8 only". Both 10-vertex parents above sit in the unsampled 20 classes. Exhaustive
enumeration (G1 below) is the authority; at small n, enumerate — never sample.

## G3 — Signature under σ

| tree | sig(C) | copy | conjugate |
|---|---|---|---|
| A₄ | (4,0) | (2,0) | (2,0) |
| D₆ | (6,0) | (3,0) | (3,0) |
| E₈ | (8,0) | (4,0) | (4,0) |
| cat8 | (7,1) | **(4,0)** | **(3,1)** |

Folding diagrams split definite/definite; the hyperbolic tree splits into a definite copy and
an indefinite conjugate — the (1,1) indefinite signature of the golden norm a² + ab − b²,
realized as the σ-split of a concrete operator. Which copy is definite is σ-asymmetric: the
negative eigenvalue belongs to one specific golden quadratic factor (σ-image positive).

## G6 — Galois structure (the mechanism of non-cyclotomic goldenness)

| tree | ℚ-irreducible factors | discriminant | Galois group |
|---|---|---|---|
| A₄ | two quadratics | 5 | C₂ |
| D₆ | (t−2)² · quartic | 2⁴·5³ | C₄ |
| E₈ | one octic | 2⁸·3⁴·5⁶ | (degree 8; abelian, cyclotomic) |
| cat8 | two quartics | 5²·29 each | **D₄** (dihedral, non-abelian) |

ADE goldenness is cyclotomic (abelian Galois groups, exponents mod 5 — exp_06's law).
Hyperbolic goldenness is not: cat8's quartics have dihedral Galois group with ℚ(√5) as the
quadratic subfield, and 29 (a split prime, 29 ≡ 4 mod 5) in the discriminant. Two
mechanisms, one ledger.

## G1 — exhaustive multi-field census, n ≤ 13 (2,275 trees, 894 with any content)

Grades over ℚ(√5) (strict / ledger-with-rational-core / partial); other fields analogous:

| n | trees | √5 strict | √5 core | √5 partial | √2 s/c/p | √3 s/c/p | √13 s/c/p |
|---|---|---|---|---|---|---|---|
| 4 | 2 | **1** | 0 | 0 | 0/0/0 | 0/1/0 | 0/0/0 |
| 6 | 6 | 0 | **2** | 0 | 0/0/1 | 0/0/1 | 0/1/0 |
| 8 | 23 | **2** | 1 | 1 | 0/0/4 | 1/1/0 | 0/2/1 |
| 10 | 106 | 0 | **7** | 7 | 0/0/18 | 0/5/13 | 0/4/1 |
| 12 | 551 | **4** | 15 | 38 | 0/0/105 | 0/10/50 | 1/8/9 |
| odd n | 1,597 | 0 | 0 | (partial only) | | | |

Two trivial theorems visible in the table and provable in a line: a ledger (strict or core)
forces even n (q·σ(q) plus an even-multiplicity rational core has even degree); strict
forces the absence of any rational eigenvalue.

### The fold classification (join on characteristic polynomial)

For each k ≤ 6, every tree-shaped Coxeter diagram of 3s with exactly one 5-bond was
enumerated and its q·σ(q) looked up among the Cartan polynomials of all 2k-vertex trees:

| k | one-5 diagrams | with a tree parent | golden trees at n=2k | of which folds | non-folding |
|---|---|---|---|---|---|
| 2 | 1 | 1 | 1 | 1 | 0 |
| 3 | 1 | 1 | 2 | 1 | 1 (star₆) |
| 4 | 3 | 3 | 3 | 3 | 0 |
| 5 | 6 | 6 | 7 | 6 | 1 |
| 6 | 15 | 15 | 18 | 15 | 3 |

**Zero orphan diagrams at every size: every one-5 tree-shaped diagram has a tree parent.**
Every *strict* golden tree is a fold. The non-folding golden trees (1, 1, 3) are all
core-grade, high-degree (max degree 5, 3, 6, 6, 10) — star-like objects whose √5 arrives
"accidentally" (star₆: adjacency eigenvalues ±√5 = ±√(n−1)); they carry the ledger form but
no diagram. The fold test, not the grade, is the classifying invariant.

### Why odd diagrams ramify (provable)

Odd-k parents (D₆ ← [3,5]; the 10-vertex parents of [3,5,3,3] and H₅) are core-grade; even-k
parents are strict. Reason: the k-node diagram is bipartite, so its spectrum is self-dual
under λ ↔ 4−λ; odd k forces a fixed point λ = 2 — a rational root of q, hence (t−2)² in the
parent. The ramified core of exp_06 is the Mirror's relation-free level, forced by parity.

### Field resonance (predicted, then verified on all 21 ADE trees n ≤ 12)

A cyclotomic tree with Coxeter number h pairs over ℚ(√d) iff √d ∈ ℚ(ζ_{2h}). Checked for
d ∈ {2, 3, 5, 13, 15} on A₄–A₁₂, D₄–D₁₂, E₆–E₈: **21/21**. E₈ (2h = 60) is strict over
√3, √5 and √15 — all three real quadratic subfields of its Coxeter field — and over nothing
else; A₁₂ (2h = 26) over √13 alone; A₄ over √5 alone. The grade follows from exponent
multiplicities (rational roots ⇒ core; unpaired factors ⇒ partial).

### Correction to exp_08 (forward, per lineage rule)

exp_08's Prüfer census reported complete pairing at "n = 8 only". Exhaustive enumeration
shows strict pairing at n = 4, 8, 12 (1, 2, 4 trees) and core-grade ledgers at n = 6, 8, 10,
12; the two 10-vertex folds sat among the 20 of 106 classes sampling never drew. The
registered 1/2 stands; the "n = 8 only" sentence is withdrawn here.

## Why 5 (the bond question), and the Eisenstein non-analogue

Among the bond types whose cosine generates a quadratic field (m = 5, 8, 10, 12), only the
5-bond series has simply-laced tree parents: 5/5 one-bond tree diagrams through k = 4 fold
from trees, against 0/5 for m = 8, 10, 12. Allowing √2 and √3 bonds in the parent recovers
the classical non-simply-laced foldings — [8] ← the B₄ shape (one √2 bond), [12] ← the F₄
shape — while [10] has no rank-4 parent even then. So the instrument rediscovers the
crystallographic folding classification from the tree side, and isolates the fact that
2cos(π/5) = φ is the only quadratic bond weight that is itself a unit of its field: the
5-bond is the unique one whose Galois-fold parents are simply-laced (ADE-type). This is the
exact content behind "Δ_φ = 5 is significant" (Farmer, gap U-GRA-09).

Eisenstein: real symmetric Cartan spectra never pair over ℚ(√−3) (E₈'s polynomial is
irreducible there; A₄'s ℚ(√−3)-factorization is just its ℚ-factorization). The ℤ[ω]
analogue of the ledger cannot live on Cartan matrices; it would need Hermitian objects. The
invitation extended in the summary is corrected accordingly.

**Forward correction (same day):** the sentence above — "2cos(π/5) = φ is the only quadratic
bond weight that is itself a unit of its field" — is false: 2cos(π/12) = √(2+√3) has minimal
polynomial t⁴ − 4t² + 1 and is a unit. The correct discriminator is degree: among m ≥ 4,
2cos(π/m) has degree 2 for m ∈ {4, 5, 6} and degree 4 for m ∈ {8, 10, 12}; of the degree-2
cases only m = 5 has an irrational square (φ² = φ + 1), so only the 5-bond yields a q that
is genuinely golden (σ(q) ≠ q) with the bond weight inside the field. Caught on re-derivation
before anything downstream used it.

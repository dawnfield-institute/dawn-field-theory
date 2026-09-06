# 2026-09-06 — the lattice end and the scaling end: Farmer's 3 against this corpus's 5

**Layer: mathematics, exposition.** Nothing here is registered, scored or predicted. It records
what a second exchange with Andy Farmer (WOLF) turned up when checked against the corpus, and it
writes down three sentences the corpus did not yet contain. The mode is *exploring*, not
*predicting* (STANDARDS §2.7.5); one candidate claim is named at the end as a thread, not a
registration.

## What arrived

Two screenshots (Discord, 2026-09-06), LLM-assisted text, so each claim was checked on its own.

**"Advance 3: representation-theoretic proof of decoupled notation."** The claim that writing
½ + √5/2 rather than (1 + √5)/2 has a rigorous justification: the trace midpoint ½·Tr(α) = ½
transforms under the trivial Galois representation, the "radical scar" ±√d/2 under the sign
representation, and the fraction conflates them.

*Check.* True, and elementary. A quadratic field K/ℚ with Gal = ℤ₂ is K = ℚ ⊕ ℚ√d as a Galois
module, the trivial and sign isotypic components, and every element is uniquely Tr(α)/2 plus an
element of ker Tr. Decoupled notation displays that decomposition; the fraction hides it. It is a
notational convenience with a correct footnote, not a theorem, and the corpus adopted the
discipline on 2026-08-31 as canonical form a + bφ (`journals/2026-08-31_founding.md`). His remark
that the split notation helps language models pattern-match is about prompting, not fields.

**"TEST-01: the tangent fallacy."** The claim that tan(θ/2) ∉ ℚ ⇏ θ/τ ∉ ℚ, with the counterexample
tan 30° = 1/√3 ∉ ℚ while 30° = τ/12, followed by: "this proves why the non-monic minimal
polynomial (13x⁴ − …) is mathematically necessary to establish irrationality."

*Check.* The fallacy is real and the counterexample is the right one. The sentence after it is
wrong for tangents and right for cosines. tan 30° has minimal polynomial 3x² − 1 — non-monic — and
30° is a rational angle, so "non-monic ⇒ irrational angle" fails on the example that precedes it.
The criterion that works is Niven's: θ/π ∈ ℚ ⇒ 2cos θ is an algebraic integer, so a non-monic
primitive minimal polynomial for 2cos θ proves the angle irrational. Whether his 13x⁴ − … is the
polynomial of a tangent or of 2cos θ decides whether the test proves anything. Asked in the reply.

**"3 is everything."** Peter's reading of the thesis: everything in mathematics is built on
squares — quadratics — and moving to triangles and groups of three is more primitive.

## Three sentences the corpus did not have

### 1. Degree two is forced by symmetry; triangles are quadratic too

"Quadratic" and "squares" are different things. The smallest nontrivial symmetry is an
involution, and the invariant of an involution is a quadratic form. Reflection groups, Cartan
matrices with their bond weights 2cos(π/m), inner products, M13's complement, the two sheets of
M18's branched double covers: all are degree two because they are built on flips, not on squares.
And the triangle does not escape it. The field of the cube roots of unity has degree
φ(3) = 2, because conjugation swaps ω with ω⁻¹ — the flip sits underneath the threefold turn.
Threefold, fourfold and sixfold symmetry all live in quadratic fields (φ(3) = φ(4) = φ(6) = 2);
fivefold is the first that needs degree four (φ(5) = 4), with the golden field as its quadratic
subfield. So "triangles rather than squares" is a choice between two quadratic fields, not an
escape from quadratics, and no symmetry lives below degree two.

### 2. Within that choice, 3 comes first — exactly

The Eisenstein field ℚ(√−3) has the smallest discriminant of any number field other than ℚ,
|−3|, ahead of the Gaussian field at 4 and the golden field at 5. In the one ordering that says
"most primitive" without hand-waving, Farmer is right: 3, then 4, then 5. And S₃ — the Weyl
group of A₂, the triality of D₄ — is the smallest non-abelian group. M14 found D₄ to be the only
ADE type whose automorphism group produces genuine quantum uncertainty, because it is the only
non-abelian one. His 3 is where non-commutativity begins. That is a stronger claim than packing
density, and it is his to make.

### 3. What separates 3 from 5 is real against imaginary — and that is the cascade

An imaginary quadratic field has a finite unit group: six roots of unity for Eisenstein, four
for Gaussian, two for every other. Rotations, tilings, crystals, and nothing else. A real
quadratic field has a fundamental unit of infinite order, and for the smallest one that unit
is φ. The PAC recursion Ψ(k) = Aφ⁻ᵏ + Bψ⁻ᵏ is scaling by a unit; the cascade clock, ln φ per
level, Ξ = γ + ln φ, all need a unit that does not return to where it started. An imaginary
field cannot scale. It can only turn. So the triangle is the primitive geometry of the static
picture and φ is the primitive of the dynamical one, and the reason this corpus lands at 5
rather than 3 is that it is a theory of collapse in time, not of tilings. The same fact explains
where Farmer's notation is exact: a crystal is a cascade with the clock stopped.

In M18's own terms: the fold family γ² + wγ − 1 = 0 has discriminant w² + 4 > 0 for every w, so
every fold field is real, and w = 1 gives the smallest, ℚ(√5). The imaginary fields never enter
the family; Panel G recorded independently that a real symmetric Cartan spectrum never pairs
over ℚ(√−3) and that a ℤ[ω] ledger would need Hermitian objects
(`journals/2026-09-01_panel_g_golden_trees.md`, "the Eisenstein non-analogue").

## Where they meet, and the corpus had already measured it

**The (p, q, r) form.** The simply-laced diagrams are the three-arm trees with
1/p + 1/q + 1/r > 1. With the first two arms fixed at 2 and 3 the third can be 3, 4 or 5 — E₆,
E₇, E₈ — so 3 is the arm every exceptional diagram shares and 5 is the last finite value. The
boundary is (2, 3, 6), sum exactly 1: the triangle group of the hexagonal tiling, which is
affine Ê₈. Farmer's triangular lattice is the flat edge of the family whose finite end folds to
φ; his square lattice is (2, 4, 4), affine Ê₇; (3, 3, 3) is affine Ê₆. MED's node bound (≤ 3) is
the same combinatorial fact from the other side: a branch point of degree four is already affine
D̃₄. This form was not written anywhere in the corpus before today.

**And exp_12 measured the edge.** The field-resonance law (predicted, then 21/21 on the ADE
trees: a tree with Coxeter number h pairs over ℚ(√d) iff √d ∈ ℚ(ζ₂ₕ)) puts all three lattices
under one rule — A₁ gives ℚ(i), A₂ gives ℚ(√−3), A₄ gives ℚ(√5) — and exp_12 T1 recorded that
every finite tree pairs over exactly the predicted fields while **affine trees never pair over
any of them** (`journals/2026-09-01_exp12_outcomes.md`). The golden pairing dies exactly where
the lattice begins.

**P, A and Δ as the sheets and their difference.** Peter's reading: Farmer's split is P and A,
and the third thing is the Δ between them, which SEC sees as a phase transition. In the
quadratic-field picture the two Galois conjugates are the two sheets; the invariant half is the
ledger side and the conjugation-odd half is the side free to flip (M15's per-edge det −1 against
det +1 on every loop). Δ is their difference, twice the radical part, and the phase transition is
where it vanishes: the ramified locus. exp_06's splitting law reads the E₈ exponents mod 5 as
split, inert and ramified, calls the ramified case the gauge core, and its failed tests failed
there (`journals/2026-08-31_exp06_outcomes.md`). The 08-31 note that Farmer's mod-3 trichotomy is
the sister of that law was right and now says why: 3 is the ramified prime of his field, 5 of
ours, and each is "everything" in its own field in the sense that the sheets meet there.

**E₈ holds both.** Its Coxeter number is 30 = 2·3·5, the icosahedron's three rotation orders,
and Panel G found E₈ strict over √3, √5 and √15 and over nothing else. The 3 that appears there
is real — the triangle's real shadow beside the pentagon's. `asymmetric_conservation` exp_16 had
already called {2, 3, 5} the MED collapse basis.

## What is not claimed

- The Fibonacci 3 (F₄ in Koide's 2/3; k = 9 = F₄² as the MED transition) is a golden-side
  number, and no argument here identifies it with the arm or with the ramified prime.
- D = 3 spatial dimensions is derived elsewhere by other routes and is not identified with any
  of the above.
- Two of the five threes in play are one fact; a third is plausibly the same fact; the last two
  are recurrences until someone derives the link (the corpus's standing rule: recurrence is not
  importance unless independent).
- Nothing here predicts. The exposition is consistent with the corpus and adds a form; it does
  not add a result.

## THREAD: crystalline systems carry no cascade clock

If §3 is the right reading, it is a registrable claim: a system whose symmetry field is imaginary
quadratic (a lattice, a crystal, a tiling) has a finite unit group and therefore no scaling
invariant, so none of the cascade signatures — ln φ per level, the φ-crossing, the 0.020 Hz
clock — should appear in it, while any system that shows them has a real unit underneath.
Invariant, not coordinate; falsifiable by one crystalline system with a cascade signature. Not
registered here; recorded so the consolidation quest finds it.

## Housekeeping found on the way

The theory overview states MED's bound as depth ≤ 1 where the lexicon and THEORY_MAP say
depth ≤ 2, nodes ≤ 3. Flagged, not fixed here.

Reply to Farmer drafted in the outbox (unsigned; Peter sends or does not). Data: none; this is
exposition.

# After Phase 8: the square is a theorem, and φ is forced — not chosen (2026-09-04)

**Mode: exploring, then proving.** Nothing here was sealed; these are readings on data we already hold,
two of which closed into proofs the same day (STANDARDS §2.7 item 7, and the file-it-today rhythm).
Prompted by Peter's call after the n = 24 census: *the kills are repositions, not retractions.*

Scripts `explore_s1_resultant_square.py`, `explore_s2_metallic_family.py`,
`explore_s3_third_species.py`; outputs in `results/…_20260904.txt` (append-only).

## 1. The resultant is a perfect square — and the reason is bipartiteness (PROVED)

Phase 8 T3 killed the exact-denominator conjecture but left a sharper fact standing: on all 236 fold
halves we hold, Res(q₀, q₁) is a perfect square (1, 4, 9, 16, 81). The cause is not special to folds.

**Theorem.** *Let F and G be forests and let Res be the resultant of their characteristic
polynomials. If Res ≠ 0 then |Res| is a perfect square.*

*Proof.* A forest is bipartite, so its adjacency spectrum is symmetric about 0: λ is an eigenvalue
iff −λ is, with equal multiplicity. Res(f, g) = ∏_{i,j} (λ_i − μ_j). Restrict first to λ_i ≠ 0 and
μ_j ≠ 0; those indices partition into quadruples {±λ} × {±μ}, and each quadruple contributes

> (λ − μ)(λ + μ)(−λ − μ)(−λ + μ) = (λ² − μ²)²,

a square. If both F and G have a zero eigenvalue the product vanishes, which is excluded. If exactly
one does — say F, with multiplicity m — the remaining factors are ∏_j(0 − μ_j)^m = (±det G)^m, and
det G is, up to sign, ∏ over ± pairs of (−μ²), so |det G| is itself a square; a square raised to m is
a square. The total is a product of squares. ∎ In the Cartan channel C = 2I − A the resultant changes
by a sign only, so the statement transfers verbatim. **Corollary:** if F and G both have a zero
eigenvalue then Res = 0 — which is exactly the degenerate half the census declares and never scores.

**Corollary (the fold half).** q₁ = charpoly(D − A − B) and q₀ ≡ charpoly(D − e) (mod q₁), and both
D − e and D − A − B are forests. So Res(q₀, q₁) is a perfect square on every fold half.

*Verification (`explore_s1`):* 112/112 random forest pairs in the adjacency channel, 120/120 in the
Cartan channel, and the **control is decisive** — on non-bipartite graphs only **35/150** are squares.
Parity split: even/even 86/86, even/odd 61/61, odd/even 48/48; odd/odd yields no nonzero cases at all.

**What this repositions.** The killed conjecture routed the denominator through rad(Res), which
discards the multiplicity that carries the answer. With √Res now a well-defined integer, the
surviving relation on all 236 fold halves is

> **den(5·b) = √Res if √Res is odd; √Res / 2 if √Res is even.**

Checked: (1,1), (4,1), (9,3), (16,2), (81,9) — 236/236, on 8 halves with a non-unit resultant across
4 distinct values. Thin, exact, and filed open, not claimed.

## 2. φ is forced by simple-lacedness (PROVED)

r17 Theorem 1 builds parent(D, e*) with three lifts over the 5-bond, one of them a direct **defect**
edge, and shows the sheet-mixing subspaces W_γ = span{(v,0) + γ(v,1)} are invariant iff γ² + γ − 1 = 0.
The defect edge's weight is the construction's *only* free parameter. Redo the argument with weight w:
(B,0) ↦ w(A,0) + (A,1) and (B,1) ↦ (A,0), so (B,0) + γ(B,1) ↦ (w + γ)(A,0) + (A,1), which lies in W_γ
with weight μ = 1/γ precisely when

> **γ² + w·γ − 1 = 0,  discriminant w² + 4.**

**Theorem.** *The fold field of a weight-w construction parent is ℚ(√(w² + 4)). It is ℚ(√5), with
γ = 1/φ and bond weights φ and −1/φ, exactly at w = 1 — the simply-laced choice, where the defect
edge carries the same weight as every other edge in the cover.*

| w | disc | field | γ |
|---|---|---|---|
| 0 | 4 | ℚ (degenerate, no fold) | ±1 |
| **1** | **5** | **ℚ(√5) — golden** | **1/φ, −φ** |
| 2 | 8 | ℚ(√2) | √2 − 1 |
| 3 | 13 | ℚ(√13) | (√13 − 3)/2 |

γ² + wγ − 1 = 0 is the metallic-mean family; golden is its w = 1 member. Verified symbolically and on
the object: at w = 1 the construction returns the A₄ path and charpoly(parent) = q·σ(q) exactly.

**Why this matters for the duality.** The standing question was whether 1/φ is a fixed point or a
label we chose. It is neither chosen nor coincidental: **φ is what a double cover with integer bonds
and one cross-wire must produce.** Take the cover, demand every edge weigh the same, and the golden
ratio is the only coupling at which the sheets decouple. This answers kill-condition 2 from the
2026-09-03 duality reading in the affirmative, and by derivation rather than by sweep.

## 3. The third species: no handle yet (NULL)

41 asymmetric trees, 9 carrying an integral Galois half. Five looks at what separates the 9 from the
32: degree-4 vertex (7/9 vs 12/32, Fisher p = 0.057), degree-5 vertex, halves count, leaf count.
Nothing survives correction for five looks, and 9 objects cannot carry a claim. **Recorded as a null**
— a bearing, not a verdict. The species remains the milestone's largest unidentified object.

## Filed

- Two theorems indexed in `formal/theorems/README.md` the same day they were proved.
- `formal/conjectures/m18_open.md`: the denominator successor sharpened to the √Res form; the square
  moves out of the conjecture list into the theorem list.
- Not claimed: the third-species separation; any dynamics that drives γ to 1/φ (that remains a
  conversation before it is a registration).

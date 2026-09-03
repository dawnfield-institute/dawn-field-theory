# r21 — the polynomial form of integrality is false (2026-09-02, evening)

**Mode: exploring, not predicting.** Nothing here was sealed; the numbers are readings on the
census we already hold. The result is a *kill* of one open conjecture with a stated scope, and one
refined conjecture filed in its place.

Script `scripts/explore_r21_poly_integrality.py`; results
`results/explore_r21_poly_integrality_20260902.json` (append-only). ~2 min on 8 cores.

## What was tested

The open row in [`formal/conjectures/m18_open.md`](../../../../formal/conjectures/m18_open.md):

> **Polynomial integrality.** The minimal-degree representative b(t) of the reflection
> (R = √5·b(C)) has 5·b ∈ ℤ[t]. The matrix form 5·b(C) = S + 2Π is proved on constructions; the
> polynomial form is not. — 7/7 at n ≤ 12 (explore_r10).

Prompted by an external proof attempt of exactly this statement on construction parents
(A. Farmer, 2026-09-02, private communication; not reproduced here). The attempt's two genuine
observations are recorded in §4; its two gaps are what this exploration measures.

For every strict tree we hold — 22 at n ≤ 16 (census + explore_r16b), 66 at n = 20 (exp_15) —
and for **every Galois half** q of p = q·σ(q) (not only the fold half): the unique Bézout
polynomial v with deg v < deg q and σ(v)q + vσ(q) = 1; P(t) = v·σ(q) mod p; R(t) = 2P − 1;
b = R/√5 (σ-odd, so b ∈ ℚ[t]); the denominators of b and of 5b; and the prime content of
Res(q, σq). Halves whose gcd(q, σq) is non-constant (a repeated conjugate pair) are declared,
not scored. Class labels at n = 20 come from exp_15: construction parent = T3 ledger form ∧ T4
quotient isomorphic (the 47); degenerate partner (6); unpartnered (13).

## Findings

**1. The conjecture is false on construction parents at n = 20.** Three construction parents —
every strict law holding (T3 ledger form and anticommutation, T4 quotient with the mult-3 bond,
T5 single copy-internal defect, T6 trace 2/√5, leak 2/5, vertex law) — have a unique Galois half
whose reflection polynomial has **den(5·b) = 3**, den(b) = 15, and Res(q, σq) = 3²·5⁵ = 28125.
Their characteristic polynomials are distinct (dets −71, 769, −239); deg b = 18 in all three.

| Edges (n = 20) |
|---|
| (0,11) (0,18) (1,0) (1,2) (1,8) (2,3) (3,4) (4,5) (4,7) (5,6) (8,9) (9,10) (11,12) (12,13) (13,14) (14,15) (14,17) (15,16) (18,19) |
| (0,11) (0,18) (1,0) (1,2) (1,8) (2,3) (2,7) (3,4) (3,6) (4,5) (8,9) (9,10) (11,12) (12,13) (12,17) (13,14) (13,16) (14,15) (18,19) |
| (0,11) (0,17) (1,0) (1,2) (1,7) (2,3) (3,4) (3,6) (4,5) (7,8) (8,9) (9,10) (11,12) (12,13) (13,14) (13,16) (14,15) (17,18) (18,19) |

The matrix theorem is untouched: on these three trees √5·R(C) = S + 2Π is an integer matrix
(exp_15 T3). What fails is the step from the matrix to the polynomial: S + 2Π lies in
ℚ[C] ∩ M₂₀(ℤ) but not in ℤ[C]. The order ℤ[C] is not saturated in its endomorphism ring — the
index is divisible by 3 here — and that is precisely the step no proof can skip.

**2. Scorecard of the statement across the census** (halves scored, declared halves excluded):

| n | class | trees | with an integral half | fold-half integral |
|---|---|---|---|---|
| 4–16 | construction parents | 21 | 21 | 21/21 (all diagram halves) |
| 16 | sector-strict (no diagram half) | 1 | 0 | — (0/8 halves) |
| 20 | construction parents | 47 | 44 | 44/47 |
| 20 | degenerate partner | 6 | 0 | — (0/4 each) |
| 20 | unpartnered (sector-strict + asymmetric) | 13 | 0 | — |

So at n ≤ 16 the statement held on every fold half, which is why r10's 7/7 and today's 21/21
looked like a law. It breaks at the first size with a parent whose diagram resultant is not a unit.

**3. Where the denominators come from.** On every integral fold half Res(q, σq) = 5^{n/4} exactly
(5, 5², 5³, 5⁴ at n = 4, 8, 12, 16; 5⁵ on the 44 at n = 20). On the three failures it is 3²·5⁵. On
non-fold halves of parents it is 5⁵·ℓ² with ℓ ∈ {11, 29, 101, 131} and exactly ℓ enters
den(5b). Off the constructions (degenerate partners, unpartnered) the ramified prime itself
enters beyond first order — den(5b) ∈ {5, 25, 125, …} occurs only there. Across all
non-integral halves the denominator primes are {2, 3, 5, 7, 11, 13, 19, 29, 31, 41, 61, 101, 131}:
both inert (2, 3, 7, 13) and split (11, 19, 29, …) primes of ℚ(√5). There is no
ramification-only law.

**4. Certificate reading (exploration, not a registered claim).** "Some Galois half has
5·b ∈ ℤ[t]" holds on 44/47 parents and on 0/19 non-parents at n = 20, and on 21/21 vs 0/1 at
n ≤ 16. As a sufficient certificate of construction parenthood it has no false positives in the
census; it is not necessary (the three).

## The denominator theorem (proved the same evening; filed in `formal/theorems/`)

Any q ∈ ℤ[φ][t] is q = q₀ − φ·q₁ with q₀, q₁ ∈ ℤ[t]; on a fold half this is the expansion along
the realized 5-bond (q₁ = the diagram's charpoly with the bond's two nodes deleted), and
σq − q = √5·q₁. Write the minimal Bézout coefficient as v = a + √5·c with a, c ∈ ℚ[t]. Then
σ(v)q + vσ(q) = 1 has **no √5-part**, and its rational part reads

> a·(2q₀ − q₁) + 5c·q₁ = 1  in ℚ[t],  and  5·b = (5c)·(2q₀ − q₁) + 5a·q₁.

So (a, 5c) is the minimal Bézout pair of the two *integer* polynomials A = 2q₀ − q₁ and q₁ —
uniqueness of the minimal pair identifies it — and Sylvester's identity Res(A, q₁) = a′A + c′q₁
with a′, c′ ∈ ℤ[t] gives, since Res(A, q₁) = Res(2q₀, q₁) = ±2^{deg q₁}·Res(q₀, q₁):

> **Theorem (denominator bound).** den(5·b) divides 2^{deg q₁}·Res(q₀, q₁). In particular the
> ramified prime never enters unless 5 | Res(q₀, q₁), and 5·b ∈ ℤ[t] whenever Res(q₀, q₁) = ±1.

Verified on all 208 scorable halves (`scripts/explore_r21b_rational_reduction.py`,
`results/explore_r21b_rational_reduction_20260902.json`): the rational and ℚ(√5) Bézout
polynomials agree 208/208; the bound holds 208/208; 5 | den(5·b) ⇔ 5 | Res(q₀, q₁) (158 neither,
50 both, 0 mixed). The 144 remaining halves have gcd(A, q₁) non-constant, which is the same thing as
gcd(q, σq) non-constant — the declared repeated-pair halves.

**Direct verification on the three counterexamples (are we certain?).** For each of the three
trees, 5·b(t) computed exactly has denominator 3 (coefficients such as 10/3, 5932/3, −59264/3),
and evaluating that same polynomial on C gives an integer matrix with entries in {−1, 0, 1, 2},
squaring to 5·I, with unit diagonal and row-norm² = 5 — i.e. S + 2Π, on the object itself, with no
appeal to exp_15's flags. A third, Bézout-free computation (Lagrange interpolation of ±1/√5 on the
twenty eigenvalues) reproduces the fractional coefficients to 10⁻⁷ at double precision. The three
methods agree; the integer matrix and the non-integer polynomial coexist.

**What stays open — the exact denominator.** On the 68 fold halves of construction parents:
Res(q₀, q₁) = ±1 → den 1 (63); Res = 4 → den 1 (2); Res = 9 → den 3 (3). So the bound is not
sharp at 2, the three failures are exactly the Res = 9 parents, and whether an odd prime dividing
Res(q₀, q₁) *always* enters on a fold half rests on three cases. Filed as the open row; not a law.

This is the true content of the "(√5)-adic" idea: the ramified prime is controlled — trivially,
once the problem is reduced to ℚ[t] — and the unramified primes are controlled by the diagram
resultant, not excluded by it.

## Provenance and the external attempt

Two things in the external attempt are right and worth keeping: (i) the exact odd difference
(σq − q)/√5 = q₁ ∈ ℤ[t] along the realized 5-bond (explore_r10 recorded the same object as
"the Galois direction" without naming q₁), and (ii) the factorization
Res(q, σq) = ±(√5)^k·Res(q₀, q₁), which is the right lens — the three failures are exactly the
parents with Res(q₀, q₁) = 9. Two things are wrong: the claim that no prime ℓ ≠ 5 divides
Res(q₀, q₁) (argued from real interlacing, which says nothing about common roots mod ℓ;
refuted by the 3² above and by 2², 2⁴, 11², … on other halves already at n ≤ 16), and the step
"5·b(C) ∈ Mₙ(ℤ) ⇒ 5·b(t) ∈ ℤ[t]" (the saturation of ℤ[C], refuted by the three trees). The
attempt also cited a "det −11" example whose q₀ has t³-coefficient −7, impossible for a four-node
Coxeter diagram (trace 8); the polynomial arithmetic on it is correct, the input is not a diagram.
Its A₄, E₈, P₆ and P₈ rows reproduce ours exactly.

## What dies, what survives

- **Dies:** the polynomial-integrality conjecture in its unconditional form, on construction
  parents, at n = 20. Scope: the polynomial form only.
- **Survives untouched:** the matrix form √5·R = S + 2Π ∈ Mₙ(ℤ) (theorem on constructions), the
  construction theorem, every strict law, rigidity 47/47, the vertex law |R_vv| = 1/√5 (a matrix
  statement).
- **Changes:** the r15 Addendum-5 remark that "the integrality route is the standing suggestion
  for the matching form" loses its polynomial half; the matrix half is the theorem we already have.
- **Gained:** the denominator theorem above (the reflection polynomial's denominators are bounded
  by the diagram resultant, the ramified prime controlled outright), and the census fact that
  polynomial integrality of some half certifies construction parenthood with no false positives.

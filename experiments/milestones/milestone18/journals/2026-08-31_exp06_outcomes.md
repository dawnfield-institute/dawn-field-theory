# 2026-08-31 (night): exp_06 outcomes — 2/4, and the reason for the split

**Registration**: 74bcd0df. Score: T1 PASS, T2 PASS (sealed claims), T3 FAIL, T4 FAIL
(proved impossible). Scoring notes below; nothing relaxed after the run — one in-script
criterion that was STRICTER than the seal is documented and corrected transparently.

## Results

- **T1 PASS.** σ(P) = I − P for all three foldings with exact symbolic projectors
  (Bezout construction — P as a polynomial in C via extended Euclid on q, σq; no
  eigenvalue radicals). D₆ requires resolving the degenerate λ=2 core: the identity does
  not pick a direction — it pins a **conic**, c·N(τ) + b·Tr(τ) + a = 0 (here
  3N + 2Tr + 3 = 0), every point golden. The norm is forced; the direction is gauge.
- **T2 PASS on the sealed claims.** charpoly = q·σ(q) with q the H-partner's for
  {A₄, D₆, E₈}; zero golden content for {A₅, D₄, D₅, E₆, E₇}. Scoring note: the first
  run scored an additional criterion ("A-family golden only at n=4") that was not in the
  seal and is wrong — golden content occurs iff 5 | h = n+1, so the sweep set is {4, 9},
  as measured. The script records this history in-line.
- **T3 FAIL.** R = P − σP is not Aut-equivariant on D₆. Anatomy: Aut commutes with the
  off-core sector automatically (it is a polynomial in C); on the gauge core it acts by
  τ ↦ 1/τ; no conic point is invariant.
- **T4 FAIL, proved for every gauge.** An Aut-compatible core line requires τ = ±1;
  neither lies on the conic. The σ-ledger and the orbit quotient are incompatible at
  D₆'s core — not unaligned, irreconcilable. (Registered as either-outcome-informative;
  this is the informative outcome: the two "complements" — deletion and orthogonal —
  are distinct structures where they meet.)
- **Honesty note on A₄'s T3/T4 passes:** any automorphism commutes with any polynomial
  in C, so where P is functorial the passes are automatic. A₄'s cells carry no
  alignment content.

## Why the split (diagnosis)

Everything that passed is **functorial** — a statement about polynomials in the operator,
frame-free. Everything that failed requires **gauge** — data beyond the operator, entering
exactly at the degenerate core, which sits at the duality's fixed point λ = 2 (the
relation-free level of the Mirror journal).

The location of the core follows a **purity law**. A₄ (h=5) and E₈ (h=30) have exponent
sets equal to the full totative sets of h (rank = φ(h)); their spectra avoid λ=2 and the
ledger is fully functorial. D₆ (h=10, rank 6 ≠ φ(10)) is impure: exponent 5 ∣ h, hit
twice. Reducing exponents mod 5:

    m ≡ ±1 (mod 5): the H-copy        (E₈: {1,11,19,29} = the H₄ exponents)
    m ≡ ±2 (mod 5): the σ-copy        (E₈: {7,13,17,23})
    m ≡  0 (mod 5): the gauge core    (D₆: the doubled 5s)

This is the **splitting law of ℚ(√5)** (split at ±1, inert at ±2, ramified at 5) applied
to exponents. The failed tests failed at the ramification locus; the pair products
computed earlier already show it (N = 1, a unit; N = 5, the ramified prime). The same
trichotomy appears for ℚ(ω) in Farmer's triangular document ("Eisenstein Primes and the
Mod-3 Trichotomy") — an independent arrival at the sister field's version.

## Open

Whether the mod-5 trichotomy decides every σ-ledger property from exponents alone
(pure ⇒ functorial; ramified ⇒ gauge core with provable Aut-incompatibility) is the
candidate organizing law. The non-cyclotomic pairing found in the census preliminaries
(spectral radius > 2; Salem-type territory, cf. Smith's theorem and McKee–Smyth) is
flagged, not pursued.

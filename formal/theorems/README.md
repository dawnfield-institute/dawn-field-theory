# Theorems

Results that are **proven**, as distinct from measured, conjectured, or observed to hold.

This is an index, not a restatement. Each entry points to the journal where the derivation
lives and the experiment that verified it. A second copy of an argument is a second thing
that can drift, and this repository has already paid for that — two `meta.yaml` specs, two
journal specs, and two mutually contradictory legends for one filename scheme.

## Grades

Not everything below is the same kind of true, and the distinction matters more than the
count:

| Grade | Meaning |
|---|---|
| **derivation** | Proven algebraically or in closed form. Holds independent of any run. |
| **structural** | Exact within the framework's own construction — true by what PAC/SEC *are*, verified numerically to machine precision. |
| **negative** | A proven impossibility. Closes a door, which is worth as much as opening one. |

M11's own honest note applies across this page: roughly 60% of its tests are structural,
and a 100% score reflects internal consistency rather than empirical validation. These are
theorems *of the framework*. Whether the framework describes nature is what
[`ROADMAP.md`](../../ROADMAP.md) and the observational programs are for.

---

## The holonomy closed form

**θ(m) = m · θ_T(m)** — reproduces all ten round-1 measured angles exactly.

**C₆ = −I is a theorem**: θ(6) = π exactly. The ℤ₂ frame-inversion twist is structural,
not numerical.

**cos θ(C₄) = −7/9**, derived from first principles, matching the measured exact rational.

**lim_{m→∞} θ(m) = 8/3**, exactly — which killed the `e` candidate that a five-point
extrapolation had suggested. Derivation-first policy earning its keep.

> *grade:* derivation · *derived:* [`milestone15/journals/2026-06-12_holonomy_closed_form.md`](../../experiments/milestones/milestone15/journals/2026-06-12_holonomy_closed_form.md) · *verified:* M15 exp_04 (GATE PASS)

On 8/3 honestly: the derivation gives 8/3 = 2 · (1 + 1/3), the factor 2 from the two-mode
frame. That 8/3 = F₆/F₄ is a Fibonacci ratio is recorded as an observation, **not** claimed
— the derivation is odd-harmonic, not Fibonacci.

## The connection generator is the box momentum operator

**G[j′,j] = 4jj′/(j′² − j²)** for j′ − j odd — a parity selection rule. L_k is half the
nuclear norm of G_k.

Established across three evidential grades: closed form at k=2; disclosed postdictive
match at k=3,4; and **pre-registered prediction confirmed** — L₅ and L₆ registered before
the run, measured at 6.7×10⁻⁷ and 1.1×10⁻⁶ relative error.

Transporting the complement frame along the cycle shifts the deleted vertex — the box
translates — so **curvature of the M15 connection is momentum**.

> *grade:* derivation · *derived:* [`milestone15/journals/2026-07-17_general_k_momentum_generator.md`](../../experiments/milestones/milestone15/journals/2026-07-17_general_k_momentum_generator.md) · *verified:* M15 exp_06, pre-registration `85658499`

Both naive candidates — odd-harmonic and Fibonacci — died first, by registered
measurement. L_k is algebraic and rational only at k=2, which is why every "nice formula"
hunt failed.

## The origin of Xi

**Xi = gamma + ln(phi) is fully determined. Zero free parameters.**  (Ξ = γ + ln φ)

Two independent halves:

- **φ is uniquely selected** by gravity-time duality. Requiring g_out = g_in² gives
  b² − b − 1 = 0, whose unique positive root is φ. A scan of b from 1.01 to 5.0 found one
  solution.
- **γ is uniquely determined** by harmonic counting. A cascade where level k costs 1/k
  nats totals H_n; the excess H_n − ln(n) → γ, converging to 0.02% at n = 5000.

> *grade:* derivation (Part A algebraic, Part B a known number-theoretic limit) · *derived:* [`milestone11/README.md`](../../experiments/milestones/milestone11/README.md) Round 4 · *verified:* M11 exp_09 T1

The genuine content is the **mechanism**: DFT claims γ comes from harmonic counting and φ
from duality, and the test confirms both independently.

## PAC is spectral confinement

For any symmetric W = V D Vᵀ, the operation D → f(D) preserves the eigenvectors V
**exactly**. Drift measured at **2.4×10⁻¹⁵** — machine epsilon.

The system can change *how much* of each mode; never *which* modes.

> *grade:* structural · *derived:* [`milestone10/SYNTHESIS.md`](../../experiments/milestones/milestone10/SYNTHESIS.md) · *verified:* M10 exp_14 (4/4)

This is what makes PAC, SEC and MED non-independent: PAC is the confinement, MED the
viability threshold, SEC the condensation dynamics — all three from one operation.

## The orbit Hilbert space is positive definite

The orbit-quotient Gram matrix is the **identity for every ADE type**. Same-orbit vertices
collapse to a single basis vector.

> *grade:* structural · *derived:* [`milestone14/README.md`](../../experiments/milestones/milestone14/README.md) · *verified:* M14 exp_01 (4/4)

This resolves M13's PSD problem — and is not a repair. Same-orbit vertices are
gauge-equivalent, so collapsing them is the correct physical reading.

## D_4 is the only quantum ADE type

Among ADE types, **D_4 alone** has a non-abelian automorphism group (S₃, order 6), and it
is therefore the only type with non-commuting observables and a nontrivial Robertson
bound. Every other type is abelian (ℤ₂) or trivial, hence classical.

> *grade:* structural · *derived:* [`milestone14/README.md`](../../experiments/milestones/milestone14/README.md) · *verified:* M14 exp_07, exp_08 (4/4 each)

## PSD degeneracy is fundamental — a proven impossibility

**No isomorphism-invariant metric on ADE vertex sets can be positive definite.**

Same-orbit vertices have identical complement spectra, so any invariant construction
assigns them zero distance. Tested against complement spectra, heat kernels,
characteristic polynomials, spectral zeta and combinations: **0 of 6 diagrams positive
definite, in every case**.

> *grade:* negative · *derived:* [`milestone13/README.md`](../../experiments/milestones/milestone13/README.md) M13.5 · *verified:* M13 exp_14, exp_16 (0/4 — the failure *is* the result)

This is the load-bearing negative in the corpus. It is not a defect to fix: it is the
lemma M15's DFT-Hodge conjecture is built on. The framework's metric layer is *canonically*
a metric on the orbit quotient — class-level, cohomological — and demanding a
representative without declaring a frame is ill-posed.

---

## The σ-ledger: conjugation is complementation

For a rational symmetric operator whose spectrum splits into Galois-conjugate halves over ℚ(√5),
the spectral projector P onto one half satisfies **σ(P) = I − P, P·σ(P) = 0, (P − σP)² = I** —
Vieta's relations at operator level; the Fibonacci Q-matrix is the minimal instance. The
**ledger theorem**: charpoly(C) = q·σ(q) with q the H-partner's polynomial, on exactly
{A₄, D₆, E₈} within ADE and on every one-5 tree fold beyond it.

> *grade:* derivation · *derived:* [`milestone18/journals/2026-08-31_night_exploration_sigma_ledger.md`](../../experiments/milestones/milestone18/journals/2026-08-31_night_exploration_sigma_ledger.md) · *verified:* M18 exp_06, exp_07 (4/4, registered), exp_12 T2 (68/68)

## Odd-k parents carry a rational core; strict pairing forces even n

Bipartite self-duality of a tree parent forces λ = 2 as a rational root of the diagram polynomial
when the diagram has an odd number of nodes, so every odd-k fold is core-grade; and q·σ(q) has even
degree, so a strictly paired tree has even order.

> *grade:* derivation · *derived:* [`milestone18/journals/2026-09-01_panel_g_golden_trees.md`](../../experiments/milestones/milestone18/journals/2026-09-01_panel_g_golden_trees.md) · *verified:* exp_11 T3/T4 (n = 14: 50 core, 0 strict), exp_15 (n = 20)

## Copy and conjugate signatures; the det-sign rule

With P the projector for the diagram's own polynomial, **signature(copy) = signature(diagram)** and
**signature(conjugate) = signature(σ-diagram)** (charpoly(σM) = σ(charpoly M)); hence for a
hyperbolic diagram the conjugate is definite iff det(parent) < 0.

> *grade:* derivation · *derived:* [`milestone18/journals/2026-09-01_exp12_outcomes.md`](../../experiments/milestones/milestone18/journals/2026-09-01_exp12_outcomes.md) (T3) · *verified:* exp_12 (37/37), exp_13

## The construction theorem: parents are branched double covers

For a one-5 diagram (D, e*), **parent(D, e\*)** — two sheets, trivial over ordinary edges,
cross-wired with one direct (defect) edge over the 5-bond — satisfies
**charpoly(parent) = q·σ(q)**: the sheet-mixing subspaces span{(v,0) + γ(v,1)} are invariant iff
**γ² + γ − 1 = 0**, the two roots give bond weights φ and −1/φ, the sectors are orthogonal, and
C_parent ≅ Gram(D) ⊕ σ(Gram(D)). **Corollary: every one-5 tree diagram has a tree parent** — the
existence half of the one-5 conjecture, at every k.

> *grade:* derivation · *derived:* [`milestone18/journals/2026-09-02_r17_construction_theorem.md`](../../experiments/milestones/milestone18/journals/2026-09-02_r17_construction_theorem.md) · *verified:* 117/117 placements at k ≤ 7; zero orphans at k ≤ 10 (exp_11, exp_13, exp_15)

## The matching form, and every strict-fold law, on construction parents

On a construction parent the Bezout projector is the golden-sector projector, block-diagonal over
sheet pairs, so **√5·R = S + 2Π** (S = ±1 by sheet, Π the deck transformation) with SΠ = −ΠS. From
the form and [R, C] = 0 alone: Π is a non-adjacent perfect matching; there is a single copy-internal
defect edge over the unique multiplicity-3 quotient edge (the realized 5-bond); cut = 2; the copy
side is a spanning tree and the conjugate side has two components of the diagram's halves; and
**|R_vv| = 1/√5, tr(RD) = 2/√5, ‖(I−P)BP‖² = (1/10)Σ_v(d_v − d_{Π(v)})² = 2/5**.

> *grade:* derivation · *derived:* [`milestone18/journals/2026-09-02_r17_construction_theorem.md`](../../experiments/milestones/milestone18/journals/2026-09-02_r17_construction_theorem.md) (Thm 2) and [`milestone18/journals/2026-09-01_r15_matching_structure.md`](../../experiments/milestones/milestone18/journals/2026-09-01_r15_matching_structure.md) (Addenda 2, 4, 5) · *verified:* exp_12 T4 (7/7), exp_13 T3 (13/13), exp_15 T3–T6 (47/47)

## The off-ledger identity, and B-invariant cores

For a core-grade fold with rational-core projector Qc, **P_off + σ(P_off) = I − Qc** (Qc is a
rational polynomial in C, hence σ-fixed; CRT on the Bezout identity) at every n. If B·Qc = Qc·B
(B = 2I − D), the two mixed leakage blocks vanish identically.

> *grade:* derivation · *derived:* [`milestone18/journals/2026-09-01_provenance_and_proof_notes.md`](../../experiments/milestones/milestone18/journals/2026-09-01_provenance_and_proof_notes.md) (Lemmas 1–2) · *verified:* r13 (61/61), exp_14 (80/80)

## The cut lifts of the bond are dynamically identical

Under the heat kernel exp(−βC) the two cut lifts (A,0)–(B,1) and (A,1)–(B,0) of the 5-bond have
identical entries: both equal Σ_γ γ/(1+γ²)·K^γ_{AB} in the sector basis. The sheet asymmetry
vanishes on the bond itself and is carried by its first-neighbour pairs.

> *grade:* derivation · *derived:* [`milestone18/journals/2026-09-02_exp18_outcomes.md`](../../experiments/milestones/milestone18/journals/2026-09-02_exp18_outcomes.md) (r20 addendum) · *verified:* 47/47 at 40 digits (n = 20)

## The denominator bound on the reflection polynomial

**Statement.** Let p = q·σ(q) be a strict Galois fold with q = q₀ − φ·q₁, q₀, q₁ ∈ ℤ[t] (on a
fold half, the expansion along the realized 5-bond), and let b ∈ ℚ[t] be the reflection polynomial
R = √5·b(C) from the minimal-degree Bézout identity. Then den(5·b) divides 2^{deg q₁}·Res(q₀, q₁).
In particular the ramified prime never enters unless 5 | Res(q₀, q₁), and 5·b ∈ ℤ[t] whenever
Res(q₀, q₁) = ±1.

**Mechanism.** Writing v = a + √5·c, the identity σ(v)q + vσ(q) = 1 has no √5-part and reduces to
a·(2q₀ − q₁) + 5c·q₁ = 1 over ℚ[t], with 5·b = (5c)·(2q₀ − q₁) + 5a·q₁; uniqueness of the
minimal pair plus Sylvester's identity for the integer polynomials 2q₀ − q₁ and q₁ gives the bound.

**What it does not say.** The bound is not an equality: polynomial integrality (5·b ∈ ℤ[t]
unconditionally) is *false* — three construction parents at n = 20 have Res(q₀, q₁) = 9 and
den(5·b) = 3 while 5·b(C) = S + 2Π is an integer matrix. The exact denominator is an open row.

> *grade:* derivation · *derived:* [`milestone18/journals/2026-09-02_r21_polynomial_integrality_refuted.md`](../../experiments/milestones/milestone18/journals/2026-09-02_r21_polynomial_integrality_refuted.md) · *verified:* explore_r21b (208/208 halves: rational and ℚ(√5) Bézout agree, bound holds, 5 | den ⇔ 5 | Res)

---

## What is not here

Results that are **measured** rather than proven live with their experiments — α_EM at
5.7 ppm, the Higgs at 83 ppm, sin²θ_W = 3/13, S8 at 0.07σ. See
[`experiments/EXPERIMENTS.md`](../../experiments/EXPERIMENTS.md) and
[`THEORY_MAP.md`](../../THEORY_MAP.md), where each claim resolves across all four layers.

Attempted and unproven work is in [`../conjectures/`](../conjectures/).

## Forest resultants are perfect squares

For forests F and G, |Res(charpoly F, charpoly G)| is a **perfect square** whenever it is nonzero.
A forest is bipartite, so its adjacency spectrum is symmetric about 0; the index pairs with
λ, μ ≠ 0 partition into quadruples {±λ} × {±μ}, each contributing (λ−μ)(λ+μ)(−λ−μ)(−λ+μ) = (λ²−μ²)².
A one-sided zero eigenvalue of multiplicity m contributes (±det)^m with |det| already a square;
a two-sided zero makes the resultant vanish. C = 2I − A changes the resultant by a sign only.

**Corollaries.** Res(q₀, q₁) is a perfect square on every fold half, since q₁ = charpoly(D−A−B) and
q₀ ≡ charpoly(D−e) (mod q₁) are forest polynomials. And two forests that both carry a zero
eigenvalue have resultant 0 — exactly the degenerate half the census declares and never scores.

> *grade:* derivation · *derived:* [`milestone18/journals/2026-09-04_after_phase8_two_theorems.md`](../../experiments/milestones/milestone18/journals/2026-09-04_after_phase8_two_theorems.md) §1 · *verified:* `explore_s1` 112/112 adjacency, 120/120 Cartan, control 35/150 on non-bipartite; 236/236 fold halves in the census (exp_19)

## The golden coupling is forced by simple-lacedness

In the construction parent the defect edge's weight w is the only free parameter. The sheet-mixing
subspaces W_γ = span{(v,0) + γ(v,1)} are invariant iff **γ² + w·γ − 1 = 0**, of discriminant
**w² + 4**, so the fold field is ℚ(√(w²+4)). It is ℚ(√5) — with γ = 1/φ and bond weights φ and
−1/φ — **exactly at w = 1**, the simply-laced choice in which the defect edge carries the same weight
as every other edge of the cover. The family γ² + wγ − 1 = 0 is the metallic means; golden is its
w = 1 member, and w = 0 is degenerate (rational, no fold). Generalizes the construction theorem,
whose statement is the w = 1 case.

> *grade:* derivation · *derived:* [`milestone18/journals/2026-09-04_after_phase8_two_theorems.md`](../../experiments/milestones/milestone18/journals/2026-09-04_after_phase8_two_theorems.md) §2 · *verified:* `explore_s2` symbolic; at w = 1 the construction returns the A₄ path with charpoly(parent) = q·σ(q) exactly

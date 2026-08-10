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

## What is not here

Results that are **measured** rather than proven live with their experiments — α_EM at
5.7 ppm, the Higgs at 83 ppm, sin²θ_W = 3/13, S8 at 0.07σ. See
[`experiments/EXPERIMENTS.md`](../../experiments/EXPERIMENTS.md) and
[`THEORY_MAP.md`](../../THEORY_MAP.md), where each claim resolves across all four layers.

Attempted and unproven work is in [`../conjectures/`](../conjectures/).

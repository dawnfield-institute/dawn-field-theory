# Milestone 16: Relational Locality — How Neighbours Come to Cohere

**Status**: active (founding)
**Founded**: 2026-08-13
**Origin**: An empirical dead end in `reality-engine`, reached from the other side.

---

## The empirical fact that opened this

The reality engine evolves three fields (E, I, M) on a Möbius manifold under sixteen
operators. Its spatial autocorrelation length is **1.00 cell** — the floor, meaning
neighbouring cells are statistically independent — at t = 500, 1666 and 5000, under both
the current balance operator and an ADE-derived replacement.

It stayed at the floor under every intervention tried: gravity removed entirely, Poisson
source exponent 0.5 → 1.0, viscosity across 40×, PAC enforcement on/off, thermal noise,
the π-harmonic even-depth null, and a tiling filter whose Ξ grew 2.2× over the run.
Rendering the mass field confirms it directly: pixel-level speckle, no connected structure.

**No amount of tuning produced spatial coherence, because nothing in the model makes one
cell's state depend on another's identity.**

## The claim

> **Locality is not given by the lattice. Coherence between neighbouring regions is a
> consequence of relational identity, and a system whose elements carry intrinsic state
> cannot produce it at any coupling strength.**

The framework already says identity is relational. Three independent statements of it:

| source | statement |
|---|---|
| **M13** exp_01 (4/4) | **Identity IS complement** — a vertex's identity is the structure of the rest of the graph without it; complement spectra distinguish all orbits across A₅, D₄, E₆ |
| **`confluent-identity`** | "The identity of any PAC node is the weighted confluence of its children… what is conserved is not merely a scalar but **identity coherence** itself" |
| **M14** | States live on the **orbit quotient** — grouped by automorphism, not by position |

The engine implements none of it. Each cell holds intrinsic (E, I, M) and is evolved by
rules that are local *to that cell*. Coupling enters only through RBF's laplacian and
gravity's Poisson solve — both of which move *values* between cells without any cell's
identity being constituted by its neighbours.

So the gap is not graph-physics versus field-physics. It is that **a field of independent
points is not a field**, and relational identity is the missing ingredient that would make
it one.

## What this milestone must establish

**Q1 — Does relational identity produce correlation?** Construct a dynamics in which a
cell's state is defined by its complement (or its confluence with neighbours) rather than
held intrinsically, and measure whether correlation length rises above 1.0.

**Q2 — Is there a threshold?** If correlation appears, does it appear gradually with
relational weight, or at a transition? M10 established laws-as-equilibria with
characteristic response times; a transition would connect to that.

**Q3 — What sets the correlation length?** If a length scale emerges, is it arbitrary or
does it land on framework structure — φ-spacing, cascade depth, or Ξ as the reconciliation
threshold?

**Q4 — Does coherence survive the continuum?** M13.5 proved PSD degeneracy fundamental and
M15 reclassified the class/representative split. Relational identity is a class-level
notion; whether it survives being given metric representatives is the same boundary.

## What would falsify the claim

> **If a relational-identity dynamics still yields correlation length 1.0 across the
> parameter range, then relational identity is not what produces spatial coherence, and
> the engine's incoherence has another cause.**

Recorded as a result either way. A second, weaker failure mode also counts: if correlation
appears but only at parameter values that destroy conservation or diverge, the mechanism is
not viable even if the principle holds.

## Registered invariant, not coordinates

Per the corpus rule, the registered quantity is the **ratio of correlation length to the
relational coupling scale** — dimensionless, and independent of grid size and of the
absolute coupling. Not the correlation length in cells, which moves with resolution.

## Method commitments

- **Pre-registration before any run** (STANDARDS §2.7): hypothesis, thresholds, and
  falsification condition committed first, outcomes committed separately.
- **Render before trusting a statistic.** Hours of statistics were computed on the engine's
  mass field before it was looked at, and a box-counting estimator returning D = 2.000 for
  a filament survived that whole time. Every structural claim here carries a rendered
  frame.
- **Every metric ships a `selftest()`** against known inputs. A metric never checked
  against a known answer is not a measurement.
- **Correlation length is the cheap gate.** One number, at the floor for everything tried
  so far. Any proposed mechanism is checked against it before anything more elaborate.

## Relationship to prior milestones

This does not re-derive M13 or M14 — identity-as-complement and the orbit quotient are
established. It asks the **dynamical** question those results leave open: given that
identity is relational, what does that imply for how an extended system evolves, and does
it produce the spatial coherence that intrinsic-state dynamics demonstrably cannot?

M15's class/representative split is the likely boundary condition (Q4). M10's
laws-as-equilibria is the likely home for any threshold found (Q2).

## Experiments

| Exp | Name | Status |
|-----|------|--------|
| 01 | Correlation-length baseline for intrinsic-state dynamics | pre-registration pending |

## Core machinery (reuse, do not reimplement)

- `experiments/milestones/milestone13/core/identity_complement.py` — complement spectra
- `experiments/milestones/milestone14/core/quantum_complement.py` — orbit quotient
- `experiments/studies/asymmetric_conservation/core/async_pac.py` — Δ buffer, reconciliation
- `reality-engine/proof_of_concepts/v4/structure.py` — calibrated structure metrics

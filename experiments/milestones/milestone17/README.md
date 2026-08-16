# Milestone 17: Criticality — the boundary where identity changes scale

**Status**: active (founding)
**Founded**: 2026-08-16
**Origin**: Four independent routes hitting the same wall, and the recognition that the wall
has a standard name.

---

## The claim

> **The limits this corpus keeps encountering are critical points: the parameter values at
> which the correlation length diverges and the scale on which identity is defined changes.
> Ξ is the framework's critical point. It has already been established as such, twice
> independently, in systems that are not structure-forming — and no DFT structure-forming
> system has ever been measured relative to it.**

At a critical point the correlation length diverges. That is what criticality *is*. And when
ξ → L, every element's neighbourhood becomes the whole system — so under M13's
identity-IS-complement, a node's complement stops being local and becomes global. **The scale
at which identity is defined changes.** Below the transition elements have local identity; at
it identity is global; above, a different regime again.

Percolation states the same thing in cluster language: below p_c, finite clusters; at p_c, the
largest becomes system-spanning. A thing stops being *a clump* and becomes *the network*. Its
identity changes category, not degree.

This is not an analogy laid over critical phenomena. It is what the order parameter does.

## What the corpus has already established

| source | result |
|---|---|
| **`cellular_automata_pac_attractors`** | The four rules nearest Ξ are **all Wolfram Class IV**, p = 8.58×10⁻⁸, 42.7× enrichment. Rule 110 sits at 1.0579, 0.07% from Ξ. Random rules near Ξ: 0/1000. |
| **`sec_threshold_detection`** | "ξ ≈ 1.0571 **appears at phase transitions**" — cross-domain including the Lorenz attractor, combined p < 0.00001. |
| **`sec_prime_manifold`** | exp_20 *phase transition proof*, exp_29 *phase transition*. |
| **CAH** | "Ξ is **the maximum sustainable computational asymmetry** for closed recursive systems under PAC conservation" — a ceiling, which is what a critical point is. |

**Wolfram Class IV is the edge of chaos.** It is the critical class: Class II is frozen
(identity too rigid to change), Class III is chaotic (identity dissolves), Class IV is where
structures persist *and* transform. Gliders hold together while propagating and interacting.
That the top-4 nearest Ξ are all Class IV, at p < 10⁻⁷, is the corpus already having found
that **Ξ marks the boundary where identity can morph without dissolving.**

## The empirical fact that opened this

Four independent routes to one wall, all measuring the same quantity without naming it:

| route | measurement |
|---|---|
| v3 field engine, ~40 runs | percolation 0.007–0.019; ξ pinned at the white-noise floor |
| v4 particle substrate (attractive) | percolation 0.425 but the component spans 0.47 of the box — collapsed, not spanning |
| **exp_11, the corpus's own published 3D web** | **percolation 0.0068** against a noise control of 0.0025 at matched occupancy |
| exp_10's convention, swept 0.3 → 2.5, 5 seeds | percolation 0.011–0.017 at **every** setting, and it runs *opposite* to contrast |

ξ diverges at criticality. Every one of these measured ξ ≈ 0.63 cells — the white-noise floor.
That is not "no structure". **It is maximally sub-critical**, and it is the same statement four
times.

## The methodological finding underneath

**exp_10 concluded "NO discrete phase transition" from four one-point statistics** — density
CV, void fraction, filament fraction, and a sampled clustering coefficient. None of those is an
order parameter, and a phase transition is essentially invisible to all of them. It lives in the
correlation length, the cluster-size distribution and the susceptibility.

So the honest status of the framework's structure-forming systems is **not** "no criticality".
It is **criticality was never looked for**, because the corpus has no instrument that could see
it.

That is the capability hole this milestone exists to plug.

## Questions

**Q1 — Do the structure-forming systems have a critical point at all?** Finite-size scaling:
run at several system sizes L, measure ξ/L across the control parameter. At a critical point
ξ/L becomes size-independent and the curves for different L **cross**. Away from it, ξ/L
shrinks with L. Independent second signature from the same runs: the cluster-size distribution
is power-law at p_c and exponential away from it.

**Q2 — If a critical point exists, is it at Ξ?** This is the registered prediction and the
reason the milestone is worth running: Ξ marks transitions in Lorenz, in primes, and in
cellular automata at p < 10⁻⁵. If the crossing lands near Ξ, the framework has predicted the
*location* of a transition it was not fitted to.

**Q3 — Is the framework self-organized-critical, or must criticality be tuned?** SOC systems
drive themselves to the critical point and produce scale-free connected structure generically.
DFT speaks in cascades, avalanches and collapse thresholds — SOC vocabulary — and Ξ is already
described as a threshold and a maximum sustainable asymmetry. **M10's open thread 2** is the
same question from the other side: all M10 experiments are deterministic, and *"genuine
irreversibility needs stochastic self-application."*

**Q4 — Does criticality derive M10's finite-size correction?** Near a critical point,
finite-size corrections take a universal form set by the critical exponents. **M10 open thread
1**: φ^(−1/N) converges to φ but N = 8 is 3.3% off and *"the correction is underived."*
**M10 thread 6 / M9**: the 8.9% slope gap, *"finite-size noise, or sub-leading physics?"* If
these systems sit near criticality, both are derivable rather than fitted.

## What would falsify the claim

> **If ξ/L shows no crossing at any system size across the accessible parameter range, and the
> cluster-size distribution is exponential throughout, then the structure-forming systems are
> generically sub-critical and Ξ-at-transitions does not extend to structure formation.**

Recorded as a result either way. A second, weaker failure also counts: a crossing that exists
but sits nowhere near Ξ answers Q1 yes and Q2 no, which would separate "the framework has
criticality" from "Ξ locates it."

## Registered invariants

**ξ/L**, and the **critical exponents**. Both dimensionless and both the point of finite-size
scaling — exponents are *universal*, shared across every system in a universality class
regardless of microscopic detail, so they are the strongest possible registered quantity. Never
ξ in cells, never a percolation value without its occupancy.

## The capability holes to plug

Named plainly, because this is what has been missing rather than any physics:

| instrument | status |
|---|---|
| correlation length, per axis | **exists** — `reality-engine/proof_of_concepts/v4/structure.py` |
| percolation with occupancy | **exists**, N-D, calibrated |
| finite-size scaling (ξ/L crossing) | missing |
| cluster-size distribution + power-law fit | missing |
| susceptibility / order parameter | missing |
| critical exponent extraction | missing |
| edge-of-chaos / Wolfram-class classifier | partial — `cellular_automata_pac_attractors` has classification, not as a reusable instrument |

**Calibration reference: 2D site percolation.** p_c = 0.592746 with exact known exponents
(β = 5/36, γ = 43/18, ν = 4/3). Every instrument above gets validated against it before being
trusted on a DFT system. Seven instrument faults in the preceding round make this
non-negotiable — and unlike most calibration targets, this one has textbook answers.

## Relationship to prior milestones

This does not re-derive the CA result or the threshold detection. It asks the question those
leave open: **given that Ξ marks criticality in computation and in dynamical systems, where do
the framework's structure-forming systems sit relative to it, and why has nothing ever
checked?**

It also carries three existing open ends as potential closures rather than as separate work:
M10's finite-size correction (thread 1), M10's stochastic extension (thread 2), and the M9/M10
slope gap (thread 6). M16's percolation floor is the empirical entry point, and M13.5's
class/representative split is the identity-level statement of the same boundary.

## Experiments

| Exp | Name | Status |
|-----|------|--------|
| 01 | Finite-size scaling on 2D site percolation — instrument calibration | pre-registration pending |

## Core machinery (reuse, do not reimplement)

- `reality-engine/proof_of_concepts/v4/structure.py` — `correlation_length`, `percolation`,
  `web_metrics`, `coherent_fraction`, all N-D and selftested
- `reality-engine/proof_of_concepts/v4/particles.py` — N-D particle substrate, both force
  conventions
- `experiments/studies/cellular_automata_pac_attractors/core/invariant_metrics.py` —
  excess entropy, block entropy, correlation dimension
- `experiments/studies/sec_threshold_detection/scripts/exp_01_threshold_detector.py` — the
  existing threshold detector

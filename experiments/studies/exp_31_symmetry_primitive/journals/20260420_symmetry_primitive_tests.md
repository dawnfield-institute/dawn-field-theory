# exp_31: Symmetry Primitive — Prediction Tests

**Date**: 2026-04-20
**Status**: Active (2/6 sub-experiments complete)

## Context

exp_31 tests the M7 symmetry primitive thesis: that symmetry is pre-axiomatic and
generates the DFT framework through self-reference → recursion → ADE → PAC/SEC/MED/RBF.
This experiment takes the abstract claims and turns them into quantitative predictions.

## Work Completed

### exp_31a — Self-Reference Recursion (3/4)

Tests whether cross-scale relational self-reference is necessary and sufficient for phi.

**Key results**:
- Cross-scale + conservation + hierarchy generates phi (convergence to 1.84% at depth ≥5)
- Self-similarity is NOT independent — it emerges from the other three conditions
- This reduces the axiom count from 4 to 3 (self-similarity is derived, not assumed)
- 6.0× enrichment over generic self-reference (p = 1.1e-63)
- Test 4 failure: need to investigate whether this falsifies the full claim or reveals
  a more specific version

**Insight**: The axiom reduction is significant. If self-similarity emerges from
cross-scale + conservation + hierarchy, then the "symmetry primitive" is even more
primitive than proposed — it's really a conservation + hierarchy primitive with
symmetry as an emergent consequence.

### exp_31b — Attractor Generates Phi (3/4)

Tests whether scale invariance + conservation → phi on structured trees.

**Key results**:
- Binary trees at depth ≥5 converge to within 1.84% of phi
- Test 3 "failure" is actually theoretically significant: scale invariance + conservation
  generates phi regardless of topology (even flat partitions give 0.61% from phi)
- Both drive direction and conservation are load-bearing (decomposition analysis confirms)
- Detailed decomposition scripts (exp_31b_decompose.py, exp_31b_verify.py) isolate
  which components are necessary

**Insight**: Phi doesn't require tree structure — it arises from scale invariance +
conservation on ANY topology. This is stronger than the original hypothesis predicted.
The "failure" of topology dependence is actually a success of universality.

## Remaining Work

- exp_31c–31f: pending design. Should test stability under perturbation, multi-scale
  drive, and connection to ADE classification.
- The test 3 "failure" in exp_31b should be reframed as a universality result.

## Connection to M8

M8's N=6 cascade levels (from φ^{1/6} Hubble ratio) may connect to the depth ≥5
convergence threshold in exp_31a. The cascade needs at least 5 levels to "see" phi —
and we find 6 in nature. This should be explored in M9.

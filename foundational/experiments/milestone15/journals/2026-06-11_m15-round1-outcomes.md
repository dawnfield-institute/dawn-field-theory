# M15 Round 1 Outcomes: First Curvature Measured; One Re-Pose Killed

**Date:** 2026-06-11
**Pre-registration:** commit `7e5d467f` (founding + exp_01/exp_02; exp_03 deferred at smoke test)
**Results:** `exp_01_rapidity_one_form_20260611_113340.json`,
`exp_02_coherence_per_scope_20260611_113345.json`

---

## exp_01: 3/4 PARTIAL — the complement-frame connection has genuine curvature

| Test | Result |
|------|--------|
| T1 scalar sector exact | PASS (telescoping error = 0.0) |
| T2 chord–arc deficit is a class quantity | **PASS** (relabel-invariant; class-constant; chord = 0 ⟺ same orbit verified) |
| T3 affine holonomy nonzero, invariant, rank-stable | **PASS** (deficits ≈ 2.8; angles 2.76–3.14 rad; det = +1; CV = 0.0135) |
| T4 Â extremal among unicyclic controls | FAIL (max at \|V\|=7; interior at 9, 11) |

**The headline:** the holonomy of the complement-eigenframe connection around affine-A
cycles is nonzero, labeling-invariant, and rank-stable — the framework's first measured
curvature invariant. The registered uncertainty (vertex-transitive cycles might force
triviality) resolved in favor of curvature.

**Reported [D], not scored:**
- **C₆ saturates the k=2 holonomy bound exactly**: angle = π, deficit = 2√2 (the maximum
  ‖H−I‖_F for SO(2)). The hexagon is maximally curved in this connection — likely an exact
  symmetry result; derivation open.
- The angle sequence decreases monotonically from C₆ (3.1416, 3.0079, 2.9239, 2.8677,
  2.8283, 2.7994, 2.7777, 2.7610) with geometric-looking tail; crude extrapolation of the
  large-n limit lands at ≈ 2.70–2.72. e = 2.71828 is a candidate; five tail points cannot
  resolve it. Open question, registered for a future rank-extension test (Â_12..Â_30).

**T4's honest failure:** Â is not extremal among matched unicyclic controls — in part
because the deficit saturates near the 2√2 ceiling for many graphs (the measure compresses
near angle π). Maximal symmetry ≠ extremal holonomy. If retested, the comparison should
use the angle, not the deficit, and the saturation effect must be handled.

**What this does to M13 exp_08:** the rapidity-composition failure is formally
reclassified. The scalar sector is exact (T1) — composition along paths was never the
problem; the chord/arc comparison was a category error (T2 certifies the deficit as a
class quantity). The genuine geometric content the old test was groping for is the
connection holonomy (T3) — which exists, is invariant, and is measurable.

## exp_02: 0/3 KILLED — per-scope re-posing does not rescue a boundary-dominated observable

| Test | Result |
|------|--------|
| T1 parity classes converge | FAIL — A_even CV 0.138 (WORSE with more data), A_odd 0.059, D 0.034 ✓ |
| T2 class-ratio stability | FAIL — r₁ drifts 6.0%, r₂ 4.7% (bar: both < 5%) |
| T3 classes tighter than random | FAIL — A_odd ✓, D ✓, A_even ✗ |

The registered KILL fired: the A-family oscillation is **not** parity-class structure.
With ranks extended to 28, the even class diverges further rather than converging. The
diagnosis recorded for the next attempt: `max_deformation_rate` is an extreme statistic
(max over edges) dominated by path-boundary effects that decay slowly in rank — the
observable itself is coordinate-flavored. Scoping cannot fix a non-invariant observable;
the re-pose must replace the statistic (e.g., bulk/interior deformation rates, or
spectral-density functionals) before scoping can be tested again. D-family behaves
throughout (converges, tighter than random) — the per-scope picture holds where the
observable is sane.

**Conjecture scoring (per the founding document):** falsifier 3 is partially engaged —
one re-posed claim killed. Against it: exp_01's revival is exactly the class-pass /
representative-fail decomposition the conjecture predicts, and the kill's diagnosis
(bad observable, not bad scope) is itself a representative-level critique. Status:
conjecture ALIVE, wounded, with a sharpened requirement: re-poses must replace
coordinate-flavored observables, not merely re-scope them.

## Ledger

- exp_01: 3/4 PARTIAL — first curvature invariant; C₆ saturation; angle-limit open
  question (~2.71, [D]); chord–arc reclassification certified.
- exp_02: 0/3 KILLED — parity scoping rejected; observable critique recorded.
- exp_03: deferred at smoke test (M14's null is static, not dynamical — wrinkle recorded
  in the pre-registration journal; faithful harness required).
- Registration discipline held; invariant rule applied throughout (values reported [D],
  relations scored).

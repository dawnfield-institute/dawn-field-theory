# M17 exp_02: the 0.63 route is dead, and the kill sentence fires

## What happened

The 2026-08-17 retraction examined M17's percolation route in full and withdrew the founding
wall. It did **not** examine the fourth route — *"correlation length pinned at the white-noise
floor of 0.63 cells"* — which was withdrawn by association, never tested.

exp_02 tested it against 2D site percolation, where the answers are exact.

## The result

At **L = 256**, `structure.correlation_length` reads **0.6296 at the exact p_c**, against its
own documented white-noise floor of **1 − 1/e = 0.6321**. Its entire dynamic range across
p ∈ [0.40, 0.80] is **0.003 wide**, and it converges *onto* the floor as resolution improves:

| L | reading at exact p_c | gap to floor |
|---|---|---|
| 32 | 0.6111 | 0.0210 |
| 64 | 0.6199 | 0.0122 |
| 128 | 0.6257 | 0.0064 |
| **256** | **0.6296** | **0.0025** |

On the same lattices in the same run, the connectivity length goes 6.84 → 13.64 → 29.30 →
60.95, scaling as **L^1.057, R² = 1.000**. Discrimination power **D_A = 0.893** against
**D_B = 7.526**.

**A system sitting exactly on its critical point reads at the white-noise floor.** So *"ξ at the
floor ⇒ maximally sub-critical"* never carried the information read out of it. The instrument is
not broken — `selftest_correlation` recovers Gaussian σ=2 → 3.971/4.0 and cosine λ=16 →
3.039/3.041, and it documents its own floor. It was being asked a question it cannot answer:
in percolation the diverging quantity is **connectivity**, and the density field is i.i.d.
Bernoulli(p) at every p, critical or not.

This does **not** say the engine is critical. It removes a piece of evidence that said it was
not, leaving Q1 open. A bearing, not a verdict.

## Both predictive tests failed

T1 and T2 were disclosed **postdictive** in the pre-registration (scouted before registering),
so they score nothing under STANDARDS §2.7.4. The two predictive tests both failed.

- **T3 — ν = 1.3400** against exact 4/3, a **0.5%** recovery. But the registered claim was the
  *relation* that ν would run ~10% low like exp_01's γ/ν and τ, establishing one coherent
  milestone-wide bias. **Refuted.** Bias belongs to each estimator, not to the size range —
  **Block B may not apply a blanket correction.**
- **T4 — α(0.75) = +0.398** against a registered |α| < 0.25. α tracks ξ approaching the lattice
  scale (ξ 4.4 → α 0.085; ξ 1.9 → α 0.398; ξ 1.0 → α 0.565). Below **ξ ≈ 2 cells** the estimator
  reports the small-cluster tail, which grows with L by extreme-value sampling. **The same
  failure class as estimator A's floor, at the other end.**

**Kill sentence honoured** (§2.7.5): Block A does not complete, Block B is not licensed.

## Also

- **exp_01's write-up said its exponents run "~10% high". They run ~10% low** — γ/ν 9.4% below
  exact, τ 10.0% below. Corrected; the direction is load-bearing for T3.
- A **rejected control** is recorded with its reason: thresholded smoothed Gaussian noise is not
  a null, because a thresholded correlated field is itself a level-set percolation problem with
  its own transition. It would have passed T4 meaninglessly.
- **Instrument fault #12**, caught on the smoke run: the ν minimiser returned 2.60, the top of
  its own scan range. A minimiser on its boundary has not found a minimum. `nu_at_scan_boundary`
  is now recorded and T3 fails outright on a boundary hit.

## Added

`core/criticality.py` gains three N-D instruments: `connectivity_length` (second moment of
pair-connectedness over finite clusters, spanning excluded with the excluded count returned so
the exclusion can be asserted rather than trusted), `scaling_exponent`, and `collapse_residual`.

## Tooling fix caught in passing

`tools/generate_path.py` labelled `map.yaml`'s tree root with `basename(REPO_ROOT)`, which in a
**git worktree** is the worktree's directory name — this run first emitted `dft-m17-exp02/`
instead of `dawn-field-theory/`. Since the workspace convention routes *all* repo work through
dedicated worktrees, that is the normal case, and every worktree-based commit would have
silently rewritten the root label. Fixed by resolving the real repository name through the
`.git` file's `gitdir:` pointer, falling back to `basename` for a primary checkout (where
`.git` is a directory and behaviour is unchanged).

## Route forward

A **v2 pre-registration**, per the M16 precedent where prereg v2 superseded v1 rather than v1
being re-scored. It must register **ξ ≳ 2 cells as a domain of validity** with one control
inside it and one deliberately outside that is **required to fail** — a floor is only
demonstrated by showing the instrument break where the floor says it should.

M17 remains `status: archived`. Nothing here re-founds it.

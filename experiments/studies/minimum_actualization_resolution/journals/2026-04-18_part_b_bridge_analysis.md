# Journal: Part B Bridge Analysis — PAC Conservation as Gauss's Law

**Date**: 2026-04-18
**Status**: analysis (building on exp_31 Part B honest failure)
**Depends on**: exp_31 cascade gravity stress test, exp_31b symmetry primitive (v4)

---

## The Problem

Part B (2026-03-14) showed that local exponential interactions F ~ exp(-r/r_0)/r
do NOT produce 1/r^2 through superposition. Measured effective exponent: -8.5.
The exponential kills contributions beyond ~5 r_0 regardless of source count.

This left an open question: **what bridges the local exponential model
(gravity_from_maxwell_pac, exp_09) to the Gauss model (Part A) across scales?**

## New Evidence from exp_31b

The exp_31b decomposition (run 2026-04-18) and v4 results provide the key:

### What the decompose showed

| Component | R | delta_phi | phi? |
|-----------|---|-----------|------|
| Baseline (no evolution) | 1.452 | 10.24% | No |
| A: Random noise + conservation | 1.476 | 8.78% | No |
| B: Drive WITHOUT conservation | diverges | - | No |
| C: Conservation only | 1.452 | 10.24% | No |
| D: Scale-inv drive + conservation | 1.591 | 1.69% | **Yes** |
| E: REVERSE drive + conservation | diverges | - | No |

**Both the drive direction and conservation are load-bearing.**

### The flat partition surprise

The v4 flat partition control (genuinely flat random groups, no spectral hierarchy)
STILL produces phi (R=1.608, 0.61% error). This means:

**Scale invariance + conservation → phi regardless of topology.**

The tree structure is not required. The multi-level conservation + D_{n+1} → S_n
drive is sufficient. The tree is a natural instantiation, not the mechanism.

## The Bridge

The Part B failure is not a gap — it's a **category error**. We were looking for
a bridge through pair interactions (superposition of exp(-r/r_0)/r → 1/r^2).
This is like trying to derive pressure from individual molecular collisions when
the correct route is through the ideal gas law (a conservation/statistical argument).

The bridge is:

1. **PAC conservation at each level = Gauss's law for information flux**

   A PAC tree conserves total value: P = D + S at every node. In a 3D spatial
   embedding, the total "information flux" through any closed surface enclosing
   level n is the same as through any surface enclosing level n-1. This IS
   Gauss's law — the conserved quantity is cascade information, not charge.

2. **Scale invariance selects the ratio (phi) and the profile (1/r^2)**

   The exp_31b result shows: scale invariance + conservation → phi for the
   dominant/total ratio. The SAME constraint, applied to flux through shells
   in 3D, gives 1/r^2 (Part A). These are two faces of the same conservation law:
   - **Ratio face**: D_{n+1}/P_{n+1} → 1/phi (tree ratios)
   - **Flux face**: Phi(r) ~ 1/r → F ~ 1/r^2 (spatial flux)

3. **The exponential model works at its own scale for different reasons**

   gravity_from_maxwell_pac (exp_09) with cosmological r_0 gets 85% cosmic web
   match. This is an N-body mean-field result at cosmological scales — it doesn't
   need to bridge down to Planck-scale Gauss's law. It's a valid effective
   description at its own scale, like fluid dynamics vs kinetic theory.

## Why Part B "Should" Fail

The Yukawa potential (exp(-r/r_0)/r) has a massive force carrier. Massive carriers
give screened potentials that DON'T satisfy Gauss's law — the field lines "leak"
into the mass term. This is well-known physics:

- Massless carrier (photon, graviton): 1/r^2, Gauss's law holds
- Massive carrier (W/Z, pion): exp(-r/r_0)/r, Gauss's law fails at r > r_0

The Part B test was asking whether a massive-carrier-like interaction could mimic
a massless-carrier interaction. The answer is no, and it shouldn't — these are
structurally different theories. In DFT terms: the cascade information field is
**conserved** (PAC), which means its carrier is effectively massless, which means
the flux satisfies Gauss's law, which means 1/r^2.

## Implications

1. **Part B is not a gap — it's a constraint.** It tells us the fundamental
   interaction MUST conserve flux (PAC), ruling out screened potentials as
   fundamental. The exponential model is effective, not fundamental.

2. **The connection between exp_31b and gravity is tighter than expected.**
   Scale invariance + conservation → phi (ratios) AND 1/r^2 (flux). Same
   mechanism, different observables.

3. **The cascade route to Schwarzschild is now clean:**
   PAC conservation → Gauss's law → I(r) ~ 1/r → rho_c ~ r_s/r → metric
   No pair interaction assumed. No Newton assumed. Only conservation + 3D + isotropy.

## Updated Open Questions

- ~~What bridges the local exponential model to the Gauss model?~~ → Wrong question.
  They're independent effective descriptions at different scales.
- Can the "two faces" (phi for ratios, 1/r^2 for flux) be derived from a single
  variational principle? (MED candidate)
- Does the flat partition result (phi without tree) extend to continuous systems?
  If so, what is the continuum limit of scale invariance + conservation?
- The exp_31b verify found D_{n+1} ≈ S_n mismatch is 62-67% at equilibrium,
  yet R still ≈ phi. The drive approaches but doesn't reach its target. Why does
  phi emerge as an attractor along the way? Is this a finite-step artifact or a
  deeper phenomenon (phi as a saddle point)?

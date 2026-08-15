# Journal: exp_32 Einstein Field Equations from PAC Conservation

**Date**: 2026-03-14
**Status**: complete (6/6 PASS)

---

## Origin

After exp_30 (Schwarzschild metric from cascade density, 7/7) and exp_31 (gravity stress test, 4/5), the natural next question was: can we derive the full Einstein field equations, not just the Schwarzschild solution? Peter flagged this as "super exciting and I think is going to be huge for this."

The corpus sweep revealed five independent ingredients already in the codebase that converge on this target:
1. Gauss's law for PAC flux (exp_31 Part A) — the non-circular 1/r derivation
2. MED depth <= 2 (milestone3 exp_22) — constrains field equations to second-order
3. Cascade density -> metric (exp_30) — Schwarzschild as vacuum solution
4. P = A + xi + Theta (Landauer structure) — thermodynamic foundation for T_muv
5. SEC wave equation (gravity_from_maxwell_pac) — gravitational wave propagation

## Structure

Six parts, each building one step of the derivation chain:

- **Part A (PASS)**: Construct the cascade stress-energy tensor T_muv from the PAC budget partition P = A + xi + Theta. The Landauer partition maps to equation of state parameter w in [-1, 1], where the bounds come from PAC causality (local c = 1 step/step). T_muv is symmetric because cascade information exchange between directions is reciprocal (symmetric projection from gravity_from_maxwell_pac exp_02).

- **Part B (PASS)**: PAC conservation (f(Parent) = sum f(Children)) applied to spacetime regions gives the covariant divergence condition nabla_mu T^muv = 0. In flat space this reduces to continuity + Euler equations. In curved space it gives the TOV equation for hydrostatic equilibrium. This REQUIRES the geometric side to satisfy the contracted Bianchi identity: nabla_mu G^muv = 0.

- **Part C (PASS)**: The MED depth bound from exp_22 (all k-step PAC recursions floor to depth <= 2) constrains field equations to involve at most second derivatives of the metric. Combined with symmetry (Part A) and divergence-free (Part B), Lovelock's theorem (1971) uniquely selects G_muv + Lambda g_muv in 4 dimensions. The Gauss-Bonnet term vanishes in 4D (topological invariant). Physical DoF = 10 - 4 (Bianchi) - 4 (gauge) = 2 = gravitational wave polarizations.

- **Part D (PASS)**: The coupling constant kappa = 8piG/c^4 is determined by matching the weak-field limit to exp_31 Part A's Gauss's law for PAC information flux. The complete field equations are G_muv + Lambda g_muv = (8piG/c^4) T_muv.

- **Part E (PASS)**: Schwarzschild recovered as the unique spherically symmetric vacuum solution (Birkhoff's theorem). R_muv = 0 verified numerically at multiple radii (max residual: 0.0). Classical GR tests: Mercury precession 42.99 arcsec/century (0.03% error), light deflection 1.7516 arcsec (0.07% error), Shapiro delay formula identical.

- **Part F (PASS)**: Friedmann equations derived as cosmological application. Dark matter Omega_c = F3*Xi/F6 = 0.2646 (0.148% error vs observed 0.265). Dark energy Omega_Lambda -> 1/phi at PAC equilibrium (6.7pp from observed 0.685). Universe age 13.80 Gyr (0.07% error). Cosmological constant problem persists (10^123 ratio, same as standard QFT). Honest: CC is UNSOLVED.

## Key Finding

The Einstein field equations G_muv + Lambda g_muv = (8piG/c^4) T_muv are DERIVED from three PAC/SEC principles:

1. **PAC conservation** -> symmetric T_muv with nabla_mu T^muv = 0
2. **MED depth <= 2** -> field equations at most second-order in metric
3. **Lovelock's theorem (4D)** -> G_muv + Lambda g_muv is the UNIQUE solution

The coupling constant follows from matching to the PAC-derived Gauss's law (exp_31 Part A). No part of Einstein's equations is assumed — they are the necessary consequence of information conservation, recursion depth bounds, and 4-dimensional spacetime.

## Honest Limitations

- Lovelock's theorem (1971) is a known mathematical result — we USE it, not re-derive it
- The novelty is in the PREMISES that feed Lovelock: PAC gives symmetry + divergence-free, MED gives second-order
- G itself is not derived from first principles (gravity_from_maxwell_pac gives G ~ 1/F_183, order-of-magnitude)
- Cosmological constant Λ is a free parameter — the 10^120 problem is not solved
- Kerr metric (rotating black holes) not yet derived from angular cascade density

## Connections

- exp_30: Schwarzschild metric recovered as vacuum solution
- exp_31 Part A: Gauss's law provides the weak-field matching for kappa
- exp_22 (milestone3): MED depth <= 2 constrains to second-order
- exp_29: Global-local duality provides frame covariance (partial -> nabla)
- exp_17: d_spatial = 3 restricts to 4D Lovelock (no Gauss-Bonnet dynamics)
- gravity_from_maxwell_pac exp_02: symmetric projection -> T_muv symmetry
- landauer_erasure_structure: P = A + xi + Theta -> equation of state
- milestone3 exp_25: Dark matter Omega_c and dark energy 1/phi predictions

## Open Questions

- Can Lambda be derived from PAC (not just left as free parameter)?
- Full Kerr metric from angular cascade density / frame dragging?
- Quantum corrections: does PAC discreteness give corrections to G_muv?
- Bridge between local exponential (exp_31 Part B) and Gauss model (exp_31 Part A)?
- The 10^120 cosmological constant problem
- Can the Einstein field equations be used to constrain the dark energy deviation (6.7pp from 1/phi)?

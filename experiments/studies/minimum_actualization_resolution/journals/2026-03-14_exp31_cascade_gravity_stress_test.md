# Journal: exp_31 Cascade Gravity Stress Test

**Date**: 2026-03-14
**Status**: partial (4/5 tests passing, 1 honest failure)

---

## Origin

After exp_29 (global-local duality, 7/7) and exp_30 (cascade general relativity, 7/7), Peter asked to "try to break it" — find the weaknesses so we can keep pushing. The critical analysis identified 7 potential weaknesses. The most serious: exp_30 Part A assumed the Newtonian potential Phi = -GM/r to derive cascade density, which is circular.

The corpus search revealed three independent gravity routes already in the codebase: (1) infodynamic gravity spike (F = -kBT ln2 nabla I), (2) gravity_from_maxwell_pac (projection duality, local exponential N-body), (3) cascade budget (exp_29/30). These needed bridging and stress-testing.

## Structure

Five parts, each testing a specific weakness:

- **Part A (PASS)**: The 1/r profile follows from Gauss's law for information flux on a 3D Planck lattice. PAC conservation + 3D + isotropy = conservation of information flux through any closed surface. This is the same mathematical structure as Gauss's law for electrostatics, but the conserved quantity is cascade information, not electric charge. Crucially dimension-dependent: only d_spatial=3 gives 1/r (connects to exp_17).

- **Part B (FAIL)**: Local exponential interactions exp(-r/r_0)/r do NOT produce effective 1/r^2 through discrete superposition. Measured exponent: -8.5, expected -2. The exponential kills everything beyond r_0 regardless of source count. This means gravity_from_maxwell_pac's cosmic web result works because its r_0 is cosmological-scale, not Planck-scale. The two models (local exponential and Gauss) are independent pictures at different scales; their bridge is still open.

- **Part C (PASS, informational)**: The M4 exp_01 cascade throughput slope of 0.50 matches three candidates: exact 1/2 (equipartition, 0% error), xi_floor = 0.5196 (3.9% error), and kappa_R/2 = ln^2(2) (3.9% error). Cannot distinguish without higher precision. The gap xi_floor - 1/2 = 0.01955 has no clean PAC expression. Flagged as open.

- **Part D (PASS)**: Null hypothesis sweep. Only 1/r matches Mercury precession. 1/r^2 (4D spatial) kills precession entirely. exp(-r)/r produces zero effect at Mercury distance. ln(r)/r gives 3x too much precession. The cascade prediction is selected by observation among tested alternatives.

- **Part E (PASS, informational)**: Falsification boundary map. Cascade gravity and GR are indistinguishable at all currently testable scales. Distinctive predictions (Planck quantization, discrete horizon structure, GW strain quantization, GW-EM unification) are 16-36 orders below measurement capability. The cosmological constant problem persists — naive cascade estimate off by ~10^120, same as standard QFT.

## Key Findings

1. **The circularity in exp_30 is closed.** Part A derives 1/r from PAC conservation + 3D geometry, no Newton assumed. The cascade route to the Schwarzschild metric is now: PAC conservation → Gauss's law → I(r) ~ 1/r → rho_c ~ r_s/r → phase-cycling → metric.

2. **The local exponential model fails as a bridge.** This is an honest result, not a failure of the cascade picture. It means gravity_from_maxwell_pac and cascade gravity are independent descriptions at different scales, not connected through superposition. Their bridge remains open.

3. **The cascade picture is falsifiable in principle but not in practice.** All distinctive predictions are orders of magnitude below current technology. The only currently testable target (cosmological constant) is unsolved.

## Connections

- exp_30: Gap in Part A (assumed Phi = -GM/r) now closed by Part A here
- exp_29: Frame duality confirmed as framework for SR+GR unification
- exp_28: Multiplicative asymmetry drives round-trip factor
- exp_17: d=3+1 established; Part A shows 1/r requires d_spatial=3
- gravity_from_maxwell_pac: exp_09 cosmic web works at cosmological r_0, not Planck
- infodynamic_gravity spike: F = -kBT ln2 nabla I is consistent with Gauss route
- milestone4 exp_01: Throughput slope 0.50 still open (equipartition vs xi_floor)

## Open Questions

- What bridges the local exponential model (gravity_from_maxwell_pac) to the Gauss model (exp_31 Part A) across scales?
- Is the cascade throughput factor exactly 1/2 (geometric), or xi_floor (information-theoretic)?
- Can the Einstein field equations (not just Schwarzschild) be derived from cascade principles?
- Does the cascade picture offer any insight into the cosmological constant?
- Can the Kerr metric be derived from angular cascade density / frame dragging?

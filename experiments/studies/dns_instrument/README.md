# DNS Instrument: Real-Fluid Solver for the Re-Founded Turbulence Program

**Status**: active (v0 qualified 2026-07-17)
**Origin**: The 2026-07-17 turbulence-stack audit found the program has never had a
real fluid solver — every "turbulence" claim was engine-internal (see
`ade_cascade/`, corrected FDOs). This instrument exists so that turbulence claims
can be made against actual Navier-Stokes flow, with predictions registered before
measurement.

## v0 (CPU prototype, this directory)

2D incompressible NS, vorticity-streamfunction pseudo-spectral on [0,2π)²,
2/3-rule dealiasing, RK4. Backend-agnostic array code (numpy now; the 3090 port
swaps in torch, logic unchanged).

**Qualification (exp_00, all gates PASS):**
- Q1 Taylor-Green exactness (256², ν=0.01, t=1): max error **2.0×10⁻¹⁵**
- Q2 energy budget dE/dt = −2νZ closure (decaying random field): residual **4.1×10⁻⁹**
- Q3 resolution consistency 128² vs 256², common band k ≤ 40: **1.1×10⁻⁸**
  (gate-design history in the script docstring: raw-grid and under-resolved
  comparisons rejected as wrong tests; production runs must apply the same
  spectral-tail check to detect under-resolution)

**No physics claims are made at v0.** Instrument qualification only.

## Next (GPU phase, gated on 3090/CT103)

Forced 2D turbulence at 1024²–4096², structure functions, with pre-registered
targets per the midnight discipline: the 2D She-Leveque analog (k = 4 on real
flow — never tested; see corrected `she-leveque-fibonacci-turbulence` FDO) and
MAR's forward prediction k(2+1) = 5.18. 3D 256³ afterward.

## Scripts

| Script | Purpose |
|--------|---------|
| `core/solver2d.py` | Spectral2D solver: velocity/rhs/RK4 + spectral diagnostics |
| `scripts/exp_00_qualification.py` | Q1–Q3 qualification gates |

## FDO Links

- `cascade-turbulence-mode-count` (corrected 2026-07-17; names this instrument as the upgrade path)
- `she-leveque-fibonacci-turbulence` (corrected 2026-07-17)

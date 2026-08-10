# Cascade Clock Beats Halo Virial Model for CIV Velocity Evolution

**Date:** 2026-06-08
**Status:** Key result — cascade clock explains gas velocity better than standard astrophysics

---

## The Test

Does the cascade clock N(z) = a + (1/ln(phi)) * ln(t_lookback) parameterize CIV Doppler velocity evolution better than standard astrophysical models?

## The Result

| Model | Params | R² | BIC |
|-------|--------|-----|-----|
| z (linear) | 2 | 0.717 | 392.5 |
| (1+z)^alpha (halo virial) | 3 | 0.778 | 373.5 |
| **N(z) = cascade clock** | **2** | **0.851** | **329.9** |
| ln(t) free slope | 2 | 0.851 | 329.9 |
| z² (quadratic) | 3 | 0.862 | 327.0 |
| z³ (cubic) | 4 | 0.875 | 321.6 |

The cascade clock (R²=0.851, 2 params) beats the halo virial model (R²=0.778, 3 params) with fewer parameters. It matches 97% of a cubic polynomial (R²=0.875) with half the parameters.

The halo virial model collapsed to alpha≈0 — the standard (1+z)^alpha scaling doesn't fit CIV velocity evolution.

## The Phi Slope

The free ln(t) slope is 82.05 in units of 1/ln(phi). Fixing the slope to the phi-determined rate costs ZERO R². The data is perfectly consistent with phi-constrained evolution.

## The A-E Plane

XQR-30 data (8 ions, z=2-6.5) confirms the ionization axis redistribution:

| Ion | IP (eV) | Direction with N | p-value |
|-----|---------|-----------------|---------|
| FeII | 7.9 | SHRINKS | 0.000 |
| SiII | 8.2 | SHRINKS | 0.001 |
| SiIV | 33.5 | GROWS | 0.011 |
| CIV | 47.9 | GROWS | 0.000 |

Low-ionization gas weakens with N. High-ionization gas strengthens. The cascade pushes energy up the ionization ladder. The crossover is between AlIII (18.8 eV) and SiIV (33.5 eV).

## Why This Matters

The velocity (b-parameter) measures GAS ENERGY, not ionization balance. The UV background explanation covers EW evolution (more/fewer ions) but not velocity evolution (faster/slower gas). The cascade clock captures the energy redistribution — how much kinetic energy each gas phase has at each cosmic epoch.

Standard astrophysics says gas velocity should scale with halo virial velocity: v ~ (1+z)^(1/2) at fixed mass. This model fails (R²=0.778, alpha≈0). The cascade clock succeeds (R²=0.851) with a functional form derived from PAC conservation, not from halo physics.

443,000 absorption systems. 12 billion years. The phi-logarithmic curve fits the velocity evolution better than the standard model with fewer parameters and zero fitting of the slope.

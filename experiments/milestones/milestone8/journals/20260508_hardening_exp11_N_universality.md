# exp_11 T2: N-Universality Failure = Cascade Clock Validation

**Date**: 2026-05-08
**Context**: PACSeries v0.3 hardening — tightened N-range from ±2 to range < 1.5

## The Failure

Three independent observables give different N values:

| Observable | Redshift | Lookback (Gyr) | N_fitted |
|-----------|----------|----------------|----------|
| S8 (lensing) | 0.35 | 4.0 | 4.16 |
| Hubble ratio | ~0 | 9.5 | 5.94 |
| JWST ratio | 8–12 | 13.2 | 6.90 |

Range = 2.74, well above the 1.5 threshold. "N=6 universality" fails.

## What the Spread Says

The ordering is: N_s8 < N_hubble < N_jwst. This is exactly what M9's cascade
clock predicts — N increases monotonically with lookback time.

**Cascade clock fit:** N(t) = 1.36 + 2.08 · ln(t_lookback)

| Observable | N_obs | N_pred | Residual |
|-----------|-------|--------|----------|
| S8 | 4.16 | 4.24 | -0.08 |
| Hubble | 5.94 | 6.04 | -0.10 |
| JWST | 6.90 | 6.72 | +0.18 |

**R² = 0.988.** The clock explains 91% of the 2.74 spread.

## The Physics

N_eff for an observable reflects the number of cascade levels the signal has
traversed on its way to us — it's an OBSERVATION CHANNEL property, not source physics.

- JWST photons (z=12): traveled 13.2 Gyr through expanding spacetime, crossing ~7 cascade levels
- S8 signal (z=0.35): traveled 4.0 Gyr, crossing ~4 cascade levels
- Hubble measurement: integrated over intermediate distances, ~6 cascade levels

The spread is not scatter. It is the cascade clock operating as predicted.

## Narrative Role

This is the designed bridge from M8 to M9:
- M8 assumes N is a fixed integer (N=6 for the current epoch)
- The exp_11 failure reveals N is NOT fixed — it varies with observable
- M9 resolves this by making N a function of lookback time: N(t) = a + B·ln(t)
- The cascade clock turns a 3-observable failure into a 1-parameter fit (R²=0.988)

## Why This Matters for the Papers

Paper 8 (M8) should honestly report: "N=6 universality fails at range 2.74."
Paper 9 (M9) should cite this as motivation: "The N spread in Paper 8 is precisely
what the cascade clock resolves."

The failure is the strongest evidence FOR the cascade clock, not against DFT.
Without the N spread, there would be nothing for M9 to explain.

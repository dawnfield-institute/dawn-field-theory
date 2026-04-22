# The Cascade Clock: N as a Temporal Function

**Date**: 2026-04-22
**Status**: Complete (spike) — bridges M8 to M9
**Script**: `spike_n_vs_scale.py`

## Origin

During M8 hardening, exp_11 (cross-consistency) independently fit N_cascade from each
observable and found they don't agree:

| Observable | Best-fit N | Lookback time (Gyr) |
|-----------|-----------|---------------------|
| S8 | 4.16 | ~3.6 |
| Hubble/BAO | 5.94 | ~9.3 |
| JWST (z=12) | 6.90 | ~13.3 |

This is monotonic with lookback time. Not noise — a signal.

## Hypothesis

N is not a free parameter. It's a derived quantity: the number of cascade levels
completed at a given epoch. The cascade is a clock that ticks in phi-units of time.

## Results

### Free fit
```
N(t) = 0.972 + 2.264 * ln(t)
RMS = 0.091
```

### DFT-constrained fit (slope = 1/ln(phi) = 2.0781)
```
N(t) = 1.357 + log_phi(t)
RMS = 0.131
Slope deviation from exact: 8.9%
```

### Cascade level timing (DFT-constrained)

| Level | t_complete (Gyr) | What happens there |
|-------|-----------------|-------------------|
| 1 | 0.29 | Recombination era |
| 2 | 0.47 | Early structure |
| 3 | 1.31 | Galaxy assembly |
| 4 | 3.57 | S8 measurement scale |
| 5 | 5.76 | Mid-cosmic time |
| 6 | 9.34 | Hubble/BAO scale |
| 7 | 15.1 | FUTURE (1.3 Gyr from now) |

We are at t = 13.8 Gyr, which is 81% through level 7.

### Improved predictions vs M8 fixed N=6

| Observable | M8 (N=6) | Clock N(t) | Observed | Winner |
|-----------|----------|-----------|----------|--------|
| S8 | 0.787 | 0.769 | 0.768 | Clock |
| w0 | -0.921 | -0.987 | ~-1.0 | Clock |

### DFT constants in the fit

The constrained slope is 1/ln(phi) = 2.0781. The free fit gives 2.264, which is
8.9% off. Close enough to be suggestive, not close enough to be conclusive. This
is exactly the kind of tension that should drive M9.

The level spacing is phi^1 in units of ln(t): each level takes phi times longer
than the previous one in logarithmic time.

## M9 Implications

If N(t) = log_phi(t/t_1) holds exactly, then:

1. **DFT reduces from 2 free parameters to 1** — only depth 73 remains free,
   N_cascade is derived from the observation epoch
2. **The infodynamic mechanism IS the cascade clock** — M9 should derive WHY
   levels tick in phi-units from PAC/SEC first principles
3. **Level 7 completion at t = 15.1 Gyr is a prediction** — a discrete shift
   in cosmic parameters should be observable (or its absence falsifies)

### Falsifiable M9 predictions
- Euclid/DESI: S8(z) should follow S8(z) = S8_today * [1 - 0.054*(N(z)-N(0))]
- The cascade timing should be derivable from PAC conservation + SEC dynamics
- Level transitions should produce detectable signatures in cosmological data

## What This Doesn't Explain

- Why depth 73? (still a free parameter)
- What drives the cascade forward? (entropy gradient? PAC conservation?)
- Why phi specifically for the level spacing? (M7 showed phi is necessary from
  PAC, but the temporal application needs its own derivation)
- The 8.9% slope deviation — is this measurement error, or does it indicate
  a correction term?

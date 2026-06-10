# PACSeries Paper 12: Observational Contact in Absorption Spectroscopy

**Status**: Draft v0.1
**Date**: June 2026

## Summary

Tests the DFT cascade clock against 585,800 quasar absorption line systems. The cascade clock N(z) = 1.360 + (1/ln(phi)) * ln(t_lookback), calibrated on S8/H0/JWST, predicts CIV gas velocity evolution at R²=0.851 (beating halo virial at 0.778), velocity skewness transition at p=0.003, Fe/Mg enrichment at R²=0.89, and ionization redistribution across 8 ions.

## Reproduce

```bash
cd Code
python reproduce.py          # all experiments
python reproduce.py exp_12   # key result: smooth cascade
python reproduce.py exp_13   # key result: faster time
python reproduce.py exp_16   # key result: structural regularity
```

Requires: SDSS DR16 MgII/FeII catalogs, SDSS DR12 CIV catalog, XQR-30 multi-ion catalog. See `Code/reproduce.py` for download instructions.

## Structure

```
paper.md                    # Main paper
README.md                   # This file
Code/
  reproduce.py              # Reproduction entry point
  trace.yaml                # Source traceability
  experiments/              # 9 experiment scripts
Data/
  catalogs/                 # SDSS, XQR-30 data (download separately)
  results/                  # JSON output from experiments
```

## Pre-registration

Predictions registered at commit `193d1c8e` on GitHub before any observational data was examined (June 6, 2026).

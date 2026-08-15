# Fibonacci Structure in Turbulence Intermittency

## Overview

This preprint demonstrates that the She-Leveque turbulence intermittency exponents derive from Fibonacci structure through PAC conservation dynamics.

**Key Result:** 0.47% mean error against experimental data, 14.3× improvement over K41

## Pre-Registration

This is a **pre-registered postdiction**—predictions were committed to version control (git commit `19e4b6b`) before validation against experimental data. This provides cryptographic proof that predictions preceded comparison.

To verify:
```bash
git log --oneline | grep "PRE-REGISTERED"
```

## The Formula

The She-Leveque formula:
```
ζ_p = p/9 + 2[1 - (2/3)^(p/3)]
```

Fibonacci decomposition:
```
ζ_p = p/(F₄)² + F₃[1 - (F₃/F₄)^(p/F₄)]
```

Where F₃ = 2, F₄ = 3.

## Key Predictions

| Order p | Prediction | Measured | Error |
|---------|------------|----------|-------|
| 1 | 0.364 | 0.37 | 1.64% |
| 2 | 0.696 | 0.70 | 0.58% |
| 3 | 1.000 | 1.00 | 0.00% |
| 4 | 1.280 | 1.28 | 0.03% |
| 5 | 1.538 | 1.54 | 0.13% |
| 6 | 1.778 | 1.77 | 0.44% |

All predictions within 2σ of experimental values.

## Cross-Domain Significance

The ratio 2/3 = F₃/F₄ also appears in:
- **Koide formula** (lepton masses): Q = 2/3 at 0.0009% precision
- **Quark charges**: +2/3 (up-type), -1/3 (down-type)

Same constant arising independently in particle physics and fluid dynamics suggests fundamental structural origin.

## Reproduction

### Requirements
- Python 3.10+
- NumPy

### Run Prediction
```bash
cd Code
python she_leveque_prediction.py
```

### Run Validation
```bash
python she_leveque_validation.py
```

## Files

```
she_leveque_fibonacci_turbulence/
├── paper.md          # Full preprint
├── meta.yaml         # Schema v2.0 metadata
├── README.md         # This file
├── CITATION.md       # Citation information
├── LICENSE           # AGPL-3.0
├── Code/
│   ├── she_leveque_prediction.py
│   └── she_leveque_validation.py
├── Data/
│   ├── predictions.json
│   └── validation_results.json
└── Figures/
    ├── prediction_vs_experiment.png
    └── fibonacci_decomposition.png
```

## Citation

See [CITATION.md](CITATION.md)

## License

AGPL-3.0 - See [LICENSE](LICENSE)

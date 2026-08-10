# SEC Threshold Detection

This package contains the preprint and supplementary materials for "Symbolic Entropy Collapse Thresholds: Cross-Domain Detection and the Universal Balance Operator ξ."

## Key Findings

- **SEC threshold detection** algorithm identifies phase transitions from trajectories
- **ξ ≈ 1.0571** appears at critical points across all tested domains
- **A/B testing**: correct threshold 1.48× faster, wrong threshold 50.96× slower
- **Lorenz dimension**: D = 2 + (ξ-1) = 2.0571 matches observed D = 2.06 (0.14% error)
- **Cross-domain p < 0.00001** for ξ relationships

## Domains Tested

| Domain | ξ Relationship | Result |
|--------|---------------|--------|
| Navier-Stokes | Re*/1000 = ξ | ✅ |
| Lorenz | D = 2 + (ξ-1) | ✅ |
| Logistic Map | r*/3.37 = ξ | ✅ |
| Three-Body | m* = ξ-1 | ✅ |

## Contents

```
sec_threshold_detection/
├── Paper.md                 # Full paper
├── README.md                # This file
└── experiments/             # Supporting code and data
```

## Experiment References

The experimental code lives in the main experiments folder:
- `foundational/experiments/sec_threshold_detection/` - Detection and validation

## Citation

```bibtex
@article{hartwell2025secthreshold,
  title={Symbolic Entropy Collapse Thresholds: Cross-Domain Detection and the Universal Balance Operator ξ},
  author={Hartwell, P.L.},
  year={2025},
  journal={Dawn Field Theory Institute Preprint}
}
```

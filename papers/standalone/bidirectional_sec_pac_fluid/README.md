# Bidirectional SEC and PAC Fluid Dynamics

This package contains the preprint and supplementary materials for "Bidirectional Symbolic Entropy Collapse and Fluid Dynamics on PAC Hierarchies."

## Key Findings

- **Bidirectional SEC** operates on PAC trees: downward (differentiation) and upward (integration)
- **Root-as-calculus, leaves-as-geometry** validated computationally
- **PAC-DAG fluid simulation** maintains strict conservation with turbulent-like cascades
- **Power-law spectrum** slope ≈ -1.9 (steeper than Kolmogorov's -5/3)
- **Ξ ≈ 1.057** emerges from turbulent regime

## Contents

```
bidirectional_sec_pac_fluid/
├── Paper.md                 # Full paper
├── README.md                # This file
└── experiments/             # Supporting code and data
```

## Experiment References

The experimental code lives in the main experiments folder:
- `experiments/milestones/pac_dag_fluid/` - PAC-DAG fluid simulations

## Reproducibility

All experiments can be reproduced by running the scripts in the experiments folder. Each script outputs JSON results with timestamps.

## Citation

```bibtex
@article{hartwell2026bidirectional,
  title={Bidirectional Symbolic Entropy Collapse and Fluid Dynamics on PAC Hierarchies},
  author={Hartwell, P.L.},
  year={2026},
  journal={Dawn Field Theory Institute Preprint}
}
```

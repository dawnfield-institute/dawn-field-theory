# WorldSeed: Evolutionary Software Architecture Through PAC/SEC Dynamics

## Quick Start

```bash
# Install dependencies
pip install -r Code/requirements.txt

# Run basic evolution demo
python Code/experiments/exp_01_basic_evolution.py

# Run real GAIA evolution (requires GPU recommended)
python Code/experiments/exp_02_real_gaia_integration.py

# Full WikiText-2 benchmark
python Code/experiments/exp_03_wikitext2_evolution.py
```

## Overview

This preprint presents WorldSeed, a system that applies **evolutionary dynamics** from Dawn Field Theory to **software and ML architecture design**. Instead of manually designing architectures, WorldSeed evolves them using PAC/SEC principles:

- **PAC (Potential-Actualization Conservation)**: Architectural constraints conserved across generations
- **SEC (Symbolic Entropy Collapse)**: Coherence-based selection pressure
- **Fibonacci Structure**: Natural emergence of hierarchical bounds

## Key Results

| Metric | Baseline | Evolved | Improvement |
|--------|----------|---------|-------------|
| Overall Fitness | 1.445 | 1.500 | +3.8% |
| Speed (tok/s) | 335 | 776 | +131% |
| Quality Score | 0.77 | 0.98 | +27% |
| Concentration | 0.618 | 0.785 | +27% |

## Contents

```
worldseed_evolutionary_architecture/
├── paper.md          # Full paper (Markdown)
├── paper.tex         # Full paper (LaTeX)
├── README.md         # This file
├── CITATION.md       # How to cite
├── LICENSE           # AGPL-3.0
├── Code/
│   ├── trace.yaml
│   ├── requirements.txt
│   ├── reproduce.py
│   ├── core/
│   └── experiments/
├── Data/
│   └── results/
└── Figures/
```

## Reproduction

All experiments can be reproduced from this package. See `Code/reproduce.py` for the master reproduction script.

## Citation

See [CITATION.md](CITATION.md) for citation formats.

## License

- **Code**: AGPL-3.0
- **Paper**: CC-BY-4.0

## Links

- **Repository**: [Dawn Field Theory](https://github.com/dawnfieldlab/dawn-field-theory)
- **GAIA Documentation**: See related preprint `gaia_field_native_intelligence_comprehensive`

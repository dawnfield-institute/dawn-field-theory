# PAC/SEC Cosmology: JWST High-Redshift Black Hole Validation

**Category**: PAC  
**Version**: 1.0  
**Impact**: 5/5  
**Complexity**: 5/5  
**Evidence Type**: E (Empirical)

## Overview

This preprint applies the PAC (Potential-Actualization Conservation) and SEC (Symbolic Entropy Collapse) framework to the problem of anomalously massive high-redshift supermassive black holes (SMBHs) observed by JWST. Rather than fitting parameters to data, we test whether PAC's fixed constants (φ = 1.618..., Ξ = 1.0571) provide physical constraints that explain observations that challenge ΛCDM with realistic assumptions.

See `paper.md` for the full paper.

## Key Results

- **PAC with DC seeds**: 9/10 JWST objects achievable
- **ΛCDM realistic**: 0/10 objects achievable  
- **SEC enhancement**: 1.17× (duty cycle, not rate)
- **Four falsification criteria** established

## Quick Start

```bash
# Install dependencies
pip install -r Code/requirements.txt

# Run all experiments
python Code/reproduce.py

# Run specific experiment
python Code/reproduce.py 3

# List available experiments
python Code/reproduce.py --list
```

## Contents

```
.
├── paper.md          # Main paper (Markdown)
├── paper.tex         # LaTeX version (if generated)
├── paper.pdf         # PDF version (if generated)
├── meta.yaml         # Paper metadata
├── Code/
│   ├── trace.yaml    # Links to original source files
│   ├── core/         # PAC cosmology modules
│   ├── experiments/  # Numbered experiment scripts
│   └── reproduce.py  # Main entry point
├── Data/
│   └── results/      # Generated results (JSON)
└── Figures/          # Visualizations
```

## Code Traceability

See `Code/trace.yaml` for links to the original source files in the repository.

## Citation

See `CITATION.md` for how to cite this work.

## License

GNU AGPL v3.0 (code), CC-BY-4.0 (paper). See `LICENSE`.

---

*This is exploratory research applying Dawn Field Theory to observational astrophysics. Results require independent validation and peer review.*

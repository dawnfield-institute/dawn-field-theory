# Sec Med Framework Information Amplification

**Category**: PAC  
**Version**: 1.0  
**Impact**: 5/5  
**Complexity**: 2/5  
**Evidence Type**: E

## Overview

See `paper.md` for the full paper.

## Quick Start

```bash
# Install dependencies
pip install -r Code/requirements.txt

# Run all experiments
python Code/reproduce.py

# Run specific experiment
python Code/reproduce.py 7

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
│   ├── core/         # Reusable modules
│   ├── experiments/  # Numbered experiment scripts
│   └── reproduce.py  # Main entry point
└── Data/
    └── results/      # Generated results (JSON)
```

## Code Traceability

See `Code/trace.yaml` for links to the original source files in the repository.

## Citation

See `CITATION.md` for how to cite this work.

## License

MIT License (code), CC-BY-4.0 (paper). See `LICENSE`.

---

*This is exploratory research. Results require independent validation.*

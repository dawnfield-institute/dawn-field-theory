# Environment and Reproducibility

This repository targets Python 3.10-3.12 on Windows, macOS, and Linux. Experiments are pure Python with optional GPU acceleration via PyTorch.

## Quick Start (CPU-only)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install --upgrade pip
pip install -r requirements.txt
# Install PyTorch separately for your platform:
# https://pytorch.org/get-started/locally/
```

## Dependencies

Core experiments use numpy, scipy, matplotlib, and sympy. PyTorch is optional (used by reality-engine and some GAIA experiments).

PyTorch is intentionally excluded from requirements.txt to avoid platform wheel mismatches. Install it per your CUDA/CPU environment.

## Related Repos

| Repo | Purpose | Dependencies |
|------|---------|-------------|
| `fracton` | PAC math SDK (70+ modules) | numpy, scipy |
| `reality-engine` | GPU simulator | PyTorch, numpy |
| `dawn-models` | GAIA ML models | PyTorch, fracton |

## Reproducibility

- Experiments in `experiments/` are self-contained Python scripts
- Each experiment root has a `meta.yaml` and a `README.md` with results
- Prefer running experiments inside a fresh virtual env
- Pin exact versions in a lock file if you snapshot a result for a preprint

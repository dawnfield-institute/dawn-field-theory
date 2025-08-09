# Environment and Reproducibility

This repository targets Python 3.10–3.12 on Windows, macOS, and Linux. The core models (TinyCIMM, SCBF, legacy CIMM) are pure Python with optional GPU acceleration via PyTorch.

Quick start (CPU-only):
- Install Python 3.11
- Create a virtual environment
- Install the base requirements
- Install PyTorch separately (see below)

Windows PowerShell (example):

```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Install PyTorch last, using the official selector for your CUDA/CPU setup:
# https://pytorch.org/get-started/locally/
# Example (CPU): pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Notes:
- Torch is intentionally excluded from requirements.txt to avoid platform wheel mismatches. Install it per your CUDA/CPU environment.
- SCBF uses matplotlib/seaborn; TinyCIMM experiments use pandas/matplotlib; legacy CIMM pulls in scipy/sklearn utilities.

Tested baseline (Aug 2025):
- Python 3.11.9
- numpy 1.26.x, pandas 2.2.x, matplotlib 3.8.x, seaborn 0.13.x, scipy 1.11.x, scikit-learn 1.4.x, tqdm 4.66.x
- torch 2.3–2.4 (install per platform)

Per-submodule hints:
- TinyCIMM: requires torch, numpy, pandas, matplotlib
- SCBF: requires torch, numpy, scipy, matplotlib, seaborn
- CIMM (legacy): requires torch, numpy, scipy, scikit-learn, pandas, matplotlib

Reproducibility:
- See EVIDENCE_MAP.md for a claim→artifact index
- Prefer running experiments inside a fresh virtual env
- Pin exact versions in a lock file if you snapshot a result for a preprint

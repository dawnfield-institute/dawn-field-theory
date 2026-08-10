# Environment and Reproducibility

Python 3.10–3.12 on Windows, macOS or Linux. Experiments are pure Python; GPU is optional
and only used by `reality-engine` and some GAIA work.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # .venv\Scripts\Activate.ps1 on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

That covers the experiments and the repository tooling. **PyTorch is deliberately not in
`requirements.txt`** — its wheels are platform- and CUDA-specific and cannot be expressed
portably. Install it separately if you need it:
<https://pytorch.org/get-started/locally/>

## Running an experiment

Experiments are self-contained. From the repository root:

```bash
python experiments/milestones/milestone15/scripts/exp_04_holonomy_closed_form.py
```

Results are written to that experiment's `results/` as
`exp_NN_name_YYYYMMDD_HHMMSS.json`. Nothing overwrites a prior run — the timestamp makes
each result addressable.

On Windows, set `PYTHONIOENCODING=utf-8` before running. Several scripts print ✓ and Greek
letters, and the default console codepage (cp1252) raises `UnicodeEncodeError` on them —
which looks like a failed experiment but is only a failed print.

## Repository tooling

```bash
python tools/validate_experiment_structure.py   # structure + metadata against STANDARDS.md
python tools/generate_experiment_index.py       # regenerate experiments/EXPERIMENTS.md
python tools/generate_inventory.py              # regenerate INVENTORY.md
python tools/generate_path.py                   # regenerate map.yaml
python tools/update_meta_yamls.py               # refresh the generated meta.yaml zone
```

CI runs all of these and fails on a structural error or a stale generated file. Run the
validator before opening a PR.

## Related repos

| Repo | Purpose | Extra dependencies |
|---|---|---|
| `fracton` | PAC mathematics library | numpy, scipy |
| `reality-engine` | GPU simulator | PyTorch |
| `dawn-models` | GAIA and ML validation | PyTorch, fracton |

## Reproducibility

- Experiments live under `experiments/{milestones,sidecars,studies}/` and each has a
  `meta.yaml` and a `README.md` carrying its score and honest failures.
- Prefer a fresh virtual environment.
- **Pin exact versions in a lock file when you snapshot a result for a preprint.** The
  floors in `requirements.txt` are for working, not for archiving — a published package
  should record the versions it actually ran against, and
  `papers/registry/hardware_timeline.yaml` records the hardware.
- Archived experiments under `archive/` predate the current layout and may carry their own
  `requirements.txt`. They are preserved as-run and are not maintained against current
  dependencies.

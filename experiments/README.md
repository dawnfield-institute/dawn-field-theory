# Experiments

Experiments validating Dawn Field Theory (PAC / SEC / RBF / MED) across pure mathematics,
physics derivation, quantum phenomena, cosmology, and cross-domain applications.

## The index

**[EXPERIMENTS.md](EXPERIMENTS.md) is the index.** It is generated from each experiment's
`meta.yaml` and lists every experiment with its title, status, and score, grouped as
active / completed / archived.

This file used to carry a hand-maintained table as well. It drifted — it claimed 51
experiments against a real 73, and its links rotted. One generated index replaces it, so
the count cannot disagree with the corpus again. Regenerate with:

```bash
python tools/generate_experiment_index.py
```

## How the corpus is organised

| | |
|---|---|
| `milestone1` … `milestone15` | The core derivation chain. Each milestone builds on the previous. |
| `milestone-r`, `midnight` | Sidecars — real programs that don't continue the main chain. |
| `exp_30`–`exp_33`, thematic dirs | Standalone investigations. |
| `archive/era1`, `archive/era2` | 2025-era work, preserved as lineage — see [archive/README.md](../archive/README.md). |

Experiments carry an `era` (when the work matured) and a `status` (`active`, `completed`,
`archived`, `falsified`) in `meta.yaml`. The two are independent.

## Working here

Structure, naming, journal format, the scoring convention, and the pre-registration
protocol are specified in [STANDARDS.md](../STANDARDS.md). Check your work with:

```bash
python tools/validate_experiment_structure.py
```

A falsified experiment is a result, not a failure — it stays in the corpus with its
falsification documented.

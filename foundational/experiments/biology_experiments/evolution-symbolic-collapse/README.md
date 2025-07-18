# Symbolic Collapse Entropy Experiment

## Overview
This experiment simulates symbolic collapse in biological trees, comparing entropy waves between extinction-informed (empirical) and random (simulated) trees. It supports batch runs, parameter sweeps, biological realism, statistical testing, and full provenance logging for publication-quality reproducibility.

## Features
- Parameter sweep over tree depth, breadth, and extinction rate
- Biological realism: default and probabilistic extinction models
- Batch replicates for statistical robustness
- Statistical testing (t-test) for empirical vs simulated metrics
- Provenance logging (parameters, seeds, timestamps)
- Diagnostics and results saved as JSON and PNG
- CLI for flexible experiment configuration

## Usage
Run from the command line:

```
python main.py --depths 3,4,5 --breadths 3,4,5 --extinction_rates 1.0,0.7,0.4 --extinction_model probabilistic --n_replicates 10 --seed 42 --out_dir output
```

### Key Arguments
- `--depths`: Comma-separated tree depths to sweep
- `--breadths`: Comma-separated tree breadths to sweep
- `--extinction_rates`: Comma-separated extinction rates (fraction of leaves extinct)
- `--extinction_model`: `default` (from CSV) or `probabilistic` (random)
- `--n_replicates`: Number of replicates per parameter combo
- `--seed`: Random seed for reproducibility
- `--out_dir`: Output directory for results
- `--no_save_plots`: Disable PNG plot saving

## Output
- For each parameter combo: metrics.json, real_entropy_traces.json, sim_entropy_traces.json, ttest_results.json, provenance.json, entropy_waves.png
- For the sweep: sweep_provenance.json

## Experiment Design & Assumptions
- Trees are generated with specified depth and breadth
- Extinction can be annotated from real data or simulated probabilistically
- Entropy is traced at each depth; metrics (KL, JS, Wasserstein, QWCS) are computed
- Statistical tests compare empirical and simulated entropy means
- All parameters and seeds are logged for full reproducibility

## Extending
- Add new extinction models in `run_experiment`
- Add more metrics or statistical tests as needed
- Use output JSONs for further analysis or visualization

## Citation
If you use this pipeline, please cite the repository and acknowledge the DAWN Field Theory project.

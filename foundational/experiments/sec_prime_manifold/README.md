# SEC Prime Manifold Experiments

**Status**: Active Research  
**Version**: 1.0.0  
**Date**: December 9, 2025

## Overview

This package documents the discovery that **Symbolic Entropy Collapse (SEC)** applied to integer sequences produces a stress field that partitions at the **golden ratio (1/φ ≈ 0.618)** and cascades through **Fibonacci ratios** as parameters vary.

## Key Results

| Finding | Value | Precision |
|---------|-------|-----------|
| φ-threshold (size=9) | 0.6184 | 0.04% error vs 1/φ |
| Prime enrichment (top 1%) | 67.5% | 3.3x baseline |
| Factor base independence | 2.1x | detects primes outside basis |
| Fibonacci cascade | 2/3 → 1/φ → 3/5 | through F_n sizes |

## Quick Start

```bash
cd sec_prime_manifold

# Run all experiments with traces
python -m scripts.run_all_experiments

# Run individual experiment
python -m scripts.exp_01_baseline_validation

# Verify results
python -m scripts.verify_results
```

## Connection to PAC-SEC Duality

The same golden ratio structure appears in:
- **Bell correlations**: (2αβ)² = 4/5 exactly
- **Stress field partition**: frac(E>0) = 1/φ
- **Fibonacci gauge closure**: F₇ = 13 = 1+3+8+1

See `../standard_model_connection/` and `../pac_confluence_xi/` for physics connections.

## Directory Structure

```
sec_prime_manifold/
├── core/           # SEC implementation
├── scripts/        # Experiment scripts
├── results/        # Traced JSON outputs
├── figures/        # Publication figures
├── papers/         # Preprint drafts
└── journals/       # Discovery logs
```

## Related Work

This research connects to:
- **Euclidean Distance Validation** (`../../arithmetic/euclidean_distance_validation/`): E=mc² from information geometry
- **PAC Confluence** (`../pac_confluence_xi/`): PAC-SEC duality and 4/5 emergence
- **Standard Model Connection** (`../standard_model_connection/`): Physics constants from PAC

See `SYNTHESIS.md` for detailed cross-connections.

## Citation

If you use this work, please cite:
```
Dawn Field Institute. (2025). Symbolic Entropy Collapse and the 
Golden Ratio Partition in Prime Distribution. Dawn Field Theory.
```

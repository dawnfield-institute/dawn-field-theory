# Prime Harmonic Manifold Experiments

**Status**: Active Research  
**Version**: 2.0.0 (Validated)  
**Date**: December 12, 2025

## Overview

This package documents the discovery that **prime gap pairs form a Markov chain** with eigenvalue decay rate **-1/π² per log-decade**. This structure cannot be reproduced by random models — including shuffled real gaps.

**Note**: Earlier claims about φ = 1/λ₁ have been **refuted** by bootstrap validation. The 1/φ crossing is coincidental; the true constant is **1/π²**.

## Validated Results ✅

| Finding | Value | Evidence |
|---------|-------|----------|
| Decay rate | **-1/π² ≈ -0.101** | 95% CI: [-0.129, -0.081] |
| Real vs Cramér | z = 30.4 | λ₁: 0.597 vs 0.338 |
| Real vs Shuffled | z = 5.9 | Order matters |
| Optimal chord length | 2 gaps | 100% sig at n=2, 0% at n=4 |

## Refuted Claims ❌

| Original Claim | Reality |
|----------------|---------|
| λ₁ = 1/φ stable | Just a crossing point (1/φ outside 95% CI) |
| φ is fundamental | Coincidence; 1/π² is the real constant |
| Mean λ₁ = 0.618 | Artifact of scale selection |

## Decay Law

```
λ₁ ≈ 1.12 - (1/π²) × log₁₀(N)
```

| Scale | λ₁ | 95% CI |
|-------|-----|--------|
| 50k primes | 0.705 | [0.684, 0.728] |
| 200k primes | 0.631 | [0.617, 0.646] |
| 1M primes | 0.572 | [0.564, 0.581] |
| 2M primes | 0.550 | [0.542, 0.559] |

## Cramér Model Comparison

Real primes are **4.4× more constrained** in chord vocabulary:

| Metric | Real Primes | Cramér Random |
|--------|-------------|---------------|
| λ₁ | 0.597 | 0.338 ± 0.009 |
| Unique chords | 633 | 2,808 |
| Gap σ | 9.6 | 11.6 |

## Quick Start

```bash
cd prime_harmonic_manifold

# Run main analysis
python scripts/exp_01_chord_analysis.py

# Run eigenvalue scaling test
python scripts/exp_02_eigenvalue_scaling.py

# Run full validation suite
python scripts/run_all_experiments.py
```

## Connection to PAC/SEC Framework

This work bridges:
- **SEC Prime Manifold**: φ-threshold in stress field (validated)
- **PAC Confluence Xi**: Fibonacci Standard Model derivation (validated)
- **Standard Model Connection**: Physics mechanism search (ongoing)

**Key Insight**: 
- SEC finds φ as **static equilibrium threshold**
- PHM finds 1/π² as **dynamic decay rate**  
- PAC finds Fibonacci in **gauge couplings**

All three converge on the same mathematical substrate: primes encode structure that connects arithmetic to physics.

**Unified Picture**:
```
PAC (Ψ = Ψ + Ψ)  →  φ solution  →  SEC (φ threshold)
                                 →  Physics (Fibonacci gauge)
                                 →  PHM (π² decay → GUE → zeta)
```

See `SYNTHESIS.md` and `journals/2025-12-12_cross_experiment_synthesis.md` for full documentation.

## Directory Structure

```
prime_harmonic_manifold/
├── core/           # Implementation modules
├── scripts/        # Experiment scripts  
├── results/        # JSON traces
├── figures/        # Publication figures
├── papers/         # Preprint drafts
└── journals/       # Daily research logs
```

## Related Work

- **SEC Prime Manifold** (`../sec_prime_manifold/`): φ-threshold discovery
- **PAC Confluence Xi** (`../pac_confluence_xi/`): PAC-SEC duality
- **Standard Model Connection** (`../standard_model_connection/`): Physics bridge

## Citation

```
Dawn Field Institute. (2025). Prime Harmonic Manifold: Golden Ratio 
Eigenvalue Emergence in Chord Dynamics. Dawn Field Theory.
```

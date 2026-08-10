# Infodynamic Gravity Experiment

A comprehensive implementation and validation of infodynamic gravity theory with scale-dependent arithmetic.

## Overview

This experiment implements the unified scale-dependent infodynamic gravity theory that explains:
- **Galaxy-scale dynamics** (10% dark matter fraction)
- **Cosmic web formation** (60% dark matter fraction) 
- **Smooth transition** between scales via σ(L) function
- **Hierarchical information fields** (I_local + I_global)

## Key Scientific Achievement

Successfully unified galaxy and cosmic web physics using scale-dependent parameters:
- Same underlying physics equations
- Different regimes based on characteristic length scale L
- Natural emergence of dark matter effects at large scales

## Directory Structure

```
infodynamic_gravity/
├── src/                           # Core implementation
│   ├── infodynamic_gravity.py     # Main InfoGravityField class
│   ├── scale_dependent_arithmetic.py  # Scale transition mathematics  
│   ├── sec_dynamics.py            # Structured Entropy Collapse
│   └── galaxy_simulator.py        # Galaxy-scale simulation engine
├── tests/                         # Validation and testing
│   ├── validation_tests.py        # Comprehensive physics validation
│   ├── test_dark_matter.py        # Cosmic web dark matter tests
│   ├── test_hierarchical_information.py  # Cross-scale info exchange
│   └── test_scale_dependent_summary.py   # Scale transition summary
├── experiments/                   # Research experiments  
│   ├── sec_enhanced_cosmic_web.py # SEC-inspired cosmic web formation
│   └── run_experiments.py         # Experiment runner
├── results/                       # Output data and plots
├── docs/                          # Documentation and analysis
├── archive/                       # Historical development files
└── reference_material/            # Background research
```

## Core Theory

Based on the formalization with scale-dependent parameters:

### Information Distance
```
I(r) = I₀ × exp(-r/λ_c(L)) + I_quantum_floor(L)
```

### Landauer Force
```
F = -κ(L) × k_B T ln(2) × ∇I(r)
```

### Scale-Dependent Parameters
```
κ(L) = κ_galaxy × σ(L) + κ_cosmic × (1 - σ(L))
λ_c(L) = λ_galaxy × σ(L) + λ_cosmic × (1 - σ(L))
β_floor(L) = β_galaxy × σ(L) + β_cosmic × (1 - σ(L))
```

## Key Components

1. **InfoGravityField** - Core infodynamic gravity implementation
2. **SECDynamics** - Structured Entropy Collapse for structure formation  
3. **GalaxySimulator** - Dark matter galaxy simulation
4. **ValidationTests** - Extract scaling laws and compare to observations

## Fracton Integration

Leverages Fracton's architecture:
- `RecursiveEngine` for stable gravitational evolution
- `MemoryField` for information coherence matrix storage
- `EntropyDispatch` for SEC collapse dynamics
- `BifractalTrace` for pattern analysis
- GPU acceleration for large-scale simulations

## Expected Outcomes

1. **Quadratic Scaling**: N_bits ∝ g² relationship
2. **Dark Matter Curves**: Flat rotation curves from quantum coherence floor
3. **Structure Formation**: Galaxy structure through SEC dynamics
4. **Landauer Correspondence**: Energy changes match information erasure

## Installation

```bash
pip install fracton
```

## Usage

```python
from infodynamic_gravity import InfoGravityField, GalaxySimulator

# Create galaxy simulation
sim = GalaxySimulator(N_particles=10000, lambda_c=3.086e19)

# Run evolution
results = sim.run_simulation(n_steps=1000)

# Analyze rotation curve
curve = sim.analyze_rotation_curve()
```

## Validation

Run the validation suite to test against:
- Milky Way rotation curve data
- Dark matter density profiles
- Galaxy formation timescales
- Information conservation laws

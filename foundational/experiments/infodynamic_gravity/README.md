# Infodynamic Gravity Experiment

This experiment implements the formalized infodynamic gravity arithmetic using Fracton as the computational backend. The goal is to test whether information-based gravity can reproduce dark matter effects and galaxy structure formation.

## Core Theory

Based on the formalization:

### Information Distance
```
I(r) = I₀ × exp(-r/λ_c) + I_quantum_floor
```

### Landauer Force
```
F = -k_B T ln(2) × ∇I(r)
```

### Recursive Update
Using Fracton's recursive engine for stable field evolution with global information conservation.

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

# User Guide

## Symbolic Navier-Stokes Engine

This guide explains how to use the symbolic Navier-Stokes engine for fluid dynamics simulations.

## Quick Start

```python
from api.engine_interface import EngineInterface

# Initialize engine
engine = EngineInterface()

# Define boundary conditions
boundary_conditions = {
    "geometry": "pipe",
    "reynolds": 1000,
    "velocity": 1.0
}

# Run simulation
result = engine.run(boundary_conditions)

if result["status"] == "success":
    velocity_field = result["solution"]["velocity"]
    print(f"Simulation complete! Field shape: {velocity_field.shape}")
```

## Core Concepts

### 1. Pattern Trees
The engine uses fractal pattern trees to represent flow structures. Patterns are organized hierarchically by flow regime (laminar, turbulent) and scale.

### 2. Entropy Navigation
Boundary conditions are converted to entropy signatures that guide navigation through the pattern tree to find appropriate flow solutions.

### 3. Thermodynamic Validation
All operations respect Landauer bounds and energy conservation principles.

## Boundary Conditions

Supported boundary condition parameters:

- `geometry`: Flow geometry ("pipe", "channel", "cavity")
- `reynolds`: Reynolds number
- `velocity`: Characteristic velocity
- `pressure_gradient`: Pressure gradient (optional)
- `boundary_values`: Dict of boundary values

## Examples

See the `examples/` directory for complete examples:
- `pipe_flow.py`: Basic pipe flow simulation
- `cavity_flow.py`: Lid-driven cavity flow
- `cylinder_flow.py`: Flow around cylinder

## Configuration

Configuration files in `configs/`:
- `default.yaml`: Default settings
- `high_performance.yaml`: Performance-optimized
- `validation.yaml`: Validation-focused

## Validation

The framework includes comprehensive validation:
- Classical solution comparison
- CFD benchmark comparison  
- Thermodynamic compliance checking
- Performance metrics

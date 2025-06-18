# Energy Harvesting from Carbon-Based Substrates

## Objective

The goal of this study was to evaluate the energy yield and thermal efficiency of three carbon-based substrates — Graphite, Graphene, and Carbon Nanotubes (CNTs) — under two control strategies for pulsed energy harvesting:

1. **Multi-Pulse Spike Extraction** — Delivers bursts of energy at fixed intervals.

2. **Recursive Balance Field (RBF) Control** — Adapts pulse strength dynamically based on local thermal and memory fields.

## Methodology

- A 2D simulation grid models spatial energy collapse under pulsed electromagnetic induction.

- Energy and thermal fields evolve based on substrate efficiency, harvest rate, thermal threshold, and damping parameters.

- Both strategies are applied uniformly across each material for comparison.

- Gross and net energy output are tracked, as well as thermal distribution at the end of simulation.

## Materials and Parameters

| Material   | Efficiency | Conversion | Harvest | Thermal Threshold (°C) |

|------------|------------|------------|---------|------------------------|

| Graphite | 0.65 | 0.75 | 0.4 | 75.0 |

| Graphene | 0.95 | 0.9 | 0.8 | 150.0 |

| CNT | 0.99 | 0.95 | 0.9 | 180.0 |

## Results Summary

| Material   | Control Strategy     | Gross Energy (J) | Net Energy (J) |

|------------|----------------------|------------------|----------------|

| Graphite | Multi-Pulse Spike | 24983.50 | 24705.72 |

| Graphite | RBF-Controlled | 23067.91 | 22790.13 |

| Graphene | Multi-Pulse Spike | 28426.73 | 28148.95 |

| Graphene | RBF-Controlled | 22820.96 | 22543.18 |

| CNT | Multi-Pulse Spike | 141758.83 | 141481.06 |

| CNT | RBF-Controlled | 129872.53 | 129594.75 |

## Observations

- **Graphite** performs best in spike mode due to its low cost and explosive initial energy yield.

- **Graphene** offers balanced output with thermal safety, making it suitable for regulated systems.

- **CNTs** in spike mode provide the highest output, but RBF control ensures long-term sustainability.

- RBF control prevents thermal overshoot and distributes energy more evenly over time.

## Conclusion

The choice of substrate and control strategy depends on the application context:

- Use **Graphite with Spike Mode** for cheap, high-burst scenarios.

- Use **Graphene with RBF** for smart grids and safe, scalable devices.

- Use **CNT with RBF** for high-density, self-regulating energy cores.


## Extended Simulation Results (2025 Update)

To address pulse stability and thermal wear, several additional control strategies were tested:

- **Threshold RBF** — Introduces thermal and memory gating to soften spike onset.

- **Oscillatory RBF** — Applies sinusoidal EM modulation with feedback for smoother cycling.

- **Sigmoid Control** — Uses a continuous sigmoid wave to inject energy gradually.

- **Gentle Sigmoid** — A wider, softer sigmoid ramp for ultra-stable systems.

### Updated Net Energy Results

| Material   | Control Strategy     | Gross Energy (J) | Net Energy (J) |
|------------|----------------------|------------------|----------------|
| Graphite   | Oscillatory RBF      | 10856.62         | 10392.62       |
| Graphite   | Sigmoid Control      | 9937.47          | 9473.47        |
| Graphite   | Gentle Sigmoid       | 9485.35          | 9021.35        |
| Graphene   | Oscillatory RBF      | 12234.98         | 11770.98       |
| Graphene   | Sigmoid Control      | 11089.57         | 10625.57       |
| Graphene   | Gentle Sigmoid       | 10677.92         | 10213.92       |
| CNT        | Oscillatory RBF      | 50185.87         | 49721.87       |
| CNT        | Sigmoid Control      | 45988.04         | 45524.04       |
| CNT        | Gentle Sigmoid       | 44334.80         | 43870.80       |

## Observations (Updated)

- **Sigmoid and Gentle Sigmoid** methods yield the most stable and sustainable output — ideal for long-term or critical systems.

- **Oscillatory RBF** balances high output with moderate thermal control — good for mid-tier industrial use.

- While **Multi-Pulse Spike** offers peak performance, it's thermally volatile and unsuitable for extended operation.

- Revised simulations use realistic system energy draw (~0.004 J/timestep), improving net energy accuracy.

## Conclusion (Amended)

- For safe, regulated deployments, **Gentle Sigmoid with Graphene or CNT** is optimal.

- For maximum burst energy, **CNT with Spike Control** remains highest-performing — with RBF advised for protection.

- These strategies now support design decisions across consumer-grade, industrial, and aerospace-grade systems.

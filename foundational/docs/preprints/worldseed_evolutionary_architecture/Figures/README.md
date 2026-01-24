# Figures

This directory contains publication-quality figures for the paper.

## Generated Figures

Run `python Code/experiments/generate_figures.py` to generate:

1. **figure_1_evolution_trajectory.png/pdf** - Fitness evolution over generations
2. **figure_2_performance_comparison.png/pdf** - Baseline vs evolved performance
3. **figure_3_constant_evolution.png/pdf** - φ and Ξ tracking over generations
4. **figure_4_concentration_discovery.png/pdf** - Concentration threshold evolution

## Requirements

```bash
pip install matplotlib
python Code/experiments/generate_figures.py
```

## Figure Descriptions

### Figure 1: Evolution Trajectory
Shows fitness improvement over 5 generations, highlighting the breakthrough at Generation 3 when Ξ was mutated.

### Figure 2: Performance Comparison
Bar chart comparing baseline vs evolved configuration across fitness, speed, and quality metrics.

### Figure 3: Constant Evolution
Dual plot showing how φ and Ξ evolved relative to theoretical values.

### Figure 4: Concentration Discovery
Shows how evolution discovered that higher concentration thresholds (0.618 → 0.785) improve fitness.

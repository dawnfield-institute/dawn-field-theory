# Journal: exp_41 Actualization Ratio in Field Dynamics

**Date**: 2026-03-15
**Status**: complete (PARTIAL — hypothesis not confirmed)

---

## Origin

During Reality Engine v3 development, we replaced the EulerIntegrator with an ActualizationOperator that gates field changes through a MAR threshold and splits actualized potential into local (Landauer) and global (entanglement) fractions. The split ratio f determines what fraction stays local vs redistributes globally.

The hypothesis: f = ln(phi) = 0.4812 (the actualization ratio A/(A+xi) validated across 11 MAR experiments) is the optimal split for sustained field dynamics — maximizing structure formation while preventing heat death.

## Experiment Design

- **Part A**: Sweep f from 0.1 to 1.0, measure mean disequilibrium after 3000 ticks (64x64 grid)
- **Part B**: Mass concentration (Mmax, dense%) vs f
- **Part C**: Actualization rate stability over 5000 ticks
- **Part D**: Head-to-head comparison (ln(phi) vs 0.5 vs 1.0/Euler vs 0.05/all-global)
- **Part E**: Grid size independence (32x32, 64x64, 128x64)

## Key Results

### Part A: Disequilibrium is MONOTONIC in f

Disequilibrium increases monotonically with f. f=1.0 (pure local) gives highest diseq (0.4621), f=0.1 gives lowest (0.1950). ln(phi) = 0.3491, sitting mid-range. No peak or inflection at ln(phi).

### Part B: Mass concentration peaks at LOW f

f=0.1 gives highest Mmax (3.02) and dense% (3.1%). More global redistribution seeds more nucleation sites. ln(phi) shows modest mass formation (Mmax=2.14, dense=0.2%).

### Part C: Actualization stability

| f | stability | early | late |
|---|-----------|-------|------|
| 0.2 | 0.41 | 97 | 40 |
| 0.4812 | 0.66 | 71 | 47 |
| 0.5 | 0.71 | 66 | 46 |
| 0.8 | 0.75 | 68 | 51 |
| 1.0 | 0.00 | 0 | 0 |

f=1.0 has zero actualization events (everything is local, no threshold gate fires). Higher f = better stability ratio, but ln(phi) is decent at 0.66.

### Part D: 5000-tick head-to-head

| Config | diseq | Mmax | dense% | T |
|--------|-------|------|--------|---|
| ln(phi) | 0.316 | 2.90 | 12.3% | 0.336 |
| half | 0.315 | 3.03 | 12.4% | 0.335 |
| euler | 0.437 | 3.15 | 15.7% | 0.450 |
| all_global | 0.230 | 4.00 | 14.5% | 0.263 |

ln(phi) and f=0.5 are nearly identical. Euler has highest diseq but most thermal waste. All-global has highest Mmax (4.0) — best gravitational collapse.

### Part E: Grid independence

Optimal f for disequilibrium consistently ~0.6 across all grid sizes, not 0.4812.

## Interpretation

**The hypothesis is NOT confirmed at this level of simulation.**

ln(phi) does not emerge as optimal for any single field-dynamics metric. It sits at a balance point between:
- High f: more local retention, higher disequilibrium and temperature, less structure
- Low f: more global redistribution, more mass nucleation, lower disequilibrium

Possible explanations:
1. **The actualization ratio's significance is more fundamental than field dynamics** — it optimizes information preservation during quantum collapse (Landauer bounds), not macroscopic field evolution
2. **The simulation is too simple** — 3000 ticks on 64x64 with basic operators may not capture the full causal chain where ln(phi) matters
3. **The right metric isn't disequilibrium or mass** — ln(phi) may optimize a composite or information-theoretic quantity not measured here
4. **The ratio IS approximate** — the grid-independent optimum of ~0.6 vs 0.4812 suggests the functional form matters but the exact value depends on context

## Verdict

**PARTIAL**: The actualization gate mechanism works (prevents heat death, sustains dynamics), and the local/global split ratio matters, but ln(phi) is not a clear optimum in this experimental setup. The MAR-derived value may be correct for quantum-scale actualization while field-scale dynamics prefer a slightly different ratio.

This is an honest null-ish result. The framework is sound; the specific prediction needs refinement.

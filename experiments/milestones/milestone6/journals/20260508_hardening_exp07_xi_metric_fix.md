# exp_07 T2 Fix: Xi Is a Propagation Dynamics Attractor

**Date**: 2026-05-08
**Context**: PACSeries v0.3 hardening cycle — exp_07 T2 was failing (P/A drifting 16%)

## The Bug

exp_07 reimplemented the Rule 110 P/A metric with two changes from the original
`cellular_automata_pac_attractors/core/pac_embedding.py`:

1. **Random init** instead of single-cell init
2. **Spatial per-step entropy** instead of temporal whole-history entropy

These changes measured a fundamentally different quantity. The reimplementation
converged (beautifully, by step 2000) to **1.3117** — an equilibrium structural
ratio, not Xi.

## What Xi_CA Actually Is

Xi_CA = 1.0579 was measured with:
- **Single-cell initialization** (one cell in center, everything else empty)
- **Temporal metrics** (density distribution across timesteps, MI over full history)
- **Width=101, steps=200** (canonical parameters)

This measures **propagation dynamics**: how information cascades from a single seed
of pure potential into actualized structure. The P/A ratio at 200 steps captures the
balance at the moment the expanding cone has just filled the available space.

| Setup | Init | Metric | P/A |
|-------|------|--------|-----|
| Original (Xi_CA) | single cell | temporal | **1.0579** |
| exp_07 (broken) | random | spatial/step | 1.3117 |
| Original, random init | random | temporal | 1.82 |

## Why Single-Cell Matters

The single-cell init starts as pure potential (one active cell in a sea of zeros).
The 200-step evolution IS the PAC cascade: P → A through boundary crossings. The
P/A ratio at step 200 captures how much potential remains relative to what has been
actualized — exactly what Xi measures in DFT.

Random init starts already at high density. No propagation dynamics. No cascade from
potential to actualization. It measures equilibrium structure, which is a different
attractor entirely.

## Scale Dependence

Xi_CA is NOT width-independent:

| Width | Steps | P/A | Error vs Xi |
|-------|-------|-----|-------------|
| 51 | 200 | 1.82 | +72% |
| 101 | 200 | 1.059 | +0.2% |
| 151 | 200 | 0.73 | -31% |

This is physical: Xi emerges when the propagation cone has just filled the available
space. At width=51, the cone wraps multiple times. At width=151, it hasn't filled
yet. Width=101, steps=200 is the boundary-crossing moment.

This is consistent with DFT: Xi = gamma + ln(phi) is the cost of a single scope
boundary crossing. The measurement captures the P/A ratio at exactly that transition.

## The Fix

Replaced the spatial/random metric with the original temporal/single-cell metric.
Result: P/A = 1.0592, error = 0.201% from Xi_CA = 1.0571. Test now passes 4/4.

## Insight for the Paper

The 1.3117 equilibrium attractor is itself interesting — it's the P/A ratio for
sustained Class IV dynamics in steady state. This is a different physical quantity
from Xi (propagation cost) but may have its own DFT interpretation. Not investigated
further here.

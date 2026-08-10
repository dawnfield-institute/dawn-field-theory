# ADE Cascade: Re-founding Turbulence on Milestone Machinery

**Status**: active
**Founded**: 2026-07-17
**Origin**: The 2026-07-17 audit of the turbulence stack (see corrected lore FDOs
`cascade-turbulence-mode-count`, `she-leveque-fibonacci-turbulence`, `med-navier-stokes`)
found the strongest structural signal in the corpus — exp_15's structured-vs-random
coupling advantage (z = 52.7) and the k−1 offset appearing in all three regime tests —
posed in pre-milestone language, with the coupling kernel `exp(−|i−j|·cd)` chosen ad hoc.
This experiment re-poses the cascade on ADE Dynkin diagrams (M12 connection-as-primitive,
M15 affine-A machinery) under the midnight invariant-registration discipline.

## The re-pose

The legacy kernel `exp(−|i−j|·cd)` is exactly the graph-distance kernel
`exp(−d_G(i,j)·cd)` on the **A-family (path) Dynkin diagram**. The legacy engine was
therefore always running the A-arm without knowing it. Round 1 asks:

- **R1 (diagram selectivity)**: at equal rank, do A/D/E coupling topologies produce
  distinct spectral exponents, or does only mode count matter?
- **R2 (affine vertex)**: is the k−1 offset the affine-ADE extra node? Registered as:
  extending A_r by its affine node (→ cycle, Ã_r) shifts the exponent by far less than
  extending it by an ordinary path node (→ A_{r+1}).
- **R3 (bridge)**: does the monotone mode-count→exponent map survive the kernel swap?

## Discipline

Pre-registration per midnight protocol (relations only, decision rules with an
inconclusive branch, committed before any ADE run):
`journals/2026-07-17_ade-cascade-round1-preregistration.md`. Outcomes journals cite the
registration commit hash. The legacy engine (`milestone4/core/utils.py::energy_cascade`)
is reused with an injected-coupling extension whose default path must reproduce the
exp_14 baseline exactly before any registered run (refactor-safety gate).

## Scripts

| Script | Purpose |
|--------|---------|
| `core/coupling.py` | Graph-distance coupling matrices from Dynkin diagrams (reuses milestone12 `DynkinDiagram`, milestone15 `build_cycle`) |
| `scripts/exp_00_baseline_gate.py` | Refactor-safety gate: injected A-kernel must be bit-identical to legacy kernel; exp_14 canonical exponents must reproduce |
| `scripts/exp_01_diagram_selectivity.py` | R1 + R3: A/D/E arms at ranks 6–8 (+A_9), 100 seeds each |
| `scripts/exp_02_affine_vertex.py` | R2: Ã_r (cycle) vs A_r vs A_{r+1} shift ratios |

## FDO Links

- `cascade-turbulence-mode-count` (corrected 2026-07-17)
- `she-leveque-fibonacci-turbulence` (corrected 2026-07-17)
- `milestone15-representative-problem` (affine-A machinery)

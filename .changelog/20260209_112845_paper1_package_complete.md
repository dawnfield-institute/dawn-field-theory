# PACSeries Paper 1 — Reproducibility Package Complete

**Date**: 2026-02-09 11:28
**Type**: engineering | documentation

## Summary

Completed the full reproducibility package for PACSeries Paper 1 ("The Structure Cost of Erasure"), including running missing experiments, generating publication figures, and cleaning up section numbering.

## Changes

### Added
- `PACSeries/structure_cost_of_erasure/Data/results/exp_15_gauge_group_hierarchy_20260209_111320.json` — gauge ξ hierarchy results (SU(3) > SU(2) > U(1), p = 1.51e-11)
- `PACSeries/structure_cost_of_erasure/Data/results/exp_16_ln_phi_derivation_20260209_111320.json` — first-principles ln(φ) derivation results
- `foundational/experiments/landauer_erasure_structure/results/exp_15_gauge_group_hierarchy_20260209_111320.json` — source copy
- `foundational/experiments/landauer_erasure_structure/results/exp_16_ln_phi_derivation_20260209_111320.json` — source copy
- `PACSeries/structure_cost_of_erasure/Code/generate_figures.py` — generates 6 publication figures
- `PACSeries/structure_cost_of_erasure/Figures/fig1_coupling_topology.png` — §4.4 coupling topology
- `PACSeries/structure_cost_of_erasure/Figures/fig2_information_budget.png` — §4.5 PAC budget
- `PACSeries/structure_cost_of_erasure/Figures/fig3_decay_ratio_sweep.png` — §6 ln(φ) convergence
- `PACSeries/structure_cost_of_erasure/Figures/fig4_cascade_amplification.png` — §10.3 cascade 53×
- `PACSeries/structure_cost_of_erasure/Figures/fig5_dense_sparse_regimes.png` — §11.2 dense/sparse 69×
- `PACSeries/structure_cost_of_erasure/Figures/fig6_pac_ratio_stability.png` — §9.2 ratio stability

### Changed
- `paper.md` — Renumbered sections: §5.5→§6, §6→§7, ... §14→§15. Flattened §8.1.1/§8.1.2 to §9.2/§9.3. All 15 sections now sequential with no gaps.
- `Code/trace.yaml` — Updated section references, added exp_15/16 data entries, added figures section
- `Code/reproduce.py` — Updated section mapping in docstring
- `Code/requirements.txt` — Added matplotlib>=3.5.0 for figure generation
- `meta.yaml` — Added exp_15/16 data files, figures, and 2 new key results
- `README.md` — Updated key results table (+2 rows), package tree (correct sections, figures), requirements

## Details

### Experiments Run
- **exp_15** (gauge group ξ hierarchy): Confirmed ξ(SU(3)) > ξ(SU(2)) > ξ(U(1)) with 100% consistency across 30 seeds. Mann-Whitney p = 1.51e-11. Required `PYTHONIOENCODING=utf-8` on Windows to handle Greek characters.
- **exp_16** (ln(φ) derivation): Derived A/(A+ξ) → ln(φ) from PAC recursion Ψ(k) = Ψ(k+1) + Ψ(k+2). Predicted ξ/A = 1.078 vs measured 1.086 (0.76% error).

### Section Renumbering Map
| Old | New | Title |
|-----|-----|-------|
| 5.5 | 6 | The ratio is not arbitrary |
| 6 | 7 | On the nature of ξ |
| 7 | 8 | Open questions and limitations |
| 8 | 9 | Connections to related work |
| 8.1.1 | 9.2 | Testing PAC conservation directly |
| 8.1.2 | 9.3 | The PAC/SEC hierarchy |
| 9 | 10 | The Thermodynamic Cascade |
| 10 | 11 | Time as computational density |
| 11 | 12 | The binding interpretation |
| 12 | 13 | Summary and outlook |
| 13 | 14 | Connections to the PACSeries |
| 14 | 15 | Open Computations |

## Related
- Previous session: `.changelog/20260207_095100_thermodynamic_cascade_integration.md`
- PACSeries README: `foundational/docs/preprints/PACSeries/README.md`

# PAC-DAG Fluid Dynamics Experiments

Experiments validating bidirectional SEC on PAC hierarchies.

## Key Results

- PAC trees exhibit **root-as-calculus, leaves-as-geometry** property
- PAC-DAG fluid simulation maintains **strict conservation** (error < 10⁻¹⁵)
- Power-law spectrum slope ≈ **-1.9** (steeper than Kolmogorov -5/3)
- **Ξ ≈ 1.057** emerges from turbulent regime

## Experiments

| Script | Description | Status |
|--------|-------------|--------|
| exp_01_pac_tree_basic.py | Basic PAC tree SEC field | ✅ |
| exp_02_pac_tree_blowup.py | Blow-up operator dynamics | ✅ |
| exp_03_pac_dag_fluid.py | Full DAG fluid simulation | ✅ |
| exp_04_spectral_analysis.py | Spectrum and Reynolds scaling | ✅ |

## Related Paper

See `foundational/docs/preprints/bidirectional_sec_pac_fluid/`

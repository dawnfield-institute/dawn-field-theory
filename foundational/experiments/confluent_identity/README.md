# Confluent Identity -- Phases 1-21: Formation through Falsification

## Hypothesis

Identity in a PAC tree is the weighted confluence pattern of its children --
formalized as the harmonic component of the weighted graph Laplacian (discrete
Hodge theory). A parent's identity cannot be reconstructed from the simple
mean or sum of its children, but *can* be reconstructed from their spectral
fingerprints weighted by coupling strength to the parent's global identity.

## Status: Active -- Phases 1-21 complete

## Key Results

| Claim | Test | Result |
|-------|------|--------|
| 1. Identity is confluence | Coupling-contribution correlation | rho=0.42 (uniform), rho=0.50 (eigenvector-weighted) |
| 3. Weight = coupling, not mass | Small-in-flow vs large-out-of-flow | 1.7x impact (9 vs 69 cells) |
| 4. Global cascade | Distant change fraction | 88-97% beyond 10 cells |
| 5. Retroactive recontextualization | Bayesian + Kalman smoother | 71.5% revision, 4/4 Kalman tests verified |
| Formalization | 7 theorems + 5 operator properties | 12/12 VERIFIED |
| Sheaf cohomology | H^0, H^1 computation | 3/3 sheaf tests verified |
| Eigenvalue analysis | K_MODES=10 validation | 93.1% state energy captured, JUSTIFIED |
| Size deconfound | Partial correlation, permutation test | partial rho=0.42 (p=4e-4), perm p=0.0007, 4/4 verified |
| Multi-level H^1 | H^1 propagation across hierarchy | H^1 decays 10.34→9.12 upward, 3/3 verified |
| Fiedler coherence | Fiedler predicts perturbation sensitivity | **FALSIFIED**: signal vanishes after size deconfound (partial rho=+0.07) |
| H^1-revision synthesis | Sheaf H^1 ↔ spectral revision | **FALSIFIED**: n=96 groups, rho=0.016 — no link |
| Structured perturbation | Node removal vs Gaussian | Sensitivity rankings independent across perturbation types (tau≈0) |
| Spectral phase transition | lambda_2 attenuation | Monotonic decay (15/15), 53% show susceptibility peak, 2/4 verified |
| Coupling ceiling | 5 weighting schemes | Gradient best (partial rho=0.41), ceiling below 0.45 |
| PAC buffer hypothesis | Size ~ 1/√N scaling | **FALSIFIED**: slopes positive (bigger = MORE sensitive), 0/4 verified |
| Hodge mode decomposition | Cosine similarity of shift vectors | **PARTIALLY FALSIFIED**: cos~0.47 (not orthogonal), but topological perturbations share structure (cos=0.49) |
| Revision driver | 8 candidate predictors | perturbation_l2 (rho=0.34) and boundary_coupling (rho=0.33) predict revision; H^1 reconfirmed dead (rho=0.016) |
| Confluence complexity | Eigenvalue distribution vs Fiedler | ALL Fiedler correlations collapse after size deconfound; mode 1 dominance low everywhere (~24%) |

## Experiments

| Script | Purpose | Phase | Status |
|--------|---------|-------|--------|
| `exp_01_lattice_fluid_baseline.py` | 128x128 PAC-conservative fluid, steady state | 1 | Complete |
| `exp_02_hierarchical_partition.py` | Watershed segmentation, 5-level hierarchy | 1 | Complete |
| `exp_03_hodge_identity.py` | Spectral identity I(R), coupling weights w(S,R) | 1 | Complete |
| `exp_04_confluence_test.py` | Test: coupling weights predict spectral contribution (v2 basis-aligned) | 1 | Complete |
| `exp_05_obstacle_insertion.py` | Test: small pebble in flow > large boulder on bank | 2 | Complete |
| `exp_06_retroactive_recontextualization.py` | Test: Bayesian smoothing revises past identity | 3 | Complete |
| `exp_07_formal_operator.py` | Formal definitions (Defs 1-6), theorem verification (7/7) | -- | Complete |
| `exp_08_gradient_coupling.py` | Gradient/eigenvector-weighted perturbation (deconfound size) | 4 | Complete |
| `exp_09_eigenvalue_decay.py` | Data-driven K_MODES validation via state energy analysis | 4 | Complete |
| `exp_10_projection_operator.py` | CI as proper Hilbert space projection (5/5 properties) | 4 | Complete |
| `exp_11_sheaf_cohomology.py` | Cellular sheaf on hierarchy, H^0/H^1, identity crisis test | 4 | Complete |
| `exp_12_kalman_smoother.py` | Kalman + RTS smoother replaces ad-hoc exponential | 4 | Complete |
| `exp_13_multiseed_ensemble.py` | Multi-seed robustness (5 seeds) | 4 | Complete |
| `exp_14_partial_correlation.py` | Size deconfound: partial Spearman, stratified, permutation | 5 | Complete (4/4) |
| `exp_15_multilevel_sheaf.py` | Multi-level H^1 propagation across hierarchy | 6 | Complete (3/3) |
| `exp_16_spectral_gap_dynamics.py` | Fiedler as identity coherence predictor | 7 | Complete (2/4) |
| `exp_17_synthesis_h1_revision.py` | Synthesis: sheaf H^1 ↔ spectral revision | 8 | Complete (2/3) |
| `exp_18_fiedler_deconfound.py` | Fiedler-size deconfound + entropy reversal | 9 | Complete (1/4) |
| `exp_19_structured_perturbation.py` | Node removal + edge rewiring vs Gaussian | 10 | Complete (1/4) |
| `exp_20_spectral_phase_transition.py` | Spectral gap phase transition (lambda_2 attenuation) | 11 | Complete (2/4) |
| `exp_21_h1_revision_powered.py` | H^1-revision with n=96 sub-partitioned groups | 12 | Complete (0/4) |
| `exp_22_coupling_ceiling.py` | 5 weighting schemes: coupling ceiling analysis | 13 | Complete (2/4) |
| `exp_23_size_as_pac_buffer.py` | Size as PAC conservation buffer: 1/√N scaling test | 14 | Complete (0/4) |
| `exp_24_perturbation_mode_decomposition.py` | Hodge decomposition of vulnerability modes | 15 | Complete (1/4) |
| `exp_25_revision_driver.py` | 8 candidate predictors of backward revision | 16 | Complete (2/4) |
| `exp_26_confluence_complexity.py` | Why does coupling create entropy? | 17 | Complete (2/4) |
| `exp_27_boundary_geometry.py` | Boundary geometry census: 6 metrics vs sensitivity | 18 | Complete (2/4) |
| `exp_28_boundary_coupling_ceiling.py` | Boundary-aware coupling schemes vs 0.41 ceiling | 19 | Complete (2/4) |
| `exp_29_sensitivity_mediation.py` | Does boundary surface mediate size→sensitivity? | 20 | Complete (1/4) |
| `exp_30_scale_validation.py` | 256x256 lattice replication | 21 | Complete (1/4) |

## Formal Operator: CI(f) = Pi_harm @ f

**Phase 4 upgrade**: CI is now a proper orthogonal projection operator Pi_harm: L^2(V_R) -> ker(L_R),
verified to satisfy idempotency, self-adjointness, orthogonality, non-trivial composition,
and correct trace. Fiedler value, spectral entropy, and coefficients are derived quantities.

### Phase 1-3 Theorems (exp_07, 7/7 verified)

1. Decomposition completeness + Parseval (error 6.25e-17)
2. Conservation under decomposition (error 1.33e-15)
3. Coupling-contribution correlation (rho=0.42, p=0.0004)
4. Non-mass dependence (14 outliers, max 21.5x size fraction)
5. Spectral gap coherence (10/10 merge events reduce Fiedler)
6. Backward reweighting non-triviality (cosine rotation 0.89 -> 0.52)
7. D=3 Fibonacci-exponential uniqueness (2^d+1 = d*F_{d+1} at d=3 only)

### Phase 4 Results

8. **Operator idempotency**: Pi^2 = Pi to 2.5e-15 (exp_10)
9. **Operator self-adjointness**: Pi = Pi^T to machine precision (exp_10)
10. **Eigenvector-weighted coupling**: rho improved 0.39 -> 0.50 (exp_08)
11. **K_MODES=10 justified**: captures 93.1% state energy, median k_state_95=10 (exp_09)
12. **Sheaf cohomology**: state IS global section, perturbation creates identity crisis (H^1 up), smoothing resolves (70% recovery) (exp_11)
13. **Kalman smoother**: 4/4 tests verified, RTS revision correlates with exponential (r=0.60) (exp_12)

### Phase 5-8 Results

14. **Size deconfound**: partial rho(coupling, natural | size) = 0.42 (p=4.1e-4), 3/3 size terciles positive, permutation p=0.0007 (exp_14)
15. **Multi-level H^1**: H^1 propagates upward through hierarchy with attenuation (10.34 → 9.12), 3/3 verified (exp_15)
16. **Fiedler coherence (raw)**: Fiedler predicts noise sensitivity (rho=-0.32, p=0.015), top-quartile 1.35x less sensitive (exp_16)
17. **H^1-revision synthesis (underpowered)**: R^2=0.11, rank overlap 5/5, but n=6 insufficient (exp_17)

### Phase 9-13 Results: Falsification and Refinement

18. **Fiedler FALSIFIED as coherence predictor**: partial rho(Fiedler, noise_sens | size) = +0.07 (p=0.63). The rho=-0.32 from exp_16 was entirely a size proxy (rho(Fiedler,size)=-0.90). Entropy reversal is REAL: partial rho(Fiedler, entropy | size) = +0.29 (p=0.03). (exp_18)
19. **Perturbation types are independent**: node removal, edge rewiring, and Gaussian produce completely different sensitivity rankings (Kendall tau ≈ 0). But 20% node removal disrupts Fiedler massively (96% of regions change >10%). (exp_19)
20. **Lambda_2 decay is universal**: 15/15 regions show monotonic lambda_2 decrease under edge attenuation. Projection operator stays pristine (idempotency error < 1e-15 everywhere). 53% show susceptibility peaks. (exp_20)
21. **H^1-revision link FALSIFIED**: With n=96 sub-partitioned groups, rho(H^1, spectral_revision) = 0.016. Multi-seed medians also show no signal (rho=-0.09). H^1 and backward revision are independent phenomena. (exp_21)
22. **Coupling ceiling at ~0.41**: Gradient weighting is best (partial rho=0.41), Laplacian-response close (0.39), both beat uniform (0.35). But ceiling is below 0.45. Cross-level consistency is zero — scheme effectiveness flips between hierarchy levels. (exp_22)

### Phase 14-17 Results: PAC Interpretation of Falsifications

23. **PAC buffer FALSIFIED**: Size does NOT predict sensitivity via 1/√N conservation scaling. Power-law exponents are NEGATIVE (bigger regions are MORE sensitive). Partial rho(-size, sens | Fiedler) = -0.15 (wrong sign). Sub-partition ratios scattered (1/5 in target range). Neither size nor Fiedler explains sensitivity. (exp_23)
24. **Hodge orthogonality PARTIALLY FALSIFIED**: Gaussian and node-removal shift vectors are NOT orthogonal (mean |cos|=0.47). But topological perturbations (removal + rewiring) DO share structure (|cos|=0.49, test 4 verified). Band activation: mid modes (4-7) dominate all perturbation types — no mode separation. (exp_24)
25. **Revision drivers identified**: perturbation_l2 (rho=0.34, p=8e-4) and boundary_coupling (rho=0.33, p=1e-3) both predict backward revision. H^1 falsification replicated (rho=0.016). Multiple regression R²=0.15. Revision is driven by perturbation magnitude reaching parent-child boundaries — straightforward diffusion physics. (exp_25)
26. **Confluence-entropy is SIZE**: ALL raw Fiedler correlations with spectral shape (entropy, Gini, effective dimensionality) collapse after size deconfound. Partial rho(Fiedler, entropy | size) = -0.02. The exp_18 finding (partial rho=+0.29) used partial-spectrum entropy; full-spectrum analysis shows size mediates everything. Random graphs show similar pattern (rho=-0.37). Mode 1 dominance is universally low (~24%). (exp_26)

### Phase 18-21 Results: Boundary Geometry Hypothesis

27. **Boundary gradient is a real signal**: mean_boundary_gradient has partial rho=-0.30 (p=0.023) with sensitivity after size deconfound — the only boundary metric that survives. Compactness is independent of size (rho=-0.08, verified). But boundary_area_ratio does NOT predict sensitivity (partial rho=0.12, p=0.39). Only 1/6 metrics clear the |partial rho|>0.20 bar. (exp_27)
28. **Boundary schemes beat interior by +0.09**: boundary_fiedler achieves partial rho=0.39 vs gradient's 0.30 — a meaningful improvement. Cross-level Kendall tau=0.33 (consistent). But the 0.45 ceiling wasn't broken, and rho(coupling,size) remains >0.30. The coupling ceiling persists even with boundary-aware weighting. (exp_28)
29. **Sensitivity mediation FALSIFIED**: boundary_area_ratio does NOT mediate the size→sensitivity relationship. Size retains its direct effect (partial rho=0.28 after controlling for bar). Bootstrap indirect effect CI includes zero. Sub-partition validation passes (3/5) but the core mediation hypothesis fails. (exp_29)
30. **256x256 replication FAILS**: 174 regions (sufficient) but correlations don't replicate. The abbreviated diffusion produces low-structure fields (C std=0.0009 vs ~0.01 at 128). Gradient coupling partial rho=-0.19 (sign-flipped from 128x128 ref of +0.41). Boundary metrics show near-zero correlations. Scale-up requires full steady-state diffusion. (exp_30)

## Falsification Criteria

| Prediction | Result | Verdict |
|---|---|---|
| I(parent) ~ f(I(children), w) | rho=0.42 globally, p<0.001 | SUPPORTED |
| Small obstacle in flow > large obstacle outside | 1.7x more impact, 7.7x fewer cells | SUPPORTED |
| Global cascade from local perturbation | 88-97% distant change | SUPPORTED |
| Past identity changes after perturbation | 71.5% revision, cosine rotation | SUPPORTED |
| Harmonic component captures identity | Parseval to 1e-17, conservation to 1e-15 | SUPPORTED |
| CI is proper projection operator | 5/5 operator properties verified | SUPPORTED |
| Perturbation creates identity crisis | H^1 increases in all 6 groups | SUPPORTED |
| Smoothing resolves identity crisis | 70% recovery at alpha=0.3 | SUPPORTED |
| Coupling signal survives size deconfound | Partial rho=0.42, perm p=0.0007 | SUPPORTED |
| H^1 propagates upward through hierarchy | Attenuation 10.34→9.12 | SUPPORTED |
| High Fiedler = less perturbation sensitive | Partial rho=+0.07 after deconfound | **FALSIFIED** |
| H^1 drives backward revision | rho=0.016 with n=96, zero signal | **FALSIFIED** |
| Node removal > Gaussian perturbation | Ratio 1.02x, rankings independent | **FALSIFIED** |
| Lambda_2 decay predicts fragmentation | Monotonic but susceptibility peaks sparse | PARTIALLY SUPPORTED |
| Coupling ceiling above 0.50 | Best partial rho=0.41 (gradient) | **FALSIFIED** |
| Size is PAC conservation buffer (1/√N) | Positive slopes, alpha=-0.27 | **FALSIFIED** |
| Perturbation modes are Hodge-orthogonal | Mean cos=0.47 (not orthogonal) | **FALSIFIED** |
| Backward revision has identifiable driver | perturbation_l2 rho=0.34, boundary rho=0.33 | SUPPORTED |
| Fiedler-entropy is intrinsic to coupling | Partial rho=-0.02 after full-spectrum deconfound | **FALSIFIED** |
| Boundary area ratio predicts sensitivity | Partial rho=0.12 (p=0.39) after size deconfound | **FALSIFIED** |
| Boundary schemes break coupling ceiling | Best partial rho=0.39, below 0.45 target | **FALSIFIED** (improved but not broken) |
| Boundary surface mediates size→sensitivity | Size retains effect (partial rho=0.28) | **FALSIFIED** |
| Boundary gradient predicts sensitivity | Partial rho=-0.30 (p=0.023) after size control | SUPPORTED |
| Boundary schemes outperform interior | +0.09 improvement, cross-level consistent | SUPPORTED |
| 256x256 replicates 128x128 findings | Correlations don't replicate (abbreviated diffusion) | **INCONCLUSIVE** |

## Related FDOs

- `confluent-identity` -- core concept
- `proj-identity` -- project anchor
- `confluence-operator` -- parent formalism
- `pac-framework-unified` -- conservation framework
- `classical-physics-information-geometry` -- Hodge duality bridge
- `asymmetric-conservation` -- scope/frame dependence
- `hodge-conjecture-symbolic-collapse` -- Hodge-SEC connection

## Origin

Beach walk, New Brunswick, March 27, 2026. Fractal drainage patterns in sand
and mass-hierarchy effects around stones.

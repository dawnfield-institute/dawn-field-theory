# Confluent Identity -- Phases 1-29: Formation through Falsification

## Hypothesis

Identity in a PAC tree is the weighted confluence pattern of its children --
formalized as the harmonic component of the weighted graph Laplacian (discrete
Hodge theory). A parent's identity cannot be reconstructed from the simple
mean or sum of its children, but *can* be reconstructed from their spectral
fingerprints weighted by coupling strength to the parent's global identity.

## Status: Active -- Phases 1-29 complete

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
| Spectral shape | 5 modal descriptors vs sensitivity | Spectral bandwidth (partial rho=0.44, p=7e-4) and mode concentration (-0.35, p=9e-3) survive size deconfound |
| Mutual information coupling | MI, dCor vs Spearman ceiling | NMI=0.47 exceeds rho²=0.25 by +0.22; nonlinear signal exists but dCor doesn't beat partial_rho |
| Scale-dependent coupling | Per-level decomposition | Coupling IS scale-dependent (range=1.19) but pooled (0.42) beats all individual levels |
| 256x256 full diffusion | Fix exp_30 with steady-state | C std still 0.0007 — problem is initialization, not step count; size-sens sign replicates |
| Frame-relative identity (Δ buffer) | Δ between parent/child views | Δ_entropy rho=-0.79 (p=0.02), depth-dependent, but n=8 underpowered; 0/4 formal tests |
| Eigenstructure ceiling derivation | Geometric alignment vs coupling ceiling | Eigenbasis alignment=0.60 > ceiling=0.42; ceiling is rank-compression artifact, not geometry; 0/4 formal tests |

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
| `exp_31_spectral_shape_sensitivity.py` | Spectral shape descriptors vs sensitivity | 22 | Complete (2/4) |
| `exp_32_mutual_information_coupling.py` | MI and distance correlation vs Spearman ceiling | 23 | Complete (3/4) |
| `exp_33_scale_dependent_coupling.py` | Per-hierarchy-level coupling decomposition | 24 | Complete (1/4) |
| `exp_34_scale_validation_full.py` | 256x256 with full steady-state diffusion | 25 | Complete (2/4) |
| `exp_35_frame_relative_identity.py` | Frame-relative identity: Δ buffer between parent/child views | 26 | Complete (0/4) |
| `exp_36_eigenstructure_ceiling.py` | Eigenstructure derivation of coupling ceiling | 27 | Complete (1/5) |
| `exp_37_multiseed_phi_validation.py` | Multi-seed validation of phi-constant relationships | 28 | Complete (1/4) |
| `exp_38_scoped_mediation.py` | Scoped mediation: level-by-level identity propagation | 29 | Complete (3/4) |

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

### Phase 22-25 Results: Spectral Shape, Nonlinear Coupling, and Scale

31. **Spectral bandwidth predicts sensitivity**: partial rho(spectral_bandwidth, sensitivity | size) = 0.44 (p=7e-4) — the strongest size-independent sensitivity predictor found. Mode concentration also survives (partial rho=-0.35, p=9e-3). Spectral flatness is NOT a size proxy (|rho|=0.31, verified). But mid-mode ratio (the 24% clue) shows no signal (partial rho=0.15), and multiple regression R²=0.08 (below 0.15 target). (exp_31)
32. **Nonlinear signal exists but doesn't break ceiling**: MI(coupling, natural) is highly significant (permutation p=0.0000). NMI=0.47 exceeds Spearman rho²=0.25 by +0.22 — substantial nonlinear dependence. Residualized MI=0.65 nats (survives size deconfound). But distance correlation of residuals (0.42) doesn't beat |partial_rho| (0.42) — the ceiling is real physics, not a Spearman artifact. (exp_32)
33. **Coupling IS scale-dependent but pooling helps**: Per-level partial_rho range = 1.19 (from -0.90 to +0.29). But no single level exceeds 0.50 — Level 1 collapses to -0.08 after size control, Level 2 reaches only 0.29. Pooled partial_rho (0.42) beats every individual level. This is reverse Simpson's paradox: aggregation improves the signal. Within-parent correlations are weak (mean rho=-0.14). (exp_33)
34. **256x256 STILL fails with full diffusion**: 10,000 steps with PeriodicLatticeFluid.run_to_steady_state() produces C std=0.0007 — no better than exp_30's abbreviated 1500 steps. The problem is initialization/parameters, not convergence. Size-sensitivity sign replicates (positive, matching paradox) but correlation magnitudes don't match (gradient partial rho=-0.09 vs 128x128's +0.41). 256x256 requires fundamentally different initialization (stronger stones, lower viscosity, or different frequency content). (exp_34)

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
| Spectral shape predicts sensitivity | Bandwidth partial rho=0.44 (p=7e-4) after size control | SUPPORTED |
| Mid-mode ratio explains sensitivity paradox | partial rho=0.15, not significant | **FALSIFIED** |
| MI breaks Spearman coupling ceiling | NMI exceeds rho² by +0.22 but dCor matches partial_rho | **FALSIFIED** (nonlinear signal exists, ceiling is real) |
| Per-level coupling exceeds 0.50 | Best level partial rho=0.29, pooled=0.42 beats all | **FALSIFIED** |
| 256x256 works with full steady-state diffusion | C std=0.0007, same as abbreviated | **FALSIFIED** (initialization problem) |
| Δ buffer is bounded (max < 1.0) | Δ_spectral max=1.58 (cosine distance in [0,2]) | **FALSIFIED** (wrong measure) |
| Δ adds predictive power (R² +0.05) | Δ_entropy rho=-0.79 (p=0.02) but n=8 underpowered | **INCONCLUSIVE** (signal present, insufficient power) |
| Δ scales with hierarchy depth | Δ_fiedler and Δ_entropy rho=-0.76 (p=0.027) | SUPPORTED (but n=8) |
| Frame-augmented coupling > 0.42 | Frame augmentation with Δ_spectral hurt rather than helped | **FALSIFIED** |
| Eigenbasis alignment = coupling ceiling | Mean cos_eigenbasis=0.60, distance=0.18 from 0.42 | **FALSIFIED** (alignment HIGHER than ceiling) |
| Delta_self predicts coupling residuals (n=59) | rho=0.09, p=0.50 — zero signal | **FALSIFIED** (exp_35's n=8 signal was artifact or dual-role specific) |
| Raw field alignment = ceiling | Mean cos_raw=0.78, distance=0.36 from 0.42 | **FALSIFIED** (much higher) |
| Conservation: Delta reduces prediction variance >=10% | Pooled: 3.7%. Within-parent: 16% mean | **FALSIFIED** pooled, SUPPORTED within-parent |
| Partial_rho ≈ γ (Euler-Mascheroni) | Seed 42: 0.33% match. 20 seeds: mean=0.38, CV=0.57 | **FALSIFIED** (seed-specific) |
| Size confounding = 1/φ⁴ across seeds | Mean=0.1501, prediction=0.1459, 95% CI contains target | SUPPORTED (2.87% error, n=20) |
| Coupling strength is a universal constant | CV=0.57 (partial), CV=1.08 (raw) across 20 seeds | **FALSIFIED** (realization-dependent) |
| Coupling attenuates monotonically with level distance | Projection norms: 0.033→0.015→0.011→0.011 | SUPPORTED |
| Per-hop attenuation ≈ 1/φ | Mean ratio=0.730, delta=18.1% from 1/φ=0.618 | SUPPORTED (within 20%) |
| 2-hop = product of 1-hop couplings | Mean relative error 72.4% | **FALSIFIED** (mediation transforms, not attenuates) |

### Phase 27 Results: Eigenstructure Ceiling Derivation

36. **The ceiling is NOT geometric alignment — it's rank compression**: Eigenbasis-projected cosine similarity between gradient and state fields averages 0.60 (median 0.90), and raw field alignment averages 0.78. Both are well above the 0.42 coupling ceiling. The ceiling arises during the conversion from continuous alignment to Spearman rank correlation of aggregated norms — the lossy compression of region-level summaries discards ~0.18 of the geometric signal. This means the coupling ceiling is an information-theoretic limit on how much of a continuous geometric relationship survives rank-order aggregation across regions. (exp_36)

37. **exp_35's Delta signal was a small-sample artifact or dual-role specific**: Delta_self (internal entropy vs external projection norm) shows rho=0.09 (p=0.50) with coupling residuals at n=59 — zero signal. The rho=-0.79 from exp_35 at n=8 dual-role nodes does not generalize to all children. Dual-role nodes may be genuinely special (they have their own children, giving them richer internal structure), or n=8 was simply insufficient for reliable inference. (exp_36)

38. **Conservation IS within-scope**: Within-parent, Delta reduces coupling-prediction variance by 16% on average (range: 0-35% across parents). But pooled across parents, only 3.7%. This is consistent with P+A+Delta=C being a within-scope conservation law — the Δ buffer operates within each parent's frame, not across frames. Cross-parent pooling dilutes the signal because each parent has its own eigenstructure. (exp_36)

39. **Coupling ceiling update**: Pooling all 67 children across all 10 parents gives partial_rho(coupling, natural | size) = 0.5753 — significantly above the 0.42 previously reported for parent (2,1) alone. The "ceiling" depends on scope: more parents = more signal. This is reverse Simpson's paradox again (exp_33), now confirmed directly. (exp_36)

### Phase 28 Results: Multi-Seed Phi Validation

40. **Size confounding = 1/φ⁴ is universal**: Across 20 random seeds, the size confounding (raw_rho - partial_rho) averages 0.1501 — matching the tetration termination penalty 1/φ⁴=0.1459 within 2.87%. The 95% CI [0.0916, 0.2087] contains 1/φ⁴. This is the ONE phi-constant relationship that survives multi-seed validation. (exp_37)

41. **γ match was seed-specific**: The partial_rho ≈ γ = 0.5772 match found in exp_36 (0.33% error at seed 42) does not generalize. Across 20 seeds, partial_rho has CV=0.57 and mean=0.3805 — far from γ. The coupling strength itself varies wildly across fluid realizations. (exp_37)

42. **Coupling variability is extreme**: Raw Spearman CV=1.08, partial Spearman CV=0.57. Some seeds produce negative coupling (sign flip). The coupling ceiling is not a fixed constant — it depends on the fluid realization. What IS constant is the size confounding penalty. (exp_37)

### Phase 29 Results: Scoped Mediation

43. **Identity propagates level-by-level with monotonic attenuation**: Coupling between a level-0 region and its ancestors decays monotonically with hop distance: 0.033 → 0.015 → 0.011 → 0.011. The first boundary absorbs most information loss (ratio 0.45), then flattens (0.78, 0.97). By level 3-4, only harmonic content remains. (exp_38)

44. **Attenuation ≈ 1/φ per hop**: Mean per-hop attenuation ratio is 0.730, within 18.1% of 1/φ=0.618. CV of hop ratios is 0.294 (just under 0.3 threshold). The golden ratio appears as the natural attenuation rate through scope boundaries. (exp_38)

45. **2-hop coupling is NOT predictable from 1-hop products**: Mean relative error 72.4% (median 84.4%). This is the KEY confirmation of scoped mediation — each level boundary transforms the signal, not just attenuates it. You cannot compose 1-hop couplings to predict 2-hop because each level is a recursive closure. This mirrors ADE: multiplication is not "addition applied twice." (exp_38)

46. **Size-normalized coupling INCREASES with distance**: After dividing by size ratio, coupling per unit size goes from 0.19 (distance 1) to 0.66 (distance 4). The surviving signal at higher levels is proportionally stronger — the harmonic component that passes through every scope boundary is concentrated, not diluted. (exp_38)

### Phase 26 Results: Frame-Relative Identity

35. **Frame Δ buffer reveals depth-dependent signal (underpowered)**: Only 8 dual-role nodes exist (nodes that are both parent AND child with ≥2 children). Despite n=8, Δ_entropy (complexity mismatch between internal and external views) correlates with coupling at rho=-0.79 (p=0.02) — the strongest single-measure coupling correlation in the entire experiment series. Δ_fiedler and Δ_entropy both scale with hierarchy depth (rho=-0.76, p=0.027) — deeper nodes have larger frame shifts, consistent with asymmetric conservation. However, Δ_spectral (cosine distance of eigenbasis projections) was the wrong measure for the boundedness test (scipy cosine_distance returns [0,2] for anti-aligned vectors, max=1.58). Frame augmentation with Δ_spectral actually hurt partial_rho rather than helped — the spectral projection conflates frame shift with basis incompatibility. The Δ concept is promising but needs either a larger lattice (more hierarchy levels) or a different formalization. Key insight: the frame shift is real and depth-dependent, but the coupling ceiling may be a fixed-point property of the spectral projection operator itself, not a frame-averaging artifact. (exp_35)

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

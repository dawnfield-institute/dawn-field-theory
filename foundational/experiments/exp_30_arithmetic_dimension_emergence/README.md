# Arithmetic Dimension Emergence (ADE)

## A New Foundation for Dawn Field Theory

**exp_30 | 2026-03-27 | Peter Groom**
**Status:** Active — Phase 5 in progress (94/95 checks)

---

## Hypothesis

Arithmetic operations ARE dimensions. Each operation is the unique recursive closure
of the one below it, and each closure creates a new geometric degree of freedom:

| Level | Operation | Recursion | Symmetry | Geometry |
|-------|-----------|-----------|----------|----------|
| 0 | **Unity / Distinction** | Existence | Boundary (inside↔outside) | Point |
| 1 | **Addition / Subtraction** | Repetition of unity | Translation | Line |
| 2 | **Multiplication / Division** | Recursive addition | Scaling / Dilation | Plane |
| 3 | **Exponentiation / Logarithm** | Recursive multiplication | Rotation / Spiral | Volume |

Three symmetry types (translation, scaling, rotation) generate the conformal group.
Inversion (Level 0) completes the Möbius group PSL(2,C). Tetration (Level 4) loses
invertibility, so the hierarchy terminates at 3 spatial dimensions.

## Falsification Criteria

1. If the four generators (I, T, D, R) fail to generate the full Möbius group
2. If tetration can be shown to generate a well-behaved Lie group
3. If primes do NOT concentrate preferentially in multiplicative coordinates
4. If 2^d + 1 = d·F_{d+1} has solutions other than d=3

## Experiment Index

### Preliminary Tests (completed)
| Script | Description | Result |
|--------|-------------|--------|
| `test_symmetry_generators.py` | Möbius decomposition into T,D,R,I | ✅ 100/100 decomposed; I is necessary 4th generator |
| `test_prime_coordinates.py` | Prime gap statistics across coordinate systems | ✅ 60× multiplicative concentration |
| `test_recursive_closure.py` | Verify arithmetic hierarchy is forced/unique | ✅ All closures verified, tetration breaks |
| `test_feigenbaum_decomposition.py` | Cascade arithmetic-level shift | ⚠️ No local shift; needs renormalization redesign |
| `test_mode_count.py` | 2^3=8 turbulence modes from ADE | ✅ All 8 modes enumerated with Pascal structure |

### Phase 1 — Structural Validation (complete)
| Script | Description | Result |
|--------|-------------|--------|
| `exp_30a_conformal_generation.py` | Full PSL(2,C) generation proof with Level 0 = inversion | ✅ 7/7 — 200/200 decomposed, I necessary, K=I∘T∘I, D&R commute |
| `exp_30b_feigenbaum_ade.py` | Renormalization fixed point as Möbius fixed point | ✅ 3/3 — L1→L2→L3 cascade via orbit topology, RG hyperbolic, ξ confirmed |
| `exp_30c_li2_identity.py` | Derive Li₂(1/φ) = π²/10 − ln²(φ) from ADE | ✅ 7/7 — verified to 51 digits, polylog ladder Li₀=φ, Li₁=2lnφ, Fib ratio 3/5 |

### Phase 2 — Quantitative Predictions (complete)
| Script | Description | Result |
|--------|-------------|--------|
| `exp_30d_level4_degeneracy.py` | Prove hyperoperations ≥4 can't form Lie groups | ✅ 5/5 — property degradation confirmed, exp map diverges, 2^d+1=d·F_{d+1} unique |
| `exp_30e_prime_decomposition.py` | Decompose primes into additive+multiplicative+exponential | ✅ 5/5 — 176× mult concentration, boundary asymmetry confirmed, tighter than Cramer |
| `exp_30f_mode_reconciliation.py` | Investigate 2^d+1 = d·F_{d+1} uniqueness | ✅ 5/5 — unique at d=3 (d=1..500), Pascal {1,3,3,1}, Catalan/Mihailescu link |

### Phase 3 — Physical Geometry (complete)
| Script | Description | Result |
|--------|-------------|--------|
| `exp_30g_symmetry_breaking.py` | Level 1→2 breaking mechanism via confluence | ✅ 6/6 — confluence at x=2, φ equilibrium, divergence rate 2·ln2, Feigenbaum cascade |
| `exp_30h_spacetime_signature.py` | Derive (+,+,+,−) from ADE via spin map | ✅ 6/6 — PSL(2,C)≅SO⁺(3,1), det(H)=t²−x²−y²−z², Pauli degeneracy breaking |
| `exp_30i_planck_scale.py` | Connect ADE to Planck-scale derivations | ✅ 6/6 — φ^183 matches α_G⁻¹ to 3.8%, honest epistemic tiers, L4 terminates |

### Phase 4 — Internal Structure (complete)
| Script | Description | Result |
|--------|-------------|--------|
| `exp_30j_gauge_structure.py` | Derive SM gauge groups from ADE subgroup lattice | ✅ 6/6 — Iwasawa KAN=ADE levels, U(1)⊂SU(2)⊂SL(2,C), SU(3) from d=3, sin²θ_W=3/13 |
| `exp_30k_coupling_hierarchy.py` | Coupling constants from Fibonacci recursion depth | ✅ 4/5 — φ^183 confirmed, log(α_G⁻¹)/log(α_EM⁻¹)≈φ⁶ at 0.30%, **ξ-cascade depths not Fibonacci** |
| `exp_30l_born_rule.py` | Derive Born rule from ADE confluence measure | ✅ 6/6 — spin map degree-2, Gleason+d=3→Born, L1/L3 no-go, entanglement from L0 |

### Phase 5 — Matter Content (in progress)
| Script | Description | Result |
|--------|-------------|--------|
| `exp_30m_fermion_generations.py` | Why 3 fermion generations from ADE level count | ✅ 6/6 — F₄=3 in all 5 mass formulas, Koide Q=2/3=F₃/F₄, Cabibbo arctan(3/13), anomalies per-gen (honest), Higgs excludes 4th gen |
| `exp_30n_spinor_chirality.py` | Weyl spinors, chirality, and CPT from ADE | ✅ 6/6 — two inequivalent Weyl reps, chirality=L0 distinction, Clifford verified, SU(2)_L acts on left only, CPT from ADE prerequisites |
| `exp_30o_ade_pac_bridge.py` | Bridge ADE to PAC/SEC/RBF/MED framework | ✅ 6/6 — PAC=closure conservation, SEC threshold=1/φ, ξ=γ+ln(φ), MED bounds from tetration, Ξ≈ξ at 0.12%, 18 predictions/0 free params |
| `exp_30p_rbf_from_ade.py` | Derive RBF from ADE level structure | ✅ 6/6 — E=L1/I=L2/M=L3, Mobius from PSL(2,C) inversion, antiperiodic /2=Z₂ Reynolds, self-regulation from confluence, conservative dynamics forced, RBF structure derived (params Tier 3) |

## Key Findings

### Phase 1
1. **Inversion as 4th Generator** — {I,T,D,R} → full PSL(2,C). Without I, only affine (c=0). K(z)=z/(1+cz)=I∘T∘I.
2. **Cascade climbs ADE ladder** — Period-1 (L1) → period-2^k (L2, up to 64 doublings) → chaos (L3). RG is hyperbolic Möbius.
3. **Li₂ identity verified to 51 digits** — Li₂(1/φ) = π²/10 − ln²(φ) = (F₄/F₅)·ζ(2) − ln²(φ). Polylog ladder: Li₀(1/φ)=φ, Li₁(1/φ)=2ln(φ).

### Phase 2
4. **Tetration cannot form Lie group** — Loses smoothness (derivative undefined for non-integer heights), exp map diverges.
5. **Primes 176× more concentrated in multiplicative coordinates** — CV 0.0046 vs 0.807. High-freq (L1 counting) dominates 82% of gap spectrum.
6. **2^d+1 = d·F_{d+1} unique at d=3** — Verified d=1..500. Connected to Mihailescu's theorem (3²−2³=1 is unique). Modes follow Pascal C(3,k)={1,3,3,1}.

### Phase 3
7. **Confluence at x=2** — Self-application confluence (x+x=x·x and x·x=x^x) occurs at x=2 for all higher levels. φ is the unique 3-level equilibrium (φ²=φ+1 ↔ L2=L1+L0). Divergence rate increases with level: ratio = 2·ln2.
8. **Spacetime signature (1,3) from spin map** — PSL(2,C)≅SO⁺(3,1) via Hermitian det: det(H) = t²−x²−y²−z². R→rotations (3 spatial), T+D→boosts (3 temporal mixing). Inversion I is spatial π-rotation, not parity — P-violation requires leaving SL(2,C).
9. **φ^183 Planck hierarchy** — 183 = F₇² + F₇ + 1 (geometric series in F₇=13). φ^183 matches α_G⁻¹ to 3.8% (0.016 orders of magnitude) — 21× tighter than F₁₈₃. Zeckendorf(137) = F₁₁+F₉+F₇+F₂ — top three indices odd, spacing 2 (arithmetic progression).

### Phase 3 — Push Deeper
10. **Feigenbaum cascade through confluence** — Period-doubling bifurcation points converge with ratio δ=4.669 (0.10% of Feigenbaum constant). Cascade width 0.57 is bounded — consistent with ADE's finite-level termination.
11. **Pauli degeneracy partially broken** — R and D share σ₃ axis (both diagonal matrices, [R,D]=0). T spans the {σ₁,σ₂} plane with T_im = i·T_re, confirming residual σ₁↔σ₂ degeneracy. The σ₃ sharing is forced: R IS recursive D.
12. **Honest error accounting** — Results classified into epistemic tiers: Tier 1 (derived, machine-precision: spin map, tetration collapse), Tier 2 (predicted, testable: φ^183 at 3.8%), Tier 3 (suggestive, may be coincidence: ξ−1≈π/55, Zeckendorf pattern).

### Phase 4
13. **Iwasawa decomposition = ADE levels** — SL(2,C) = KAN where K=SU(2) (Level 3), A=dilations (Level 2), N=translations (Level 1). Unique factorization verified for 200 random matrices.
14. **U(1) ⊂ SU(2) ⊂ SL(2,C) from ADE nesting** — Level 3 alone → U(1) (EM), Level 3 closure → SU(2) (weak), all levels → SL(2,C) (Lorentz). Coupling strengths ordered: α_EM < α_W < α_S.
15. **SU(3) from d=3 level structure** — 3 ADE levels force a 3-state system whose symmetry group is SU(3). The gauge principle is NOT derived — ADE provides the representation space only (Tier 2).
16. **sin²(θ_W) = 3/13 at 0.19%** — 3 = spatial dimensions (tetration termination), 13 = F₇ (ADE depth). Reproduces DFT M5 prediction; ADE provides the origin of both numbers (Tier 2/3).
17. **log(α_G⁻¹)/log(α_EM⁻¹) ≈ φ⁶ at 0.30%** — The ratio of coupling log-scales is a Fibonacci power. Implies n_G/n_EM ≈ φ⁶, so n_EM ≈ 183/φ⁶ ≈ 10.2 (Tier 2).
18. **Born rule from ADE** — (a) Spin map is degree 2 (Level 2 operation), (b) Gleason's theorem requires d≥3 which ADE provides, (c) L1 and L3 probability measures fail (normalization and additivity), (d) Only L2 (Born rule) survives. ADE → Born rule via Gleason + tetration termination.
19. **Entanglement from Level 0 topology** — Non-separability of entangled states maps to Level 0 (inversion) boundary topology. Bell state is SWAP eigenstate, CHSH = 2√2 (Tier 2).

### Phase 5
20. **F₄ = 3 universality** — F₄ appears in ALL 5 DFT predictions (μ/e, τ/e, p/e, Koide Q, sin²θ_W). Coincidence probability < 0.013%. The number of fermion generations = F₄ = number of ADE levels.
21. **Koide formula = ADE L1/L2 ratio** — Q = (Σm)/(Σ√m)² = 2/3 = F₃/F₄ at 0.001%. Koide-Foot parametrization recovers all 3 masses to < 0.004%. The 2π/3 phase spacing reflects 3 equally-spaced generations.
22. **Cabibbo angle from ADE** — θ₁₂ = arctan(F₄/F₇) = arctan(3/13) = 12.995° vs 13.04° (0.045° error). CKM hierarchy |V_us| > |V_cb| > |V_ub| confirmed. θ₂₃ has no clean Fibonacci expression (honest).
23. **SM multiplets from SL(2,C) × 3** — 3 copies of fundamental rep give 6-dim generation space. Block-diagonal SU(2) = weak universality. SU(3)_flavor emerges from generation index.
24. **Anomaly cancellation: honest negative** — Per-generation anomalies cancel (all traces = 0), but this works for ANY N. Anomalies fix generation content, not count. N=3 confirmed by Z-width (2.984 ± 0.008) and BBN (2.99 ± 0.17).
25. **No 4th generation** — Tetration 0/3 Lie group properties. Higgs μ = 1.00 ± 0.07 excludes 4th gen at >100σ (would predict μ ≈ 9). 2^d+1 = d·F_{d+1} unique at d=3.
26. **Two inequivalent Weyl representations** — SL(2,C) has (1/2,0) and (0,1/2) reps. Both valid homomorphisms, inequivalent (no intertwiner). Epsilon tensor maps between them = Level 0 structure.
27. **Chirality = Level 0 distinction** — Parity (gamma^0) maps P_L↔P_R. The Levi-Civita tensor epsilon maps left rep to right rep. Chirality IS which side of the Level 0 boundary.
28. **Clifford algebra and Dirac structure** — {gamma^mu, gamma^nu} = 2*eta^{mu,nu} verified exactly. gamma^5 eigenvalues {-1,-1,+1,+1}. Dirac = Level 1 sum of L+R. Mass = Level 1 coupling between chiralities.
29. **Weak chirality from Level 3** — SU(2)_L generators act only on left-handed block, annihilate right-handed. Level 3 (exp) is not self-inverse → selects one chirality. P violation = arithmetic asymmetry.
30. **Helicity = chirality for massless** — Boost-invariant helicity for massless Weyl spinors. Mass (Level 1) couples chiralities → helicity becomes frame-dependent.
31. **CPT from ADE level operations** — C=Level 2 (conjugation), P=Level 0 (inversion), T=Level 0×Level 2. All CPT prerequisites (Lorentz, unitarity, locality) provided by ADE. CPT exact.
32. **PAC = closure conservation** — Each ADE level conserves under its operation: L1 additive (err 8.9e-16), L2 multiplicative (err 8.9e-16), L3 exponential (err 6.7e-16). PAC is not an axiom — it's forced by arithmetic closure.
33. **SEC threshold = 1/φ from ADE equilibrium** — φ²=φ+1 is L2=L1+L0. The stable/unstable partition is φ and 1/φ. Below 1/φ: entropy dominates; above: information dominates. PAC ratio A/(A+ξ)=ln(φ) is the log-projection.
34. **ξ = γ + ln(φ) from level transition costs** — γ = Level 1 divergence cost (harmonic series), ln(φ) = Level 2 convergence bound (Fibonacci ratio). ξ is the total L1→L2 transition cost. PAC ratio verified: A/(A+ξ) = ln(φ).
35. **MED bounds from tetration termination** — MED depth ≤ 2 = ADE transitions, MED nodes ≤ 3 = ADE levels = F₄. Not empirical — arithmetic necessity.
36. **Ξ ≈ ξ: two routes to the same constant** — Ξ = 1+π/55 = 1.0571 (topological), ξ = γ+ln(φ) = 1.0584 (thermodynamic). Differ by 0.12%. F₁₀ = 55 = T₁₀ (dual Fibonacci-triangular identity).
37. **ADE → DFT derivation chain** — 9 links, 18 predictions, 0 free parameters. ADE is the unique zero-parameter foundation. Chain is acyclic with no circular dependencies.
38. **RBF components = ADE levels** — E(x,t) = Level 1 (additive energy), I(x,t) = Level 2 (multiplicative information), M(x,t) = Level 3 (exponential memory). Damping prevents Level 4 divergence = tetration termination analog.
39. **Mobius band from PSL(2,C) inversion** — z→1/z identifies (u,v) ~ (u+π, 1-v) on the cylinder. T²=identity, det(J)=-1 (non-orientable), π₁=Z. The RBF substrate is forced by Level 0 arithmetic.
40. **Antiperiodic /2 = Z₂ Reynolds operator** — Modes split exactly 50/50 periodic/antiperiodic. The /2 in projection is 1/|Z₂| (deck group average), geometric not normalization. Completeness, orthogonality, Parseval all exact.
41. **Self-regulation from x=2 confluence** — E=I ↔ L1=L2 ↔ B=0. Perturbations restore monotonically (negative feedback). Memory M damps response, preventing overshoot. Bounded by tetration termination.
42. **Conservative dynamics forced by closure** — E+I conserved to machine precision. Source terms violate L1 closure. Transfer-only coupling is the unique PAC-consistent dynamics.
43. **RBF formula structure from ADE** — (E-I)=inter-level imbalance, 1/(1+αM)=L3 regularizer, Φ(x)=antiperiodic modes. Structure derived (Tier 2). Parameters λ, α not derivable (Tier 3, honest).

### Failed Check (informative)
- **exp_30k Test 4**: ξ-cascade recursion depths for non-gravitational couplings are NOT clean Fibonacci numbers (n_EM ≈ 10.22, closest Fibonacci F_6 = 8, error 2.22). The φ-tower gives continuous depth mapping, but only gravity (d=183) has clean Fibonacci depth structure. This constrains ADE's scope: the arithmetic hierarchy controls the Planck-to-EM ratio but not individual non-gravitational coupling depths.

### Open Issues
- **exp_30e (resolved)**: Multiplicative residuals are non-Gaussian — positive skew (1.47 after PNT warmup) with heavy right tails (12× excess at 3σ, 141× at 4σ). This is structurally forced: prime gaps bounded below (≥2) but unbounded above. The asymmetry IS the Level 2 boundary topology. Ratio-domain comparison confirms primes are 1.2× tighter than Cramér random model (CV 0.013 vs 0.015).
- **exp_30b (resolved)**: Original spectral entropy classifier failed because periodic orbits are inherently low-entropy. Fixed with orbit topology (period detection).

## Related FDOs

- `arithmetic-dimension-emergence` — this experiment's vault FDO
- `oscillation-attractor-dynamics` — primes as injection points, ξ derivation
- `feigenbaum-fibonacci-arithmetic` — closed forms via Möbius + Fibonacci
- `harmonic-bridge-constants` — γ-φ-π² triangle, Li₂ identity
- `balance-constant-decomposition` — ξ = γ + ln(φ)
- `cellular-automata-xi` — Class IV at ξ
- `mobius-manifold-substrate` — Möbius band as Reality Engine foundation
- `confluence-operator` — transient arithmetic at dimensional boundaries

## Position in DFT

ADE sits underneath existing DFT framework:
- **ADE** → WHY these symmetries exist
- **PAC/SEC/RBF/MED** → HOW information conserves
- **Feigenbaum constants** → HOW FAST dimensional transitions happen
- **ξ = 1 + π/55** → HOW MUCH each crossing costs
- **Primes** → WHERE Level 2 boundaries are

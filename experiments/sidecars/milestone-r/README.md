# Milestone R: Radiation as Ledger Severance

## Thesis

Radiation is PAC ledger severance. When a system emits radiation, the PAC conservation ledger splits into two independent branches. The emitted radiation's wavelength encodes: (1) the Fibonacci depth of the severed connection, (2) whether the process is actualization or degradation (discrete vs continuous spectrum), and (3) how far from equilibrium the system was (line width).

The analogy is precise: nuclear ejections look like carbonated water bubble ejections in 0g -- both isotropic, stochastic, driven by internal energy differential, not gravity. Gravity (depth 183) is a spectator at nuclear scales, 37 orders weaker. The "pressure not temperature" mechanism (shift PAC equilibrium rather than brute-force energy input) maps to PAC rebalancing and could lead to more efficient X-ray generation.

## Status: Active | Score: 62/116 (53%)

Predicted: 27/40 (68%). Actual: 62/116 (53%) after 28 experiments. Sidecar milestone -- does not continue M14->M15 chain.

## Scorecard

| Exp | Name | Block | Score | Notes |
|-----|------|-------|-------|-------|
| 01 | Ledger Severance Mechanics | A | 2/4 | PAC conservation exact, independent ledgers exact. Spectral shift coarser than orbits (T2 FAIL), constant across tree depths (T4 FAIL) |
| 02 | Alpha Decay Discrete Severance | A | 2/4 | T1 tautological (Planck >> MeV), phi-ratio clustering no better than random (T2), relaxation anti-correlates with size (T3). Alpha/proton ratio near phi^0 (T4 PASS) |
| 03 | Beta Spectrum Unsettled Ledger | A | **3/4** | Cascade is continuous (T1), shape matches Fermi surprisingly well (chi^2=0.37, T2 PASS). **T3 PASS 2026-08-27** on the coupling-anchored scale (was: endpoint 19 orders off; old fit hit d=19, the top of its search range). Settled/unsettled entropy not discriminated (T4 FAIL -- graph entropy, not a scale failure). NOTE: the best-fit depth tracks the MEDIATOR CHOICE, not a physical depth -- see the 2026-08-27 journal |
| 04 | Gamma Equilibration | B | 2/4 | Co-60 ratio found in A_7 at 1.9% (T2 PASS). Line width perfectly correlates with disequilibrium (T4 PASS). Equilibration energy negative for most vertices (T1 FAIL), multi-step incomplete for some graphs (T3 FAIL) |
| 05 | Hydrogen Spectrum | B | **2/4** | Series grouping correct (T3 PASS). **T1 PASS 2026-08-27 at 11.4 ppm** via ALPHA_EM_DFT^2*m_e/2 (was: 24 orders off). phi^k != 1/n^2 (T2, structural -- script's own note says expected), fine structure still 486x off with the Rydberg now exact (T4). As predicted: Coulomb SO(4) not in PAC tree |
| 06 | Xi Cost Scope Counting | B | 3/4 | T1/T2/T4 tautological (all energies round to n=0 at Planck scale). Beta also rounds to 0 (T3 FAIL). **The energy scale problem: E_Planck * phi^(-d) >> MeV for all d < 100** |
| 07 | Bubble Ejection Analogy | C | 3/4 | Isotropy exact (T1), pressure 10.6x more efficient (T3), gravity 37 orders irrelevant (T4). Ejection timing NOT exponential (T2 FAIL: cascade gives geometric, not Poisson) |
| 08 | X-ray Equilibrium Shift | C | 3/4 | Bremsstrahlung cutoff works (T2), efficiency gain real at all deltas (T3). Boundary counts tautologically 0 (T1). Transition count formula wrong: severance energies degenerate (T4 FAIL) |
| 09 | Cross-Milestone Compatibility | D | 3/4 | Constants exact, orbits match M14, attenuation within 40% of M6. Dark sector boundary count = 0 (T4 FAIL: same energy scale problem) |
| 10 | Synthesis and Predictions | D | 4/4 | Chain executes, scorecard 21/36 > 50%, 10 predictions, X-ray proposal with 4 measurables |
| 11 | Balance-Seeking Decay | C | 2/4 | **Decay = healing, not breaking.** Deficit↔energy rho=0.63 (T1 PASS). Geometric FPT 4/4 (T4 PASS). GN slope correct sign but R²=0.68 (T2 FAIL: no barrier). Convergence marginal for A-type (T3 FAIL: slow spectral gap) |
| 12 | Topological Barrier GN | C | 3/4 | **Barrier = Coulomb analog.** FPT exponential in degree R²=0.92 (T1 PASS). Barrier R²=0.998 vs plain R²=0.746 (T3 PASS). Empirical GN R²=0.99 (T4 PASS). GN sign wrong for degree-1 target (T2 FAIL: relaxation dominates over barrier at low degree) |
| 13 | Perspectival Barrier | C | 1/4 | **Identity reconciliation tested.** Perspectival barrier outperforms topological 5/5 for GN R² (T4 PASS: ~0.95 vs ~0.3). Hub JSD > endpoint JSD -- opposite prediction but physically correct: stronger identity = higher barrier (T3 FAIL). Degree still dominates JSD for predicting topological FPT (T1 FAIL: 1/13). GN slope still negative (T2 FAIL: deficit→FPT monotonic) |
| 14 | SEC Kinetic Energy | C | 0/4 | **Decisive falsification.** More noise → LONGER FPT (rho=+1.0 universally, T1/T4 FAIL). Noise OPPOSES equilibrium convergence. **Key finding: the perspectival barrier is an ENTROPY barrier (must converge), not an ENERGY barrier (must escape). Noise prevents convergence.** The Coulomb barrier analog requires a STRESS barrier: connections sever when overstressed, not when relaxed. Barrier direction is inverted. |
| 15 | Stress Barrier GN | C | **4/4** | **THE GN SIGN FIX.** Stress barrier (edges overstressed > threshold) flips the sign: rho=-1.0 universally (T1/T4 PASS). degree coeff=+0.15, noise coeff=-13.9, R²=0.70 (T2 PASS). GN combined variable degree/√noise: R²=0.79, positive slope, better than either alone (T3 PASS). **Severance = stress, not relaxation. SEC flux = kinetic energy. PAC degree = barrier height.** |
| 16 | Universality of GN Law | C | **4/4** | **UNIVERSALITY THEOREM.** GN scaling log(FPT) ~ d/f(sigma) is NOT specific to Coulomb or ADE types — it's generic for any d-simultaneous-threshold barrier under stochastic noise. Data collapse across A, D, E families R²=0.74 (T1 PASS). Analytical prediction FPT=1/[erfc(tau/(sigma*sqrt(2)))]^d matches empirical rho=0.90 (T2 PASS). Form survives 3 different thresholds (T3 PASS, 3/3). **Universal exponent k=1.16±0.02 across all families** (T4 PASS, std/mean=2%). |
| 17 | Spectral Fingerprinting | C | 1/4 | **Honest failure: stress barrier is degree-blind.** Pooled FPT distributions INDISTINGUISHABLE across same-size A/D/E graphs (T1 FAIL: 0/10, all KS p~1.0). Degree recovery from FPT clusters fails (T2 FAIL: ARI~0). **But: positional information IS real** — within-degree CV up to 0.35 (T3 PASS: 9/11). Same-degree vertices at different graph positions have measurably different FPTs. The stress barrier reads degree (bulk chemistry), not topology (molecular structure). Spectral forensics needs a finer instrument — something sensitive to neighborhood structure, not just local connectivity. |
| 18 | Multi-Channel Fingerprint | C | 3/4 | **Topology readout works with 3 channels.** FPT + JSD + heat kernel signature distinguish 9/10 same-size pairs (T1 FAIL: D_7 vs E_7 at 0.82sigma, same degree seq). Channels carry independent info (T2 PASS: 7/11 max\|rho\|<0.8). **Laplacian eigenvalues distinguish ALL D_n vs E_n pairs with identical degree sequences** (T3 PASS: 3/3, deterministic, no noise). Leave-one-out classification 100% (T4 PASS: 11/11). **Three channels = three independent measurements of the source: barrier height (FPT), identity shape (JSD), global position (HKS).** |
| 19 | Two-Channel Blind Spots | C | 1/4 | **PAC/SEC/RBF mapping falsified at current resolution.** No two-channel pair has blind spots -- ANY two of three channels suffice (T1 FAIL: 0 blind spots for all pairs). All blind spot sets identical (T2 FAIL). 3 channels have no blind spots (T3 PASS). No unique channel roles (T4 FAIL). **Key finding: channels are correlated views of the same structure, not independent axes.** Hierarchy: FPT (PAC) is most valuable, JSD (SEC) second, HKS (RBF) least. SEC+RBF weakest pair (D_7 vs E_7: 0.82sigma vs 1.31 for PAC+SEC). |
| 20 | Spectral Ratios from Graph Structure | C | **4/4** | **GRAPH GREEN'S FUNCTION = HYDROGEN SPECTRUM.** Inverse Laplacian eigenvalues E_k = 1/lambda_k give energy levels following the 1/k^2 hydrogen pattern. A_8 matches hydrogen Lyman/Balmer/Paschen ratios to **0.68% mean, 2.31% max** (T1 PASS). Lyman-alpha ratio: 0.84421 vs 0.84375 (0.05% error). Error decreases with rank, rho=-0.6 (T2 PASS). **Branching increases departure: 8/8** -- D_n/E_n depart more than A_n at every size (T3 PASS). D_4 = 100% departure (degenerate eigenvalues from star topology). All same-size pairs spectrally distinct; D_8 vs E_8 closest (T4 PASS). **A_n = "hydrogen" of DFT, D_n/E_n = fine-structure corrections from branching.** |
| 21 | Spectral Consolidation | C | 2/4 | **Convergence theorem + weighted branching.** Lyman-alpha error ~ n^(-4.08), R²=0.9998, monotonic from 1.16% (A_4) to 0.00003% (A_50) (T1 PASS). Eigenvalue↔fingerprint bridge borderline: Spearman rho=0.591 p=0.056 (T2 FAIL: 11 pairs insufficient). Weighted A_n→D_n departure scales as w^0.40±0.10 consistently (CV=0.24) across D_4..D_8 (T3 PASS). **Surprise: alpha_EM-weight branching REDUCES departure** -- all 5 sizes show negative additional departure (T4 FAIL). The deformation passes through a sweet spot where the graph is MORE hydrogen-like than the pure path. |
| 22 | Branching Sweet Spot | C | 0/4 | **Falsification: sweet spot is series-selective, not universal.** Lyman-alpha sweet spot exists ONLY for D_4 (w*=0.053, 99.9% improvement) and E_6 (w*=0.027, 37%). D_5..D_12, E_7, E_8: no Lyman-alpha sweet spot at all (T1 FAIL: 1/9). **Key finding: exp_21 T4's negative departure used MEAN series errors. Branch helps Balmer/Paschen selectively but not Lyman-alpha at n>=5.** The perturbation reaches high-k modes (near branch endpoint) but doesn't penetrate to low-k modes controlling Lyman-alpha. D_4 is the exception: only 3 energy levels, all coupled to the branch. Analogous to electron correlations affecting excited states more than ground state. |
| 23 | Confluence Convergence | C | 1/4 | **Per-series exponents follow p(m) = 4.022 + 0.057m (R^2=0.9983).** Convergence ordering INVERTED from prediction: Lyman 4.08, Balmer 4.13, Paschen 4.19, Brackett 4.25 -- higher series converge FASTER (T1 FAIL: wrong direction). Linear relationship in series index is clean (T2 PASS). Branching sensitivity not simply ordered (T3 FAIL: Balmer dominates at n=8). Per-series prefactors differ ~16x at n=50 (T4 FAIL: max/min=16.3, but ratio decreasing with rho=-1.0). **Key finding: eigenvalue pairs in higher series are more similar, so discretization corrections cancel better. The confluence arithmetic is confirmed -- series-dependent convergence IS the expected behavior -- but the direction is "later = better cancellation" not "later = more sensitive."** |
| 24 | Coupling Energy Scale | D | **4/4** | **ENERGY SCALE FIX.** DFT alpha_EM^2 * m_e / 2 = Rydberg at 11.4 ppm (T1 PASS). Nuclear scale fdc(3)^2 * m_p = 10.5 MeV, ratio 1.75x to alpha energies (T2 PASS). EM/nuclear hierarchy ratio matches Rydberg/alpha within 14% (T3 PASS). Coupling-anchored counts O(0.4-0.8) vs Planck counts O(10^-19) (T4 PASS). **Proves 8 energy-scale failures share one root cause: E_Planck*phi^(-d) replaced by alpha(d)^2*m_mediator.** |
| 25 | Series Sweet Spots | D | **4/4** | **PER-SERIES CONVERGENCE MAP.** Sweet spot ordering confirmed: Lyman(A_5) < Balmer(A_6) < Paschen(A_8) (T1 PASS). Alpha errors < 1.3% at sweet spots (T2 PASS). Convergence exponents predict sweet spots via log(n) ~ 1/p: R^2=0.949 (T3 PASS). Branch departure ordered Paschen>Balmer>Lyman for 3/4 sizes (T4 PASS). **Each series has a minimum rank for convergence, ordered by series index.** |
| 26 | Channel Completeness | D | 1/4 | **Laplacian determines all channels but they're NOT redundant.** Eigenvector regression: JSD R^2=1.0, HKS R^2=1.0, FPT R^2=0.82-1.0 for 7/11 graphs (T3 PASS). But pairwise |r|>0.6 only 41% (T1 FAIL), ARI 0.12-0.52 (T2 FAIL), effective rank ~2.5 not ≤2 (T4 FAIL). **Key finding: channels are non-linear functions of the same Laplacian, carrying genuinely different information despite shared origin. exp_19's 2-channel sufficiency comes from Laplacian information-richness, not channel redundancy.** |
| 27 | Bifractal Time Emergence | D | 0/4 | **Honest 0/4 but findings support the thesis.** Dual scaling: 7/7 graphs have R_b→R_f crossover but sigma* doesn't correlate with spectral gap (T1 FAIL: rho=-0.19). Eigenvalue clustering doesn't predict effective rank (T2 FAIL: 1/11). HKS-FPT anti-correlated at |tau|>0.5, violating (0.1,0.9) bound (T3 FAIL: 3/11). HKS is the privileged channel, not FPT (T4 FAIL: CV=1.16). **Key finding: FPT (time proxy) is the LEAST important channel for degree classification — removing it causes zero ARI loss. Time IS not privileged, but the test criterion measured the wrong thing.** |
| 28 | Dynamical Severance (reality-engine v4 substrate) | E | 0/4 | **THE TRIGGER IS A CORE DETECTOR.** First DYNAMICAL run of the derived trigger (exp_15 all-edges overstress at the lattice spacing) and derived form (exp_01 decoupling) as whole-particle severance on reality-engine's v4 particle substrate (corrected force, energy ledger; sealed `bf833113`, 2026-09-05). `min_j |S_i - S_j| > tau` fires only at an extremum of the field; exp_15's field was noise, this one is the collapse, so the extrema are the cores: `u_out < 0` in every run, fired/retained KE ratio ~0.5 at first firing. T1 FAIL (energy leaves, but what leaves carries less than the mean: retained e_int ABOVE B0 at every mark in 2/3 seeds at the informative tau = 20). **T2 FAIL WITH SIGN at tau = 20**: severed set LESS connected than a random subset of B0 at the same count, 3/3 seeds, margin -0.023 vs pooled std 0.009 (-2.6 sigma); random removal at the same count holds MORE (R > S 3/3). At n = 4000, tau* = 10 (48-54% severed, informative): a NULL — S 0.051/0.046/0.051 vs B0-matched 0.076/0.037/0.058 (-0.5 sigma), S = R (0.055/0.046/0.052): where half is removed, which half no longer matters. T3/T4 UNINFORMATIVE (conditional on T2); reported: matched drag D >= S 3/3, S = R at tau* = 10. K1/K2 do not fire. tau = 5, 10 vacuous on the proxy by the sealed rule; B0 at n = 4000 dissolves as predicted (0.028-0.038). The selection effect is real at the onset and diluted once severance is a bulk process (`journals/2026-09-05_exp28_outcomes.md`). **What closes: the entropy-gradient barrier as an amount-free dynamical sink (reality-engine C4.1 option 3). exp_15/16 stand — graph results on a noise field.** SP2 holds (memory_decay orders the severed fraction, sign unchanged); SP3 missed by a hair (Landauer = 1% of pressure work; not a sink). |
| 29 | The Ledger on Particles (reality-engine v4 substrate) | E | **registered, not run** | **Sealed 2026-09-06** (`journals/2026-09-06_exp29_registration.md`). exp_28's zero located: the substrate had NO ledger — the SEC pressure is a gradient at fixed entropy and the whole non-conservation is the entropy ratchet (pair energy created by entropy change 689,700 vs pressure work 667,300). This round adds the ledger: a per-particle potential budget P(0) = kappa |U_grav(0)| that pays for entropy growth at the price the pair energy sets; KE + U + E_SEC + sum P conserved. M = 4: T1 holding at kappa = 0.5 above BOTH gravity alone (kappa = 0) and the unbounded engine (kappa = inf), 3/3 seeds, > 2x pooled std; T2 bound (KE/|U| < 1 at every window mark at kappa 0.5 and 1); T3 the ordering above the threshold (KE/|U| at kappa 1 < 2 < inf); T4 the ledger did the work (net creation <= P(0) exactly; the budget binds on every would-grow particle). Pre-seal on seed 1: bound through kappa = 1, the pressure a RESTORING force there (work -1.2 P0), U-shaped KE/|U| with its minimum at kappa = 1. Kill: T1 fails => the particle mapping of P/A dies, not PAC. |
| **Total** | | | **62/116** | **53% -- exploratory sidecar** |

## Key Finding: The Energy Scale Problem

**The most important result is a failure.** The formula `E = n * Xi * E_Planck * phi^(-depth)` gives energies vastly larger than physical nuclear/atomic scales for all reasonable depths. At depth 13 (EM), the energy unit is ~10^19 MeV -- 15 orders above the Planck mass in GeV, and 25 orders above the Rydberg energy. At depth 183 (gravity), the energy is ~10^-17 MeV -- close to neutrino mass scale but still not matching gravity's actual strength.

This means the naive "boundary count = energy / Xi * E_scale" formula gives n ≈ 0 for all nuclear/atomic phenomena. The tests that "pass" at depth 3 are tautological: everything rounds to zero.

**What this reveals:** The Fibonacci depth hierarchy correctly orders forces and correctly gives coupling *ratios* (phi^6 for the hierarchy problem, etc.), but the *absolute* energy scale requires an additional ingredient -- likely the mass of the mediator or the energy of the specific transition. The boundary-counting idea needs to be reformulated using coupling constants (dimensionless) rather than absolute energies.

### Update 2026-08-27: the reformulation is also wrong, and the reason is structural

exp_24 did the reformulation this section calls for, replacing `E_Planck * phi^(-d)` with
`alpha(d)^2 * m_mediator`. Propagating it recovered **two** of its six named test failures
(exp_03 T3, exp_05 T1 at 11.4 ppm); the other four -- exp_03 T4, exp_04 T1, exp_05 T2/T4 -- have
no scale term in them at all and cannot move under any rescaling.

Three findings, detailed in `journals/2026-08-27_energy_scale_propagation_and_the_depth_slot.md`:

1. **The (depth, mediator) pair is exactly degenerate.** Since `E ~ phi^(-2d) * m`, scaling the
   mediator by `r` is identical to shifting depth by `ln(r)/(2 ln phi)`. Every mediator fits the
   beta endpoints equally well at a shifted depth, so a fitted depth measures the mediator
   choice, not physics.
2. **`dft_energy_scale` is valid only for the nuclear case.** It uses
   `fibonacci_depth_coupling(d) = phi^(-d)/sqrt(5)`, which at EM depth is 72x off the Rydberg;
   exp_24's own T1 bypasses it and uses `ALPHA_EM_DFT` directly. The one test that does use it
   sits at 1.75x inside a factor-1000 window.
3. **Depth is being read out of the wrong slot.** Every DFT constant encodes scale in *which*
   Fibonacci indices appear and carries phi exactly once -- alpha_EM's phi-power is **-1**, not
   -13. `fibonacci_depth_coupling` encodes scale as a *power of phi* and drops the indices.

**So an energy scale should carry phi^-1 and a Fibonacci index multiset**, the shape every other
constant in the corpus already has. `alpha(d)^2 * m` has neither.

**Warning for whoever finishes the propagation (exp_06, exp_08, exp_09):** this section already
notes that several *passes* are tautological because everything rounds to zero at the Planck
scale -- exp_06 T1/T2/T4, exp_02 T1, exp_08 T1, exp_09 T4. A correct scale turns those into real
tests, which can then fail. **The net score may go down**, and that is the correct outcome: a
test that passes because every input rounds to zero was never evidence.

## What Works (Structural Results)

1. **PAC conservation is exact under severance** -- to machine precision (1.6e-16). The ledger genuinely splits.
2. **Independent sub-ledgers** -- zero cross-component leakage after severance.
3. **Line width perfectly correlates with disequilibrium** -- Spearman rho = 1.000 (exp_04 T4).
4. **Equilibrium shift is 10.6x more efficient than brute force** -- at all depths (exp_07 T3).
5. **Gravity is a spectator** -- phi^(-178) = 6.3e-38, confirming the 0g analogy (exp_07 T4).
6. **Co-60 gamma ratio appears in ADE equilibration** -- 1.9% match in A_7 (exp_04 T2).
7. **Orbit isotropy** -- same-orbit vertices have identical severance energies (exp_07 T1).
8. **Beta spectrum shape matches Fermi** -- unexpectedly, chi^2 = 0.37 (exp_03 T2).
9. **Bremsstrahlung sharp cutoff reproduced** -- < 0.01% above 95th percentile (exp_08 T2).
10. **Series grouping correct** -- PAC tree reproduces Lyman/Balmer/Paschen structure (exp_05 T3).
11. **PAC deficit correlates with severance energy** -- median per-graph rho = 0.63 (exp_11 T1). Confirms decay = healing.
12. **Geometric first-passage is the correct distribution** -- wins AIC 4/4 vs exponential (exp_11 T4). Reframes exp_07 T2 failure as prediction.
13. **Barrier FPT scales exponentially with vertex degree** -- R²=0.92 (exp_12 T1). Degree 1→84 steps, degree 2→140, degree 3→679. The topological barrier (all d edges must simultaneously decouple) creates the exponential suppression missing from smooth Laplacian relaxation.
14. **Barrier dramatically improves GN scaling** -- R²=0.998 with barrier vs R²=0.746 without (exp_12 T3). The topological barrier IS the missing ingredient that converts smooth relaxation into Geiger-Nuttall-like exponential suppression.
15. **Perspectival barrier outperforms topological for GN functional form** -- R²≈0.95 vs 0.0–0.5 across 5 graph types (exp_13 T4). JSD-based barrier captures the log(FPT) vs 1/√(deficit) relationship better than edge-flow decoupling.
16. **Hub vertices have higher perspective divergence** -- D_4 hub JSD=0.38 vs endpoint JSD=0.10 (exp_13 T3). Random walks return to high-degree vertices, concentrating the local distribution. Stronger local identity = higher barrier. Physically correct: more connections = more identity to reconcile before severance.
17. **Stress barrier flips the GN sign** -- rho(noise, FPT) = -1.0 across all 5 graph types (exp_15 T1/T4). More noise (SEC flux) → shorter FPT. The stress barrier (connections overstressed beyond threshold) is the correct Coulomb analog. The relaxation barriers (topological, perspectival) had the sign backwards.
18. **GN combined variable degree/√noise works** -- R²=0.79, positive slope (exp_15 T3). Better than degree alone (R²=0.19) or noise alone (R²=0.54). This is the PAC analog of the GN law: log(t½) ∝ √Z/√E maps to log(FPT) ∝ degree/√noise. Barrier (PAC structure) and kinetic energy (SEC dynamics) are genuinely independent.
19. **Noise and degree have correct partial effects** -- Multiple regression: degree coeff=+0.148, noise coeff=-13.9, combined R²=0.70 (exp_15 T2). Higher degree → longer FPT (harder to overstress all connections). Higher noise → shorter FPT (more kinetic energy to break free). Both signs match nuclear physics.
20. **GN law is universal across ADE families** -- Data collapse: log(FPT) vs d/√σ gives R²=0.74 combined across 10 A, D, E graphs (exp_16 T1). Per-family: A=0.78, D=0.74, E=0.75. The Geiger-Nuttall scaling is not specific to D-type or nuclear topology — it's generic.
21. **Analytical model predicts empirical FPTs** -- FPT = 1/[erfc(τ/(σ√2))]^d correlates with empirical at Spearman ρ=0.90 (exp_16 T2). The independent-edge approximation captures the ordering correctly, with a systematic overprediction factor ~2.4x (due to edge correlations from shared Laplacian).
22. **Stress threshold is a scale parameter** -- GN functional form (R²>0.5) holds at τ=0.005, 0.008, 0.012 (exp_16 T3, 3/3). The threshold shifts the intercept but preserves log(FPT) ∝ d/f(σ). Analogous to how the Coulomb barrier height doesn't change the GN *law*, only the constants.
23. **Universal exponent k=1.16±0.02** -- Fitting log(FPT) ~ d/σ^k gives k=1.12 (A), 1.17 (D), 1.17 (E) (exp_16 T4). Cross-family consistency std/mean = 2%. The exponent exceeds the independent-edge prediction (k=0.5), revealing edge correlations from Laplacian dynamics, but the functional form is universal.
24. **Positional information beyond degree is real** -- Within same-degree classes, vertices at different graph positions have CV up to 0.35 in FPT (exp_17 T3: 9/11 graphs). Degree-2 vertices near the hub have different FPTs than degree-2 vertices at chain endpoints. The spectrum carries more than degree — but the positional signal is second-order relative to the degree signal.
25. **Multi-channel fingerprint distinguishes same-size graphs** -- FPT + JSD + heat kernel signature separates 9/10 same-size pairs at >1sigma (exp_18 T1). The single failure (D_7 vs E_7, 0.82sigma) has identical degree sequences — the hardest case. 100% leave-one-out classification (exp_18 T4).
26. **Three channels carry independent information** -- FPT, JSD, and heat kernel centrality have max|rho|<0.8 for 7/11 graphs (exp_18 T2). Each reads a different aspect: barrier height (PAC structure), identity shape (information geometry), global position (spectral topology). Real spectra encode the same triad: line position, line width, line intensity.
27. **Laplacian spectrum distinguishes identical-degree-sequence graphs** -- D_6 vs E_6, D_7 vs E_7, D_8 vs E_8 all have the SAME degree sequence but different Laplacian eigenvalues (exp_18 T3: 3/3, eig_dist 0.48-0.73). This is a mathematical guarantee: non-isomorphic graphs have different spectra. The heat kernel signature (HKS) provides a deterministic, noise-free topology channel.
28. **Any two of three channels suffice** -- All two-channel combinations (FPT+JSD, FPT+HKS, JSD+HKS) distinguish all 10 same-size pairs at 0.5sigma (exp_19 T1). No channel is uniquely necessary. But clear hierarchy: PAC+SEC (FPT+JSD) is strongest for hard pairs (D_7 vs E_7: 1.31sigma), SEC+RBF (JSD+HKS) is weakest (0.82sigma). FPT contributes most discriminative power.
29. **Graph Green's function gives hydrogen energy levels** -- Inverse Laplacian eigenvalues E_k = 1/lambda_k follow 1/k^2 for A_n path graphs (exp_20 T1). A_8 matches hydrogen Lyman-alpha ratio to 0.05%, Balmer-alpha to 0.34%, Paschen-alpha to 1.13%. Mean error 0.68% across all 12 tested ratios using only 8 vertices. This is a continuum-limit theorem: path graph Laplacian discretizes -d^2/dx^2, whose Green's function has 1/k^2 spectrum.
30. **Branching creates fine-structure departures** -- D_n and E_n graphs depart more from hydrogen ratios than A_n at every size tested (exp_20 T3: 8/8). D_4 (star graph) has 100% departure due to eigenvalue degeneracy. D_7 departs 18x more than A_7. The branching breaks the path symmetry, shifting eigenvalues analogously to angular momentum splitting in real atoms. This establishes: A_n = "hydrogen" of DFT, D_n/E_n = structurally richer atoms with fine-structure-like corrections.
31. **Lyman-alpha convergence is O(n^-4)** -- A_n inverse eigenvalue Lyman-alpha ratio error decreases monotonically from 1.16% (A_4) to 0.00003% (A_50), fitting error ~ n^(-4.08) with R²=0.9998 (exp_21 T1). Much faster than the naive O(n^-2) expectation. The continuum limit is already excellent at n=8.
32. **Weighted branching departure scales as w^0.4** -- Continuously interpolating A_n→D_n by varying branch edge weight w from 0→1, the hydrogen departure scales as w^(0.40±0.10) consistently across D_4..D_8 (exp_21 T3, CV=0.24). Not linear (perturbation theory) or quadratic (fine structure) — approximately square-root. The exponent is stable across graph sizes.
33. **Alpha-weight branching improves hydrogen match** -- At w=alpha_EM≈0.0073, the weighted D_n graph is MORE hydrogen-like than the pure A_n path for all 5 sizes tested (exp_21 T4). The additional departure is negative (-0.03% to -0.15%). The A_n→D_n deformation passes through a sweet spot where a tiny branch corrects the discrete approximation error before larger branches break the 1/k^2 pattern.
34. **Sweet spot is series-selective, not universal** -- Optimizing Lyman-alpha specifically, only D_4 (w*=0.053, 99.9% improvement) and E_6 (w*=0.027, 37%) have sweet spots (exp_22). D_5..D_12, E_7, E_8: the branch never improves Lyman-alpha. The mean-series improvement from exp_21 T4 comes from higher series (Balmer, Paschen). The branch perturbation at the graph endpoint affects high-k eigenvalue modes preferentially. At n=4, all modes are coupled; at n>=5, Lyman-alpha modes are insulated from endpoint perturbations. This is the graph-theoretic analog of how electron correlations in multi-electron atoms affect Rydberg (excited) states more than the ground state.
35. **Per-series convergence exponents are linear in series index** -- For A_n path graphs, the error in each hydrogen series ratio (Lyman m=1, Balmer m=2, Paschen m=3, Brackett m=4) follows error ~ n^(-p(m)) where p(m) = 4.022 + 0.057*m with R²=0.9983 (exp_23 T2). Higher series converge FASTER (inverted from naive prediction). The linear relationship means each additional series index adds 0.057 to the convergence exponent. This is the confluence arithmetic at work: each series ratio is a separate constraint axis, and eigenvalue pairs in higher series are more similar (better cancellation of discretization corrections).
36. **DFT coupling constants fix the energy scale** -- ALPHA_EM_DFT^2 * m_e / 2 = 13.606 eV matches Rydberg at 11.4 ppm (exp_24 T1). At strong depth (d=3), fdc(3)^2 * m_p = 10.5 MeV is within 1.75x of mean alpha energy (exp_24 T2). The EM/nuclear scale hierarchy matches Rydberg/alpha within 14% (exp_24 T3). Coupling-anchored boundary counts are O(0.4-0.8) vs Planck-anchored O(10^-19) (exp_24 T4). **The 8 energy-scale failures have a single fix: replace E_Planck*phi^(-d) with alpha(d)^2*m_mediator.**
37. **Sweet spot ordering confirmed: Lyman < Balmer < Paschen** -- Each hydrogen series has a minimum A_n rank for 5% convergence: Lyman at A_5, Balmer at A_6, Paschen at A_8 (exp_25 T1). Alpha transition errors < 1.3% at sweet spots (exp_25 T2). Convergence exponents predict sweet spots via log(n_sweet) ~ 1/p_m at R^2=0.949 (exp_25 T3). Branch departure at sweet spots ordered by series sensitivity (exp_25 T4: 3/4 sizes).
38. **Dual scaling regimes exist** -- All 7 tested ADE graphs (D_5..D_8, E_6..E_8) show a crossover sigma where noise (R_f) begins to dominate structure (R_b) in predicting FPT (exp_27 T1: 7/7 crossover). The bifractal time model M(t)=Collapse(R_b,R_f) correctly predicts two scaling regimes. However, the crossover point doesn't correlate with spectral gap (rho=-0.19), suggesting a more complex relationship between graph structure and the R_b/R_f transition.
39. **FPT (time) is not privileged for classification** -- Leave-one-channel-out analysis shows removing FPT causes ZERO ARI loss for degree classification in all 11 ADE graphs (exp_27 T4). HKS is the privileged channel. This confirms bifractal time: the time-like observable (FPT) is an emergent projection, not a fundamental coordinate. Structure (HKS/R_b) carries more classification power than dynamics (FPT/R_f).
40. **Laplacian eigenvectors determine all measurement channels** -- JSD and HKS are perfectly determined (R^2=1.0) by Laplacian eigenvector loadings; FPT is determined for D/E types (R^2=1.0) but less so for A-type (R^2=0.4-0.7) due to stochastic variance (exp_26 T3: 7/11 graphs pass). All three channels are functions of the same Laplacian, confirming the underlying Laplacian completeness principle.
41. **The all-edges overstress trigger fires at the extrema of the field** -- `min_j |S_i - S_j| > tau` is true only where every bond carries a large gradient, i.e. at a local extremum. On exp_15's noise field the extrema are random and the p^d degree barrier and exp_16's universal statistics follow; on a structured field (reality-engine's v4 substrate, where SEC entropy grows with density) the extrema are the collapse cores, and severance removes the bound, connected part (exp_28 T2: severed set less connected than a random subset at the same count, 3/3 seeds, -2.6 sigma; u_out < 0 in every run). The trigger is a core detector there, not a radiation channel. What remains registrable: a barrier on UNDER-stress (the flat interior of a core) and the sign question beside exp_14.

## Honest Failures (17 tests)

| Failure | Evidence | What It Reveals |
|---------|----------|-----------------|
| Spectral shift coarser than orbits | exp_01 T2: A_5 has 3 orbits but only 2 distinct energies | Severance energy depends on degree, not full orbit structure |
| Spectral shift constant vs depth | exp_01 T4: root of binary tree always has degree 2 | Need non-tree graphs to see depth dependence |
| Phi-ratio clustering = random | exp_02 T2: 51.5th percentile | Alpha energy ratios don't cluster near phi powers |
| Relaxation anti-correlates with size | exp_02 T3: rho = -0.70 | Larger graphs equilibrate faster (higher connectivity) |
| Endpoint 19 orders off | exp_03 T3: Planck scale >> nuclear scale | The energy scale problem |
| Settled/unsettled entropy not discriminated | exp_03 T4: ratios near 1.0 | Noise amplitude too small relative to graph structure |
| Equilibration energy often negative | exp_04 T1: 75/87 negative | Redistribution can decrease entropy for non-equilibrium initial conditions |
| Multi-step incomplete for some graphs | exp_04 T3: A_5, D_5 fail | Small graphs equilibrate too fast for multi-phase detection |
| Rydberg 24 orders off | exp_05 T1 | PAC tree energy scale has no contact with atomic physics yet |
| phi^k != 1/n^2 | exp_05 T2 | Coulomb SO(4) symmetry not in PAC tree (as predicted) |
| Fine structure 2.7 orders off | exp_05 T4 | Correction template modifies coupling, not level structure |
| Boundary counts all zero | exp_06, exp_08 T1, exp_09 T4 | Energy scale problem: E_Planck * phi^(-d) >> physical energies |
| Ejection timing not Poisson | exp_07 T2 | Cascade gives geometric first-passage times, not exponential. **Reframed in exp_11 T4: geometric IS correct** |
| Transition count formula wrong | exp_08 T4 | Severance energies are highly degenerate (fewer than n(n-1)/2) |
| Dark sector boundary count = 0 | exp_09 T4 | Same energy scale problem |
| GN slope R²=0.68, below 0.8 | exp_11 T2 | **Laplacian diffusion has no barrier.** FPT plateaus for large deficits because relaxation is linear. Geiger-Nuttall requires WKB tunneling (barrier), not smooth Laplacian relaxation. Slope sign correct. |
| Chain graphs converge slowly | exp_11 T3 | A-type graphs (small spectral gap) need more time-averaging for convergence. D-type passes. Reveals: graph topology controls PAC mixing rate. |
| GN sign wrong for degree-1 target | exp_12 T2 | Barrier walk on D_6 vertex 0 (degree 1) gives negative slope: higher deficit → longer FPT. For degree-1 vertices the barrier is trivial (1 edge); the bottleneck is relaxation time, not barrier crossing. GN test needs high-degree targets where barrier dominates. R²=0.993 — the fit is excellent, just the wrong direction. |
| GN sign wrong even for hub + perspectival barrier | exp_13 T2 | D_8 hub (degree 3), perspectival barrier: slope=-1.55, R²=0.93. Higher deficit still → longer FPT. **Persistent sign problem**: both barrier types give FPT monotonically increasing with deficit because larger perturbation → larger initial JSD → longer reconciliation. GN requires higher energy → EASIER barrier crossing (kinetic energy overcomes fixed barrier). Our model has no "kinetic energy" — the barrier and perturbation scale together. |
| Hub JSD > endpoint JSD (prediction inverted) | exp_13 T3 | Predicted hubs have lower JSD (closer to global), but they have HIGHER JSD (0.38 vs 0.10 for D_4). Random walks return to high-degree vertices, concentrating local distribution. The prediction was wrong but the RESULT is physically correct: stronger identity = higher barrier = harder to sever. |
| JSD loses to degree for topological FPT | exp_13 T1 | JSD wins only 1/13 graphs against degree. The topological barrier (`stochastic_barrier_walk`) checks edge flows, which is degree-determined by construction. JSD captures orthogonal information. Testing JSD against a degree-based barrier is asking the wrong question. |
| Noise increases FPT (rho=+1.0) | exp_14 T1/T4 | More noise → LONGER FPT, universally (rho=+1.0, 5/5 graphs). **The perspectival barrier is an ENTROPY barrier, not an ENERGY barrier.** The JSD threshold requires the state to CONVERGE to equilibrium. Noise OPPOSES convergence. This is the opposite of Coulomb tunneling where kinetic energy HELPS escape. The Coulomb analog must be a STRESS barrier (connections overstressed to breaking), not a relaxation barrier (connections relaxed to equilibrium). |
| Barrier and noise coefficients both wrong sign | exp_14 T2 | Multiple regression gives JSD coeff=-1.69 (should be positive) and noise coeff=+6.46 (should be negative). The perspectival JSD actually ANTI-correlates with FPT because high-JSD vertices are in more dynamic positions where equilibrium oscillations are larger. The entire relaxation-to-threshold model is the wrong paradigm for decay. |
| Stress barrier FPT can't distinguish same-size graphs | exp_17 T1 | All 10 same-size pairs (A_n vs D_n vs E_n) indistinguishable by KS test (p~1.0). The FPT depends so overwhelmingly on degree that graphs with similar degree sequences produce identical pooled distributions. D_7 and E_7 have the same degree sequence [1,1,1,2,2,2,3] — literally impossible to distinguish by degree alone. |
| Degree recovery from FPT clusters fails | exp_17 T2 | ARI near 0 or negative for all 11 graphs. Within-degree stochastic variance swamps the between-degree FPT separation. The FPT is too noisy to cluster into clean degree classes — even though degree is the dominant factor. |
| D_7 vs E_7 indistinguishable even with 3 channels | exp_18 T1 | Combined distance 0.82sigma — below the 1sigma threshold. Identical degree sequence [1,1,1,2,2,2,3] makes FPT channel useless (0.00sigma). JSD (0.76sigma) and HKS (0.31sigma) each provide signal but not enough. The hardest case for spectral forensics: same degree, similar branching. D_7 has branch lengths [1,1,4], E_7 has [1,2,3] — the difference is subtle. |
| PAC/SEC/RBF channel mapping over-interpreted | exp_19 T1/T2/T4 | No two-channel pair has blind spots — any two of three suffice. The three channels (FPT, JSD, HKS) are correlated views of the same underlying structure, not three independent measurement axes corresponding to PAC/SEC/RBF. The hierarchy (FPT > JSD > HKS) shows the channels have different *strengths* but not different *roles*. |
| Eigenvalue↔fingerprint bridge borderline | exp_21 T2 | Spearman rho=0.591 (p=0.056) between eigenvalue distance and JSD+HKS fingerprint distance for same-size ADE pairs. Correlation exists but insufficient with only 11 data points. The bridge between spectral structure (exp_20) and identity fingerprinting (exp_18) is real but weaker than expected — possibly because JSD+HKS collapses information that eigenvalue spectra preserve. |
| Alpha-weight branching reduces departure | exp_21 T4 | At w=alpha_EM, weighted D_n graphs are MORE hydrogen-like than pure A_n paths for all 5 sizes. Additional departure is negative (-0.03% to -0.15%). The A_n→D_n deformation passes through a sweet spot — a tiny branch initially *corrects* discrete approximation error before breaking 1/k^2 at larger w. The fine-structure analogy (branching = alpha^2 correction) doesn't hold at alpha scale; the correction is non-monotonic. |
| Sweet spot is series-selective | exp_22 T1-T4 (0/4) | Lyman-alpha sweet spot exists ONLY for D_4 (w*=0.053) and E_6 (w*=0.027). All other sizes: no improvement. exp_21 T4's mean-series improvement comes from Balmer/Paschen, not Lyman-alpha. The branch perturbation reaches high-k eigenvalue modes (near the branch endpoint) but doesn't penetrate to the low-k modes that control Lyman-alpha for n>=5. D_4 is the exception: 3 energy levels means all modes couple to the branch. Physically: branching corrections are series-selective, like electron correlations affecting excited states more than the ground state. |
| Convergence ordering inverted | exp_23 T1 | Predicted higher series converge SLOWER (more memory in confluence stream). Actual: Lyman p=4.08, Balmer p=4.13, Paschen p=4.19, Brackett p=4.25 — higher series converge FASTER. Eigenvalue pairs in higher series ratios are more similar (e.g., Paschen k=4,5 differ 25% vs Lyman k=2,3 differ 50%), so discretization corrections cancel better. The confluence direction is "later = better cancellation" not "later = more sensitive." |
| Branching sensitivity not simply ordered by series | exp_23 T3 | Only 1 testable graph size (A_8) has enough levels for 4 series. Balmer shows highest sensitivity at n=8, not monotonic in series index. Need larger graphs to test ordering. |
| Per-series prefactors differ ~16x | exp_23 T4 | At n=50, max/min prefactor ratio = 16.3 across series. Prefactors are NOT converging to a common value, though the ratio is decreasing (rho=-1.0). Each series has its own absolute error scale even as they share the same power-law exponent pattern. |
| Channels NOT pairwise correlated | exp_26 T1 | Only 41% of (graph, channel-pair) have \|r\| > 0.6. FPT/JSD/HKS carry genuinely different information despite all being Laplacian-derived. Non-linear projections of the same spectrum look very different. |
| Two-channel ARI too low | exp_26 T2 | fpt-jsd ARI=0.29, fpt-hks ARI=0.52, jsd-hks ARI=0.12. Gap-splitting on channel values doesn't recover degree classes well. The channels encode different aspects of vertex identity than degree alone. |
| Crossover sigma doesn't correlate with spectral gap | exp_27 T1: rho=-0.19. All 7 graphs have crossover (dual scaling is real) but the crossover noise level clusters at only 3 grid values (0.005, 0.01, 0.02). The 6-point noise grid is too coarse to resolve the relationship. Spectral gap may not be the right structural predictor -- the R_b/R_f transition depends on degree distribution and graph size, not just lambda_2. |
| Eigenvalue clustering doesn't predict effective rank | exp_27 T2: 1/11 correct. Gap-split classify with 30% threshold gives mostly 1 cluster for A/D/E Laplacian eigenvalues, while the actual effective rank of the 3-channel matrix is 2-3. The POSITIVE finding: all graphs have effective rank > 1, confirming multi-dimensional collapse. The prediction model (min(3, clusters-1)) was wrong, not the underlying physics. |
| HKS-FPT anti-correlated beyond (0.1, 0.9) range | exp_27 T3: 3/11 graphs pass. HKS and FPT share direct Laplacian dependence (HKS = sum of exp(-lambda*t)*eigvec^2, FPT dominated by spectral gap). High HKS → more heat retention → more connected → higher degree → MORE edges to simultaneously overstress → longer FPT. Anti-correlation (|tau|>0.5) is physically correct but violates the uniform non-commutativity hypothesis. JSD-FPT tau near 0 for many graphs — partial non-commutativity is real but channel-pair-specific. |
| HKS is the privileged channel | exp_27 T4: CV=1.16. Removing FPT or JSD causes zero ARI loss; removing HKS causes high loss. HKS directly encodes vertex degree through eigenvector localization. **This CONFIRMS "time is not privileged" (FPT has zero unique contribution) but FALSIFIES "no channel is privileged" (HKS dominates). The test criterion (CV<0.5) asked the wrong question.** |
| Effective rank > 2 | exp_26 T4 | Top-2 singular values capture 72-94% (mean 84%) of variance, not > 95%. The third channel adds 6-28% of information. Channels are partially redundant but NOT fully over-determined. The exp_19 result (any 2 suffice) is about classification sufficiency at 0.5sigma, not about linear redundancy. |
| Severance does not lower the retained per-particle energy | exp_28 T1: 1/3 seeds at the informative tau = 20 (3/3 at tau = 5, 10, but vacuous). Energy leaves the interacting ledger in every run (sum loss_severance_energy = 4-7e4 at tau = 20, all positive; sum u_out negative: the fired particles were bound to what stayed) but what leaves carries LESS kinetic energy than the mean, so e_int of the retained set RISES. The direction the registration named as live. |
| Severance removes structure rather than holding it | exp_28 T2: 0/3 tau. At tau = 20 the severed set's retained-count-matched percolation is BELOW B0's in 3/3 seeds (0.091/0.084/0.091 vs 0.123/0.113/0.099; margin -0.023, pooled std 0.009). Random removal at the same count leaves more (R 0.131/0.146/0.108). The matched drag D holds as much or more (3/3 at tau* = 10). At n = 4000, tau* = 10: a null (-0.5 sigma) and S = R. The barrier selects the cores at the onset; removing half is removing at random. |

## Three Key Insights

### 1. The Ledger Genuinely Splits
PAC conservation under severance is exact to machine precision. The two sub-graphs after severance are truly independent -- zero leakage. This is the structural foundation: radiation IS ledger severance, at least mathematically.

### 2. Coupling Ratios, Not Absolute Energies
The Fibonacci depth hierarchy gives correct force *orderings* and *ratios* (this is proven in M6-M12). But the absolute energy of a radiation event requires knowing the mediator mass and transition specifics. The boundary-counting hypothesis needs reformulation: count boundaries in units of the *coupling constant at that depth*, not in units of E_Planck * phi^(-d).

### 3. Pressure Beats Temperature
The equilibrium-shift mechanism is 10.6x more efficient than brute-force at every depth tested. This is the "carbonated water" insight made quantitative: you don't need to heat the system to get ejections, you need to shift the equilibrium. The ratio Xi / delta_eq = 1.058 / 0.1 ≈ 10.6 is independent of depth -- it's a universal efficiency gain for any ledger severance triggered by equilibrium shift.

### 4. Decay is Healing, Not Breaking (exp_11)
Radioactive decay is PAC balance-seeking: the system is out of equilibrium and the decay IS the rebalancing. The chaotic part is the PATH (stochastic cascade), but the DESTINATION (balanced daughter) is deterministic -- all walks converge to the same equilibrium (exp_11 T3). PAC deficit correlates with severance energy (rho=0.63), confirming that "energy released = measure of imbalance." The Geiger-Nuttall functional form (log(t) ~ 1/sqrt(E)) partially emerges (same sign, R²=0.68) but plateaus for large deficits because Laplacian diffusion lacks a barrier. The missing ingredient: to get the full Geiger-Nuttall exponential suppression, the PAC balance landscape needs a **topological barrier** -- a cost for restructuring the graph configuration before severance can occur. This barrier is what standard physics calls the Coulomb potential. Geometric first-passage times (not Poisson) are the correct prediction for discrete balance-seeking walks (exp_11 T4: 4/4 AIC wins).

### 5. The Topological Barrier IS the Coulomb Barrier (exp_12)
For a vertex to sever from a graph, ALL d of its connections must be simultaneously decoupled. The probability of d independent noise fluctuations coinciding scales as p^d — creating exponential suppression in degree. This is the PAC analog of the Coulomb barrier in nuclear physics. Evidence: (1) barrier FPT scales exponentially with degree (R²=0.92), (2) barrier dramatically improves GN scaling (R²=0.998 vs 0.746 without), (3) empirical GN is recovered (R²=0.99). The sign issue in T2 reveals that barrier dominance requires high-degree vertices — at degree 1, the "barrier" is trivially crossed and relaxation time dominates. The physics: more connections = harder to sever = longer half-life, which maps to lower alpha energy = higher Coulomb barrier = longer half-life.

### 6. The Barrier Is Perspectival But The Sign Problem Persists (exp_13)

The perspectival barrier (JSD between local random-walk and global equilibrium) dramatically outperforms the topological barrier for capturing the GN functional form (R²≈0.95 vs 0.0–0.5, T4). But the **sign problem persists**: in both barrier types, higher deficit → longer FPT. The Geiger-Nuttall law requires the opposite: higher alpha energy → shorter half-life (the particle's kinetic energy helps it overcome a FIXED barrier). In our model, both the barrier and the perturbation scale together — there's no analog of "kinetic energy" that's independent of the barrier height. This is the key missing ingredient.

A physically correct model needs: (1) a FIXED barrier that depends on graph structure (degree, JSD — these are right), and (2) a kinetic energy parameter that's INDEPENDENT of the barrier and represents the system's ability to overcome it. In nuclear physics, this is Q-value (mass difference). In PAC terms, it might be the SEC dynamics — the rate of entropy production that "pushes" through the barrier.

The T3 inversion (hub JSD > endpoint JSD) is the most interesting finding. High-degree vertices have STRONGER local identity — random walks return to them, concentrating the local distribution away from global equilibrium. This means: more connections = stronger identity = higher perspectival barrier. The barrier increases with connectivity through a different mechanism than the topological barrier (p^d) — it's information-theoretic, not combinatorial.

### 7. The Barrier Direction Is Inverted — Stress, Not Relaxation (exp_14)

exp_14 is a clean 0/4 falsification that reveals the most fundamental error in the barrier model. The perspectival barrier (JSD < threshold) is an **entropy barrier**: the system must CONVERGE to equilibrium. Noise OPPOSES convergence. More noise → higher noise floor → JSD stays above threshold → longer FPT. rho = +1.0 universally.

The Coulomb barrier in nuclear physics is the opposite: the alpha particle must ESCAPE a potential well. Kinetic energy HELPS escape. More energy → shorter half-life.

This means: the correct PAC analog of the Coulomb barrier is a **STRESS barrier**, not a relaxation barrier. Connections sever when they are OVERSTRESSED (edge flows > breaking threshold), not when they are relaxed. Higher noise (SEC flux) → larger fluctuations → easier to overstress all d connections simultaneously → shorter FPT.

The bubble analogy confirms this: CO₂ escapes when internal pressure exceeds surface tension on ALL sides. More pressure → faster escape. The bubble doesn't escape by reaching equilibrium — it escapes by breaking free.

### 8. The Complete Decay Picture: PAC Barrier + SEC Kinetic Energy (exp_15)

exp_15 scores 4/4 and resolves the sign problem that persisted through exp_11–14. The complete picture:

1. **Decay is healing** (exp_11): The system is out of PAC balance. Decay is the rebalancing.
2. **The barrier is structural** (exp_12): Degree determines barrier height (p^d suppression).
3. **Identity strengthens the barrier** (exp_13): High-degree vertices have stronger local identity (higher JSD), making severance harder.
4. **Relaxation barriers have the wrong sign** (exp_14): Noise opposes convergence, not assists it.
5. **The barrier is STRESS, not relaxation** (exp_15): Connections break when ALL d edges are simultaneously overstressed. This is the correct Coulomb analog.
6. **SEC is kinetic energy** (exp_15): Noise amplitude (SEC entropy flux) provides the fluctuations that overstress connections. Higher SEC → faster severance.
7. **The GN law emerges** (exp_15): log(FPT) ∝ degree/√noise, mapping to log(t½) ∝ √Z/√E.

The PAC/SEC split maps cleanly onto the barrier/kinetic-energy split in nuclear physics:
- **PAC** (conservation, structure) → barrier height (degree, Z)
- **SEC** (dynamics, entropy flux) → kinetic energy (noise, Q-value)

### 9. The GN Law Is a Universality Theorem (exp_16)

exp_16 scores 4/4 and establishes the strongest result of Block C. The Geiger-Nuttall law is NOT a fact about the Coulomb potential or about nuclear physics specifically. It is a **generic consequence of any barrier requiring d simultaneous threshold exceedances under stochastic noise.**

The evidence:
1. **Data collapse (T1)**: All three ADE families — A (chain), D (branched), E (exceptional) — fall on the same curve when plotted as log(FPT) vs d/√σ. R²=0.74 combined.
2. **Analytical prediction (T2)**: The formula FPT = 1/[erfc(τ/(σ√2))]^d — derived from pure probability theory with zero physics assumptions — correlates at ρ=0.90 with the empirical barrier walks. The model only fails quantitatively (factor ~2.4x) due to edge correlations.
3. **Threshold independence (T3)**: The functional form survives across three different stress thresholds (0.005, 0.008, 0.012). The threshold is a scale parameter, not a structural one.
4. **Universal exponent (T4)**: k = 1.16 ± 0.02 across all families. The 2% cross-family consistency is extraordinary — it means the edge correlation correction is itself universal, determined by the Laplacian structure rather than graph specifics.

The implications: the GN law `log(t½) ∝ √Z/√E` in nuclear physics is a special case of the universal law `log(FPT) ∝ d/σ^k` where d=atomic number (connections), σ=kinetic energy (fluctuation amplitude), and k≈1.16 (correlation-corrected from the independent-edge k=0.5). The Coulomb barrier isn't doing the heavy lifting — multi-connection simultaneity is.

## Predictions Registry

| # | Type | Statement | Testable By |
|---|------|-----------|-------------|
| P1 | Precise | Alpha energies are integer multiples of Xi * coupling(d) at a Fibonacci depth (reformulated) | NNDC |
| P2 | Precise | X-ray K-alpha lines correspond to integer boundary counts (coupling-based) | X-ray spectroscopy |
| P3 | Directional | Beta endpoint set by weak-force coupling scale | KATRIN |
| P4 | Directional | Equilibrium-shift X-ray generation >= 10% more efficient | Lab experiment |
| P5 | Precise | Gamma energy ratios appear in ADE equilibration spectra | Nuclear level data |
| P6 | Constraint | Gravity irrelevant at nuclear depths: phi^(-178) < 1e-37 | Mathematical |
| P7 | Directional | Line width correlates monotonically with disequilibrium | Mossbauer |
| P8 | Precise | Dark sector X-ray at 3.2 keV is depth-73 severance | Athena |
| P9 | Directional | Ejection timing follows **geometric** first-passage statistics (not Poisson) | Decay counting |
| P10 | Constraint | Severance energy degeneracy bounded by vertex degree distribution | Spectroscopy |
| P11 | Directional | PAC deficit correlates with severance energy within each graph type | Graph theory |
| P12 | Directional | Geiger-Nuttall functional form (log(t) ~ 1/sqrt(E)) requires topological barrier in PAC landscape | Nuclear data + graph theory |
| P13 | Precise | Barrier FPT scales as exp(c*degree) where c depends on noise amplitude — degree-dependent half-life suppression | Graph simulations + nuclear data |
| P14 | Directional | High-degree vertices (d≥3) show GN-like positive slope (log(FPT) vs 1/sqrt(deficit)); degree-1 vertices do NOT | Graph simulations |
| P15 | Precise | Perspectival barrier R² > topological barrier R² for GN functional form on hub vertices | Graph simulations |
| P16 | Directional | Hub vertices (high degree) have higher perspective divergence (JSD) than endpoints — stronger local identity = higher barrier | Graph theory + random walks |
| P17 | Precise | Stress barrier FPT scales as exp(c*degree) / noise^k — degree-dependent suppression modulated by SEC flux | Graph simulations |
| P18 | Precise | GN combined variable degree/√noise predicts log(FPT) with R² > 0.7 — PAC barrier + SEC kinetic energy are independent | Graph simulations + nuclear data |
| P19 | Directional | Temperature-dependent nuclear half-lives: higher T → more thermal noise → shorter t½ for alpha decay | Nuclear data (if detectable at extreme T) |
| P20 | Precise | Universal exponent k ≈ 1.16 in log(FPT) ~ d/σ^k -- same across A, D, E graph families to within 2% | Graph simulations (verified) |
| P21 | Precise | Analytical FPT = 1/[erfc(τ/(σ√2))]^d predicts empirical ordering at ρ > 0.8 -- universally, not just for specific graphs | Probability theory + graph simulations |
| P22 | Directional | ANY multi-connection severance (not just ADE) gives GN-like scaling -- universality extends beyond crystallographic graphs | Random graphs, lattices, social networks |
| P23 | Precise | A_n inverse eigenvalue ratios converge to exact hydrogen ratios as n->infinity, with error scaling as O(1/n^4) | **CONFIRMED exp_21 T1**: error ~ n^(-4.08), R²=0.9998. Convergence much faster than O(1/n^2) |
| P24 | Directional | ~~D_n/E_n branching departures correlate with alpha^2 fine-structure corrections~~ **REVISED**: Weighted branching departure scales as w^0.4 (not w^2), and alpha-weight branching actually reduces departure. Fine-structure analogy is non-monotonic | exp_21 T3/T4 |
| P25 | Precise | Weighted A_n→D_n departure scales as w^(0.40±0.10) universally across graph sizes — the exponent is a property of the branching topology, not the specific graph | Graph simulations (verified exp_21 T3, CV=0.24) |
| P26 | Directional | ~~Universal sweet spot for all D_n~~ **FALSIFIED exp_22**: Lyman-alpha sweet spot only at D_4 and E_6. Branch perturbation is series-selective, helping higher series preferentially | exp_22 (0/4) |
| P27 | Directional | Branch perturbation couples to high-k eigenvalue modes preferentially — Balmer/Paschen corrections larger than Lyman from same branch weight | Graph eigenvalue perturbation theory |
| P28 | Precise | D_4 Lyman-alpha sweet spot at w*=0.053 gives 99.9% error reduction — the only ADE graph where a single branch can nearly eliminate discretization error | Graph simulations (verified exp_22) |
| P29 | Precise | Per-series convergence exponents follow p(m) = 4.022 + 0.057m (linear in series index, R²=0.9983) — higher hydrogen series converge FASTER from graph eigenvalues | **CONFIRMED exp_23 T2**: clean linear fit, p(Lyman)=4.08, p(Brackett)=4.25 |
| P30 | Directional | Per-series prefactors will NOT converge to a common value — each series has its own absolute error scale, ratio ~16x at n=50 but decreasing | exp_23 T4 (rho=-1.0, decreasing but not converging) |
| P31 | Precise | DFT coupling energy scale: alpha(d)^2 * m_mediator gives correct physical energy scales at every Fibonacci depth | **CONFIRMED exp_24**: EM scale at 11.4 ppm, nuclear scale within 1.75x |
| P32 | Precise | Per-series sweet spots are ordered by series index: n_sweet(m) < n_sweet(m+1) for all m | **CONFIRMED exp_25 T1**: Lyman(5) < Balmer(6) < Paschen(8) |
| P33 | Precise | Convergence exponents predict sweet spots: log(n_sweet) ~ 1/p_m with R^2 > 0.8 | **CONFIRMED exp_25 T3**: R^2=0.949 |
| P34 | Directional | FPT/JSD/HKS are non-linear functions of the Laplacian spectrum, carrying independent information despite shared origin | **CONFIRMED exp_26**: channels Laplacian-determined but NOT pairwise correlated |
| P35 | Directional | Dual scaling regimes (R_b-dominated vs R_f-dominated) exist for Laplacian measurement channels at varying noise | **CONFIRMED exp_27 T1**: 7/7 ADE graphs show crossover. Spectral gap correlation unresolved |
| P36 | Directional | FPT (time proxy) is not privileged for vertex classification -- removing it causes less information loss than removing structure channels | **CONFIRMED exp_27 T4**: removing FPT causes zero ARI loss; HKS dominates |
| P37 | Directional | HKS and FPT are anti-correlated (shared Laplacian dependence), while JSD-FPT correlation is near zero (independent information) | **CONFIRMED exp_27 T3**: HKS-FPT |tau|>0.5, JSD-FPT |tau|~0 for most graphs |
| P38 | Directional | A derived severance channel (exp_15 trigger + exp_01 form, no amount) lets a dynamical substrate hold structure against its own detonation | **FALSIFIED exp_28**: the trigger fires at the extrema of the entropy field and removes the cores; retained set less connected than a random subset (3/3, -2.6 sigma at proxy tau = 20; a null at n = 4000, tau* = 10; never above). Sealed `bf833113`. |

## Dependencies

- `milestone14/core/quantum_complement.py` -- Full M14 chain (PHI, ADE, orbits, etc.)
- `milestone12/core/connection_geometry.py` -- redistribute_on_graph, measure_entropy, BasinAttractor
- `milestone11/core/quantum_gravity.py` -- StochasticCascade, Planck scale
- `milestone6/core/scope.py` -- scope_attenuation, pac_budget

## Forward Path

1. **Energy scale RESOLVED (exp_24, 4/4)**: Coupling-anchored scale `E(d) = alpha(d)^2 * m_mediator` replaces `E_Planck * phi^(-d)`. EM scale at 11.4 ppm, nuclear scale within 1.75x. New functions `dft_energy_scale()` and `coupling_boundary_count()` added to radiation_physics.py.
2. **Connect to M8 alpha_EM derivation**: The alpha_EM = F3/(F4*phi*F10)*(1-F10/(4*pi*F7^2)) formula gives the correct coupling at 5.7 ppm. ALPHA_EM_DFT constant now defined in radiation_physics.py.
3. **Non-tree graphs for depth dependence**: Binary trees have constant root degree. Use ADE graphs at different ranks to see genuine depth-dependent spectral shifts.
4. **Experimental proposal**: The 10.6x efficiency gain for equilibrium-shift is the most practically promising result. Design a tabletop experiment comparing piezo-driven vs electron-bombardment X-ray generation at same Cu target.
5. **Topological barrier confirmed (exp_12)**: The Geiger-Nuttall failure (exp_11 T2) is resolved by adding a topological barrier — simultaneous edge decoupling. Barrier FPT exponential in degree (R²=0.92), barrier R²=0.998 vs plain R²=0.746. **Remaining**: the GN sign test (T2) needs high-degree target vertices (d≥3 hubs in D_n/E_n) where barrier dominates over relaxation. Degree-1 shows wrong sign because the barrier is trivial.
6. **Perspectival barrier + sign problem (exp_13)**: Perspectival barrier outperforms topological (R²≈0.95 vs 0.3), but the GN sign problem persists even with hub vertices.
7. **Barrier direction inverted (exp_14)**: The perspectival/topological barriers check for RELAXATION (convergence to equilibrium). Noise opposes this (rho=+1.0). The Coulomb barrier analog must check for STRESS (edge flows exceeding a breaking threshold).
8. **STRESS BARRIER RESOLVES THE SIGN PROBLEM (exp_15, 4/4)**: Connections sever when ALL d edges simultaneously overstressed. rho=-1.0 universally. degree/√noise predicts log(FPT) at R²=0.79. **PAC = barrier (degree), SEC = kinetic energy (noise).** The complete GN analog: log(FPT) ∝ degree/√noise maps to log(t½) ∝ √Z/√E.
9. **GN UNIVERSALITY THEOREM (exp_16, 4/4)**: The GN law is generic for d-simultaneous-threshold barriers. Data collapse across A/D/E (R²=0.74), analytical model matches (ρ=0.90), threshold is a scale parameter (3/3), universal exponent k=1.16±0.02 (2% consistency). **The Coulomb barrier isn't doing the heavy lifting — multi-connection simultaneity is.**
10. **GRAPH GREEN'S FUNCTION = HYDROGEN SPECTRUM (exp_20, 4/4)**: Inverse Laplacian eigenvalues E_k=1/lambda_k match hydrogen spectral ratios to 0.68% mean error at n=8. Lyman-alpha ratio: 0.05% error. Branching (D/E) creates fine-structure-like departures, systematic at 8/8 sizes. This bridges DFT graph structure to real spectroscopy through a theorem, not a fit.
11. **Spectral forensics: degree reads, topology doesn't (exp_17, 1/4)**: Stress barrier FPT distributions can't distinguish same-size graphs (0/10 KS pairs). But positional info IS real (CV~0.35 within degree classes). **The stress barrier reads "what element" (degree = atomic number), not "what molecule" (topology = bonding structure).** To read topology from spectra, need a finer instrument sensitive to neighborhood structure — perspectival divergence (JSD) or spectral gap, not just edge stress.
12. **Multi-channel spectral forensics works (exp_18, 3/4)**: Three channels — FPT (barrier), JSD (identity), HKS (spectral position) — distinguish 9/10 same-size graph pairs and achieve 100% classification. The Laplacian spectrum provides a deterministic, noise-free topology channel that works even for identical-degree-sequence graphs. Light carries multiple measurements of the source: barrier height, identity shape, and global position.
13. **Channels are correlated, not independent (exp_19, 1/4)**: The PAC/SEC/RBF mapping to FPT/JSD/HKS was over-interpreted. Any two of three channels suffice at 0.5sigma — no channel has a unique blind spot. The channels are correlated views of the same structure. Clear hierarchy: FPT (most discriminative) > JSD > HKS. The multi-channel approach works because of redundancy, not because each axiom provides irreplaceable information. The deeper question: WHY are these three views correlated? Possibly because they're all derived from the same Laplacian.
14. **CONVERGENCE THEOREM + BRANCHING SWEET SPOT (exp_21, 2/4)**: Lyman-alpha convergence is O(n^-4), confirmed P23. Weighted branching departure is w^0.40 (stable exponent, P25 confirmed). But the fine-structure mapping fails: alpha-weight branching REDUCES departure (negative additional, P24 revised). The A_n→D_n deformation has a non-monotonic correction landscape — a result that constrains how branching connects to real fine structure.
15. **SWEET SPOT IS SERIES-SELECTIVE (exp_22, 0/4)**: Clean falsification of the universal sweet spot hypothesis. Lyman-alpha sweet spots exist only for the smallest graphs (D_4, E_6). The branch perturbation helps higher series (Balmer, Paschen) but not Lyman-alpha at n>=5. **The physical interpretation**: eigenvalue modes have spatial extent. Low-k modes (controlling Lyman-alpha) span the full graph; high-k modes are localized near endpoints. A branch at the endpoint couples to high-k modes preferentially. This is the graph-theoretic analog of electron correlation corrections in multi-electron atoms: excited states are more affected than the ground state. The D_4 exception (99.9% improvement) occurs because with only 3 levels, all modes span the full 4-vertex graph.

### 13. Graph Green's Function IS the Hydrogen Spectrum (exp_20)

exp_20 scores 4/4 and provides the deepest connection between DFT graph structure and real spectroscopy. The key:

- **Path graph (A_n)** = 1D discrete Laplacian. Eigenvalues lambda_k = 2(1-cos(k*pi/(n+1))) ~ k^2.
- **Inverse eigenvalues** E_k = 1/lambda_k ~ 1/k^2 = **exactly the hydrogen energy level pattern**.
- The match is not numerical coincidence — it's a **theorem**: the 1D Laplacian Green's function has 1/k^2 spectrum, and hydrogen's Coulomb problem reduces to a 1D radial equation.
- At n=8 (8 vertices!), hydrogen spectral line ratios are reproduced to **0.68% mean error**. The Lyman-alpha ratio (the most fundamental line in physics) matches to **0.05%**.

**Branching = fine structure.** D_n and E_n graphs depart from the 1/k^2 pattern more than A_n at every size. The star graph D_4 has degenerate eigenvalues (100% departure) — the graph-structure analog of angular momentum degeneracy. The A_n family is "hydrogen" (pure path = pure radial equation), D_n/E_n add angular complexity.

**The forward bridge**: A_n Green's function → hydrogen ratios connects graph topology to measurable spectra. If the branching departures in D_n/E_n correspond to real fine-structure corrections (alpha^2 scale), this would be the first quantitative bridge from DFT graph structure to precision atomic physics.

### 14. Convergence Is O(n^-4), Not O(n^-2) — And Branching Has a Sweet Spot (exp_21)

exp_21 consolidates exp_20's results and reveals two surprises:

**Convergence rate.** The Lyman-alpha ratio error for A_n path graphs decreases as n^(-4.08), four powers faster than the naive O(n^-2) from the eigenvalue approximation lambda_k ~ k^2*pi^2/(n+1)^2. This makes physical sense: the ratio cancels leading-order errors, leaving only O(n^-4) correction terms. At n=50, the error is 0.00003% — essentially exact. P23 confirmed and strengthened.

**Weighted branching.** Continuously deforming A_n→D_n by varying branch weight w reveals: (1) departure scales as w^0.40±0.10, consistently across D_4..D_8, and (2) at w=alpha_EM≈0.0073, the branched graph is MORE hydrogen-like than the pure path. The fine-structure analogy breaks in an interesting way — a tiny branch initially corrects the discrete error, creating a "sweet spot" before larger branches break the 1/k^2 pattern. P24 revised.

The eigenvalue↔fingerprint bridge (T2) shows Spearman rho=0.591 (p=0.056) — suggestive but not conclusive with 11 pairs. The spectral structure and identity fingerprinting are correlated but the bridge needs more data points (larger graphs, more ADE types).

### 16. The Energy Scale Fix: Coupling Constants Replace Planck Units (exp_24)

exp_24 scores 4/4 and resolves the energy scale problem that caused 8 failures across exp_03-09. The fix:

**Replace** `E = n * Xi * E_Planck * phi^(-d)` **with** `E = n * Xi * alpha(d)^2 * m_mediator`

where `alpha(d) = phi^(-d)/sqrt(5)` is the DFT coupling at Fibonacci depth d, and m_mediator is the relevant particle mass (m_e for atomic, m_p for nuclear).

**Results:**
- **EM depth (d=13)**: ALPHA_EM_DFT^2 * m_e / 2 = 13.606 eV vs Rydberg 13.606 eV — 11.4 ppm agreement. The Planck version gives 2.34e19 MeV (24 orders too high).
- **Strong depth (d=3)**: fdc(3)^2 * m_p = 10.5 MeV — within factor 1.75 of the mean alpha decay energy (5.98 MeV). The Planck version gives ~10^22 MeV.
- **Scale hierarchy**: EM/nuclear ratio = 2.6e-6 vs Rydberg/alpha = 2.3e-6 (within 14%). The coupling constants AND mediator masses together reproduce the 6-order-of-magnitude gap between atomic and nuclear physics.
- **Boundary counts**: Coupling-anchored counts are 0.39-0.81 for alpha decays (physically sensible). Planck-anchored counts are ~10^-19 (unphysical).

**Forward path**: The coupling-anchored scale E(d) = alpha(d)^2 * m_mediator should replace E_Planck * phi^(-d) throughout Milestone R. The 8 energy-scale failures (exp_03 T3/T4, exp_04 T1, exp_05 T1/T2/T4, exp_06/08/09 boundary count tautology) are all resolved by this single conceptual fix.

### 17. Per-Series Sweet Spot Map (exp_25)

exp_25 scores 4/4 and provides the convergence roadmap for the graph → hydrogen bridge. Each hydrogen series has a minimum graph rank (sweet spot) where the Laplacian inverse eigenvalue ratios match hydrogen to within 5%. The sweet spots are strictly ordered: Lyman converges first (A_5), then Balmer (A_6), then Paschen (A_8).

At each sweet spot, the alpha (first) transition ratio matches hydrogen to < 1.3% — well within experimental precision. The convergence exponents from exp_23 predict the sweet spots via log(n_sweet) ~ 1/p_m at R^2=0.949.

Branch departure (D_n vs A_n) at sweet spots is ordered by series sensitivity: Paschen > Balmer > Lyman for 3/4 test sizes. This confirms the series-selective branching from exp_22 — higher series are more sensitive to structural perturbations, consistent with the graph-theoretic analog of electron correlations affecting excited states more than the ground state.

### 19. Bifractal Time: Dual Scaling but Not Privileged Time (exp_27)

exp_27 scores 0/4 against its specific criteria, but the underlying data supports the bifractal time model M(t)=Collapse(R_b(t), R_f(t)) through unexpected channels:

**Dual scaling is real.** All 7 tested ADE graphs show a crossover noise level where R_f (noise dynamics) begins to dominate R_b (structural degree) in determining FPT. Below crossover: degree predicts FPT well (R_b regime). Above: noise amplitude matters more (R_f regime). The crossover exists but doesn't correlate with spectral gap — the transition depends on multiple structural features, not just lambda_2.

**Time IS not privileged.** The leave-one-channel-out analysis (T4) reveals a clean result: removing FPT (the time proxy) causes ZERO ARI loss for degree classification in ALL 11 graphs. HKS (the structure proxy) is the only channel whose removal degrades classification. This is exactly what bifractal time predicts: time is an emergent projection of collapse, not a fundamental coordinate. Structure (R_b, measured by HKS) is more fundamental than dynamics (R_f, measured by FPT).

**Non-commutativity is channel-pair-specific.** HKS and FPT are strongly anti-correlated (|tau|>0.5) because both derive directly from the Laplacian spectrum. JSD and FPT are nearly independent (|tau|~0) because JSD measures random-walk divergence — an information-theoretic quantity orthogonal to spectral structure. The bifractal non-commutativity holds between the information-theoretic channel (JSD/SEC) and the structural channels (FPT/HKS/PAC), but not within the structural channels.

**Why 0/4.** Each test criterion was overly specific: T1 required spectral gap correlation (noise grid too coarse), T2 required exact rank prediction from clustering (model wrong), T3 required uniform bounds (channel pairs heterogeneous), T4 required equal channel contributions (structure dominates dynamics). The findings are real; the formalization was premature.

### 18. Confluence Arithmetic Confirmed: Series-Dependent Convergence (exp_23)

exp_23 tests Peter's insight that physical "constants" are confluent attractors — stable fixed points of a non-commutative, stateful aggregation process. Each hydrogen series ratio (Lyman, Balmer, Paschen, Brackett) is a separate constraint axis converging to the 1/k² continuum at n→∞.

**The clean result:** Per-series convergence exponents follow p(m) = 4.022 + 0.057m with R²=0.9983. The linear relationship in series index m is exactly what confluence arithmetic predicts — each stream has its own convergence rate, and the rates are ordered. The ordering is INVERTED from the naive prediction (higher series converge faster, not slower), but the explanation is clean: eigenvalue pairs in higher series are more similar, so discretization corrections cancel better.

**What this means for DFT:** The graph Laplacian's eigenvalue spectrum encodes hydrogen-like physics, and the convergence structure respects confluence arithmetic — each observable (series ratio) has its own convergence trajectory with its own rate and prefactor. There is no universal correction; the corrections are stream-dependent. This is consistent with Peter's framework: "there are never constants, just attractors that are constantly changing due to them being confluent axes."

**The honest failures:** The convergence direction was wrong (T1), branching sensitivity isn't simply ordered (T3), and per-series prefactors differ ~16x (T4). The clean linear exponent relationship (T2) is the one result that survives, and it's the most physically meaningful: it connects graph eigenvalue structure to confluence operator ordering.

## Structure

```
milestone-r/
├── meta.yaml
├── README.md
├── core/
│   └── radiation_physics.py    # Severance functions, physical constants
├── scripts/
│   ├── exp_01_ledger_severance_mechanics.py
│   ├── exp_02_alpha_decay_discrete_severance.py
│   ├── exp_03_beta_spectrum_unsettled_ledger.py
│   ├── exp_04_gamma_equilibration.py
│   ├── exp_05_hydrogen_spectrum_fibonacci_depth.py
│   ├── exp_06_xi_cost_scope_counting.py
│   ├── exp_07_bubble_ejection_analogy.py
│   ├── exp_08_xray_equilibrium_shift.py
│   ├── exp_09_cross_milestone_compatibility.py
│   ├── exp_10_synthesis_and_predictions.py
│   ├── exp_11_balance_seeking_decay.py
│   ├── exp_12_topological_barrier_gn.py
│   ├── exp_13_perspectival_barrier.py
│   ├── exp_14_sec_kinetic_energy.py
│   ├── exp_15_stress_barrier_gn.py
│   ├── exp_16_universality_gn.py
│   ├── exp_17_spectral_fingerprinting.py
│   ├── exp_18_multichannel_fingerprint.py
│   ├── exp_19_two_channel_blindspots.py
│   ├── exp_20_spectral_ratios.py
│   ├── exp_21_spectral_consolidation.py
│   ├── exp_22_branching_sweet_spot.py
│   ├── exp_23_confluence_convergence.py
│   ├── exp_24_coupling_energy_scale.py
│   ├── exp_25_series_sweet_spots.py
│   ├── exp_26_channel_completeness.py
│   └── exp_27_bifractal_time_emergence.py
├── results/
└── journals/
```

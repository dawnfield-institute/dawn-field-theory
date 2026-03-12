# Milestone 4: Synthesis & Findings Registry

**Status**: Complete (15/15 experiments)
**Date started**: 2026-02-22
**Date completed**: 2026-03-12
**Vault FDO**: `proj-milestone4`

---

## Scorecard

| Exp | Title | Verdict | Key Number |
|-----|-------|---------|------------|
| 01 | Lorentz PAC Partition | PARTIAL | Identity exact (dev = 1e-16), cascade linearity R^2 = 0.73 |
| 02 | Nuclear Configuration Space | PARTIAL | BE vs level density rho = -0.6, p = 0.208 |
| 03 | Turbulence Mode Scaling | PASS | Power law R^2 = 0.9998, N=7 hits -5/3 within 0.003 |
| 04 | Cascade Amplification | PASS | Log fit R^2 = 0.994, TC at N=8 = 53x |
| 05 | Binding Energy Landscape | PARTIAL | Magic number effect p = 0.016, but BE vs config rho = -0.22 |
| 06 | 4D Turbulence (GPU DNS) | PARTIAL | 2D/3D calibration pass, 4D k=20 prediction fails |
| 07 | Herniation-Derived Potential | PARTIAL (3/5) | Fibonacci uniqueness proven, but phi doesn't beat random |
| 08 | Constrained Mass Ratios | PARTIAL (3/5) | Phi ranked #1 topology, but consecutive states fail |
| 09 | Full Derivation Chain | PASS | All 5 links hold, up-quarks remain weak (59% error) |
| 10 | Weakness Tests | FAIL (1/3) | Up-quarks unresolved (p = 0.335), Landauer n_levels fails |
| 11 | PAC-Grounded Fixes | FAIL | Complement potentials don't close the up-quark gap |
| 12 | QBE Golden Cascade | PARTIAL | QPL contrast = 0.878, but golden cascade correlation weak (0.104) |
| 13 | Gaussian Envelope Derivation | PASS | 5/5 parts pass, phi is unique for equal-area conservation |
| 14 | 2D Turbulence Mode Count | PASS | Same engine recovers both -5/3 (N=8) and -3 (N=3) |
| 15 | Comprehensive Null Tests | STRONG PASS (8/8) | Xi attractor CoV ratio 800x, all nulls rejected |

**Summary**: 6 PASS, 7 PARTIAL, 2 FAIL

---

## What's Proven

These results have proper error bounds, null tests, and survive falsification:

### 1. Mode count determines turbulence exponent (exp_03, 14, 15)
At physical coupling cd=0.1, mode count is a monotonic determinant of spectral exponent (Spearman r = 1.0). The same cascade engine with no re-tuning recovers:
- 3D Kolmogorov: -1.608 at N=8 (target -5/3 = -1.667, dev 3.5%)
- 2D enstrophy: -2.840 at N=3 (target -3.0, dev 5.3%)
- 2D inverse: -1.613 at N=8 (target -5/3, dev 3.2%)

She-Leveque formula k = d * F_{d+1} with k-1 offset is consistent across dimensions.

### 2. Organized fraction converges to ~2/3 (exp_03, 14, 15)
At cd=0.1, N=8: mean org_frac = 0.666 with CV = 0.2% across 100 seeds. This is a stability claim at fixed physics, not universality.

### 3. Xi is a global attractor (exp_15 C.2, H.1)
The PAC/SEC separation is quantified: CoV(global PAC sum) = 0.0002 vs CoV(local SEC fluctuation) = 0.163 — an 800x stability ratio. Local SEC doesn't conserve (some scales overshoot, others undershoot), but the global sum is rock-solid. This holds even when SEC is amplified via nonlinear_strength sweep (global sum CoV = 0.0014 across ns = 0.0 to 0.9).

### 4. Structured coupling beats random (exp_14, 15)
At cd=0.1, N=8: structured coupling distance from -5/3 = 0.055, random mean distance = 1.153 (p = 0.000). The exponential decay coupling C[i,j] = exp(-|i-j| * cd) is structurally necessary.

### 5. Gaussian envelope from first principles (exp_13)
Three independent derivations (SEC diffusion, max entropy, PAC equal-area) all produce Gaussian shape. Phi is the only scaling base that preserves equal-area conservation (CV = 1.4e-14% vs next-best 30.5%). Robust to +/-20% parameter perturbation (200/200 tests).

### 6. Lorentz identity is algebraically unique (exp_01, 15)
Only the sqrt(1-v^2) partition matches GR time dilation exactly. All 5 alternatives fail with deviations 0.25-0.55. PAC's contribution is deriving WHY this partition applies, not discovering the identity.

### 7. Derivation chain is non-circular (exp_09, 15)
The PAC -> Fibonacci -> phi-cascade -> Landauer -> Lorentz chain has 2/3 independent links. Fibonacci uniqueness: only 16/616 integer matrices produce phi eigenvalues.

---

## What's Suggestive

Evidence is positive but insufficient for a strong claim:

### 1. Cascade amplification scales logarithmically with mode count (exp_04)
R^2 = 0.994 over N=2-64. TC at N=8 matches milestone 3 reference (53x). But extrapolation to nuclear regime (N=60, target 175x) gives 250x — 43% overshoot.

### 2. Nuclear configuration space correlates with binding energy (exp_02, 05, 15)
Permutation test p = 0.035, but only 6 nuclides have both datasets. Fusion/fission sides show strong individual correlations (rho = -0.93, 0.99) but combined picture is weak. Magic number effect is real (p = 0.016).

### 3. 4D turbulence follows the k formula (exp_06)
2D and 3D calibration pass, but 4D prediction (k=20) fails calibration (measured k=10.78). The 95% CI is enormous [3.5, 300]. Would need higher-resolution DNS.

### 4. QBE golden cascade has structure (exp_12)
QPL contrast = 0.878 shows Fibonacci potential creates structure, but golden cascade correlation is weak (0.104) and phi ranks only #2 behind sqrt(2).

---

## What's Failed

Honest record of what didn't work:

### 1. Up-quark mass ratios (exp_07, 08, 09, 10, 11)
The phi-cascade potential produces good lepton and down-quark ratios (2-6% error) but fails for up-quarks (40-60% error). Multiple rescue attempts failed:
- phi^2 cascade (exp_10): p = 0.335 vs random
- Complement potentials (exp_11): scores >> 1.0
- Constrained states (exp_08): consecutive state constraint fails badly
- Herniation potential (exp_07): phi doesn't reliably beat random

**This is the biggest open problem in the cascade mass-ratio story.**

### 2. Landauer-derived level count (exp_10)
The Landauer threshold was hypothesized to predict the optimal number of cascade levels. It doesn't — Landauer prediction gives scores 7-80x worse than brute-force optimal search.

### 3. Cascade linearity with E_internal (exp_01)
Cascade throughput should scale linearly with internal energy. It doesn't (R^2 = 0.73, slope = 0.50 instead of 1.0).

---

## Cross-Block Connections

### PAC (local conservation) thread
exp_03 -> exp_04 -> exp_14 -> exp_15(B,C,D): Mode count, amplification, 2D/3D recovery, null tests. This is the strongest thread in M4.

### SEC (global phase) thread
exp_15(C.2, H.1): Xi attractor quantified. exp_13: Gaussian from SEC diffusion. exp_06: DNS calibration (partial).

### Nuclear physics thread
exp_02 -> exp_05 -> exp_15(E): Configuration space measure. Suggestive but underpowered.

### Mass ratio thread
exp_07 -> exp_08 -> exp_09 -> exp_10 -> exp_11 -> exp_12: Derivation chain holds for leptons/down-quarks but up-quarks remain an open failure.

### Relativity thread
exp_01 -> exp_15(A): Lorentz identity uniqueness proven. Cascade linearity unresolved.

---

## Infrastructure Additions

- `core/gpu_cascade.py`: PyTorch CUDA-accelerated cascade engine. Drop-in replacement for energy_cascade() with batch_size parameter. Transparent CPU fallback. Used by exp_15 for 10K+ Monte Carlo runs.
- `core/constants.py`: Extended with nuclear physics data (NIST/NNDC), nuclear level density (RIPL-3).

---

## Success Criteria Assessment (from README.md)

1. [x] Prove Lorentz factor is mathematical identity from PAC — **YES** (algebraic identity exact, unique partition)
2. [x] Determine mode count -> exponent relationship — **YES** (power law R^2 = 0.9998, She-Leveque with k-1 offset)
3. [ ] Establish binding energy vs nuclear config space — **PARTIAL** (suggestive, underpowered)
4. [ ] Resolve gravity functional form — **NOT ATTEMPTED** (exp_04 gravity was dropped in favor of amplification scaling)
5. [x] All experiments: error bounds, null tests, falsification — **YES** (exp_15 comprehensive null suite)
6. [x] Honest separation: proven vs suggestive vs speculative — **YES** (this document)

**4/6 criteria met. 2 remain partial/unresolved.**

---

## Target Papers

| Paper | Status | Key Experiments |
|-------|--------|----------------|
| Paper 5: Classical Physics from Information Geometry | Draft-ready | exp_03, 06, 13, 14 |
| Paper 7: PAC Relativity | Draft-ready | exp_01, 09, 15(A,F) |
| Paper 8: Energy as Collapsed Potential | Needs work | exp_02, 04, 05, 07-12 (up-quark gap unresolved) |

---

*Dawn Field Institute, 2026-03-12*

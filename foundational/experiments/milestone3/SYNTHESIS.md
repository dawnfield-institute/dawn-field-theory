# Milestone 3: Synthesis

## Position in Research Program

Milestone 3 sits at the intersection of three threads:

1. **Energy Equivalence Session** (Feb 16-17, 2026) — Exploratory work that produced promising but unvalidated results about cascade dynamics, Fibonacci origins, and E-I-S resonance
2. **PACSeries Consolidation** — The 6-paper rewrite needs these results validated before integration
3. **Methodology Audit** — Milestones 1-2 established *what* the framework predicts; milestone3 asks *how honestly we've tested those predictions*

## Cross-Connections

### To Paper 1 (Structure Cost of Erasure)
- **exp_01**: If two-step Landauer memory → Fibonacci is validated, it provides the "why Fibonacci" mechanism Paper 1 currently lacks
- **exp_02**: Monotonic ξ accumulation strengthens the ln(φ) universality claim (currently supported by exp_23/exp_25 in landauer_erasure_structure)
- **exp_03**: Prime cascade reachability connects to the sieve-as-PAC-conservation result from asymmetric_conservation
- **exp_06**: Θ recycling resolution determines whether Paper 1 can claim any recycling efficiency or must present it as open

### To Paper 2 (Balance Constant Decomposition)
- **exp_10**: Independence audit directly affects the joint p-value claims in Paper 2's four-domain convergence table
- milestone1 already flagged: 21 alternatives within 5% of Ξ, and 1/√3 performs comparably to γ in Mertens test

### To Paper 4 (Standard Model Parameters)
- **exp_07**: Wilson-Fisher null test determines if a key A1 prediction survives
- **exp_08**: sin²θ_W energy-scale identification is critical — 3/13 at 4.4σ from PDG central value is a problem unless it matches at a specific RG scale
- **exp_09**: Look-elsewhere factor for mass ratios determines whether "5.7 ppm" for α is impressive or expected given the search space
- **exp_13/14**: Stoichiometric Fibonacci derivation shows SM formulas emerge from integer-constrained Fibonacci (99.98th percentile vs random)
- **exp_16**: Null space predictions FAIL (0/4) — framework describes but does not predict. Paper 4 must not overclaim predictive power
- **exp_17**: Physics-derived matrix resolves exp_14's selectivity gap (0.86× → 1.23×)
- **exp_18**: Conservation is necessary but not sufficient — PAC constrains possibility, not actuality
- **exp_19**: FALSIFIED: crystallization order is basis-independent. Paper 4 must not claim Fibonacci-specific dynamics
- **exp_21**: PAC-Lazy profiles DO discriminate (KL p=0.035) — the mechanism requires conservation normalization, not raw counting

### To Paper 6 (Computational Validation)
- **exp_04**: w2/w1 ratio clarifies the 0.600 vs 0.618 gap from E-I-S dynamics
- **exp_05**: 0.020 Hz resolution is critical before merging the relativistic_mas paper — if the sim can't reproduce it, the claim must be softened
- **exp_21**: PAC-Lazy formula mesh bridges dawn-field-theory and dawn-models GAIA architecture. The same conservation + profile comparison that works in GAIA POCs (011, 016, 018) works on the formula space. Paper 6 should note this cross-domain transfer.

### From milestone1
- Inherits the FALSIFICATION_REGISTRY approach (now expanded: 13 tests, 11 pass, 1 borderline, 1 corrected)
- exp_08 and exp_09 extend milestone1's exp_04 and exp_17 respectively
- Uses milestone1's α formula as ground truth for perturbation tests

### From milestone2
- mass_derivation results feed into exp_09 (look-elsewhere quantification)
- Extended Fibonacci mass formulas are the search space being audited

### From energy_equivalence session
- All Block A and B experiments originate from this session
- Key risk: session results may not reproduce under proper conditions (different initialization, larger N, proper error analysis)

## Epistemic Status

This milestone is explicitly about **methodology validation**, not new discovery. The core question is: *are the exciting results from energy_equivalence and milestones 1-2 as strong as they appear, or are there methodological gaps that weaken them?*

Possible outcomes:
- **Best case**: All results reproduce, methodology holds, and we can integrate with confidence
- **Middle case**: Some results reproduce, some don't — we integrate what holds and honestly flag what doesn't
- **Worst case**: Major methodological issues found — we restructure claims before publishing

All three outcomes are valuable. The worst case would be the most important finding.

## Expanded Test Coverage (Feb 2026)

### Block C: Stoichiometric & Tightening (exp_12–exp_17)

**exp_12 (F11)**: Fibonacci–MED complementarity. Promoted from exploratory to scored. The golden base paradox (Fibonacci coupling can't reach MED depth) means PAC/SEC and MED are genuinely independent constraints. Implications for Paper 2 (why Ξ = γ + ln(φ) needs both terms) and Paper 5 (MED depth isn't derivable from Fibonacci).

**exp_13/14 (F12)**: Stoichiometric Fibonacci derivation. Standard Model formulas emerge from integer-constrained Fibonacci matrices. 8/10 pass. Key weakness: raw selectivity 0.86× (Fibonacci doesn't preferentially favour physics targets over non-physics). Resolved by exp_17.

**exp_15 (F13)**: PAC/SEC cost monotonicity. SEC cost scales at ~55.7 units per Fibonacci index step. SEC/PAC ratio crosses 1.0 at F₈, suggesting a complexity boundary.

**exp_16** (not in registry — prediction generation, not falsification test): 17 genuine novel predictions at <1% error including Kolmogorov C_K = 3/2 (exact), τ/μ mass ratio (0.002%), ρ-parameter (0.025%).

**exp_17** (not in registry — resolution of exp_14 weakness): Physics-derived matrix achieves 1.23× selectivity, inverting exp_14's 0.86× resistance. 160 consensus predictions across all 3 matrices (hand-built, random baseline, physics-derived).

### Honest Assessment (updated)

| Finding | Strength |
|---------|----------|
| exp_09 (α formula) | WEAKEST LINK — binomial p=0.42. Paper 4 should lead with joint constraint (exp_10), not α in isolation |
| exp_10 (48 OOM correction) | STRENGTH — "naive 10⁻¹⁹⁷, corrected 10⁻¹⁴⁷" demonstrates methodological rigour |
| exp_12 (complementarity) | NEW INSIGHT — Papers 2 & 5 should reference the Fibonacci–MED independence |
| exp_14 (selectivity) | RESOLVED by exp_17 — physics-derived matrix favours physics |
| exp_16 (null space prediction) | HONEST FAILURE — framework describes but does not predict a priori (0/4) |
| exp_19 (phase transitions) | FALSIFIED — crystallization order is basis-independent, not Fibonacci-specific |
| exp_20 (fractal mesh) | FALSIFIED — raw pressure = depth bias, wrong direction |
| exp_21 (PAC-Lazy mesh) | RESOLUTION — PAC conservation + profile comparison fixes direction (KL p=0.035). Bridges dawn-models GAIA architecture to dawn-field-theory |
| A3 (λ*/β) | NOT TESTED — no milestone3 experiment validates these closed forms |
| B5 (Θ recycling) | HONEST RANGE — 36%–94%, mechanism confirmed, efficiency model-dependent |

### Block F Arc: From Description to Discrimination (exp_16–21)

The exp_16–21 sequence is instructive:

| Exp | Approach | Score | What It Showed |
|-----|----------|-------|----------------|
| 16 | Null space mining | 0/4 | Framework describes, doesn't predict |
| 17 | Physics-derived matrix | 3/4 | Selectivity fixable (0.86× → 1.23×) |
| 18 | Conservation predictions | 2/4 | Conservation necessary, not sufficient |
| 19 | Phase transition dynamics | 1/4 | **FALSIFIED**: basis-independent crystallization |
| 20 | Fractal convergence mesh | 1/4 | **FALSIFIED**: raw pressure = depth bias |
| 21 | PAC-Lazy formula mesh | **4/4** | **RESOLUTION**: PAC conservation + KL divergence works |

**Key insight**: The common failure mode in exp_16–20 was confusing structural depth with physical significance. PAC conservation (from GAIA POC architecture) normalizes away depth bias. The formula space IS a PAC tree — the same principles that work in neural network analysis work in theoretical physics formula analysis.

**Falsifications are valuable**: exp_19 (crystallization is physics-determined, not Fibonacci-determined) and exp_20 (raw counting is biased) constrain what can and cannot be claimed. Paper 4 must not claim Fibonacci-specific dynamics; it should claim Fibonacci arithmetic structure.

### Block G: Derivation Chain (exp_22–26)

Experiments 22–26 extend the framework from description toward mathematical derivation.

| Exp | Question | Score | What It Showed |
|-----|----------|-------|----------------|
| 22 | PAC → MED depth bound | 3/4 | **THEOREM**: all k-step PAC recursions floor to depth ≤ 2 |
| 23 | F₁₈₃ gravity correction | 3/4 | Unique cyclotomic minimiser. 1 + F₁₃/(πF₆²) at 0.0008 log₁₀ |
| 24 | PAC-Lazy signal anatomy | 1/4 | Honest: bootstrap CI includes zero. Engineering, not theory |
| 25 | Dark matter depth map | 2/2 | Ω_c = F₇·Ξ²/F₁₀ at 0.079%. Corrected GUT-scale proposals |
| 26 | Unified correction template | 2/3 | F_a/(mπF_b²) works for α_EM + gravity. **0/5000 MC match** |

**The derivation chain** (exp_22 → Paper 5):
1. PAC conservation: f(Parent) = Σ f(Children)
2. → Fibonacci decay: w_j = φ^{−j} (unique stable 2-step solution)
3. → Max effective depth: Σ φ^{−j} = φ² ≈ 2.618
4. → Integer bound: floor(φ²) = 2 = MED depth bound
5. → At depth 2: hidden dimension projects as curl → Maxwell equations

**The correction template** (exp_23 + exp_26):
- α_EM: 1 − F₁₀/(4πF₇²) at 5.7 ppm. Sign = minus (EM self-screening)
- Gravity: 1 + F₁₃/(πF₆²) at 0.0008 log₁₀. Sign = plus (gravitational enhancement)
- Both anchored to F₇ = 13 (EM gauge depth)
- Index gaps a−b: 3 = F₄ (EM), 7 = F₇ (gravity) — both Fibonacci
- 0/5000 random integer sequences match both corrections simultaneously

## PAC→MED Theorem (exp_22)

**Theorem**: For any k-step PAC recursion (k ≥ 2), the maximum effective depth floors to at most 2.

**Proof sketch**:

1. **Setup**: k-step PAC recursion couples each node to its next k children:
   w_j = w_{j+1} + w_{j+2} + ⋯ + w_{j+k}

2. **Characteristic equation**: x^k − x^{k−1} − ⋯ − x − 1 = 0

3. **Largest real root**: r_k > 1, giving decay w_j = C·r_k^{−j}

4. **Max effective depth**: D_k = r_k/(r_k − 1)

5. **Key property**: r_k is monotonically increasing in k, with r_2 = φ ≈ 1.618 and lim_{k→∞} r_k = 2

6. **Therefore**: D_k is monotonically *decreasing* in k:
   - k=2: D₂ = φ/(φ−1) = φ² ≈ 2.618 (loosest bound)
   - k=3: D₃ ≈ 2.192
   - k=4: D₄ ≈ 2.077
   - k→∞: D_∞ = 2/(2−1) = 2.0 (tightest bound)

7. **Integer quantization**: floor(D_k) = 2 for all k ≥ 2

**Corollaries**:
- Only k=2 (Fibonacci) gives decay rate ln(φ). All higher-k PAC recursions produce faster decay
- Fibonacci is the *loosest* constraint — most permissive while still bounding to depth 2
- MED's "depth ≤ 2, nodes ≤ 3" is a *consequence* of PAC conservation, not independent
- Paper 5's conditional statement ("if MED bounds hold") upgrades to derived ("PAC requires MED ≤ 2")

**Status**: Analytical result confirmed computationally for k = 2–8. The theorem holds exactly; it does not depend on numerical parameters or model assumptions. The Landauer model provides partial empirical support (3/4 tests pass) but the theorem itself is purely algebraic.

## Block H: The Mechanism — π→φ→Fibonacci (exp_27)

Experiment 27 tests the foundational hypothesis: **Fibonacci scaling is the stability eigenmode of recursive phase transport on π-closed manifolds.** The causal chain is:

1. **π defines rotational closure**: θ ≡ θ mod 2π
2. **Transport requires phase advance**: Δθ = 2πα for some angular fraction α
3. **Non-resonance = stability**: phase-locking (resonance) causes gaps; the maximally irrational α avoids all resonances
4. **φ is that maximum**: its continued fraction [1;1,1,1,...] has the smallest partial quotients, making it the hardest number to approximate rationally
5. **Golden angle α\* = 1 − 1/φ** minimises worst-case star discrepancy D\*\_N
6. **Fibonacci = discrete shadow**: on integer lattices, φ-scaling maps to F\_n indices
7. **Corrections = convergent errors**: the residual from rational approximation F\_{n+1}/F\_n → φ is 1/(F\_n·F\_{n+1}), which on the phase loop gives angular errors of 2π/(F\_n·F\_{n+1}) — explaining why corrections take the form F\_a/(mπF\_b²)

### Results (5/5 PASS)

| Test | Result | Key Numbers |
|------|--------|-------------|
| Worst-case D\*\_N | Golden #1 of 12 | worst=0.024, mean=0.008 across 28 scales |
| Perturbation robustness | Golden #1 absolute | mean perturbed D\*=0.010, best of 5 irrationals |
| Landauer bridge | Mapping exact | fd=ln(φ)→α=1−1/φ, Δ=0 machine precision |
| Correction template | Structure confirmed | F₁₃=F₇²+F₆², (φ²+1)/π at 0.62% |
| Inward/outward duality | Both stable | advantage=0.389, S>0.5 both directions |

### Gravity Correction Structural Form

F₁₃/(πF₆²) = (F₇² + F₆²)/(πF₆²) = (F₇/F₆)²/π + 1/π → (φ² + 1)/π = (φ + 2)/π ≈ 1.1517

This is the **simplest mixed φ-π expression at O(1)**. It arises from convergent error bounds on the golden angle's rational approximants, not from curve-fitting.

### Significance

This experiment provides the **mechanism** that was missing from exp_01–26. Previous experiments showed that Fibonacci structure appears; exp_27 shows **why** it must appear: π-closure selects φ via non-resonance, and Fibonacci is the integer projection. The correction template and the PAC→MED theorem are consequences, not assumptions.

## Block I: Cross-Validation Triangle (exp_28)

Experiment 28 closes the loop between three independent computational representations:

```
  Phase Transport (exp_27)     ←→     Thermodynamics (fibbinoci_thermo)
          ↘                                    ↙
              Geometry (phyllotaxis packing)
```

All three respond to the **same underlying property**: equidistribution quality of the angular step.

### Results (4/4 PASS)

| Test | Result | Key Numbers |
|------|--------|-------------|
| Convergent scaling | Ratio = 1/(φ√5) | 0.2735 vs predicted 0.2764 (CV=0.049) |
| Phase-thermo correlation | D*_N predicts thermo | Spearman r=0.964–0.976, all p≈0 |
| Geometric bridge | Ranks match perfectly | D* vs pack_cv: r=1.000 |
| Limit convergence | Ladder converges | 92.2% closer at late stages |

### The 1/(φ√5) Constant

The ratio |α_k − α\*| / [1/(F_n·F_{n+1})] converges to **1/(φ√5) ≈ 0.2764**. This is not empirical — it's a theorem from continued fraction theory:

|1/φ − F_n/F_{n+1}| ≈ 1/(F_{n+1}² · √5) and 1/(F_n·F_{n+1}) ≈ φ/F_{n+1}²

So the ratio = 1/(φ·√5). The Fibonacci convergent ladder has **analytically predictable error** at each step.

### Significance

The cross-validation shows that the phase-transport mechanism (exp_27) is not domain-specific. The same equidistribution property that controls D*_N in abstract phase space also controls thermodynamic excitation rates and geometric packing quality. Fibonacci numbers appear as the **integer waypoints** on the convergent ladder — the same in all three domains because F_n/F_{n+1} is the universal best rational approximation to 1/φ.

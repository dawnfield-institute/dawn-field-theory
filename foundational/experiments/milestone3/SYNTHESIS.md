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

# Theory Map

A guide to navigating Dawn Field Theory — the derivation chain, the experiments, and where each concept lives in the repo. For the theory itself, see [dawn-field-theory.md](theory/dawn-field-theory.md). For repo conventions, see [CLAUDE.md](./CLAUDE.md).

---

## The Derivation Chain

Everything flows from a single observation: self-applied symmetry is the unique generative primitive.

```
Self-Applied Symmetry (iddea.md)
  |
  ├── Forces iteration (self-application doesn't terminate)
  ├── Forces polarity (mutual closure of two directions)
  └── Forces conservation (symmetric enactment preserves balance)
        |
        v
PAC: Potential-Actualization Conservation
  Psi(k) = Psi(k+1) + Psi(k+2)
        |
        ├── Unique stable solution: phi = (1+sqrt(5))/2
        ├── Fibonacci arithmetic as the language of physics
        └── Cascade structure with ln(phi) cost per level
              |
              v
SEC: Symbolic Entropy Collapse
  dS/dt = alpha*grad(I) - beta*grad(H)
        |
        ├── Structure forms where information gradients dominate
        ├── Prime sieve conservation, 1/phi threshold
        └── Entropy-to-information conversion
              |
              v
MED: Macro Emergence Dynamics                RBF: Recursive Balance Field
  depth <= 1, nodes <= 3                       B(x,t) = grad^2(E-I) + ...
  (complexity bounds)                          (adaptive geometry)
        |                                            |
        └────────────────┬───────────────────────────┘
                         |
                         v
              Three Derived Constants
              ________________________
              phi    = 1.618034...    (PAC fixed point)
              ln(phi) = 0.48121...   (Landauer cost per level)
              Xi     = 1.0584...     (gamma + ln(phi), balance constant)
                         |
          ┌──────────────┼──────────────────┐
          v              v                  v
   Standard Model    Cosmology       Quantum Gravity
   alpha 5.7 ppm     CC 0.09 orders  Planck scale derived
   sin^2(tW)=3/13    S8 resolved     Hawking T*M=1/(8pi)
   Higgs 83 ppm      H0 matched      Graviton from cascade
   Feigenbaum 13d    Cascade clock    Singularity resolved
```

---

## Axioms & Principles

| Principle | Role | Where Established | Where Tested |
|-----------|------|-------------------|--------------|
| **PAC** | Conservation | [milestone1](experiments/milestones/milestone1/) | Every milestone; [pac_confluence_xi](archive/era2-prefield/pac_confluence_xi/) has 45+ scripts |
| **SEC** | Dynamics | [milestone1](experiments/milestones/milestone1/) | [sec_prime_manifold](papers/standalone/golden_ratio_prime_distribution/Code/), [sec_threshold_detection](papers/series/PACSeries/v0.2/feigenbaum_fibonacci_arithmetic/Data/) |
| **MED** | Optimization | [milestone3](papers/series/PACSeries/v0.2/balance_constant_decomposition/Data/) | [navier-stokes](archive/era2-prefield/navier-stokes/), [milestone10 exp_15](experiments/milestones/milestone10/) |
| **RBF** | Geometry | [milestone7 exp_08](experiments/milestones/milestone9/) | [pre_field_recursion](archive/era2-prefield/pre_field_recursion/) |

The theory argument (why these axioms and not others) is in [iddea.md](../iddea.md) at the workspace root, tested in [milestone10](experiments/milestones/milestone10/).

---

## Three Derived Constants

### phi = (1+sqrt(5))/2 = 1.618034...

The unique stable fixed point of PAC recursion. Selected by b^2 - b - 1 = 0 (gravity-time duality).

| Where It Appears | Experiment | Result |
|-----------------|------------|--------|
| Cascade attenuation | [milestone7/exp_04](experiments/milestones/milestone9/) | Emergent 1/phi (R^2=0.995) |
| Higgs self-coupling | [milestone5/exp_07](experiments/milestones/milestone5/) | lambda = phi/(4*pi), 83 ppm |
| Hubble correction | [milestone8/exp_07](papers/series/PACSeries/v0.3/cosmological_predictions_cascade_clock/Data/) | phi^(1/6) matches SH0ES |
| Force hierarchy | [milestone6/exp_04](papers/series/PACSeries/v0.3/symmetry_primitive_scoped_mediation/Data/) | phi^6 = strong/EM ratio |
| Neural networks | [ml_validation_pythia_gpt2](papers/ml_validation_pythia_gpt2/) | phi-crossing at step 512 |
| Prime stress field | [sec_prime_manifold](papers/standalone/golden_ratio_prime_distribution/Code/) | 1/phi threshold |
| Cellular automata | [cellular_automata_pac_attractors](papers/standalone/cellular_automata_xi_clustering/Code/) | P/A = 1.0579 at Rule 110 |
| Artifact test | [phi_artifact_test](experiments/milestones/phi_artifact_test/) | Not an artifact |

### ln(phi) = 0.48121...

The Landauer erasure cost per cascade level. Independently grounded in thermodynamics.

| Where It Appears | Experiment | Result |
|-----------------|------------|--------|
| Erasure structure | [landauer_erasure_structure](papers/series/PACSeries/v0.2/structure_cost_of_erasure/Data/) | A/(A+xi) = ln(phi), 0.76% |
| Cascade clock slope | [milestone9/exp_01](experiments/sidecars/midnight/) | 1/ln(phi) = 2.0781 exact |
| Arrow of time | [milestone9/exp_06](experiments/sidecars/midnight/) | 0.665 nats/level entropy production |
| Landauer universality | [milestone11/exp_09](experiments/milestones/milestone11/) | Forward/reverse ratio super-exponential |
| JWST z-cascade | [milestone8/exp_09](papers/series/PACSeries/v0.3/cosmological_predictions_cascade_clock/Data/) | z_cascade = ln(phi)*6 |

### Xi = gamma + ln(phi) = 1.0584...

The universal balance constant. Zero free parameters — gamma from harmonic counting, phi from duality.

| Where It Appears | Experiment | Result |
|-----------------|------------|--------|
| Cellular automata | [cellular_automata_pac_attractors](papers/standalone/cellular_automata_xi_clustering/Code/) | Class IV clusters at Xi, p < 8.58e-8 |
| Scope fixed point | [milestone6/exp_07](papers/series/PACSeries/v0.3/symmetry_primitive_scoped_mediation/Data/) | Xi attractor, Euler gap 0.09% |
| Algebraic uniqueness | [milestone9/exp_02](experiments/sidecars/midnight/) | g_out = g_in^2 selects Xi uniquely |
| Boundary crossing | [milestone7/exp_03](experiments/milestones/milestone9/) | Xi per symmetric restoration |
| Mobius pairing | [oscillation_attractor_dynamics](experiments/milestones/oscillation_attractor_dynamics/) | Xi = pi/55 to 6.8e-9 |
| Wilson-Fisher | [milestone3](papers/series/PACSeries/v0.2/balance_constant_decomposition/Data/) | nu = 2/(3*Xi) |
| Xi universality | [milestone10/exp_08](experiments/milestones/milestone10/) | Extension across domains |

---

## The Milestone Stack

### M1: Standard Model Derivation Chain
**Thesis:** PAC + SEC derive SM parameters from Fibonacci arithmetic.
**Key Results:** alpha_EM, sin^2(theta_W) = 3/13, Koide = 2/3, Feigenbaum 13 digits, D=3 spatial dimensions.
**Experiments:** [milestone1/](experiments/milestones/milestone1/)

### M2: Mass Derivations
**Thesis:** Particle mass ratios from PAC recursion levels.
**Key Results:** mu/e at 5 ppm, p/e at 0.0083%, Casimir effect, RG running.
**Experiments:** [milestone2/](experiments/milestones/milestone2/)

### M3: Energy Equivalence & Falsification
**Thesis:** Test DFT against energy conservation, quantum phenomena, and Landauer bounds.
**Key Results:** Phase transitions consistent, Wilson-Fisher nu = 2/(3*Xi), MED depth<=2 proof.
**Experiments:** [milestone3/](papers/series/PACSeries/v0.2/balance_constant_decomposition/Data/) | [quantum_validation/](archive/era1-symbolic/quantum_validation/)

### M4: PAC Relativity & Turbulence
**Thesis:** Lorentz invariance as unique PAC partition; turbulence from Fibonacci structure.
**Key Results:** Lorentz uniqueness, Kolmogorov -5/3 from PAC, She-Leveque k = d*F_{d+1}.
**Experiments:** [milestone4/](experiments/milestones/milestone4/) | [navier-stokes/](archive/era2-prefield/navier-stokes/)

### M5: Standard Model Completion — 13/13
**Thesis:** Close remaining SM gaps and fix simulator drift.
**Key Results:** Higgs mass 83 ppm (lambda = phi/(4*pi)), CKM and PMNS from Fibonacci arctangents, sin^2(theta_W) = tan(theta_C) = 3/13, strong force from cascade-depth tiling, de-actualization completes PAC cycle.
**Experiments:** [milestone5/](experiments/milestones/milestone5/) — 13 experiments across 4 blocks.

### M6: Scoped Mediation — 35/40 (88%)
**Thesis:** Forces differ by Fibonacci depth. Constants are ratios of what survives scope boundaries.
**Key Results:** alpha_EM 5.7 ppm (#1 of 10,440 combinations), phi^6 force hierarchy, dark sector at depth 73, Xi as scope fixed point.
**Experiments:** [milestone6/](papers/series/PACSeries/v0.3/symmetry_primitive_scoped_mediation/Data/) — 10 experiments.

### M7: The Symmetry Primitive — 37/40 (93%)
**Thesis:** Symmetry is pre-axiomatic — PAC/SEC/MED/RBF derive from self-applied symmetry via ADE classification.
**Key Results:** Phi from cross-scale self-reference, emergent 1/phi attenuation (R^2=0.995), D=3 uniqueness, Xi per boundary crossing. Honest failures: RBF memory damping (2/4), cross-topology consistency (3/4).
**Experiments:** [milestone7/](experiments/milestones/milestone9/) — 10 experiments.

### M8: BSM Predictions & Observational Contact — 48/48 (100%)
**Thesis:** Pre-register falsifiable predictions from the cascade framework.
**Key Results:** CC at -122.09 (0.09 orders), Z' at 395 GeV (not excluded, 9x margin), dark matter 6.44 keV (X-ray line ~3.2 keV), S8 = 0.787, H0 = 73.0 km/s/Mpc, JWST cascade floor.
**Experiments:** [milestone8/](papers/series/PACSeries/v0.3/cosmological_predictions_cascade_clock/Data/) — 12 experiments (10 original + hardening).

### M9: The Infodynamic Mechanism — 37/40 (92%)
**Thesis:** The cascade is a temporal clock: N(t) = a + (1/ln(phi))*ln(t_lookback).
**Key Results:** S8 tension 3.22sigma -> 0.07sigma (98% reduction), Xi algebraically unique, 1 free parameter (t1=520 Myr), phi^{1/N_floor} discrete H0 matches SH0ES at 0.05sigma. Honest failures: DESI wa tension (-0.15 vs -0.75).
**Experiments:** [milestone9/](experiments/sidecars/midnight/) — 10 experiments.

### M10: Symmetry Self-Application — 90/115 (78%)
**Thesis:** The framework examines itself — self-applied symmetry as the unique generative primitive.
**Key Results:** Three exhaustive cases (A/B/C), time from symmetric self-enactment, laws as continuously maintained equilibria, arithmetic as fossilized closure pattern. Full derivation chain from primitive to constants.
**Experiments:** [milestone10/](experiments/milestones/milestone10/) — 17 experiments. | See also [iddea.md](../iddea.md).

### M11: Quantum Gravity — 52/52 (100%)
**Thesis:** Quantum gravity is where gravitational response-time is exceeded by perturbation timescale.
**Key Results:** Planck scale derived (depth-183 crossover), singularity resolved (cascade saturation), Hawking T*M = 1/(8*pi) exact, graviton spin-2/massless/2-pol from cascade, area law from gradient, bounce time = 1 t_Planck. 12 falsifiable predictions (7P+2D+3C).
**Core Module:** [quantum_gravity.py](experiments/milestones/milestone11/core/quantum_gravity.py) — builds on [infodynamics.py](experiments/milestones/milestone9/core/infodynamics.py) -> [bsm.py](experiments/milestones/milestone8/core/bsm.py)
**Experiments:** [milestone11/](experiments/milestones/milestone11/) — 13 experiments across 5 blocks.

---

## Standalone Experiments

These are outside the milestone structure but validated key results.

| Experiment | What It Tests | Key Result | Status |
|-----------|---------------|------------|--------|
| [pac_confluence_xi](archive/era2-prefield/pac_confluence_xi/) | PAC-Xi convergence proofs | sin^2(theta_W) = 3/13, (2*alpha*beta)^2 = 4/5, 45+ scripts | active |
| [sec_prime_manifold](papers/standalone/golden_ratio_prime_distribution/Code/) | SEC in number theory | frac(E>0) = 1/phi at criticality | active |
| [confluent_identity](experiments/milestones/confluent_identity/) | Confluent identity phases | Multi-phase validation of arithmetic identity | complete |
| [exp_30_arithmetic_dimension_emergence](experiments/milestones/exp_30_arithmetic_dimension_emergence/) | Arithmetic dimension emergence | 94/95 — ADE closure, all 4 axioms derived | complete |
| [exp_31_symmetry_primitive](experiments/milestones/exp_31_symmetry_primitive/) | Symmetry primitive validation | Companion to M7 | complete |
| [exp_32_geometric_primacy](experiments/milestones/exp_32_geometric_primacy/) | Geometric primacy of phi | Geometric derivation path | complete |
| [exp_33_black_hole_cascade](experiments/milestones/exp_33_black_hole_cascade/) | Black hole cascade | 16/16, ghost heart mechanism | complete |
| [cellular_automata_pac_attractors](papers/standalone/cellular_automata_xi_clustering/Code/) | Wolfram Class IV at Xi | Rule 110 P/A = 1.0579, p = 3.5e-10 | validated |
| [sec_threshold_detection](papers/series/PACSeries/v0.2/feigenbaum_fibonacci_arithmetic/Data/) | Feigenbaum closed forms | r_inf to 13 digits, delta to 8 digits | validated |
| [landauer_erasure_structure](papers/series/PACSeries/v0.2/structure_cost_of_erasure/Data/) | Landauer bound from PAC | A/(A+xi) = ln(phi), 25 experiments | validated |
| [oscillation_attractor_dynamics](experiments/milestones/oscillation_attractor_dynamics/) | Mobius pairing dynamics | Xi = pi/55 to 6.8e-9, 24x enrichment | validated |
| [navier-stokes](archive/era2-prefield/navier-stokes/) | She-Leveque turbulence | k = d*F_{d+1}, 14.3x more accurate than K41 | validated |
| [maxwell_from_pac_sec](experiments/milestones/maxwell_from_pac_sec/) | Maxwell from information dynamics | c^2 = alpha*gamma + beta*delta, alpha_EM = 1/137.036 | active |
| [gravity_from_maxwell_pac](experiments/milestones/gravity_from_maxwell_pac/) | Gravity from Maxwell via PAC | Information geometry bridge | active |
| [prime_harmonic_manifold](archive/era2-prefield/prime_harmonic_manifold/) | Golden ratio eigenvalue emergence | lambda_1 = 1/phi, +18.8% predictive improvement | validated |
| [asymmetric_conservation](papers/series/PACSeries/v0.2/balance_constant_decomposition/Data/) | Frame-dependent PAC | Sieve conservation EXACT over 126 steps | active |
| [minimum_actualization_resolution](experiments/milestones/minimum_actualization_resolution/) | MVAE, Planck derivation | Minimum resolution from PAC | active |
| [pre_field_recursion](archive/era2-prefield/pre_field_recursion/) | Mobius topology as substrate | 5.11x speedup, Xi = 1.0571 confirmed | active |
| [standard_model_connection](experiments/milestones/standard_model_connection/) | PAC-SM physical connection | RG flow mapping, gauge group structure | active |
| [base_agnostic_pac](papers/series/PACSeries/v0.2/balance_constant_decomposition/Data/) | PAC invariance across number bases | Confirmed | validated |
| [phi_artifact_test](experiments/milestones/phi_artifact_test/) | Is phi genuine or framework artifact? | Genuine domain property | exploratory |
| [information_amplification](papers/standalone/symbolic_entropy_collapse/Code/) | Information amplification proof | Framework established | validated |
| [unified_emergence_v2](archive/era1-symbolic/unified_emergence_v2/) | Production-grade validation | 87.5% Phase 1 success | validated |
| [landauer_erasure_field_cost_map](archive/era1-symbolic/unified_emergence_v2/) | Landauer field cost mapping | Cost landscape analysis | active |

---

## Quantum Validation Suite

Six sub-experiments testing DFT predictions against quantum phenomena. All in [quantum_validation/](archive/era1-symbolic/quantum_validation/).

| Sub-experiment | What It Tests |
|---------------|--------------|
| [born_rule](archive/era1-symbolic/quantum_validation/born_rule/) | Born rule reproduction from SEC |
| [symbolic_entanglement](experiments/studies/base_agnostic_pac/) | Entanglement via symbolic fields |
| [symbolic_interference](experiments/studies/wealth_field_dynamics/) | Two-slit interference patterns |
| [symbolic_reversability](archive/era1-symbolic/quantum_validation/symbolic_reversability/) | Reversibility in symbolic collapse |
| [symbolic_entropy_collapse_vs_quantum_decoherence](experiments/studies/phi_artifact_test/) | SEC vs decoherence comparison |
| [landauer_symbolic_erasure_energy_validation](experiments/milestones/milestone2/) | Landauer bound validation |

---

## Exploratory Experiments

Early-stage or cross-domain experiments testing novel applications.

| Experiment | Domain | What It Tests |
|-----------|--------|--------------|
| [biology_experiments](archive/era1-symbolic/biology_experiments/) | Biology | Symbolic collapse in biological systems, evolution |
| [DNA_repair](archive/era1-symbolic/symbolic_bifractal/) | Biology | BRCA1 mutation detection from entropy profiles |
| [dna_prime_structure](experiments/milestones/dna_prime_structure/) | Biology | Prime-interval patterns in DNA/protein sequences |
| [hodge_conjecture](archive/era1-symbolic/symbolic_memory_agentic_decay_test/) | Mathematics | Prime-modulated symbolic collapse in arithmetic geometry |
| [prime_growth_dynamics](papers/series/PACSeries/v0.2/balance_constant_decomposition/Data/) | Number Theory | Primes as residual roughness with conserved memory |
| [prime_growth_dynamics_v2](experiments/milestones/prime_growth_dynamics_v2/) | Number Theory | Multi-stage emergence pipeline |
| [pac_cosmology_validation](papers/standalone/pac_cosmology_jwst_validation/Code/) | Cosmology | PAC/SEC vs JWST observations |
| [pac_knowledge_discovery](experiments/milestones/pac_knowledge_discovery/) | Information | N^2 convergence for unknown children detection |
| [pac_dag_fluid](experiments/milestones/pac_dag_fluid/) | Information | Bidirectional SEC on hierarchical DAG structures |
| [wealth_field_dynamics](experiments/milestones/wealth_field_dynamics/) | Economics | Non-equilibrium wealth field analysis |
| [recursive_knot_actualization](experiments/milestones/recursive_knot_actualization/) | Topology | Partial recursive functions as topological knots |
| [algebra_geometry_interface](experiments/milestones/algebra_geometry_interface/) | Mathematics | Phi and Xi at the algebra-geometry interface |
| [prefield_em_emergence](experiments/milestones/prefield_em_emergence/) | Physics | Pre-field EM emergence |
| [symbolic_entropy_collapse](archive/era1-symbolic/symbolic_entropy_collapse/) | Core | Core SEC experiment framework |
| [symbolic_emergence](archive/era1-symbolic/symbolic_emergence/) | Language | Language-like structure emergence between agents |
| [symbolic_bifractal](archive/era1-symbolic/symbolic_bifractal/) | Fractals | Bifractal expansion and collapse patterns |
| [predictive_collapse](archive/era2-prefield/pac_confluence_xi/) | Prediction | Forecasting symbolic collapse outcomes |
| [recursive_entropy](archive/era1-symbolic/recursive_entropy/) | Core | Recursive entropy emergence simulations |
| [recursive_gravity](archive/era1-symbolic/recursive_gravity/) | Gravity | Recursive gravitational field modeling |
| [recursive_tree](archive/era1-symbolic/recursive_tree/) | Structure | Recursive tree pattern formation under entropy |
| [symbolic_fractal_pruning](archive/era1-symbolic/symbolic_fractal_pruning/) | Pruning | Recursive calculus-based symbolic pruning |
| [symbolic_memory_agentic_decay_test](archive/era1-symbolic/symbolic_memory_agentic_decay_test/) | Memory | Memory reinforcement/decay in agentic fields |
| [symbolic_superfluid_collapse_pi](archive/era1-symbolic/symbolic_superfluid_collapse_pi/) | Superfluids | Pi-field collapse in superfluid states |
| [pi_harmonics](archive/era2-prefield/navier-stokes/) | Mathematics | Pi harmonic resonance in symbolic structure |
| [language_to_logic](archive/era2-prefield/information_amplification/) | Logic | Natural language to structured symbolic logic |
| [entropy_information_polarity_field](archive/era1-symbolic/entropy_information_polarity_field/) | Polarity | Black/white hole polarity field simulations |

---

## Spikes

Exploratory work not yet promoted to experiments. No structure requirements.

| Spike | What It Explores |
|-------|-----------------|
| [darkmatter_SEC_WIP](archive/spike-darkmatter-sec/) | Dark matter via SEC (work in progress) |
| [infodynamic_gravity](spikes/infodynamic_gravity/) | Gravity from infodynamics |
| [n_scale_dependence](spikes/n_scale_dependence/) | N scale dependence |

---

## Concept Index

Where to find specific physics concepts in the experiments.

| Concept | Primary Experiments | Key Result |
|---------|-------------------|------------|
| **Alpha (fine structure)** | [M6/exp_09](papers/series/PACSeries/v0.3/symmetry_primitive_scoped_mediation/Data/), [M8/exp_01](papers/series/PACSeries/v0.3/cosmological_predictions_cascade_clock/Data/) | 5.7 ppm, #1 of 10,440 combinations |
| **Area law** | [M11/exp_04](experiments/milestones/milestone11/) | Information scales as M^2 from gradient |
| **Arrow of time** | [M9/exp_06](experiments/sidecars/midnight/), [M11/exp_09](experiments/milestones/milestone11/) | Landauer irreversibility |
| **Born rule** | [quantum_validation/born_rule](archive/era1-symbolic/quantum_validation/born_rule/) | Reproduced from SEC |
| **Bounce (Planck star)** | [M11/exp_11](experiments/milestones/milestone11/) | t_bounce = 1 t_Planck, constant across masses |
| **Cascade clock** | [M9/exp_01-03](experiments/sidecars/midnight/) | N(t) = a + (1/ln(phi))*ln(t_lookback) |
| **Casimir effect** | [M2](experiments/milestones/milestone2/) | From PAC recursion |
| **CKM matrix** | [M5/exp_08](experiments/milestones/milestone5/) | Fibonacci arctangent ratios |
| **Cosmological constant** | [M8/exp_08](papers/series/PACSeries/v0.3/cosmological_predictions_cascade_clock/Data/) | log10(rho_Lambda) = -122.09, 0.09 orders |
| **D=3 (spatial dimensions)** | [M1](experiments/milestones/milestone1/), [M7/exp_10](experiments/milestones/milestone9/) | D=3 unique from MED/symmetry |
| **Dark energy (w)** | [M9/exp_09](experiments/sidecars/midnight/) | w(z=0) = -0.987; DESI tension (honest failure) |
| **Dark matter** | [M6/exp_05](papers/series/PACSeries/v0.3/symmetry_primitive_scoped_mediation/Data/), [M8/exp_02-03](papers/series/PACSeries/v0.3/cosmological_predictions_cascade_clock/Data/) | 6.44 keV, depth 73 |
| **De-actualization** | [M5/exp_12-13](experiments/milestones/milestone5/) | Completes PAC cycle, 24% drift reduction |
| **Feigenbaum constants** | [sec_threshold_detection](papers/series/PACSeries/v0.2/feigenbaum_fibonacci_arithmetic/Data/) | r_inf 13 digits, delta 8 digits |
| **Force hierarchy** | [M6/exp_04](papers/series/PACSeries/v0.3/symmetry_primitive_scoped_mediation/Data/) | log(alpha_G^-1)/log(alpha_EM^-1) = phi^6 |
| **Graviton** | [M11/exp_07-08](experiments/milestones/milestone11/) | Spin-2, massless, 2 polarizations from cascade |
| **Gravity-time duality** | [M9/exp_05](experiments/sidecars/midnight/) | g_out = g_in^2, exact for phi only |
| **Hawking radiation** | [M11/exp_05](experiments/milestones/milestone11/) | T*M = 1/(8*pi), CV = 7.8e-17 |
| **Higgs mass** | [M5/exp_07](experiments/milestones/milestone5/) | lambda = phi/(4*pi), 83 ppm |
| **Hubble constant** | [M8/exp_07](papers/series/PACSeries/v0.3/cosmological_predictions_cascade_clock/Data/), [M9/exp_08](experiments/sidecars/midnight/) | phi^(1/6)*H_CMB, 0.05sigma of SH0ES |
| **JWST** | [M8/exp_09](papers/series/PACSeries/v0.3/cosmological_predictions_cascade_clock/Data/) | z-dependent cascade floor, z_cascade = ln(phi)*6 |
| **Koide formula** | [M1](experiments/milestones/milestone1/) | Q = F3/(F3+F2) = 2/3, 0.5 ppm |
| **Kolmogorov -5/3** | [M4](experiments/milestones/milestone4/), [navier-stokes](archive/era2-prefield/navier-stokes/) | From PAC cascade |
| **Landauer principle** | [landauer_erasure_structure](papers/series/PACSeries/v0.2/structure_cost_of_erasure/Data/), [M11/exp_09](experiments/milestones/milestone11/) | A/(A+xi) = ln(phi), grounds cascade |
| **Lorentz invariance** | [M4](experiments/milestones/milestone4/) | Unique PAC partition |
| **Maxwell equations** | [maxwell_from_pac_sec](experiments/milestones/maxwell_from_pac_sec/) | Depth-2 recursion, D=3 from MED |
| **Neutrino masses** | [M6/exp_06](papers/series/PACSeries/v0.3/symmetry_primitive_scoped_mediation/Data/), [M8/exp_05](papers/series/PACSeries/v0.3/cosmological_predictions_cascade_clock/Data/) | sum < 0.12 eV, normal hierarchy |
| **Page curve** | [M11/exp_06](experiments/milestones/milestone11/) | Peaks at k/N = 0.5, epsilon-PAC violation |
| **Phi artifact test** | [phi_artifact_test](experiments/milestones/phi_artifact_test/) | Phi is genuine, not framework artifact |
| **Planck scale** | [M11/exp_01-02](experiments/milestones/milestone11/) | Response-time crossover at depth-183 |
| **PMNS matrix** | [M5/exp_08](experiments/milestones/milestone5/) | Fibonacci arctangent ratios, < 0.3 deg |
| **Primes** | [sec_prime_manifold](papers/standalone/golden_ratio_prime_distribution/Code/), [prime_growth_dynamics](papers/series/PACSeries/v0.2/balance_constant_decomposition/Data/) | SEC stress field separates primes at 1/phi |
| **RG running** | [M2](experiments/milestones/milestone2/), [M5/exp_05](experiments/milestones/milestone5/) | PAC-consistent renormalization |
| **Rule 110** | [cellular_automata_pac_attractors](papers/standalone/cellular_automata_xi_clustering/Code/) | P/A = 1.0579, Class IV clusters at Xi |
| **S8 tension** | [M9/exp_07](experiments/sidecars/midnight/) | 3.22sigma -> 0.07sigma, 98% reduction |
| **She-Leveque** | [navier-stokes](archive/era2-prefield/navier-stokes/) | k = d*F_{d+1}, 14.3x more accurate |
| **Singularity resolution** | [M11/exp_04](experiments/milestones/milestone11/) | Cascade saturation, Kretschner finite |
| **Strong force** | [M5/exp_01-05](experiments/milestones/milestone5/) | Already in cascade-depth tiling operator |
| **Weinberg angle** | [M1](experiments/milestones/milestone1/), [pac_confluence_xi](archive/era2-prefield/pac_confluence_xi/) | sin^2(theta_W) = F4/F7 = 3/13 |
| **Xi (balance constant)** | See [Three Derived Constants](#three-derived-constants) above | gamma + ln(phi) = 1.0584 |
| **Z' boson** | [M8/exp_04](papers/series/PACSeries/v0.3/cosmological_predictions_cascade_clock/Data/) | 395 GeV, width 64 MeV, not excluded |

---

## Core Module Chain

The milestones build on each other through a Python module chain:

```
milestone8/core/bsm.py          # BSM predictions engine
    |
    v
milestone9/core/infodynamics.py  # Cascade clock, S8, H0
    |
    v
milestone10/core/foundations.py   # Self-application, laws-as-equilibria
    |
    v
milestone11/core/quantum_gravity.py  # Planck scale, Hawking, graviton
```

Each module imports its predecessor. The chain ensures consistency — M11 results can't contradict M8 predictions because they're built on the same computation.

---

## Other Resources

| Resource | Path | Description |
|----------|------|-------------|
| Theory overview | [dawn-field-theory.md](theory/dawn-field-theory.md) | Framework, results, predictions, limitations |
| Infodynamics | [infodynamics.md](theory/infodynamics.md) | Information as generator of structure |
| For AI Labs | [for_ai_labs.md](theory/for_ai_labs.md) | AI/ML-focused overview |
| Origin story | [origin_of_infodynamics.md](theory/origin_of_infodynamics.md) | How DFT started (historical) |
| Published papers | [preprints/](papers/) | PACSeries and individual papers on Zenodo |
| Formal theorems | [09_FORMAL_THEOREMS.md](archive/era2-prefield/pac_confluence_xi/papers/09_FORMAL_THEOREMS.md) | Collected formal statements |
| Corrections registry | [theory/corrections.md](./theory/corrections.md) | Honest record of what we got wrong |
| Current roadmap | [roadmaps/current_roadmap.md](./roadmaps/current_roadmap.md) | M12 planning, open problems |
| Lexicon | [theory/lexicon.yaml](./theory/lexicon.yaml) | Formal term definitions |
| Environment | [ENVIRONMENT.md](./ENVIRONMENT.md) | Setup and reproducibility |

---

## Entry Points by Audience

**Physicist:** Start with [dawn-field-theory.md](theory/dawn-field-theory.md) (the framework and results), then pick a milestone that interests you and read its README. The [concept index](#concept-index) maps specific topics to experiments.

**Mathematician:** Start with [pac_confluence_xi/](archive/era2-prefield/pac_confluence_xi/) (Fibonacci arithmetic proofs) and [sec_prime_manifold/](papers/standalone/golden_ratio_prime_distribution/Code/) (SEC in number theory). The formal theorems are in [09_FORMAL_THEOREMS.md](archive/era2-prefield/pac_confluence_xi/papers/09_FORMAL_THEOREMS.md).

**ML/AI Researcher:** Start with [for_ai_labs.md](theory/for_ai_labs.md), then see the [GAIA POCs](https://github.com/dawnfield-institute/dawn-models) and the [Pythia validation](papers/ml_validation_pythia_gpt2/).

**Agent/LLM:** Use `meta.yaml` files for structured navigation. Start with [CLAUDE.md](./CLAUDE.md) for repo conventions, then use the [concept index](#concept-index) to find specific experiments. The `kronos_search` and `kronos_navigate` MCP tools provide semantic search.

**Skeptic:** Start with [theory/corrections.md](./theory/corrections.md) (what we got wrong) and Section 7 of [dawn-field-theory.md](theory/dawn-field-theory.md) (honest limitations). Then pick any experiment and run the scripts yourself.

---

*This map covers 73 experiment directories (752 numbered experiments) across 15 milestones plus sidecars, and 3 spikes. The authoritative per-experiment list is [EXPERIMENTS.md](experiments/EXPERIMENTS.md). Last updated: August 2026.*

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
  depth <= 2, nodes <= 3                       B(x,t) = grad^2(E-I) + ...
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
| **SEC** | Dynamics | [milestone1](experiments/milestones/milestone1/) | [sec_prime_manifold](experiments/studies/sec_prime_manifold/), [sec_threshold_detection](experiments/studies/sec_threshold_detection/) |
| **MED** | Optimization | [milestone3](experiments/milestones/milestone3/) | [navier-stokes](archive/era2-prefield/navier-stokes/), [milestone10 exp_15](experiments/milestones/milestone10/) |
| **RBF** | Geometry | [milestone7 exp_08](experiments/milestones/milestone7/) | [pre_field_recursion](archive/era2-prefield/pre_field_recursion/) |

The theory argument (why these axioms and not others) is in M10 SYNTHESIS at the workspace root, tested in [milestone10](experiments/milestones/milestone10/).

---

## Three Derived Constants

### phi = (1+sqrt(5))/2 = 1.618034...

The unique stable fixed point of PAC recursion. Selected by b^2 - b - 1 = 0 (gravity-time duality).

| Where It Appears | Experiment | Result |
|-----------------|------------|--------|
| Cascade attenuation | [milestone7/exp_04](experiments/milestones/milestone7/) | Emergent 1/phi (R^2=0.995) |
| Higgs self-coupling | [milestone5/exp_07](experiments/milestones/milestone5/) | lambda = phi/(4*pi), 83 ppm |
| Hubble correction | [milestone8/exp_07](experiments/milestones/milestone8/) | phi^(1/6) matches SH0ES |
| Force hierarchy | [milestone6/exp_04](experiments/milestones/milestone6/) | phi^6 = strong/EM ratio |
| Neural networks | [ml_validation_pythia_gpt2](papers/standalone/ml_validation_pythia_gpt2/) | phi-crossing at step 512 |
| Prime stress field | [sec_prime_manifold](experiments/studies/sec_prime_manifold/) | 1/phi threshold |
| Cellular automata | [cellular_automata_pac_attractors](experiments/studies/cellular_automata_pac_attractors/) | P/A = 1.0579 at Rule 110 |
| Artifact test | [phi_artifact_test](experiments/studies/phi_artifact_test/) | Not an artifact |

### ln(phi) = 0.48121...

The Landauer erasure cost per cascade level. Independently grounded in thermodynamics.

| Where It Appears | Experiment | Result |
|-----------------|------------|--------|
| Erasure structure | [landauer_erasure_structure](experiments/studies/landauer_erasure_structure/) | A/(A+xi) = ln(phi), 0.76% |
| Cascade clock slope | [milestone9/exp_01](experiments/milestones/milestone9/) | 1/ln(phi) = 2.0781 exact |
| Arrow of time | [milestone9/exp_06](experiments/milestones/milestone9/) | 0.665 nats/level entropy production |
| Landauer universality | [milestone11/exp_09](experiments/milestones/milestone11/) | Forward/reverse ratio super-exponential |
| JWST z-cascade | [milestone8/exp_09](experiments/milestones/milestone8/) | z_cascade = ln(phi)*6 |

### Xi = gamma + ln(phi) = 1.0584...

The universal balance constant. Zero free parameters — gamma from harmonic counting, phi from duality.

| Where It Appears | Experiment | Result |
|-----------------|------------|--------|
| Cellular automata | [cellular_automata_pac_attractors](experiments/studies/cellular_automata_pac_attractors/) | Class IV clusters at Xi, p < 8.58e-8 |
| Scope fixed point | [milestone6/exp_07](experiments/milestones/milestone6/) | Xi attractor, Euler gap 0.09% |
| Algebraic uniqueness | [milestone9/exp_02](experiments/milestones/milestone9/) | g_out = g_in^2 selects Xi uniquely |
| Boundary crossing | [milestone7/exp_03](experiments/milestones/milestone7/) | Xi per symmetric restoration |
| Mobius pairing | [oscillation_attractor_dynamics](experiments/studies/oscillation_attractor_dynamics/) | Xi = pi/55 to 6.8e-9 |
| Wilson-Fisher | [milestone3](experiments/milestones/milestone3/) | nu = 2/(3*Xi) |
| Xi universality | [milestone10/exp_08](experiments/milestones/milestone10/) | Extension across domains |

---

## Claims, resolved across layers

The point of this map. Every load-bearing claim, and where each of its four layers lives:
what asserts it, what proves it, what measured it, where it was published.

| Claim | Status | Proved | Measured | Published |
|---|---|---|---|---|
| sin^2(theta_W) = tan(theta_C) = 3/13 | settled | — | [M5](experiments/milestones/milestone5/), [M1](experiments/milestones/milestone1/) | PACSeries v0.2 |
| alpha_EM at 5.7 ppm, #1 of 10,440 | settled | — | [M6 exp_09](experiments/milestones/milestone6/) | PACSeries v0.2 |
| Higgs mass, lambda = phi/4pi, 83 ppm | settled | — | [M5](experiments/milestones/milestone5/) | PACSeries v0.2 |
| **Xi = gamma + ln(phi), zero free parameters** | **proven** | [theorems](formal/theorems/README.md#the-origin-of-xi) | [M11 exp_09](experiments/milestones/milestone11/) | PACSeries v0.2 |
| **PAC is spectral confinement (drift 2.4e-15)** | **proven** | [theorems](formal/theorems/README.md#pac-is-spectral-confinement) | [M10 exp_14](experiments/milestones/milestone10/) | — |
| Planck scale from depth-183, zero free parameters | settled | — | [M11](experiments/milestones/milestone11/) | PACSeries v0.3 |
| S8 tension 3.22 sigma -> 0.09 sigma (blind) | settled | — | [M9](experiments/milestones/milestone9/) | PACSeries v0.3 |
| **C_6 = -I; theta(m) = m theta_T closed form** | **proven** | [theorems](formal/theorems/README.md#the-holonomy-closed-form) | [M15 exp_04](experiments/milestones/milestone15/) | — |
| **Connection generator = box momentum operator** | **proven** | [theorems](formal/theorems/README.md#the-connection-generator-is-the-box-momentum-operator) | [M15 exp_06](experiments/milestones/milestone15/) | — |
| **No isomorphism-invariant PD metric on ADE** | **proven negative** | [theorems](formal/theorems/README.md#psd-degeneracy-is-fundamental--a-proven-impossibility) | [M13.5 exp_16](experiments/milestones/milestone13/) | — |
| Geiger-Nuttall is a universality theorem | settled | — | [R exp_16](experiments/sidecars/milestone-r/) | — |
| Coherence limit is universal | **falsified** | — | [M13.5 exp_15](experiments/milestones/milestone13/) (0/4) | — |
| Holonomy is dynamically active | **open** | — | [M15 Phase 2](experiments/milestones/milestone15/) | — |
| SEC <-> Navier-Stokes correspondence | **conjecture** | [conjectures](formal/conjectures/README.md) | [ade_cascade](experiments/studies/ade_cascade/) | — |
| phi enters as a projection, not a magnitude (M18 thesis) | **open** | — | [M18 Blocks A/C](experiments/milestones/milestone18/) | — |
| **sigma-ledger: conjugation = complementation; charpoly = q·sigma(q) on the one-5 family** | **proven** | [theorems](formal/theorems/README.md#the-σ-ledger-conjugation-is-complementation) | [M18 exp_06/07/12](experiments/milestones/milestone18/) | — |
| **Every one-5 Coxeter diagram has a tree parent: the branched double cover** | **proven** | [theorems](formal/theorems/README.md#the-construction-theorem-parents-are-branched-double-covers) | [M18 exp_11/13/15](experiments/milestones/milestone18/) (zero orphans, k ≤ 10) | — |
| **Matching form sqrt5·R = S + 2Pi and the strict-fold laws on constructions** | **proven** | [theorems](formal/theorems/README.md#the-matching-form-and-every-strict-fold-law-on-construction-parents) | [M18 exp_12/13/15](experiments/milestones/milestone18/) (7/7, 13/13, 47/47) | — |
| **Denominator bound: den(5·b) divides 2^deg(q1)·Res(q0, q1) for the reflection polynomial** | **proven** | [theorems](formal/theorems/README.md#the-denominator-bound-on-the-reflection-polynomial) | [M18 r21b](experiments/milestones/milestone18/) (208/208 halves) | — |
| Polynomial integrality 5·b(t) ∈ ℤ[t] unconditionally | **falsified** (2026-09-02) | — | [M18 r21](experiments/milestones/milestone18/) (three construction parents at n = 20, den 3; the matrix form stands) | — |
| Every strict Galois fold is a construction parent (rigidity) | **conjecture** | [conjectures](formal/conjectures/m18_open.md) | [M18 exp_15](experiments/milestones/milestone18/) (47/47 at n = 20) | — |
| Heat-kernel sheet asymmetry peaks at the fold's branch | settled | — | [M18 exp_18](experiments/milestones/milestone18/) (47/47, sealed null) | — |
| The fold is physically reached by the corpus's own operators (Block D) | **falsified at measured sizes** (2026-09-03) | — | [M18 exp_09](experiments/milestones/milestone18/) (0 carriers / 173 objects: complete PAC trees d ≤ 8, 166 growth trees, 4 unicyclics; the H₂/H₃-type sectors carry no certificate — class without representative) | — |

A dash under *Proved* means the result is measured, not derived — the distinction
[`formal/`](formal/README.md) exists to keep. **Open** carries a standing kill-sentence:
*if holonomy is dynamically inert, it is mathematics, not physics.* M18's open rows carry
theirs: *if E₈-derived spectra do not split into two φ-scaled families and the golden form adds
no separating power at the orbit boundary, φ is not structural in this corpus* — status
wounded, not dead (clause C met, clause B not).

---

## The Milestone Stack

M1-M5 derive the Standard Model. M6-M9 find the mechanism. M10-M15 turn the framework on
itself and ask why these axioms. Scores are `passed/possible`; see
[`STANDARDS.md`](STANDARDS.md) section 2.6 for the convention and why totals are not
multiples of four.

| | Thesis | Score | Key results |
|---|---|---|---|
| **M1** [Standard Model chain](experiments/milestones/milestone1/) | PAC + SEC derive SM parameters from Fibonacci arithmetic | — | alpha_EM 5.7 ppm, sin^2(theta_W) = 3/13, Koide, D = 3 from five paths, Z' at 395 GeV |
| **M2** [Mass derivations](experiments/milestones/milestone2/) | Mass ratios from PAC recursion levels | — | mu/e at 5 ppm, p/e at 0.0083%, She-Leveque from first principles |
| **M3** [Energy & falsification](experiments/milestones/milestone3/) | Test against energy conservation and Landauer bounds | — | Wilson-Fisher nu = 2/(3 Xi); two honest falsifications recorded |
| **M4** [Relativity & turbulence](experiments/milestones/milestone4/) | Lorentz as unique PAC partition | — | Kolmogorov -5/3 from PAC; cascade engine dimension-independent |
| **M5** [SM completion](experiments/milestones/milestone5/) | Close remaining SM gaps | 13 exp | Higgs 83 ppm (lambda = phi/4pi), PMNS within 0.3 deg, sin^2(theta_W) = tan(theta_C) |
| **M6** [Scoped mediation](experiments/milestones/milestone6/) | Forces differ by Fibonacci depth | **34/40** | alpha_EM #1 of 10,440 combinations; phi^6 hierarchy; dark sector at depth 73 |
| **M7** [Symmetry primitive](experiments/milestones/milestone7/) | Symmetry is pre-axiomatic | **37/40** | phi from cross-scale self-reference; emergent 1/phi at R^2 = 0.995 |
| **M8** [BSM predictions](experiments/milestones/milestone8/) | Pre-register falsifiable predictions | **47/48** | CC at 0.09 orders; 10 predictions, 0 excluded by data |
| **M9** [Infodynamic mechanism](experiments/milestones/milestone9/) | The cascade is a temporal clock | **37/40** | S8 tension 3.22 sigma -> 0.09 sigma (blind); free parameters 2 -> 1 |
| **M10** [Symmetry self-application](experiments/milestones/milestone10/) | Self-applied symmetry is the unique generative primitive | **64/71** | PAC/SEC/MED shown non-independent; PAC = spectral confinement, drift 2.4e-15 |
| **M11** [Quantum gravity](experiments/milestones/milestone11/) | Where gravitational response-time is exceeded | **52/52** | Planck from depth-183, zero free parameters; **origin of Xi proven**; no singularity |
| **M12** [Connection as primitive](experiments/milestones/milestone12/) | Connection = addition = ADE | **49/52** | SU(2) and SU(3) the only Fibonacci-compatible types; Lorentz from SEC complexification |
| **M13 + 13.5** [Identity as complement](experiments/milestones/milestone13/) | Identity *is* complement; relativity as complement-transformation | **53/68** | Definitional parallax; proper time = dt/cosh(eta); PSD degeneracy proven fundamental |
| **M14** [Quantum mechanics](experiments/milestones/milestone14/) | QM as complement-indeterminacy on the orbit quotient | **40/44** | Born rule from orbit measure; D_4 the only ADE type with genuine uncertainty |
| **M15** [The representative problem](experiments/milestones/milestone15/) | The framework computes cohomology; observers supply gauge | Phase 1 closed | C_6 = -I proven; connection generator = box momentum operator |
| **M16** [Relational locality](experiments/milestones/milestone16/) | How neighbours come to cohere | active (re-founded 2026-08-14) | The engine's large-scale component is real (coherent-power excess +12.5 sigma under PAC balance) and has no web geometry; mechanism re-founded |
| **M17** [Criticality](experiments/milestones/milestone17/) | The corpus's limits are critical points where identity changes scale | 6/9 (Block A) | Instrumentation front-loaded; exp_03 open; Blocks B-E unstarted |
| **M18** [Non-crystallographic completion](experiments/milestones/milestone18/) | phi enters as a projection (A4->H2, D6->H3, E8->H4), not a magnitude | **51/67** | Construction theorem (parents are branched double covers); rigidity 47/47 at n = 20; the fold's branch is dynamically visible (47/47); seven theorems filed |

### Sidecars

Real programs that do not continue the M14 -> M15 chain.

| | Thesis | Score | Key results |
|---|---|---|---|
| **R** [Radiation as ledger severance](experiments/sidecars/milestone-r/) | Radiation is PAC ledger severance | **60/112** | Geiger-Nuttall shown to be a universality theorem; graph Green's function reproduces the hydrogen spectrum to 0.68%; energy scale fixed by alpha(d)^2 m_mediator |
| **Midnight** [Observational contact](experiments/sidecars/midnight/) | Take DFT to observational data | **22/32** | Phase-rate primitive; PAC/SEC separation in SDSS; source of the invariant-registration rule |

---

## Every other experiment

The 33 studies, 2 spikes and 25 archived experiments are **not listed here**. They were,
in four hand-maintained tables, and those tables drifted — one of them claimed 51
experiments against a real 73 and its links rotted for months.

**[`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md)** is the index. It is
generated from each experiment's `meta.yaml`, grouped by lifecycle, and cannot disagree
with the corpus. Archived work is in [`archive/`](archive/) with its
[era guide](archive/README.md).

---

## Concept Index

Where to find specific physics concepts in the experiments.

| Concept | Primary Experiments | Key Result |
|---------|-------------------|------------|
| **Alpha (fine structure)** | [M6/exp_09](experiments/milestones/milestone6/), [M8/exp_01](experiments/milestones/milestone8/) | 5.7 ppm, #1 of 10,440 combinations |
| **Area law** | [M11/exp_04](experiments/milestones/milestone11/) | Information scales as M^2 from gradient |
| **Arrow of time** | [M9/exp_06](experiments/milestones/milestone9/), [M11/exp_09](experiments/milestones/milestone11/) | Landauer irreversibility |
| **Born rule** | [quantum_validation/born_rule](archive/era1-symbolic/quantum_validation/born_rule/) | Reproduced from SEC |
| **Bounce (Planck star)** | [M11/exp_11](experiments/milestones/milestone11/) | t_bounce = 1 t_Planck, constant across masses |
| **Cascade clock** | [M9/exp_01-03](experiments/milestones/milestone9/) | N(t) = a + (1/ln(phi))*ln(t_lookback) |
| **Casimir effect** | [M2](experiments/milestones/milestone2/) | From PAC recursion |
| **CKM matrix** | [M5/exp_08](experiments/milestones/milestone5/) | Fibonacci arctangent ratios |
| **Cosmological constant** | [M8/exp_08](experiments/milestones/milestone8/) | log10(rho_Lambda) = -122.09, 0.09 orders |
| **D=3 (spatial dimensions)** | [M1](experiments/milestones/milestone1/), [M7/exp_10](experiments/milestones/milestone7/) | D=3 unique from MED/symmetry |
| **Dark energy (w)** | [M9/exp_09](experiments/milestones/milestone9/) | w(z=0) = -0.987; DESI tension (honest failure) |
| **Dark matter** | [M6/exp_05](experiments/milestones/milestone6/), [M8/exp_02-03](experiments/milestones/milestone8/) | 6.44 keV, depth 73 |
| **De-actualization** | [M5/exp_12-13](experiments/milestones/milestone5/) | Completes PAC cycle, 24% drift reduction |
| **Feigenbaum constants** | [sec_threshold_detection](experiments/studies/sec_threshold_detection/) | r_inf 13 digits, delta 8 digits |
| **Force hierarchy** | [M6/exp_04](experiments/milestones/milestone6/) | log(alpha_G^-1)/log(alpha_EM^-1) = phi^6 |
| **Graviton** | [M11/exp_07-08](experiments/milestones/milestone11/) | Spin-2, massless, 2 polarizations from cascade |
| **Gravity-time duality** | [M9/exp_05](experiments/milestones/milestone9/) | g_out = g_in^2, exact for phi only |
| **Hawking radiation** | [M11/exp_05](experiments/milestones/milestone11/) | T*M = 1/(8*pi) from cascade geometry; the CV = 7.8e-17 figure was retired by M11's own Round-2 hardening as an algebraic identity (1/(8*pi*M) x M cancelling), not a measurement |
| **Higgs mass** | [M5/exp_07](experiments/milestones/milestone5/) | lambda = phi/(4*pi), 83 ppm |
| **Hubble constant** | [M8/exp_07](experiments/milestones/milestone8/), [M9/exp_08](experiments/milestones/milestone9/) | phi^(1/6)*H_CMB, 0.05sigma of SH0ES |
| **JWST** | [M8/exp_09](experiments/milestones/milestone8/) | z-dependent cascade floor, z_cascade = ln(phi)*6 |
| **Koide formula** | [M1](experiments/milestones/milestone1/) | Q = F3/F4 = 2/3, 0.0009% (9 ppm) |
| **Kolmogorov -5/3** | [M4](experiments/milestones/milestone4/), [navier-stokes](archive/era2-prefield/navier-stokes/) | From PAC cascade |
| **Landauer principle** | [landauer_erasure_structure](experiments/studies/landauer_erasure_structure/), [M11/exp_09](experiments/milestones/milestone11/) | A/(A+xi) = ln(phi), grounds cascade |
| **Lorentz invariance** | [M4](experiments/milestones/milestone4/) | Unique PAC partition |
| **Maxwell equations** | [maxwell_from_pac_sec](experiments/studies/maxwell_from_pac_sec/) | Depth-2 recursion, D=3 from MED |
| **Neutrino masses** | [M6/exp_06](experiments/milestones/milestone6/), [M8/exp_05](experiments/milestones/milestone8/) | sum < 0.12 eV, normal hierarchy |
| **Page curve** | [M11/exp_06](experiments/milestones/milestone11/) | Peaks at k/N = 0.5, epsilon-PAC violation |
| **Phi artifact test** | [phi_artifact_test](experiments/studies/phi_artifact_test/) | Phi is genuine, not framework artifact |
| **Planck scale** | [M11/exp_01-02](experiments/milestones/milestone11/) | Response-time crossover at depth-183 |
| **PMNS matrix** | [M5/exp_08](experiments/milestones/milestone5/) | Fibonacci arctangent ratios, < 0.3 deg |
| **Primes** | [sec_prime_manifold](experiments/studies/sec_prime_manifold/), [prime_growth_dynamics](experiments/studies/prime_growth_dynamics/) | SEC stress field separates primes at 1/phi |
| **RG running** | [M2](experiments/milestones/milestone2/), [M5/exp_05](experiments/milestones/milestone5/) | PAC-consistent renormalization |
| **Rule 110** | [cellular_automata_pac_attractors](experiments/studies/cellular_automata_pac_attractors/) | P/A = 1.0579, Class IV clusters at Xi |
| **S8 tension** | [M9/exp_07](experiments/milestones/milestone9/) | 3.22sigma -> 0.07sigma, 98% reduction |
| **She-Leveque** | [navier-stokes](archive/era2-prefield/navier-stokes/) | k = d*F_{d+1}, 14.3x more accurate |
| **Singularity resolution** | [M11/exp_04](experiments/milestones/milestone11/) | Cascade saturation, Kretschner finite |
| **Strong force** | [M5/exp_01-05](experiments/milestones/milestone5/) | Already in cascade-depth tiling operator |
| **Weinberg angle** | [M1](experiments/milestones/milestone1/), [pac_confluence_xi](archive/era2-prefield/pac_confluence_xi/) | sin^2(theta_W) = F4/F7 = 3/13 |
| **Xi (balance constant)** | See [Three Derived Constants](#three-derived-constants) above | gamma + ln(phi) = 1.0584 |
| **Z' boson** | [M8/exp_04](experiments/milestones/milestone8/) | 395 GeV, width 64 MeV, not excluded |

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
| Current roadmap | [roadmaps/current_roadmap.md](ROADMAP.md) | M12 planning, open problems |
| Lexicon | [theory/lexicon.yaml](./theory/lexicon.yaml) | Formal term definitions |
| Environment | [ENVIRONMENT.md](./ENVIRONMENT.md) | Setup and reproducibility |

---

## Entry Points by Audience

**Physicist:** Start with [dawn-field-theory.md](theory/dawn-field-theory.md) (the framework and results), then pick a milestone that interests you and read its README. The [concept index](#concept-index) maps specific topics to experiments.

**Mathematician:** Start with [pac_confluence_xi/](archive/era2-prefield/pac_confluence_xi/) (Fibonacci arithmetic proofs) and [`sec_prime_manifold/`](experiments/studies/sec_prime_manifold/) (SEC in number theory). The formal theorems are in [09_FORMAL_THEOREMS.md](archive/era2-prefield/pac_confluence_xi/papers/09_FORMAL_THEOREMS.md).

**ML/AI Researcher:** Start with [for_ai_labs.md](theory/for_ai_labs.md), then see the [GAIA POCs](https://github.com/dawnfield-institute/dawn-models) and the [Pythia validation](papers/standalone/ml_validation_pythia_gpt2/).

**Agent/LLM:** Use `meta.yaml` files for structured navigation. Start with [CLAUDE.md](./CLAUDE.md) for repo conventions, then use the [concept index](#concept-index) to find specific experiments. The `kronos_search` and `kronos_navigate` MCP tools provide semantic search.

**Skeptic:** Start with [theory/corrections.md](./theory/corrections.md) (what we got wrong) and Section 7 of [dawn-field-theory.md](theory/dawn-field-theory.md) (honest limitations). Then pick any experiment and run the scripts yourself.

---

*This map covers the experiments across 18 milestones plus sidecars and spikes; the authoritative per-experiment list and count is [EXPERIMENTS.md](experiments/EXPERIMENTS.md) (generated). Last updated: September 2026.*

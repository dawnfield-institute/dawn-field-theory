# Dawn Field Theory

## What This Is

The core physics repository for Dawn Field Institute. Contains the theoretical framework, 170+ experiments across 14 domains, and published papers for Dawn Field Theory (DFT) — a framework that derives physical constants and dynamics from two information-theoretic axioms: PAC (Potential-Actualization Conservation) and SEC (Symbolic Entropy Collapse).

This is the **primary source of truth** for the physics. The Kronos vault's 56 physics FDOs reference experiments and documents in this repo.

## Architecture

```
dawn-field-theory/
├── foundational/
│   ├── experiments/          # 61+ experiment directories (THE MAIN CONTENT)
│   │   ├── milestone1/       # Standard Model parameter derivations
│   │   ├── milestone2/       # Mass derivations, Navier-Stokes, Koide
│   │   ├── milestone3/       # Quantum validation, Landauer erasure
│   │   ├── milestone6/       # Scoped Mediation (10 experiments, 27/40)
│   │   ├── milestone7/       # Symmetry Primitive (10 experiments, 15/40)
│   │   ├── pac_confluence_xi/ # PAC-Ξ convergence proofs
│   │   ├── sec_prime_manifold/ # SEC in number theory
│   │   └── ... (61+ total)
│   ├── arithmetic/           # PACEngine — core mathematical tools
│   │   ├── PACEngine/        # Conservation math, geometric SEC
│   │   ├── EuclideanDistanceValidation/
│   │   └── HodgeMapping/
│   ├── docs/                 # Bridges, preprints, empirical alignment
│   │   └── preprints/        # Packaged papers with Code/ directories
│   └── lexicon.yaml          # Formal term definitions
├── spikes/                   # Exploratory work (not yet promoted to experiments)
│   ├── darkmatter_SEC_WIP/
│   └── infodynamic_gravity/
├── blueprints/               # Speculative applications
├── citations/                # DOI and citation management
├── models/                   # (minimal)
├── tools/                    # Utility scripts for repo maintenance
├── resources/                # External resources
├── roadmaps/                 # Planning documents
└── [ROOT .md files]          # Theory overview docs
```

## Key Root Documents

| File | Purpose |
|------|---------|
| `dawn-field-theory.md` | Full theory overview (start here for physics) |
| `infodynamics.md` | Infodynamics foundation |
| `origin_of_infodynamics.md` | Origin story and motivation |
| `for_ai_labs.md` | AI-targeted overview |
| `EPISTEMIC_CORRECTIONS_REGISTRY.md` | Honest record of corrections |
| `CITATION.cff` | Citation metadata (requires DOI verification to modify) |
| `map.yaml` | Generated CIP navigation map (~104KB, DO NOT edit manually) |

## Conventions

### Experiment Structure (REQUIRED)
Every experiment in `foundational/experiments/` must have:
- `meta.yaml` — schema v2.0 metadata
- `README.md` — hypothesis, status, key results, FDO links
- `scripts/` — numbered scripts (`exp_NN_name.py`)
- `results/` — output data (if scripts produce any)
- `journals/` — daily research logs (recommended for active work)
- `SYNTHESIS.md` — cross-connections (recommended)

See `STANDARDS.md` at workspace root for full spec.

### Script Naming
- `exp_01_baseline.py`, `exp_02_scaling.py`, etc.
- Results: `results/exp_NN_name_YYYYMMDD_HHMMSS.json`

### Spikes vs Experiments
- `spikes/` — exploratory, no structure requirements, may be promoted to experiments
- `foundational/experiments/` — structured, documented, must meet standards

### Status Values for Experiments
- `active` — currently being worked on
- `completed` — validated, results documented
- `archived` — historical, kept for reference
- `falsified` — hypothesis disproven (these are valuable)

## Related Repos

| Repo | Relationship |
|------|-------------|
| `kronos-vault` | 56 physics FDOs reference experiments here via `source_paths` |
| `fracton` | PAC math library consumed by experiments |
| `reality-engine` | Simulator that implements DFT dynamics |
| `dawn-models` | GAIA ML models that validate DFT predictions |
| `GRIM` | AI companion with skills for experiment management |

## Current State

- **130+ experiments** in `foundational/experiments/`
  - M11 score: 52/52 (100%) after hardening cycle (52→49→52)
  - M12 score: 49/52 (94%) — connection as primitive, ADE geometry, Lorentz derivation
  - M13 score: 48/52 core (92%) + 5/16 investigation = 53/68 (78%) after M13.5 stress testing + refinement. PSD degeneracy proven fundamental (exp_16: 0/4, no invariant metric can fix it). Random graph paradox partially explained (exp_17: 2/4, density + spectral radius). 7 honest failures total.
  - M14 score: 40/44 (91%) — quantum mechanics as complement-indeterminacy. Orbit Hilbert space resolves PSD. D_4 (S_3) uniquely non-abelian: only ADE type with genuine quantum uncertainty. Born rule from orbit measure, interference from SEC complexification, entanglement from product graphs. 12 predictions, 4 honest failures (algebraic not positional).
  - M6 score: 27/40 → 35/40 → 34/40 (85%) after v0.3 hardening cycle
- **Milestones 1-14** complete (SM parameters, mass derivations, quantum validation, relativity/gravity, SM completion, scoped mediation, symmetry primitive, BSM predictions, infodynamic mechanism, symmetry self-application, quantum gravity, connection primitive, identity as complement, quantum mechanics)
- **Milestone 5** complete — SM completion & simulator validation (13 experiments)
  - Higgs mass 83 ppm (lambda = phi/4pi), PMNS < 0.3 deg, sin^2(theta_W) = tan(theta_C) = 3/13
  - De-actualization completes PAC cycle, 24% scorecard improvement
- **Milestone 6** complete — Scoped Mediation: The Propagation Mechanism of DFT (10 experiments, 34/40 = 85%)
  - Transfer matrices, harmonic fixed-point convergence, force hierarchy from Fibonacci depth
  - alpha_EM 5.7 ppm (#1 of 10,440 Fibonacci combinations, 300x better than next), phi^6 0.30%, Euler gap 0.09%
  - KAN arithmetic transition (K increases rho=1.0, N decreases rho=-1.0), scope attenuation base=0.42 in phi range
  - Three key insights: weak force = actualization mechanism, Xi = conditional attractor, neutrinos complete PAC
  - Dark sector prediction: depth 73, alpha_73 = 2.48e-16, mass ~5.8 keV
- **Milestone 7** complete — The Symmetry Primitive (10 experiments, 37/40 = 93%)
  - Tests symmetry as pre-axiomatic foundation: Symmetry → Self-reference → Recursion → ADE → PAC/SEC/MED/RBF
  - Phi from cross-scale relational self-reference (not arbitrary maps), generalizes to b-nacci constants
  - Nothing unstable under multi-scale drive + conservation; Xi = gamma + ln(phi) per boundary crossing
  - Emergent 1/phi attenuation from dynamics (R²=0.995, non-tautological); 4/5 break types improve phi-balance, phi optimal
  - 100% compatibility with M1-M6, 60% directly illuminated, 12 new derivation paths
  - Cosmological constant 0.9 orders, neutrino splitting 4.4%, D=3 unique, w = -1 + 10⁻⁶¹
  - RBF memory damping fails (2/4), cross-topology consistency partial (3/4) — honest failures
- **Milestone 8** complete — BSM Predictions & Observational Contact (12 experiments, 47/48 = 98% after v0.3 hardening)
  - 10 pre-registered falsifiable predictions, 0 excluded by current data
  - CC at -122.09 (0.09 orders!), Hubble ratio phi^{1/6} at 0.075%, Omega_c at 0.46%
  - Z' at 395 GeV: not excluded (9× margin), width 64 MeV, 4/4 tests pass
  - Dark matter: 6.44 keV from cascade routes (0.09 orders), X-ray line at 3.2 keV ≈ 3.55 keV observed
  - Neutrino splitting improved 44% → 17% with PMNS correction
  - S8 = 0.787 (per-level dissipation), H₀ = 73.0 km/s/Mpc (BAO φ^{-1/6} correction)
  - JWST: z-dependent cascade floor matches z=8 (16%) and z=12 (4%), z_cascade = ln(φ)×6
- **Milestone 9** complete — The Infodynamic Mechanism (10 experiments, 37/40 = 92%)
  - Cascade clock N(t) = a + (1/ln(φ))·ln(t_lookback) unifies S8/Hubble/JWST data points
  - S8 tension resolved: 3.22σ → 0.07σ (98% reduction), S8(z=0.35)=0.769 vs 0.768 observed
  - Xi = γ + ln(φ) algebraically unique transition cost (g_out = g_in²)
  - Parameter reduction: 2→1 free parameter (t1=520 Myr anchors to first stars)
  - N_physical boundary handling: z=0→N_max, t<t1→N_max, t≥t1→clock formula floored at 1
  - Phi self-similarity in splitting algebra (interval ratios, not cumulative sums)
  - Discrete H0 tension: phi^{1/N_floor} matches SH0ES at 0.05σ
  - 4 new falsifiable predictions for Euclid, DESI, TDSL
  - Honest failures: 8.9% slope gap (noise w/ 3 points), DESI w(z) tension (wa=-0.15 vs -0.75)
- **Milestone 12** complete — Connection as Primitive (13 experiments, 49/52 = 94%)
  - Connection = addition = ADE (algebraic identity), SU(2)+SU(3) ONLY Fibonacci-compatible ADE types
  - F_7=13=1+3+8+1 gauge closure, laws as basin attractors with phi^(d2-d1) relaxation ratios
  - SEC complexification A_1→SL(2,C)≅SO(3,1) (Lorentz from connection+entropy)
  - 3 honest failures: rate-density proportionality too simplistic, info-Fiedler not proportional, basin depth needs physical coupling
  - 8 predictions (4P+2D+2C), zero M1-M11 contradictions
- **Milestone 13 + 13.5** complete — Identity as Complement (15 experiments, 51/60 = 85% after M13.5 stress testing)
  - Identity IS complement: vertex identity = complement spectrum (outside-in definition)
  - Definitional parallax: different observers compute different complements of same target
  - Complement-transformations form Weyl group; SEC extends Z_2 → SL(2,C) → SO(3,1)
  - Speed of light = coherence limit on complement-deformation, proper time = dt/cosh(eta)
  - Curvature from connection-density gradients, invariant interval from Killing form uniqueness
  - Complete derivation chain: self-loop → phi → PAC → ADE → complement → parallax → Weyl → SEC → Lorentz → ds² → c → proper time
  - 8 predictions (4P+2D+2C), zero M1-M12 contradictions
- **PACSeries** published on Zenodo (DOI: 10.5281/zenodo.15783623)
- **Active organization effort**: bringing all experiments to full standard, adding FDO source links

## Do Not

- Edit `map.yaml` manually (it's generated, ~104KB)
- Modify `CITATION.cff` without DOI verification
- Create experiments outside `foundational/experiments/`
- Create new root-level .md files (use `.changelog/` entries instead)
- Remove or rename experiment directories without updating Kronos FDO `source_paths`

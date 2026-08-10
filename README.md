# Dawn Field Theory

> When a hammer shatters glass, thermodynamics tells us where the energy goes — heat, sound, kinetic motion. But **new information was created**: each shard now has unique geometry, distinct edges, specific boundaries. Standard physics has no framework for where this structural information comes from.
>
> Dawn Field Theory explores what might be the missing half of physics — how information organizes, crystallizes, and drives the emergence of structure across every scale.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15783623.svg)](https://zenodo.org/records/15783623)

[Full theory →](theory/dawn-field-theory.md) · [Infodynamics →](theory/infodynamics.md) · [For AI Labs →](theory/for_ai_labs.md)

---

## Machine-Native Navigation

This repository supports structured, machine-readable exploration via directory-level `meta.yaml` metadata:

* **Entry points:** `map.yaml` and directory-level `meta.yaml` files
* **Semantic search:** Use `kronos_search` / `kronos_navigate` to explore by concept, experiment, or theory
* **For AI agents/scrapers:** See [`for_ai_labs.md`](theory/for_ai_labs.md) for a targeted overview

---

## Table of Contents

* [Two Axioms, One Derivation Chain](#-two-axioms-one-derivation-chain)
* [Key Discovery: Universal 0.020 Hz Resonance](#-key-discovery-universal-0020-hz-resonance)
* [Dawn Field Ecosystem](#-dawn-field-ecosystem)
* [Models](#-models)
* [Project Structure](#-project-structure)
* [Recommended Starting Points](#-recommended-starting-points)
* [Publications](#-publications)
* [Contributing & Community](#-contributing--community)
* [Topics](#-topics)

---

## 🎯 Key Discovery: Universal 0.020 Hz Resonance

Our PACSeries research identifies a universal organizing frequency emerging across systems from quantum to cosmic scales:
- **Mathematical identity**: r = 11/(8π) 
- **Convergence point**: Iteration 91 = √2 × π phase coverage
- **Validation**: 100% reproducibility, r = -0.999632 cosmic correlation
- **Scale range**: 20+ orders of magnitude (brain waves to quasars)
- **[Read the complete papers →](https://zenodo.org/records/17295103)**

---

## 🌍 Two Axioms, One Derivation Chain

The theory is not complex. It starts from two constraints:

| Primitive | Statement | Consequence |
|-----------|-----------|-------------|
| **PAC** (Potential-Actualization Conservation) | f(Parent) = Σ f(Children) | Unique stable solution: φ^(−k). The golden ratio isn't found — it's *necessary*. |
| **SEC** (Symbolic Entropy Collapse) | ∂S/∂t = α∇I − β∇H | Structure forms where information gradients dominate entropy gradients. |

From these two, everything else derives — not as curve-fitting, but as **necessary mathematical consequences**:

```
PAC axiom → φ cascade → ln(φ) per level → Ξ = γ + ln(φ)
         → Fibonacci structure → Feigenbaum constants (13 digits)
         → Standard Model parameters (5.7 ppm α)
         → Maxwell equations from depth-2 recursion
```

### What Has Actually Been Validated (170+ Experiments, 14 Domains)

| Domain | Key Finding | Precision | Source |
|--------|------------|-----------|--------|
| **Number Theory** | SEC partition → 1/φ at k=9; sieve conservation EXACT over 126 steps | 0.04% / exact | sec_prime_manifold, asymmetric_conservation |
| **Particle Physics** | sin²θ_W = 3/13, α from Fibonacci formula, μ/e mass ratio | 0.19% / 5.7 ppm / 5 ppm | pac_confluence_xi, milestone2 |
| **Chaos Theory** | Feigenbaum r∞ and δ from Fibonacci closed forms | 13 digits / 8 digits | milestone1 |
| **Cellular Automata** | Class IV (Turing-complete) rules cluster at Ξ | p < 8.58×10⁻⁸, 42.7× enrichment | cellular_automata_xi_clustering |
| **Neural Networks** | Pythia-70M φ-crossing at step 512 (143k checkpoints) | p = 0.0014 | ml_validation_pythia_gpt2 |
| **Information Geometry** | E=mc² in embedding spaces; model-specific c² constants | R²=1.0, 3σ | euclidean_distance_validation |
| **Fluid Dynamics** | Bounded complexity; She-Leveque k = d × F_{d+1} exactly | 3,375 parameter combos | navier-stokes, milestone2 |
| **Landauer Physics** | Erasure structure A/(A+ξ) ≈ ln(φ); ξ/A = 1.086 | 0.76% proximity | landauer_erasure_structure |
| **Electromagnetism** | Maxwell from PAC depth-2 recursion; D=3 from MED bounds | Derived, not fitted | maxwell_from_pac_sec |
| **Cosmology** | Universal 0.020 Hz resonance across 20+ orders of magnitude | r = −0.999632 | pac_series |
| **Biological Evolution** | Entropy wave correlations with phylogenetic trees | r > 0.8, p < 0.001 | evolution experiments |
| **DNA Repair** | BRCA1 mutation detection from entropy profiles alone | Without alignment | dna_repair |
| **Quantum** | Born rule, Landauer erasure, interference — all consistent | 3 validation modules | quantum_validation |
| **ML Architecture** | Zero-backprop learning with 100% transfer (GAIA) | Implemented | GAIA POC-019/020/021 |

### The Balance Constant: Ξ = γ + ln(φ) = 1.0584

Independently validated from four sources — a formula, a cellular automaton simulation, analytic derivation, and prime number theory:

| Source | Ξ | Error from γ + ln(φ) |
|--------|---|----------------------|
| Formula (1+π/55) | 1.0571 | 0.124% |
| Rule 110 measured | 1.0579 | 0.050% |
| Analytic (γ+ln(φ)) | 1.0584 | 0.000% |
| Mertens-derived | 1.0584 | 0.000% (algebraic) |

### Falsifiability

This framework is designed for testing. If any of the following are observed, the theory is wrong:

- PAC conservation fails in a hierarchical system that reaches equilibrium
- φ-scaling disappears from independent domains when sampling bias is controlled
- Ξ convergence from independent sources is shown to be coincidental
- Fibonacci-derived Standard Model parameters are numerologically equivalent to alternatives

*These are computational results across 752 numbered experiments in 73 experiment directories. Independent validation and physical experimentation are actively sought. See [UNIFIED_EVIDENCE.md](./papers/UNIFIED_EVIDENCE.md) for the complete derivation chain with full statistical details.*

---

## 🧩 Dawn Field Ecosystem

Dawn Field Theory is implemented across specialized repositories:

### 💎 [Fracton](https://github.com/dawnfield-institute/fracton)
**Core math library** — PAC operators, recursive-core exports, Fibonacci arithmetic, conservation math. The computational backbone consumed by all experiments.

### 🔧 [Reality Engine](https://github.com/dawnfield-institute/reality-engine)
**GPU simulator** — Möbius topology, Poincaré activation, 18 operators. Implements DFT dynamics as a running simulation with emergent coupling constants.

### 🧠 [Dawn Models](https://github.com/dawnfield-institute/dawn-models)
**GAIA ML validation** — modular intelligence architecture built on PAC conservation. Validates DFT predictions in neural network training dynamics (phi-convergence in Pythia-70M, zero-backprop learning).

### 🤖 [GRIM](https://github.com/dawnfield-institute/GRIM)
**AI research companion** — Agent SDK, 34 skills, 38 MCP tools, WebSocket streaming. Manages experiments, vault operations, and research workflows.

---

## 🧠 Models

*For implementation details, see the [Dawn Models repository](https://github.com/dawnfield-institute/dawn-models)*

### GAIA: Field-Native Intelligence
GAIA (Generalized Architectures for Intelligent Actualization) treats intelligence as emergent field balance between energy, information, entropy, and structure. Learning without backpropagation — through PAC conservation dynamics. Phi-convergence appears naturally in activation ratios (validated in Pythia-70M at 143k checkpoints, p = 0.0014). 100% knowledge transfer, zero catastrophic forgetting.

🌍 **Implementation**: [dawn-models/research/GAIA/](https://github.com/dawnfield-institute/dawn-models/tree/main/research/GAIA)

---

## 📂 Project Structure

| Path                        | Purpose                                                                 |
| --------------------------- | ----------------------------------------------------------------------- |
| `experiments/milestones/` | 73 experiments across 15 milestones — the main content ([index](experiments/EXPERIMENTS.md)) |
| `archive/era1-symbolic/`        | Theory documents, preprint packages with code/data/figures              |
| `archive/era1-symbolic/`  | PACEngine — core mathematical tools and conservation math               |
| `roadmaps/`                 | Current roadmap (M12 planning) and historical planning docs             |
| `spikes/`                   | Exploratory work not yet promoted to experiments                        |
| `archive/blueprints/`               | Speculative applications (energy, nuclear containment)                  |
| `citations/`                | DOI registry, contributor citations, and external references            |
| `resources/`                | Publication registry and supplementary materials                        |
| `tools/`                    | Utility scripts for repo maintenance                                    |

---

## 📚 Recommended Starting Points

1. **[Full Theory Document →](theory/dawn-field-theory.md)** - The framework, results, milestones, predictions, and honest limitations
2. **[Infodynamics →](theory/infodynamics.md)** - The theory paradigm: information as generator of structure
3. **[Theory Map →](./THEORY_MAP.md)** - Navigate the derivation chain, find experiments by concept
4. **[PACSeries Papers →](https://zenodo.org/records/17295103)** - Published papers with complete validation code
5. [Experiment Index →](experiments/EXPERIMENTS.md) - all 73 experiments by status, with scores
6. [Environment & Reproducibility →](./ENVIRONMENT.md)

---

## 🧪 Environment & Reproducibility

- Environment setup and version hints: see `ENVIRONMENT.md`
- PyTorch is not pinned in a global requirements file; install via the official selector per your CUDA/CPU setup
- All experiments are documented with reproducible code and data in the PACSeries package

---

## 📖 License

AGPL-3.0 — See [LICENSE](./LICENSE) and [LICENSE_APPENDIX.md](./LICENSE_APPENDIX.md) for the Epistemic Constraint Framework.

Maintained by **The Dawn Field Institute**. See [MISSION.md](./MISSION.md) for institutional guidelines.

---

## 🤝 Contributing & Community

**Ready to contribute?** See our comprehensive [CONTRIBUTION.md](./CONTRIBUTION.md) for:
- 📝 **Contributor registration** (required for PRs)
- 🎯 **Contribution guidelines** and project boundaries
- 🏷️ **Automated citation system** for substantial contributions
- 📋 **Quick start checklist** for new contributors
- ⚖️ **Publishing & attribution boundaries** 

**Citation & Attribution:**
- Substantial contributions are automatically cited via our GitHub Actions workflow
- See [`citations/README.md`](./citations/README.md) for the full citation system
- External references and theory literature: [`citations/external_citations/`](./citations/external_citations/)

**Community Channels:**
- [Visit Dawn Field website for more info](https://dawnfield.ca/)
- **Discord** (canonical announcements): [https://discord.gg/bR8mrbHP](https://discord.gg/bR8mrbHP)
- **Twitter/X**: [https://x.com/lornecodes](https://x.com/lornecodes)
- **Follow the author on Medium**: [https://medium.com/@lornecodes](https://medium.com/@lornecodes)

**Project Governance:** See [MISSION.md](./MISSION.md) for institutional guidelines.

---

## 🏷️ Topics

### Core

* `infodynamics` `information-theory` `pac-conservation` `symbolic-entropy-collapse`
* `golden-ratio` `fibonacci-arithmetic` `recursive-systems` `field-theory`

### Physics

* `theoretical-physics` `particle-physics` `cosmology` `quantum-gravity`
* `standard-model` `fine-structure-constant` `dark-matter` `dark-energy`

### Cross-Domain

* `nonlinear-dynamics` `chaos-theory` `cellular-automata` `number-theory`
* `fluid-dynamics` `landauer-principle` `complex-systems`

### AI & ML

* `machine-learning` `neural-network-validation` `field-native-intelligence`
* `open-research` `reproducible-science`

---

## 📚 Publications

All preprints are open access on Zenodo with complete code, data, and figures.

### Core Theory
- **[Dawn Field Theory Synthesis v2.0](https://zenodo.org/records/18087136)** — Unified framework for symbolic entropy collapse, recursive intelligence, and field dynamics
- **[Infodynamics v2.0](https://zenodo.org/records/18087191)** — Collapse as crystallization: recursive balance and the Dawn Field Theory
- **[Symbolic Entropy Collapse](https://zenodo.org/records/17024434)** — Topological dynamics, recursive harmonics, and quantum correspondence

### Validation & Cross-Domain
- **[PACSeries: Universal Resonance at 0.020 Hz](https://zenodo.org/records/17295103)** — 4 papers + complete validation code
- **[Cellular Automata Ξ Clustering](https://zenodo.org/records/18086711)** — Edge-of-chaos rules cluster at the universal balance operator
- **[Golden Ratio in Prime Distribution](https://zenodo.org/records/18086778)** — Fibonacci resonance in symbolic entropy collapse
- **[ML Validation: Pythia & GPT-2](https://zenodo.org/records/18086821)** — SEC/PAC dynamics in neural network training
- **[PAC Necessity Proof](https://zenodo.org/records/18086893)** — The golden ratio as universal attractor
- **[PAC Comprehensive Framework](https://zenodo.org/records/18087020)** — Unifying mathematics for physics, information theory, and intelligent systems

### Mathematical & Engineering
- **[MED Navier-Stokes v2.0](https://zenodo.org/records/18087212)** — Bounded symbolic principles in fluid dynamics complexity
- **[Recursive Mathematical Plasticity](https://zenodo.org/records/17041249)** — Entropy architecture for adaptive intelligence systems

### AI & Applications
- **[GAIA Field-Native Intelligence](https://zenodo.org/records/18086999)** — Learning without backpropagation through physics-based dynamics
- **[Human-Agent Resonance](https://zenodo.org/records/17023921)** — Framework for human-agent co-computational ecology

> Full metadata: [`citations/doi_registry.yaml`](./citations/doi_registry.yaml) · [`resources/publications_registry.yaml`](./resources/publications_registry.yaml)

---

> **Cite this work:**  
> Groom, P. (2025). Dawn Field Theory. Zenodo. [https://doi.org/10.5281/zenodo.15783623](https://doi.org/10.5281/zenodo.15783623)

> **Disclaimer:**  
> This repository is an open, exploratory research project. All results, models, and theoretical frameworks are preliminary and provided for community investigation, critique, and extension.  
> **No claims of finality or completeness are made.**  
> Observations, hypotheses, and experiments are documented transparently, and theoretical gaps or open questions are intentional areas for future exploration.  
> Users are encouraged to replicate, challenge, and build upon this work.  
> See `MISSION.md` and `CONTRIBUTION.md` for engagement guidelines.


© 2026 The Dawn Field Institute  
All rights reserved under AGPL-3.0 + Epistemic Constraint Framework



information conservation, potential-actualization conservation, PAC theory,
Dawn Field Theory, information geometry, infodynamics, information physics,
symbolic entropy collapse, macro emergence dynamics, Fibonacci arithmetic,
golden ratio, fine structure constant, Landauer principle, cascade dynamics,
conservation laws, emergence, recursive balance field, quantum gravity,
artificial intelligence, machine learning, field-native intelligence,
information theory, computational physics, cosmological constant
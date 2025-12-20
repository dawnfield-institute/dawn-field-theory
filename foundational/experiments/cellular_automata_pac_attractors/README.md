# Cellular Automata as PAC Attractor States

**Status**: 🔄 In Progress  
**Started**: December 20, 2025  
**Preregistration**: [CA.md](../../../CA.md)

---

## Overview

This experiment tests whether Cellular Automata (CA) rules represent discrete attractor states in PAC (Potential-Actualization-Conservation) phase space.

### Core Hypothesis

If CA rules are PAC attractors, the **same dimensionless invariants** should emerge from three independent measurement frameworks:

1. **Conservation Physics** (PAC dynamics)
2. **Geometric Topology** (Betti numbers, genus, Euler characteristic)
3. **Information Theory** (excess entropy, mutual information)

### Key Innovation

This is NOT a search for specific numbers (φ, e, π). This IS a test of **cross-framework structural convergence**.

---

## Connection to Existing Work

This experiment builds on validated findings from:

| Prior Work | Key Finding | Application Here |
|------------|-------------|------------------|
| [sec_prime_manifold](../sec_prime_manifold/) | φ emerges at edge of chaos | Test if CA Rule 110 shows same criticality |
| [information_amplification](../information_amplification/) | Attractor detection via potential gradients | Reuse attractor clustering algorithms |
| [PACEngine](../../arithmetic/PACEngine/) | Cross-framework validation pipeline | Adapt for CA domain |

---

## Experiment Structure

```
cellular_automata_pac_attractors/
├── core/                    # Reusable modules
│   ├── ca_simulator.py      # Elementary CA engine
│   ├── pac_embedding.py     # Map rules → PAC phase space
│   └── invariant_metrics.py # Cross-framework invariant computation
├── scripts/                 # Numbered experiment scripts
│   ├── exp_01_baseline_ca.py
│   ├── exp_02_pac_embedding.py
│   └── ...
├── results/                 # JSON output from experiments
├── journals/                # Daily research logs
└── figures/                 # Visualizations
```

---

## Quick Start

```bash
cd cellular_automata_pac_attractors
python scripts/exp_01_baseline_ca.py
```

---

## Falsification Criteria

From [preregistration](../../../CA.md):

1. **Cross-path invariant deviation > 5%** → Hypothesis falsified
2. **Rules cluster randomly in PAC space** → No attractor structure
3. **Topological invariants fail to predict dynamics** → Geometric framework fails

---

## Success Criteria

1. Cross-framework invariants agree within 5%
2. Wolfram Class III/IV rules cluster distinctly from Class I/II
3. Rule 110 shows invariants near φ (1.618 ± 0.05)

---

## References

- Wolfram, S. (2002). A New Kind of Science
- Langton, C. (1990). Computation at the Edge of Chaos
- [SEC Prime Manifold SYNTHESIS](../sec_prime_manifold/SYNTHESIS.md)

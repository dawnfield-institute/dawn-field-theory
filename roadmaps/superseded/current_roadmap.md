# Dawn Field Theory — Current Roadmap

**Updated:** 2026-04-29
**Status:** M11 complete. Framework stable through quantum gravity. Publication and extension phase.

---

## Where We Are

11 milestones complete. 117+ experiments. PACSeries v0.2 published on Zenodo.

| Milestone | Score | Status |
|-----------|-------|--------|
| M1-M4 | Complete | Stable foundation (SM parameters, masses, relativity) |
| M5 | 13/13 | SM completion (Higgs 83 ppm, PMNS, CKM) |
| M6 | 35/40 (88%) | Scoped mediation (alpha 5.7 ppm, force hierarchy) |
| M7 | 37/40 (93%) | Symmetry primitive (pre-axiomatic foundation) |
| M8 | 48/48 (100%) | BSM predictions (CC 0.09 orders, 10 falsifiable predictions) |
| M9 | 37/40 (92%) | Cascade clock (S8 resolved, 1 free parameter) |
| M10 | 90/115 (78%) | Symmetry self-application (laws as equilibria, iddea.md) |
| M11 | 52/52 (100%) | Quantum gravity (Planck derived, Hawking from PAC, graviton) |

---

## Near-Term: Publications

### PACSeries v0.3
Incorporate M4-M11 results into the paper series. Planning doc at `papers/series/PACSeries/v0.3_PLANNING.md`.

Papers to update or add:
- Paper 1 (Erasure Cost): Add M11 Landauer universality (contraction rate = ln(b) for any b)
- Paper 2 (Xi Decomposition): Add M9 algebraic uniqueness proof and M11 phi selection
- Paper 3 (Feigenbaum): Stable — 13 digits already published
- Paper 4 (Standard Model): Add M5-M6 results (Higgs, PMNS, alpha 5.7 ppm)
- Paper 5 (Classical Physics): Add M4 Kolmogorov, M10 laws-as-equilibria
- Paper 6 (Computational Validation): Add M7-M11 experiment results
- New Paper 7: Cascade Clock and Cosmological Tensions (M8-M9)
- New Paper 8: Quantum Gravity from Response-Time Crossover (M11)

### Formal Theorems
Update `09_FORMAL_THEOREMS.md` with Theorems 11-16 from M7-M11.

### pac_necessity_proof
Update through M11 (currently stops at M4).

---

## Medium-Term: M12 — Topology Change

Deferred from M11. Core questions:

1. **Topology change at MVAE**: What happens to the Mobius topology when cascade density reaches the minimum viable actualization event? Does topology change at the bounce?
2. **Graviton self-interaction**: Cascade depth > 1 interactions. Non-perturbative regime.
3. **Full non-perturbative calculations**: M11 is semi-classical. M12 would need to go beyond.

Prerequisites: M11 results stable, PACSeries v0.3 drafted.

---

## Open Problems (No Timeline)

### Physics
- **Alpha formula derivation**: Ranks #1 of 10,440 Fibonacci combinations but lacks first-principles explanation of WHY this specific combination
- **DESI wa tension**: DFT predicts wa ~ -0.15, observed wa ~ -0.75. Needs DESI DR2/DR3
- **Why depth 73?**: The dark sector prediction uses cascade depth 73 without derivation
- **Structural vs empirical tests**: ~60% of M11 tests are structural. Hard empirical tests await LISA, CTA, Euclid

### Infrastructure
- **External peer review**: PACSeries needs submission to refereed journals
- **Independent replication**: Framework needs external validation
- **Fracton SDK**: Integration with GAIA v2 conservation bus architecture

---

## Completed (Reference)

- M1-M11 milestone stack (see milestone directories in `experiments/milestones/`)
- PACSeries v0.2 on Zenodo (DOI: 10.5281/zenodo.15783623)
- Root document rewrite (April 2026)
- M11 hardening cycle (4 rounds, 7 tautologies found and fixed)
- Epistemic Corrections Registry entries 1-3

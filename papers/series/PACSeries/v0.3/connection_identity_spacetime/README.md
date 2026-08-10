# PACSeries Paper 10: Connection, Identity, and Spacetime

## Overview

This paper derives the Lorentz group SO(3,1), the invariant interval, and the speed of light from the graph-theoretic foundations of Dawn Field Theory (DFT) — with no spacetime postulates. The central claims are that *connection is addition* (connecting two nodes is algebraically identical to addition on the root lattice, and the graphs that close under it are exactly the ADE Dynkin diagrams), and that *identity is complement* (a vertex's identity is the spectrum of what it is not). SEC complexification of the simplest ADE type A₁ turns the discrete Weyl symmetry ℤ₂ into SL(2,ℂ) ≅ SO(3,1); the Killing form forces the Minkowski signature; and the speed of light is the coherence limit on complement-deformation.

## Key Results

| Result | Value | Section |
|--------|-------|---------|
| Connection = addition (PAC recursion IS ADE) | transfer-matrix spectral radius = φ | §2 |
| SU(2), SU(3) uniquely Fibonacci-compatible | 2 of 99 SU(N) tested | §2.2 |
| Gauge closure | F₇ = 13 = 1 + 3 + 8 + 1 | §2.2 |
| SEC complexification A₁ → sl(2,ℂ) ≅ so(3,1) | all 15 commutators close to 1.1×10⁻¹⁶ | §4.1 |
| Killing form signature (6-dim algebra) | (3,3) | §4.1, §6.3 |
| Killing form on vector representation | (1,3) = Minkowski | §6.3 |
| Metric uniqueness | 1-dim null space, proportionality −1.0 | §6.3 |
| Complement spectrum uniquely identifies vertex | all ADE orbits ≤ rank 8 | §5.1 |
| Complement-transformations = Weyl group | orbit counts match ⌈(n+1)/2⌉ | §6.1 |
| ADE complement-deformation rates bounded | [0.59, 2.00] | §7.1 |
| Proper time | dτ = dt / cosh(η) | §7.2 |

**Scores (from the frozen result snapshot in `Data/results/`):**

- **M12 (Connection as Primitive):** 49/52 (94%)
- **M13 core (Identity as Complement, exp_01–13):** 48/52 (92%)
- **M13.5 investigation (exp_14–17):** 5/16 (31%)
- **M13 + M13.5 total:** 53/68 (78%)

(The paper's headline M13 figure, 51/60 = 85%, counts M13 core + investigation exp_14 and exp_15 only; the full investigation set is 53/68 = 78%, matching the milestone README. See "Overclaims flagged" below.)

## Honest Failures

- **Complement inner product is PSD, not PD (exp_05 1/4, exp_14 3/4, exp_16 0/4):** same-orbit vertices have isomorphic complements, so any isomorphism-invariant metric gives zero distance. Proven **fundamental** — no invariant metric fixes it. (Resolved downstream by M14's orbit Hilbert space, Paper 11, where PSD becomes a feature.)
- **Coherence limit is NOT universal (exp_08 T3 3/4, exp_15 0/4):** the maximum complement-deformation rate is graph- and family-dependent (A-family oscillates, D-family converges to a different limit); discrete complement distances do not compose like relativistic rapidities. The speed limit is geometric, not Fibonacci-arithmetic.
- **M12 quantitative failures (3/52):** rate–density proportionality too simplistic (exp_04 T4), info–Fiedler proportionality fails (exp_05 T3), basin-depth discrimination needs physical coupling spanning 10³⁵⁺ orders (exp_06 T2).
- **Random-graph paradox (exp_17 2/4):** between-family variance does not dominate and the paradox persists at matched sizes; but CV decreases with density (r = −0.74) and spectral radius explains rate clustering (r = 0.69).

## Falsifiable Predictions

16 registered (8 per milestone). Highlights: SU(2)/SU(3) are the only Fibonacci-compatible ADE gauge groups; F₇ = 13 gauge closure; SEC complexification of A₁ gives exactly SO(3,1); Killing signature = (−,+,+,+) for complexified A₁; proper time dτ = dt/cosh(η) from graph boost; PSD orbit Gram matrix is fundamental (not a defect).

## Source Experiments

- `experiments/milestones/milestone12/` — 13 scripts (exp_01–exp_13): Connection as Primitive
- `experiments/milestones/milestone13/` — 17 scripts (exp_01–exp_17): Identity as Complement + M13.5 investigation

The packaged scripts in `Code/experiments/` are verbatim copies, kept in per-milestone subdirectories (`milestone12/`, `milestone13/`) because the two milestones share the filename `exp_12_cross_milestone_compatibility.py` and both number 01–13. They import the shared DFT core chain (`identity_complement → connection_geometry → quantum_gravity → foundations`, milestones 13 → 12 → 11 → 10); see `Code/trace.yaml` for full provenance.

## Dependencies

- Paper 7: The Symmetry Primitive (φ and ADE from self-reference)
- Also builds on Paper 1 (PAC conservation) and Paper 8 (quantum gravity / response-time framework)

Paper 10 enables Paper 11 (Quantum Mechanics from Graph Structure), which builds the orbit Hilbert space on these same ADE graphs.

## Reproduction

```bash
# from a checkout of the dawn-field-theory repo, with numpy/scipy/matplotlib installed:
cd papers/legacy/cognition_index_protocol/v0.3/connection_identity_spacetime/Code
python reproduce.py --list      # list all 30 experiments
python reproduce.py             # run all 30
python reproduce.py 12_03       # run milestone12 exp_03 only
python reproduce.py 13_09       # run milestone13 exp_09 only
```

Fresh runs regenerate timestamped JSON into the source `milestone12/results/` and `milestone13/results/` directories (the experiments' native output location, via the core `save_m*_results` helper). The frozen snapshot used to write the paper is in `Data/results/`. Figures are regenerated by `Code/generate_figures.py` from `Data/results/`.

**Known reproduction note:** the two synthesis scripts (`exp_13_m12_synthesis.py`, `exp_13_m13_synthesis.py`) each contain a meta-scorecard test (T2) that scans a **`__file__`-relative** `../results/` directory for the sibling experiments' JSON files. Run from the packaged location that directory does not exist, so T2 reports 0 files found and each synthesis drops to 3/4. This is a packaging path artifact, not a physics failure — the underlying tests (T1, T3, T4) pass, all 30 scripts execute and exit 0, and the authoritative scores are the frozen snapshot values (M12 49/52, M13 core 48/52). The scripts are shipped verbatim and are not modified.

## Status

- [x] Draft complete
- [x] Code package assembled
- [x] Reproduction verified (venv: numpy/scipy/matplotlib) — all 30 scripts execute
- [x] Figures generated
- [ ] paper.tex final proof
- [ ] Overclaims resolved (see below — ds² precision, M13 score accounting)
- [ ] Zenodo deposit (v0.3)

## Overclaims flagged (not corrected — paper.md left untouched)

1. **ds² preservation precision (§6.3):** paper says the interval is "preserved … to machine precision (maximum relative error below 3×10⁻¹⁶)". The backing data (exp_09 m13, T3) reports **max relative error = 2.86×10⁻⁶** (at rapidity η = 10, from matrix exponentiation; 130/135 transforms pass at 1×10⁻¹⁰, none at 1×10⁻¹⁶). The uniqueness/proportionality claims (1-dim null space, constant −1.0, residual 4×10⁻¹⁷) do check out.
2. **M13 score accounting (Abstract, §9.1):** paper headlines "51/60 (85%)" for M13/13.5 and "M13.5 investigation: 3/8 (38%)", combined "100/112 = 89%". The frozen JSON data gives M13 core 48/52 + M13.5 investigation 5/16 = **53/68 (78%)** (matching the milestone README). The paper's figure silently omits investigation experiments exp_16 (0/4) and exp_17 (2/4). Including all experiments the combined score is 102/120 = 85%, not 89%.

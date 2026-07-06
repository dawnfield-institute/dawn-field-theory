# PACSeries v0.3

**Date**: May–July 2026
**Status**: In preparation (drafts complete; reproducibility packages being assembled)
**Papers**: 6 (Papers 7–12)
**Previous**: [v0.2](../v0.2/) (February 2026, Zenodo [10.5281/zenodo.18743674](https://zenodo.org/records/18743674), Papers 1–6) · [v0.1](../v0.1/) (October 2025, Zenodo [10.5281/zenodo.17295103](https://zenodo.org/records/17295103))
**Series concept DOI**: [10.5281/zenodo.15783623](https://zenodo.org/records/15783623) (resolves to the latest published version)

v0.3 adds Papers 7–12 to the series. Per the [v0.3 planning resolution](../v0.3_PLANNING.md) (2026-06-10), **scope is locked to the new Papers 7–12 only** — no revisions to Papers 1–6 ship in this release (those are deferred to v0.4).

## Papers

| # | Paper | Focus | Milestones | Type |
|---|-------|-------|-----------|------|
| 7 | [The Symmetry Primitive and Scoped Mediation](symmetry_primitive_scoped_mediation/) | Pre-axiomatic origin (φ from self-reference); force hierarchy from Fibonacci depth | M7 + M6 | B |
| 8 | [Quantum Gravity from Information Conservation](quantum_gravity_information_conservation/) | Planck scale from PAC/SEC response-time crossover; singularity resolution; Hawking, Page curve, graviton | M11 | B+C |
| 9 | [Cosmological Predictions and the Cascade Clock](cosmological_predictions_cascade_clock/) | Cascade clock (slope 1/ln φ); S8, Hubble φ^{1/6}, JWST unified; Z′ 395 GeV; dark matter 6.44 keV | M8 + M9 | A+B+C |
| 10 | [Connection, Identity, and Spacetime](connection_identity_spacetime/) | Lorentz group from ADE + SEC; Minkowski signature from the Killing form | M12 + M13/13.5 | A |
| 11 | [Quantum Mechanics from Graph Structure](quantum_mechanics_graph_structure/) | Orbit Hilbert space; Born rule from counting; D₄ triality; Bell from Aut(G); Schrödinger, path integral, decoherence | M14 + P13–P16 | A+B |
| 12 | [First Observational Contact](observational_contact_absorption_spectroscopy/) | Pre-registered line-width oscillation falsified against 443K quasar absorbers; PAC/SEC two-channel partition | Midnight | A+B+C |

## Reading Order

Papers 7–12 continue the series arc; the two strongest structural results (Papers 10, 11) are the most independently checkable.

1. **Paper 7** — Pushes the axioms back one level: symmetry → self-reference → φ → PAC/SEC; then the propagation mechanism (scoped mediation) and force hierarchy.
2. **Paper 8** — Quantum gravity: the Planck scale as a response-time crossover, with singularity resolution and Hawking/Page/graviton results.
3. **Paper 9** — Cosmology: the cascade clock unifies S8, Hubble, and JWST; registers Z′ and dark-matter predictions.
4. **Paper 10** — Spacetime: the Lorentz group and Minkowski metric derived from ADE graph structure + SEC. (Structural, Type A.)
5. **Paper 11** — Quantum mechanics from the same ADE graphs; the crown-jewel derivation. (Structural, Type A.)
6. **Paper 12** — First observational contact: a pre-registered, DFT-specific prediction tested against ~443,000 quasar absorption systems — and falsified. The honest result.

## The Derivation Chain (continued from v0.2)

```
v0.2 established: PAC → φ → ln(φ) → Ξ → Feigenbaum → SM parameters → Maxwell → ML validation
v0.3 extends:
  → SYMMETRY PRIMITIVE: symmetry → self-reference → φ → PAC/SEC (Paper 7)
  → QUANTUM GRAVITY:    Planck scale from PAC/SEC crossover, depth 183 (Paper 8)
  → COSMOLOGY:          cascade clock N(t) = a + (1/ln φ)·ln(t) (Paper 9)
  → SPACETIME:          A₁ --SEC--> SL(2,ℂ) ≅ SO(3,1); ds² from Killing form (Paper 10)
  → QUANTUM MECHANICS:  orbit Hilbert space L²(V/Aut(G)); Born rule = counting; D₄ (Paper 11)
  → OBSERVATIONAL TEST: pre-registered oscillation falsified vs 443K absorbers (Paper 12)
```

## Each Paper Contains (target package, per the v0.2 gold standard)

```
paper_name/
├── paper.md          # Full paper text
├── paper.tex         # LaTeX manuscript
├── README.md         # Overview and reproduction instructions
├── meta.yaml         # Schema v2.0 metadata
├── Code/
│   ├── experiments/  # Numbered experiment scripts (exp_01..exp_NN)
│   ├── reproduce.py  # Run all experiments
│   ├── trace.yaml    # Provenance: traces code/data to source repos
│   └── requirements.txt
├── Data/results/     # JSON outputs from experiments
└── Figures/          # Publication-quality PNGs
```

**Packaging status (2026-07-05)**: all six manuscripts are drafted and content-reviewed. Reproducibility packages (Code/Data/Figures) and `paper.tex` are being assembled from the backing experiments (milestones 6–14, `minimum_actualization_resolution`, `midnight`). See [`../state_of_the_pac_series.md`](../state_of_the_pac_series.md) for the series-wide precision/failure classification.

# PACSeries Paper 6: Computational Validation of PAC Conservation

## Overview

This paper tests whether PAC conservation — established across mathematics and physics in Papers 1–5 — also operates in artificial computational systems. Seven transformer models (Pythia 70M–1B, GPT-2 family) are analysed using a zero-parameter PAC tree framework, then a minimal architecture (TinyCIMM-Boltzmann) tests whether enforcing conservation improves behaviour.

## Key Results

| Result | Status | Section |
|--------|--------|---------|
| SEC phase → accuracy (monotonic) | Validated, 7 models | §3 |
| Phi enrichment in top-2 ratios | **Falsified** (softmax artifact) | §3.2 |
| PAC ratio magnitude → correctness | p < 0.0001 at 1B | §3.3 |
| Ξ in weight spectra, 2.36× enrichment | χ² = 5511 | §4 |
| Ξ preferentially in attention layers | All scales | §4.3 |
| 5 attention metrics significant | All p < 0.001 | §5 |
| Phase transition delay ~1.43× | 7 models, universal | §5.4 |
| Hallucination = +9.6% PAC violation | p = 4.8 × 10⁻⁵ | §6 |
| GPT-2 zero compensation | 0.000 ratio | §6.3 |
| Conservation reduces noise violation | p = 0.008 | §7 |
| 16× less transition shock | 27.3 vs 1.7 | §7.5 |
| No cost to factual learning | p = 0.42 (n.s.) | §7.6 |
| Training converges toward φ | p = 0.0014 | §8.2 |
| Landauer full-stack chain | 6 layers validated | §8.1 |

## Source Experiments

- `dawn-models/research/scbf/experiments/token_pac_tree/` — 12 scripts (PAC tree, SEC phase, attention, Xi, conservation)
- `dawn-models/research/tinycimm/TinyCIMM-Boltzmann/` — 1 script (conservation enforcement)
- `dawn-field-theory/foundational/experiments/landauer_erasure_structure/` — exp_22–25 (Landauer bridge)
- `dawn-models/research/scbf/experiments/huggingface_bifractal_validation/` — bifractal validation

## Dependencies

- Paper 1: Structure Cost of Erasure (PAC/SEC framework, Landauer bridge)
- Paper 2: Balance Constant (Ξ = 1 + π/55)
- Paper 5: Classical Physics (SEC wave equation analogy)

## Reproduction

```bash
cd Code/
pip install -r requirements.txt
python reproduce.py
```

Or run individual experiments:

```bash
cd Code/experiments/
python exp_02_multi_model_scale.py
python exp_06_xi_weight_clustering.py
python exp_07_attention_pac.py
python exp_12_pac_conservation.py
python exp_tinycimm_conservation.py
```

## Status

- [x] Draft complete
- [ ] Internal review
- [ ] Final voice pass
- [ ] Figures generated
- [ ] Code package assembled

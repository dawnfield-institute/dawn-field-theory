# Proposal: Applied DFT — PAC Knowledge Discovery

**Date**: 2026-03-25
**Thread**: Milestone 6, Thread 6 (Novel / Medium Priority)
**FDO**: `pac-knowledge-discovery` (confidence 0.40, status: seed)
**Target Paper**: "PAC Discovery: Information-Theoretic Feature Importance via Conservation Residuals"

---

## 1. Executive Summary

This proposal develops the first *applied* DFT publication — a practical tool that uses PAC conservation residuals to detect missing variables in datasets. The core idea: when f(Parent) ≠ Σf(Children), the residual doesn't just measure prediction error — it measures *missing information*. The structure of that residual (how it clusters, what it correlates with) tells you what you don't know and suggests what to measure next.

**Why this matters for DFT**: Every other DFT publication targets theoretical physics — deriving constants, predicting masses, explaining forces. This thread targets a completely different audience: machine learning researchers, data scientists, and applied statisticians. If PAC conservation is a real principle, it should work beyond physics. Demonstrating practical utility in ML validates the theory's generality and opens DFT to thousands of potential users who will never read a physics paper.

**The honest pitch**: We have a theoretically motivated approach (PAC residuals as gap detectors), early experiments on real data (MovieLens, UCI health, yeast proteome, wine quality), and a discovered phenomenon (N² convergence). What we don't have: validated tier-1 success criteria, comparison to existing causal discovery methods, or a packaged tool. This proposal is about closing those gaps.

---

## 2. Current State: What Exists

### Experiments Completed (5)
| Exp | Focus | Key Result |
|-----|-------|------------|
| exp_01 | Real data validation, convergence bridging | N² convergence discovered in multi-space datasets |
| exp_02 | Diverse branch mapping | Cross-domain convergence patterns mapped |
| exp_03 | Property convergence | High convergence zones show exploitable structure |
| exp_04 | Biological convergence (yeast proteome, ~5K proteins) | Convergence holds in biological data |
| exp_05 | PAC tree construction | Residual clustering identifies missing features |

### Core Modules Built
- `convergence_analyzer.py` — N² convergence computation (k-NN Jaccard overlap between feature spaces)
- `hypothesis_generator.py` — Proposes missing children based on residual structure
- `mesh_builder.py` — Trains model meshes across feature space pairs
- `pac_residual_tracker.py` — Computes and tracks PAC conservation residuals

### Datasets Used
- **MovieLens** (recommendations) — correlational structure, low convergence (~0.02)
- **UCI Health** (clinical) — causal structure, higher convergence
- **Yeast Proteome** (~5K proteins) — biological causal structure
- **Wine Quality** (~6.5K samples) — chemical → sensory causation

### Architecture Arsenal (MED-Bounded, ≤10)
RF, XGBoost, LightGBM, ExtraTrees, Ridge, Lasso, ElasticNet, MLP(64,32), MLP(128,64,32), SVR(rbf)

### What's Been Established
- N² convergence is a real, measurable phenomenon
- Domains with causal structure show higher convergence than correlational domains
- Residual clustering correlates with missing features
- MED bounds (≤10 architectures) prevent overfitting without information gain

### What's NOT Established
- **Tier-1 criteria not validated**: No formal test of "PAC residual decreases when missing feature added"
- **No baseline comparison**: Haven't compared to PC algorithm, NOTEARS, or any causal discovery method
- **No novel discovery**: Haven't found something PAC discovers that existing methods miss
- **No packaged tool**: Scripts exist but no API, no documentation, no pip-installable package
- **Convergence threshold unclear**: Is 0.05 universal or domain-specific?

---

## 3. The Vision: PAC Residuals as a Universal Gap Detector

### Theoretical Foundation

PAC conservation states: **f(Parent) = Σf(Children)**

In a complete causal model, the parent variable is fully explained by its children. When it isn't:

```
ε = f(Parent) - Σf(Children_known) = Σf(Children_unknown)
```

This residual ε isn't noise — it's the *signal of missing information*. The structure of ε reveals:
- **Magnitude**: How much information is missing
- **Clustering**: How many missing children there are (distinct residual modes = distinct missing factors)
- **Correlation with known features**: What the missing children look like (partial identification)
- **Spatial distribution in feature space**: Where to look (SEC entropy map — high entropy = unexplored territory)

### The N² Convergence Discovery

When you compute k-NN overlap (Jaccard similarity) between every pair of N feature spaces, you get an N×N convergence matrix. High convergence (>0.05) indicates shared latent structure — what we call "entangled roots." This is SEC collapse made measurable: entropy has crystallized into shared organization.

The convergence landscape IS the SEC entropy field. Low-convergence zones = high entropy = potential for collapse = discovery opportunity.

### Why This Isn't Just Feature Importance

Standard feature importance (SHAP, permutation importance, etc.) tells you which *known* features matter. PAC residual analysis tells you that *unknown* features exist and characterizes them. It's the difference between "Feature X is important" and "There's a missing variable that looks like this."

---

## 4. Validation Plan

### Tier 1: Core Mechanism Validation (MUST PASS)

**Test 1.1: Residual Decrease on Feature Removal/Addition**
- Take a dataset with known causal structure (synthetic or real)
- Remove one feature ("hide a child")
- Train PAC mesh on remaining features
- Compute residual ε_before
- Add the hidden feature back
- Compute residual ε_after
- **Success**: ε_after < ε_before by ≥20% consistently across datasets
- **Implementation**: Both synthetic (controlled) and real (UCI datasets with known structure)

**Test 1.2: Residual Cluster Count = Missing Children Count**
- Generate synthetic hierarchies with 1, 2, 3, 5 hidden children
- Compute PAC residuals
- Cluster residuals (DBSCAN or Gaussian mixture)
- **Success**: Cluster count correlates with true missing count (r > 0.7)

**Test 1.3: Convergence-Performance Correlation**
- Across ≥5 domains, compute convergence and downstream model R²
- **Success**: Spearman correlation r > 0.5 between convergence and predictive performance

### Tier 2: Comparative Validation (SHOULD PASS)

**Test 2.1: PAC vs PC Algorithm**
- Run both on identical datasets with known causal DAG
- Compare: structural Hamming distance to true DAG
- PAC doesn't need to beat PC on DAG recovery — it should find *missing* variables that PC assumes don't exist

**Test 2.2: PAC vs NOTEARS**
- NOTEARS (Zheng et al. 2018) learns DAGs via continuous optimization
- Compare on datasets where both causal structure AND missing variables are known
- **Key distinction**: NOTEARS finds structure among observed variables; PAC detects *unobserved* variables

**Test 2.3: PAC vs SHAP Residual Analysis**
- Train model, compute SHAP values, analyze residuals via SHAP
- Compare to PAC residual analysis
- **Expected result**: SHAP identifies important known features; PAC identifies missing unknown features (complementary, not competing)

### Tier 3: Novel Discovery (STRETCH GOAL)

**Test 3.1: Discover Something New**
- Apply PAC residual analysis to a dataset where ground truth is partially known
- Candidates: protein-protein interaction networks (known interactions + unknown), clinical datasets (known risk factors + unknown)
- PAC residuals suggest missing variable → validate by acquiring additional data or literature search
- **Success**: At least 1 genuinely novel finding that can be independently validated

---

## 5. Experiment List

| # | Experiment | Description | Tier |
|---|-----------|-------------|------|
| 1 | `exp_06_synthetic_validation` | Controlled test: generate synthetic hierarchies with known missing children. Remove features, measure residual, add back, confirm decrease. 1000 trials across 5 hierarchy structures | 1 |
| 2 | `exp_07_cluster_calibration` | Calibrate residual clustering against true missing child count. Test DBSCAN, GMM, spectral clustering. Determine reliability bounds | 1 |
| 3 | `exp_08_cross_domain_convergence` | Formal convergence-performance correlation test across ≥5 domains. Compute Spearman r with bootstrap CI | 1 |
| 4 | `exp_09_pc_comparison` | Head-to-head comparison with PC algorithm on bnlearn benchmark DAGs (Asia, Alarm, Insurance networks). Focus on missing variable detection, not DAG recovery | 2 |
| 5 | `exp_10_notears_comparison` | Comparison with NOTEARS on synthetic + semi-synthetic datasets. Measure: can PAC detect variables that NOTEARS misses? | 2 |
| 6 | `exp_11_shap_complement` | Run SHAP + PAC residuals on same datasets. Show they're complementary: SHAP for known features, PAC for unknown gaps | 2 |
| 7 | `exp_12_novel_discovery` | Apply full PAC discovery pipeline to protein interaction or clinical dataset. Attempt genuine novel finding | 3 |
| 8 | `exp_13_packaging_test` | End-to-end test of packaged tool: API, documentation, example notebooks. User testing on novel dataset | — |

---

## 6. Packaging: Python Tool

### API Design

```python
from pac_discovery import PACDiscovery

# Initialize with dataset
disco = PACDiscovery(
    data=df,
    target='outcome',
    feature_groups={'clinical': [...], 'genetic': [...], 'lifestyle': [...]},
    architectures='auto',  # MED-bounded selection, ≤10
)

# Run convergence analysis
convergence = disco.analyze_convergence()
# Returns N×N convergence matrix + SEC entropy map

# Train model mesh and compute residuals
residuals = disco.compute_residuals()
# Returns PAC residuals per sample with conservation quality score

# Detect missing children
gaps = disco.detect_gaps(
    min_reduction=0.20,  # minimum residual reduction to flag
    cluster_method='dbscan',
)
# Returns: GapReport with
#   - n_missing_estimated: int
#   - gap_profiles: list of feature-space projections
#   - acquisition_suggestions: ranked list of what to measure next
#   - confidence: float (based on residual structure clarity)

# Generate discovery report
report = disco.report()
```

### Package Structure

```
pac-discovery/
├── pyproject.toml
├── src/
│   └── pac_discovery/
│       ├── __init__.py
│       ├── core/
│       │   ├── convergence.py      # N² convergence analysis
│       │   ├── mesh.py             # Model mesh training (MED-bounded)
│       │   ├── residuals.py        # PAC conservation residual computation
│       │   └── gaps.py             # Missing children detection + clustering
│       ├── models/
│       │   └── architectures.py    # 10 bounded architectures
│       ├── viz/
│       │   ├── entropy_map.py      # SEC entropy field visualization
│       │   └── gap_report.py       # Discovery report generation
│       └── examples/
│           ├── quickstart.ipynb
│           ├── clinical_discovery.ipynb
│           └── protein_interactions.ipynb
├── tests/
└── docs/
```

### Dependencies
- scikit-learn, xgboost, lightgbm (model mesh)
- numpy, pandas, scipy (core computation)
- matplotlib, seaborn (visualization)
- No DFT-specific dependencies — this must be standalone

---

## 7. Target Paper

**Title**: "PAC Discovery: Information-Theoretic Feature Importance via Conservation Residuals"

**Abstract sketch**: We introduce PAC Discovery, a method for detecting missing variables in datasets using information conservation residuals. Grounded in Preferential Attachment Coupling (PAC) theory, our approach measures the gap between observed outcomes and predictions from known features. Unlike standard feature importance methods (SHAP, permutation importance), which rank known variables, PAC residual analysis detects and characterizes *unknown* variables. We validate on synthetic hierarchies and benchmark datasets, showing that (a) residuals decrease by ≥20% when missing features are added, (b) residual clustering estimates the number of missing factors, and (c) the method complements existing causal discovery tools (PC algorithm, NOTEARS). We release an open-source Python package for practical use.

**Target venues** (in order of preference):
1. **NeurIPS** (main conference or workshop) — highest impact for ML audience
2. **ICML** — alternative top ML venue
3. **JMLR** (journal) — if conference timing doesn't work
4. **Nature Machine Intelligence** — if novel discovery result (Tier 3) is strong enough

**Key selling points for reviewers**:
- Novel: residuals as missing-variable detectors, not just error metrics
- Theoretically grounded: PAC conservation provides principled foundation
- Practical: open-source tool with API and examples
- Complementary: works alongside existing methods, not a replacement

---

## 8. Why This Matters

### For DFT
- **Generality validation**: If PAC conservation works in ML, it's not just a physics trick
- **New audience**: ML researchers who validate PAC empirically become potential DFT advocates
- **Practical utility**: Demonstrates DFT has value beyond theoretical beauty
- **Funding angle**: Applied tools are easier to fund than theoretical physics

### For ML/Data Science
- **Missing variable detection** is an unsolved problem — current methods (causal discovery) assume all relevant variables are observed
- **Information-theoretic grounding** provides principled alternative to ad hoc residual analysis
- **Bounded complexity** (MED constraints) is a novel regularization principle
- **Discovery-oriented**: Tells you what to measure next, not just what you already have

### For Science Generally
- **Automated hypothesis generation**: PAC residuals suggest what's missing, not just that something is missing
- **Domain-agnostic**: Same method for proteomics, clinical research, economics, physics
- **Quantified ignorance**: Conservation quality score tells you how complete your model is

### Risks
- **Tier-1 failure**: If PAC residuals don't decrease when missing features are added, the entire premise fails. This is the make-or-break experiment (exp_06).
- **Not novel enough**: If SHAP residual analysis gives equivalent results, the PAC framing adds theory but not practice. exp_11 tests this directly.
- **Domain-specific**: If it only works for causal domains (not correlational), the applicability is narrower than claimed. Early evidence suggests this IS the case (MovieLens convergence ~0.02 vs clinical >0.05).
- **Comparison losses**: If PC or NOTEARS already detect missing variables effectively, PAC adds marginal value. Our bet: existing methods find structure among *observed* variables; PAC finds *unobserved* variables. Different capabilities.

---

## 9. Timeline & Dependencies

```
Week 1:  exp_06 (synthetic validation) ──────────────────► GO/NO-GO GATE
Week 1:  exp_07 (cluster calibration) ──────────────────►
Week 2:  exp_08 (cross-domain) ─────────────────────────►
Week 2:  exp_09 (PC comparison) ────────────────────────►
Week 3:  exp_10 (NOTEARS comparison) ───────────────────►
Week 3:  exp_11 (SHAP complement) ──────────────────────►
Week 4:  exp_12 (novel discovery) ──────────────────────►
Week 4:  exp_13 (packaging) ────────────────────────────►
Week 5:  Paper writing ─────────────────────────────────►
```

**GO/NO-GO gate after Week 1**: exp_06 (synthetic validation) is the critical experiment. If PAC residuals don't decrease when missing features are added across multiple synthetic hierarchies, the thread is not viable. Don't invest in comparison experiments until this passes.

**Dependencies**:
- exp_06–08 depend on existing core modules (convergence_analyzer, pac_residual_tracker) — ready
- exp_09–10 require installing causal discovery packages (causal-learn for PC, notears for NOTEARS) — check pip availability
- exp_12 requires dataset selection and potentially data acquisition
- Packaging (exp_13) depends on API design stabilizing after validation experiments

**Cross-thread connections**:
- Thread 5 (JWST cosmology) — PAC residual methodology is shared; improvements here improve cosmology analysis
- MAR exp_22 (Eddington regulator) — demonstrates PAC conservation as a diagnostic in physics; analogous to using it in ML
- `pac-dag-fluid` FDO — DAG structure dynamics theory underpins the convergence analysis

---

*This proposal takes DFT's most novel thread — applied information theory for practical data science — and maps the path from "interesting idea with early experiments" to "validated, published, and packaged tool." The honest assessment: confidence is low (0.40), tier-1 criteria are unvalidated, and there's a real chance the approach doesn't generalize. But the potential reward — opening DFT to the entire ML community — justifies the focused investment of 4–5 weeks.*

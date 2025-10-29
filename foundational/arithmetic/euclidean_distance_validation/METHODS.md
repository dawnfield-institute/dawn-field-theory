# Experimental Methods

**PAC Theory Euclidean Distance Validation Framework**  
**Version**: 2.0  
**Date**: October 28, 2025

## Overview

This document describes the experimental procedures, data generation methods, and reproducibility guidelines for the geometric validation of PAC (Potential-Actualization Conservation) theory through Euclidean distance metrics.

---

## Framework Architecture

### Core Components

```
euclidean_distance_validation/
├── core/
│   ├── pac_hierarchy.py          # PACNode, PACHierarchy classes
│   └── embedding_generator.py    # Embedding strategies
├── experiments/
│   ├── experiment_01_distance_conservation.py
│   ├── experiment_02_fractal_dimension.py
│   ├── experiment_03_depth_width_tradeoff.py
│   ├── experiment_04_context_relative_invariance.py
│   ├── experiment_05_ratio_preserving_transformations.py
│   ├── experiment_06_emc2_quantification.py
│   └── experiment_07_real_embeddings.py
├── tests/
│   └── test_hierarchy.py         # 11 unit tests
├── results/
│   └── experiment_*.json         # Experimental outputs
└── docs/
    ├── PROPOSAL.md               # Theoretical foundation
    ├── RESULTS.md                # Experimental results
    └── METHODS.md                # This file
```

---

## Data Generation

### Synthetic Hierarchies

#### Method: Bottom-Up Construction
```python
def generate_synthetic_hierarchy(depth=4, branching=3, seed=42):
    """
    Create balanced tree with PAC-compliant synthetic embeddings.
    
    Process:
    1. Build tree structure (top-down)
    2. Assign values ensuring f(P) = Σf(C)
    3. Generate embeddings (bottom-up)
       - Leaves: random normalized * sqrt(value)
       - Parents: weighted_sum(children)
    
    Result: Perfect PAC conservation by construction
    """
    pass
```

**Parameters**:
- `depth`: Tree depth (1-5 tested)
- `branching`: Children per node (2-4 tested)
- `dimension`: Embedding dimensionality (64-4096 tested)
- `seed`: Random seed for reproducibility

**Validation**: All generated hierarchies satisfy f(P) = Σf(C) within machine precision (<10⁻¹⁵)

### Real Embeddings (Ollama)

#### Setup Requirements
1. **Ollama installation**:
   ```bash
   # Install Ollama (https://ollama.ai)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull model
   ollama pull llama3.2
   
   # Verify
   ollama list
   ```

2. **Python dependencies**:
   ```bash
   pip install requests numpy scipy matplotlib
   ```

#### Embedding Generation Process
```python
def generate_real_embeddings(hierarchy, model='llama3.2:latest'):
    """
    Generate embeddings for semantic concepts.
    
    Process:
    1. For each leaf node:
       - Call Ollama API with concept text
       - Store embedding (3072D for llama3.2)
    2. For each parent node:
       - Compute weighted_sum(children_embeddings)
       - Preserves PAC structure
    
    API: POST http://localhost:11434/api/embeddings
    """
    pass
```

**Performance**:
- Time per embedding: ~200ms (llama3.2)
- Total time for 9-node hierarchy: ~2s
- API timeout: 30s
- Caching: Enabled (prevents redundant calls)

---

## Experimental Procedures

### Experiment 1: Distance Conservation

**Objective**: Test if ||e(P)||² ≈ Σ αᵢ·||e(Cᵢ)||²

**Procedure**:
1. Generate synthetic hierarchy (121 nodes)
2. Compute distance residual for each parent:
   ```
   residual = |parent_norm² - Σ(weight × child_norm²)|
   ```
3. Measure correlation with PAC residual
4. Compute success rate (residual < 0.1)

**Metrics**:
- Mean/max distance residual
- Pearson correlation (distance vs. PAC residual)
- Success rate percentage

**Expected**: r > 0.7, success rate = 100%

### Experiment 2: Fractal Dimension

**Objective**: Measure geometric scaling of hierarchies

**Procedure**:
1. For each subtree depth k:
   - Compute mean distance to parent
2. Fit power law: distance ~ depth^λ
3. Compute fractal dimension: D = log(N) / log(λ)

**Metrics**:
- Fractal dimension D
- Scaling factor λ
- R² of power law fit

**Expected**: D > 1, λ ≈ 1 (convergent for synthetic)

### Experiment 3: Depth-Width Tradeoff

**Objective**: Validate complexity symmetry principle

**Procedure**:
1. For each parent node:
   - Measure recursive depth (max child depth)
   - Compute distance spread (std of child distances)
2. Correlate depth with spread

**Metrics**:
- Pearson correlation
- Spearman correlation (rank)
- Mean spread per depth level

**Expected**: r > 0.5, p < 0.001

### Experiment 4: Context-Relative Invariance

**Objective**: Test Einstein-like relativity in information space

**Procedure**:
1. Select random parent P and sibling group S
2. Measure within-context distances (siblings in S)
3. Measure cross-context distances (S vs. other siblings)
4. Compare coefficient of variation

**Metrics**:
- Within-context CV
- Cross-context CV
- Ratio (context sensitivity)

**Expected**: Within-CV < 0.1, Ratio > 5×

### Experiment 5: Ratio-Preserving Transformations

**Objective**: Test Axiom 4 and Axiom 5

**Procedure**:
1. Apply transformations to embeddings:
   - Sibling permutation
   - Uniform scaling
   - Orthogonal rotation
2. Measure ratio preservation (before/after)
3. Test collapse reversibility:
   - Given parent embedding, reconstruct children
   - Measure error

**Metrics**:
- Ratio residual per transformation
- Irreversibility index (reconstruction error)

**Expected**: Ratio residual = 0 for valid transforms, Irreversibility > 0.01 for real embeddings

### Experiment 6: E=mc² Quantification

**Objective**: Test information-energy equivalence

**Procedure**:
1. For each node: compute E = ||e||², m = f(v)
2. Fit linear model: E = c² × m
3. Separate analysis for leaves vs. parents

**Metrics**:
- c² value (slope)
- R² (fit quality)
- Relative error

**Expected**: c² = 1.0 for synthetic leaves, R² > 0.99

### Experiment 7: Real Embeddings (Ollama)

**Objective**: Validate on actual semantic information

**Procedure**:
1. Create semantic hierarchy (science concepts)
2. Generate Ollama embeddings
3. Run all tests from Experiments 1, 5, 6

**Metrics**:
- c² value (model-specific)
- Binding energy ratio
- Irreversibility index
- Context sensitivity

**Expected**: c² ≠ 1.0, irreversibility > 0%, amplification (not reduction)

---

## Statistical Methods

### Correlation Analysis
- **Pearson r**: Linear correlation strength
- **Spearman ρ**: Rank correlation (non-linear)
- **p-values**: Computed via scipy.stats
- **Significance**: * p<0.05, ** p<0.01, *** p<0.001

### Regression Analysis
- **Linear models**: scipy.stats.linregress
- **R² computation**: Coefficient of determination
- **Residual analysis**: Mean absolute error, max error

### Power Law Fitting
```python
from scipy.optimize import curve_fit
def power_law(x, a, b): return a * x**b
params, _ = curve_fit(power_law, depth, distance)
```

### Coefficient of Variation
```
CV = std / mean
```
Used for measuring relative dispersion in context-invariance tests.

---

## Reproducibility Guidelines

### Environment Setup

**Python Version**: 3.11.9 (tested)

**Dependencies**:
```bash
pip install numpy==1.24.3 scipy==1.11.1 matplotlib==3.7.2 requests==2.32.3
```

**Optional**:
```bash
# For pretrained embeddings (not used in current experiments)
pip install transformers sentence-transformers
```

### Random Seeds
All experiments use fixed seeds for reproducibility:
- Synthetic generation: `seed=42`
- Hierarchy creation: `seed=123`
- Random sampling: `seed=456`

### Hardware Requirements
- **CPU**: Any modern processor (tested on AMD/Intel)
- **RAM**: 4GB minimum (8GB recommended for large hierarchies)
- **GPU**: Not required (CPU-only)
- **Storage**: <100MB for all experiments

### Running Experiments

**Individual experiment**:
```bash
cd euclidean_distance_validation
python -m experiments.experiment_01_distance_conservation
```

**All experiments**:
```bash
for i in {01..07}; do
    python -m experiments.experiment_${i}_*
done
```

**With output capture**:
```powershell
# PowerShell (Windows)
python -m experiments.experiment_07_real_embeddings 2>&1 | Tee-Object output.log
```

### Verification

**Unit tests**:
```bash
pytest tests/test_hierarchy.py -v
# Expected: 11/11 passing
```

**Result validation**:
```python
import json
with open('results/experiment_06_results.json') as f:
    data = json.load(f)
    assert data['c_squared_leaf'] == 1.0  # Within tolerance
```

---

## Data Formats

### Hierarchy Structure (JSON)
```json
{
  "node_id": {
    "value": 100.0,
    "children": ["child1", "child2"]
  }
}
```

### Experiment Results (JSON)
```json
{
  "experiment": "E=mc² Quantification",
  "date": "2025-10-28",
  "results": {
    "c_squared_leaf": 1.0000,
    "r_squared": 1.0000,
    "error_percent": 0.00
  }
}
```

### Embeddings (NumPy)
```python
# Shape: (n_nodes, embedding_dim)
embeddings = np.load('embeddings.npy')
```

---

## Quality Control

### Validation Checks
1. **PAC Conservation**: All hierarchies satisfy f(P) = Σf(C) < 10⁻⁶
2. **Embedding Norms**: No NaN or Inf values
3. **Numerical Stability**: Residuals < 10⁻¹⁴ for synthetic
4. **API Reliability**: Retry logic for Ollama (3 attempts)

### Error Handling
- **Missing embeddings**: Fallback to random (logged)
- **API timeout**: 30s limit, retry with exponential backoff
- **Dimension mismatch**: Validate before computation
- **Overflow/underflow**: Use float64 precision

---

## Limitations & Assumptions

### Current Limitations
1. **Synthetic bias**: Perfect PAC by construction
2. **Small hierarchies**: Largest tested = 364 nodes
3. **Single embedding strategy**: No comparison across methods yet
4. **Limited models**: Only llama3.2 tested with real embeddings

### Assumptions
1. **Embedding quality**: Assumes Ollama embeddings are semantically meaningful
2. **Hierarchy validity**: Assumes input hierarchies are well-formed
3. **Linear scaling**: E=mc² assumes linear relationship (validated empirically)
4. **Context independence**: Assumes embedding model is deterministic

---

## Future Enhancements

### Planned Improvements
1. **Multi-model testing**: Compare 5+ Ollama models
2. **Large-scale validation**: WordNet (117k nodes), Wikipedia (1.5M)
3. **DAG support**: Full multi-parent structures
4. **Learned embeddings**: Train custom PAC-optimized models
5. **Visualization**: Interactive 3D embedding space plots

### Code Optimizations
1. **Parallel processing**: Batch embedding generation
2. **Caching**: Persistent embedding storage
3. **GPU acceleration**: Optional CUDA support
4. **Memory efficiency**: Streaming for large hierarchies

---

## Citation

If using this framework, please cite:

```bibtex
@article{pac_geometric_validation_2025,
  title={Geometric Validation of Information Conservation: E=mc² in Semantic Space},
  author={Dawn Field Institute},
  journal={TBD},
  year={2025},
  note={Code: github.com/dawnfield-institute/pac-validator}
}
```

---

**Last Updated**: October 28, 2025  
**Version**: 2.0  
**Status**: Complete and validated

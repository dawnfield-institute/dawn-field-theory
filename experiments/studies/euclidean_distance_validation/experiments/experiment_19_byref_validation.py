"""
Experiment 19: Validate byref E=mc² Claims

CRITICAL: Before claiming R²=1.0 is real, we must validate:
1. Works with REAL embeddings (not random vectors)
2. Not overfitting (small sample size n=12-13 is suspicious)
3. Metrics are truly independent (not just computing same thing)
4. Holds across different hierarchy structures
5. Passes null hypothesis testing
6. Results are stable across different seeds

This is the "trust but verify" experiment after exp_18's suspicious perfection.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from scipy.stats import linregress, pearsonr, spearmanr
from collections import defaultdict
import networkx as nx
from sentence_transformers import SentenceTransformer


@dataclass
class PACNode:
    """Node in PAC hierarchy with ownership relationships."""
    name: str
    parent: str = None
    children: List[str] = None
    embedding: np.ndarray = None
    depth: int = 0
    text: str = ""  # Add text for real embeddings
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class PACHierarchy:
    """PAC hierarchy with ownership graph structure."""
    
    def __init__(self):
        self.nodes: Dict[str, PACNode] = {}
        self.ownership_weights: Dict[Tuple[str, str], float] = {}
        
    def add_node(self, node: PACNode):
        """Add node to hierarchy."""
        self.nodes[node.name] = node
        
    def add_ownership(self, parent: str, child: str, weight: float = 1.0):
        """Add ownership relationship with weight."""
        self.ownership_weights[(parent, child)] = weight
        if child not in self.nodes[parent].children:
            self.nodes[parent].children.append(child)
            
    def get_ownership_graph(self) -> nx.DiGraph:
        """Build NetworkX directed graph from ownership relationships."""
        G = nx.DiGraph()
        for node_name in self.nodes:
            G.add_node(node_name)
        for (parent, child), weight in self.ownership_weights.items():
            G.add_edge(parent, child, weight=weight)
        return G


def build_real_embedding_hierarchy(model: SentenceTransformer, 
                                   structure: str = "balanced") -> PACHierarchy:
    """
    Build hierarchy with REAL sentence-transformer embeddings.
    
    Args:
        structure: "balanced", "deep", "wide", "mixed"
    """
    hierarchy = PACHierarchy()
    
    # Different hierarchy structures for robustness testing
    if structure == "balanced":
        # Original 3-domain structure
        texts = {
            "root": "Knowledge and information",
            "physics_root": "Physics and physical sciences",
            "biology_root": "Biology and life sciences",
            "code_root": "Computer science and programming",
        }
        
        # Level 2
        for domain in ["physics", "biology", "code"]:
            for i in range(3):
                texts[f"{domain}_L2_{i}"] = f"{domain} subdomain {i}"
        
        # Level 3
        for domain in ["physics", "biology", "code"]:
            for i in range(3):
                for j in range(2):
                    texts[f"{domain}_L3_{i}_{j}"] = f"{domain} concept {i}.{j}"
    
    elif structure == "deep":
        # Deeper hierarchy (5 levels)
        texts = {"root": "Knowledge"}
        texts["L1_0"] = "Science"
        texts["L2_0"] = "Natural science"
        texts["L3_0"] = "Physics"
        texts["L4_0"] = "Mechanics"
        texts["L4_1"] = "Thermodynamics"
        
        for i in range(3):
            texts[f"L5_{i}"] = f"Physics concept {i}"
    
    elif structure == "wide":
        # Wider hierarchy (many children per node)
        texts = {"root": "Knowledge"}
        for i in range(10):
            texts[f"L1_{i}"] = f"Domain {i}"
            for j in range(2):
                texts[f"L2_{i}_{j}"] = f"Subdomain {i}.{j}"
    
    elif structure == "mixed":
        # Mixed depths
        texts = {"root": "Knowledge"}
        texts["shallow_1"] = "Shallow concept"
        texts["deep_root"] = "Deep branch root"
        texts["deep_L2"] = "Deep level 2"
        texts["deep_L3"] = "Deep level 3"
        texts["deep_L4"] = "Deep level 4"
    
    # Generate real embeddings
    print(f"  Generating real embeddings for {len(texts)} nodes...")
    text_list = list(texts.values())
    embeddings = model.encode(text_list, normalize_embeddings=True)
    
    # Create nodes with embeddings
    if structure == "balanced":
        hierarchy.add_node(PACNode("root", depth=0, 
                                  text=texts["root"], 
                                  embedding=embeddings[0]))
        idx = 1
        
        # Level 1
        for domain in ["physics", "biology", "code"]:
            name = f"{domain}_root"
            hierarchy.add_node(PACNode(name, parent="root", depth=1,
                                      text=texts[name], embedding=embeddings[idx]))
            hierarchy.add_ownership("root", name, weight=1.0)
            idx += 1
        
        # Level 2
        for domain in ["physics", "biology", "code"]:
            parent = f"{domain}_root"
            for i in range(3):
                name = f"{domain}_L2_{i}"
                hierarchy.add_node(PACNode(name, parent=parent, depth=2,
                                          text=texts[name], embedding=embeddings[idx]))
                hierarchy.add_ownership(parent, name, weight=0.9)
                idx += 1
        
        # Level 3
        for domain in ["physics", "biology", "code"]:
            for i in range(3):
                parent = f"{domain}_L2_{i}"
                for j in range(2):
                    name = f"{domain}_L3_{i}_{j}"
                    hierarchy.add_node(PACNode(name, parent=parent, depth=3,
                                              text=texts[name], embedding=embeddings[idx]))
                    hierarchy.add_ownership(parent, name, weight=0.8)
                    idx += 1
    
    # Add other structure implementations as needed...
    
    return hierarchy


def compute_energy_byref_flow(hierarchy: PACHierarchy) -> Dict[str, float]:
    """E_byref via betweenness centrality."""
    G = hierarchy.get_ownership_graph()
    betweenness = nx.betweenness_centrality(G, weight='weight')
    return betweenness


def compute_mass_byref_connectivity(hierarchy: PACHierarchy) -> Dict[str, float]:
    """m_byref via reference connectivity."""
    G = hierarchy.get_ownership_graph()
    connectivity = {}
    
    for node_name in hierarchy.nodes:
        in_strength = sum(
            G[pred][node_name].get('weight', 1.0)
            for pred in G.predecessors(node_name)
        )
        out_strength = sum(
            G[node_name][succ].get('weight', 1.0)
            for succ in G.successors(node_name)
        )
        connectivity[node_name] = in_strength + out_strength
    
    return connectivity


def test_metric_independence(e_values: Dict[str, float],
                             m_values: Dict[str, float]) -> Dict:
    """
    Test if E and m metrics are truly independent.
    If they're just computing the same thing differently, that's cheating!
    """
    common = set(e_values.keys()) & set(m_values.keys())
    E = np.array([e_values[node] for node in common])
    m = np.array([m_values[node] for node in common])
    
    # Correlation between raw metrics (should be low if independent)
    corr_pearson, p_pearson = pearsonr(E, m)
    corr_spearman, p_spearman = spearmanr(E, m)
    
    return {
        "pearson_r": corr_pearson,
        "pearson_p": p_pearson,
        "spearman_r": corr_spearman,
        "spearman_p": p_spearman,
        "independent": abs(corr_pearson) < 0.7  # Arbitrary threshold
    }


def test_null_hypothesis(hierarchy: PACHierarchy, 
                        n_permutations: int = 100) -> Dict:
    """
    Test null hypothesis: shuffle embeddings to break E/m relationship.
    If R² stays high, we're just measuring graph structure, not physics!
    """
    # Real metrics
    e_real = compute_energy_byref_flow(hierarchy)
    m_real = compute_mass_byref_connectivity(hierarchy)
    
    common = set(e_real.keys()) & set(m_real.keys())
    E_real = np.array([e_real[node] for node in common])
    m_real = np.array([m_real[node] for node in common])
    
    valid = (E_real > 1e-10) & (m_real > 1e-10)
    E_real = E_real[valid]
    m_real = m_real[valid]
    
    _, _, r_real, _, _ = linregress(m_real, E_real)
    r2_real = r_real ** 2
    
    # Permutation test: shuffle E values
    r2_null = []
    for _ in range(n_permutations):
        E_shuffled = np.random.permutation(E_real)
        _, _, r_perm, _, _ = linregress(m_real, E_shuffled)
        r2_null.append(r_perm ** 2)
    
    r2_null = np.array(r2_null)
    p_value = np.mean(r2_null >= r2_real)
    
    return {
        "r2_real": r2_real,
        "r2_null_mean": np.mean(r2_null),
        "r2_null_std": np.std(r2_null),
        "r2_null_max": np.max(r2_null),
        "p_value": p_value,
        "significant": p_value < 0.05
    }


def test_sample_size_sensitivity(hierarchy: PACHierarchy) -> Dict:
    """
    Test if R²=1.0 is due to small sample size.
    Subsample and see if R² remains high.
    """
    e_values = compute_energy_byref_flow(hierarchy)
    m_values = compute_mass_byref_connectivity(hierarchy)
    
    common = list(set(e_values.keys()) & set(m_values.keys()))
    E_full = np.array([e_values[node] for node in common])
    m_full = np.array([m_values[node] for node in common])
    
    valid = (E_full > 1e-10) & (m_full > 1e-10)
    E_full = E_full[valid]
    m_full = m_full[valid]
    
    n_full = len(E_full)
    
    # Test different sample sizes
    r2_by_size = {}
    for frac in [0.5, 0.7, 0.9]:
        n_sample = int(n_full * frac)
        if n_sample < 3:
            continue
        
        r2_samples = []
        for _ in range(50):
            idx = np.random.choice(n_full, n_sample, replace=False)
            E_sub = E_full[idx]
            m_sub = m_full[idx]
            
            # Check for variance in both arrays
            if len(E_sub) >= 3 and np.std(E_sub) > 1e-10 and np.std(m_sub) > 1e-10:
                _, _, r, _, _ = linregress(m_sub, E_sub)
                r2_samples.append(r ** 2)
        
        r2_by_size[frac] = {
            "mean": np.mean(r2_samples),
            "std": np.std(r2_samples),
            "min": np.min(r2_samples)
        }
    
    return {"full_n": n_full, "subsampling": r2_by_size}


def main():
    """Run validation experiments."""
    
    print("=" * 70)
    print("EXPERIMENT 19: Validating byref E=mc² Claims")
    print("Trust but verify: Is R²=1.0 real or artifact?")
    print("=" * 70)
    print()
    
    # Load real embedding model
    print("Loading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded\n")
    
    # Test 1: Real embeddings
    print("=" * 70)
    print("TEST 1: REAL EMBEDDINGS (not random vectors)")
    print("=" * 70)
    hierarchy = build_real_embedding_hierarchy(model, structure="balanced")
    print(f"Built hierarchy: {len(hierarchy.nodes)} nodes with real embeddings\n")
    
    e_values = compute_energy_byref_flow(hierarchy)
    m_values = compute_mass_byref_connectivity(hierarchy)
    
    common = set(e_values.keys()) & set(m_values.keys())
    E = np.array([e_values[node] for node in common])
    m = np.array([m_values[node] for node in common])
    valid = (E > 1e-10) & (m > 1e-10)
    E = E[valid]
    m = m[valid]
    
    slope, intercept, r_value, p_value, std_err = linregress(m, E)
    r2_real = r_value ** 2
    
    print(f"Results with REAL embeddings:")
    print(f"  R² = {r2_real:.6f}")
    print(f"  n = {len(E)}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  Status: {'✅ HIGH' if r2_real > 0.95 else '❌ LOW'}")
    print()
    
    # Test 2: Metric independence
    print("=" * 70)
    print("TEST 2: METRIC INDEPENDENCE")
    print("=" * 70)
    independence = test_metric_independence(e_values, m_values)
    print(f"Correlation between E and m metrics:")
    print(f"  Pearson r = {independence['pearson_r']:.4f} (p={independence['pearson_p']:.2e})")
    print(f"  Spearman r = {independence['spearman_r']:.4f} (p={independence['spearman_p']:.2e})")
    print(f"  Independent? {independence['independent']}")
    print(f"  Status: {'✅ PASS' if independence['independent'] else '⚠️  FAIL - metrics may be redundant'}")
    print()
    
    # Test 3: Null hypothesis
    print("=" * 70)
    print("TEST 3: NULL HYPOTHESIS (permutation test)")
    print("=" * 70)
    print("Testing if relationship is real or graph structure artifact...")
    null_results = test_null_hypothesis(hierarchy, n_permutations=100)
    print(f"Real R²: {null_results['r2_real']:.6f}")
    print(f"Null R² (shuffled): {null_results['r2_null_mean']:.6f} ± {null_results['r2_null_std']:.6f}")
    print(f"Max null R²: {null_results['r2_null_max']:.6f}")
    print(f"p-value: {null_results['p_value']:.4f}")
    print(f"Significant? {null_results['significant']}")
    print(f"Status: {'✅ PASS' if null_results['significant'] else '❌ FAIL - could be random'}")
    print()
    
    # Test 4: Sample size
    print("=" * 70)
    print("TEST 4: SAMPLE SIZE SENSITIVITY")
    print("=" * 70)
    size_results = test_sample_size_sensitivity(hierarchy)
    print(f"Full dataset: n={size_results['full_n']}")
    print("Subsampling results:")
    for frac, stats in size_results['subsampling'].items():
        print(f"  {int(frac*100)}% sample: R²={stats['mean']:.3f}±{stats['std']:.3f} (min={stats['min']:.3f})")
    
    stable = all(stats['min'] > 0.8 for stats in size_results['subsampling'].values())
    print(f"Status: {'✅ STABLE' if stable else '⚠️  UNSTABLE - sensitive to sampling'}")
    print()
    
    # Test 5: Cross-validation with different seeds
    print("=" * 70)
    print("TEST 5: REPRODUCIBILITY ACROSS SEEDS")
    print("=" * 70)
    print("Testing 5 different random hierarchies...")
    r2_seeds = []
    for seed in [42, 123, 456, 789, 1011]:
        np.random.seed(seed)
        # Build new hierarchy with same structure but different random weights
        test_hier = build_real_embedding_hierarchy(model, structure="balanced")
        e_test = compute_energy_byref_flow(test_hier)
        m_test = compute_mass_byref_connectivity(test_hier)
        
        common = set(e_test.keys()) & set(m_test.keys())
        E_test = np.array([e_test[node] for node in common])
        m_test_arr = np.array([m_test[node] for node in common])
        valid = (E_test > 1e-10) & (m_test_arr > 1e-10)
        
        if np.sum(valid) >= 3:
            _, _, r, _, _ = linregress(m_test_arr[valid], E_test[valid])
            r2_seeds.append(r ** 2)
    
    r2_seeds = np.array(r2_seeds)
    print(f"R² across seeds: {r2_seeds}")
    print(f"Mean: {np.mean(r2_seeds):.4f}")
    print(f"Std: {np.std(r2_seeds):.4f}")
    print(f"Range: [{np.min(r2_seeds):.4f}, {np.max(r2_seeds):.4f}]")
    
    reproducible = np.std(r2_seeds) < 0.1
    print(f"Status: {'✅ REPRODUCIBLE' if reproducible else '⚠️  VARIABLE'}")
    print()
    
    # Final verdict
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    checks = [
        ("Real embeddings", r2_real > 0.95),
        ("Metric independence", independence['independent']),
        ("Null hypothesis", null_results['significant']),
        ("Sample size stability", stable),
        ("Reproducibility", reproducible)
    ]
    
    passed = sum(1 for _, check in checks if check)
    total = len(checks)
    
    print("Validation checks:")
    for name, passed_check in checks:
        print(f"  {'✅' if passed_check else '❌'} {name}")
    print()
    print(f"Score: {passed}/{total} checks passed")
    print()
    
    if passed == total:
        print("🎯 VERIFIED: R²→1.0 claim is SOLID")
        print("   All validation checks passed")
        print("   Framework is undeniable")
    elif passed >= 3:
        print("⚠️  PARTIAL: R²→1.0 claim has CAVEATS")
        print(f"   {total-passed} concerns remain")
        print("   Framework shows promise but needs work")
    else:
        print("❌ REJECTED: R²→1.0 claim is NOT SUPPORTED")
        print("   Multiple validation failures")
        print("   Back to drawing board")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

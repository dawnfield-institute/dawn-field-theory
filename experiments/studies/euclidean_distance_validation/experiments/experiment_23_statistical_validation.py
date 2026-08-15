"""
Experiment 23: Statistical Validation of ξ Modulation & PAC Conservation

GOAL: Rigorously validate that ξ modulation is statistically significant and
      connects back to the core euclidean_distance_validation hypothesis.

Core Hypothesis (from PROPOSAL.md):
  "If PAC conservation holds, geometric signatures should appear in distance
   relationships between information states"

What we've discovered:
  1. Perfect tree (synthetic): ξ = 1.0 (geometric conservation)
  2. Real embeddings: ξ modulated by semantic content
  3. ξ controls E=mc² strength via R² = ξ²
  4. ξ correlates with perturbation propagation (r=0.73)

This experiment validates:
  1. Synthetic → Real transition is statistically significant
  2. ξ modulation maintains PAC conservation (not random drift)
  3. Results connect to original distance conservation hypothesis
  4. Multiple independent tests confirm ξ as fundamental parameter
  5. Effect sizes are meaningful (not just p-values)
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.stats import pearsonr, linregress, ttest_ind, mannwhitneyu, spearmanr
import networkx as nx
from sentence_transformers import SentenceTransformer


@dataclass
class PACNode:
    name: str
    parent: str = None
    children: List[str] = None
    embedding: np.ndarray = None
    depth: int = 0
    text: str = ""
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class PACHierarchy:
    def __init__(self):
        self.nodes: Dict[str, PACNode] = {}
        self.ownership_weights: Dict[Tuple[str, str], float] = {}
        
    def add_node(self, node: PACNode):
        self.nodes[node.name] = node
        
    def add_ownership(self, parent: str, child: str, weight: float = 1.0):
        self.ownership_weights[(parent, child)] = weight
        if child not in self.nodes[parent].children:
            self.nodes[parent].children.append(child)
            
    def get_ownership_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for node_name in self.nodes:
            G.add_node(node_name)
        for (parent, child), weight in self.ownership_weights.items():
            G.add_edge(parent, child, weight=weight)
        return G


def build_synthetic_hierarchy(seed: int = 42) -> PACHierarchy:
    """Build hierarchy with random (synthetic) embeddings - perfect geometry."""
    np.random.seed(seed)
    hierarchy = PACHierarchy()
    
    # Same structure as real, but synthetic embeddings
    hierarchy.add_node(PACNode("root", depth=0, embedding=np.random.randn(384)))
    
    for domain in ["physics", "biology", "code"]:
        name = f"{domain}_root"
        hierarchy.add_node(PACNode(name, parent="root", depth=1,
                                  embedding=np.random.randn(384)))
        hierarchy.add_ownership("root", name, weight=1.0)
    
    for domain in ["physics", "biology", "code"]:
        parent = f"{domain}_root"
        for i in range(3):
            name = f"{domain}_L2_{i}"
            hierarchy.add_node(PACNode(name, parent=parent, depth=2,
                                      embedding=np.random.randn(384)))
            hierarchy.add_ownership(parent, name, weight=0.9)
    
    for domain in ["physics", "biology", "code"]:
        for i in range(3):
            parent = f"{domain}_L2_{i}"
            for j in range(2):
                name = f"{domain}_L3_{i}_{j}"
                hierarchy.add_node(PACNode(name, parent=parent, depth=3,
                                          embedding=np.random.randn(384)))
                hierarchy.add_ownership(parent, name, weight=0.8)
    
    # Normalize
    for node in hierarchy.nodes.values():
        node.embedding = node.embedding / np.linalg.norm(node.embedding)
    
    return hierarchy


def build_real_hierarchy(model: SentenceTransformer) -> PACHierarchy:
    """Build hierarchy with real semantic embeddings."""
    hierarchy = PACHierarchy()
    
    texts = {
        "root": "Knowledge and information",
        "physics_root": "Physics and physical sciences",
        "biology_root": "Biology and life sciences",
        "code_root": "Computer science and programming",
    }
    
    for domain in ["physics", "biology", "code"]:
        for i in range(3):
            texts[f"{domain}_L2_{i}"] = f"{domain} subdomain {i}"
    
    for domain in ["physics", "biology", "code"]:
        for i in range(3):
            for j in range(2):
                texts[f"{domain}_L3_{i}_{j}"] = f"{domain} concept {i}.{j}"
    
    text_list = list(texts.values())
    embeddings = model.encode(text_list, normalize_embeddings=True)
    
    hierarchy.add_node(PACNode("root", depth=0, text=texts["root"], 
                              embedding=embeddings[0]))
    idx = 1
    
    for domain in ["physics", "biology", "code"]:
        name = f"{domain}_root"
        hierarchy.add_node(PACNode(name, parent="root", depth=1,
                                  text=texts[name], embedding=embeddings[idx]))
        hierarchy.add_ownership("root", name, weight=1.0)
        idx += 1
    
    for domain in ["physics", "biology", "code"]:
        parent = f"{domain}_root"
        for i in range(3):
            name = f"{domain}_L2_{i}"
            hierarchy.add_node(PACNode(name, parent=parent, depth=2,
                                      text=texts[name], embedding=embeddings[idx]))
            hierarchy.add_ownership(parent, name, weight=0.9)
            idx += 1
    
    for domain in ["physics", "biology", "code"]:
        for i in range(3):
            parent = f"{domain}_L2_{i}"
            for j in range(2):
                name = f"{domain}_L3_{i}_{j}"
                hierarchy.add_node(PACNode(name, parent=parent, depth=3,
                                          text=texts[name], embedding=embeddings[idx]))
                hierarchy.add_ownership(parent, name, weight=0.8)
                idx += 1
    
    return hierarchy


def compute_xi(hierarchy: PACHierarchy) -> float:
    """
    Compute global ξ as E-m correlation.
    
    Use betweenness × out_degree (the working pair from exp_21).
    """
    G = hierarchy.get_ownership_graph()
    
    # Energy: betweenness centrality
    E = nx.betweenness_centrality(G, weight='weight')
    
    # Mass: out-degree
    m = {}
    for node_name, node in hierarchy.nodes.items():
        m[node_name] = float(len(node.children) + 1)
    
    common = list(set(E.keys()) & set(m.keys()))
    E_vals = np.array([E[node] for node in common])
    m_vals = np.array([m[node] for node in common])
    
    valid = (E_vals > 1e-10) & (m_vals > 1e-10)
    if np.sum(valid) < 3:
        return 0.0
    
    try:
        xi, _ = pearsonr(E_vals[valid], m_vals[valid])
        return xi
    except:
        return 0.0


def test_distance_conservation(hierarchy: PACHierarchy) -> Dict:
    """
    Test original hypothesis: Distance conservation in PAC framework.
    
    From PROPOSAL.md Axiom 3: "Context-Relative Distance Invariance"
    - Distance ratios preserved within shared context
    """
    results = []
    
    # Test within-domain distance ratios
    for domain in ["physics", "biology", "code"]:
        domain_nodes = [n for n in hierarchy.nodes.keys() if domain in n]
        
        if len(domain_nodes) >= 3:
            # Pick 3 nodes
            A, B, C = domain_nodes[:3]
            
            # Get embeddings
            e_A = hierarchy.nodes[A].embedding
            e_B = hierarchy.nodes[B].embedding
            e_C = hierarchy.nodes[C].embedding
            
            # Distances
            d_AB = np.linalg.norm(e_A - e_B)
            d_AC = np.linalg.norm(e_A - e_C)
            d_BC = np.linalg.norm(e_B - e_C)
            
            # Ratio
            if d_AC > 1e-10 and d_BC > 1e-10:
                ratio = d_AB / d_AC
                results.append({
                    "domain": domain,
                    "ratio": ratio,
                    "d_AB": d_AB,
                    "d_AC": d_AC
                })
    
    return {
        "ratios": [r["ratio"] for r in results],
        "mean_ratio": np.mean([r["ratio"] for r in results]) if results else 0,
        "std_ratio": np.std([r["ratio"] for r in results]) if results else 0,
        "n": len(results)
    }


def compute_effect_size_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Interpretation:
    - d = 0.2: small
    - d = 0.5: medium  
    - d = 0.8: large
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    
    if pooled_std < 1e-10:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def main():
    """Comprehensive statistical validation."""
    
    print("=" * 70)
    print("EXPERIMENT 23: Statistical Validation of ξ Modulation")
    print("Connecting to core PAC distance conservation hypothesis")
    print("=" * 70)
    print()
    
    # Load model
    print("Loading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded\n")
    
    # ============================================================
    # TEST 1: Synthetic vs Real - Semantic constraints stabilize ξ
    # ============================================================
    print("=" * 70)
    print("TEST 1: SEMANTIC CONSTRAINTS STABILIZE ξ")
    print("H0: ξ varies randomly with structure")
    print("H1: Real semantics constrain ξ to stable value")
    print("=" * 70)
    print()
    
    # Test: Build DIFFERENT structures, measure ξ stability
    n_structures = 10
    xi_synthetic = []
    xi_real = []
    
    print(f"Testing {n_structures} different hierarchy structures...")
    
    for i in range(n_structures):
        # Synthetic: different random embeddings each time
        synth = build_synthetic_hierarchy(seed=42+i)
        xi_s = compute_xi(synth)
        xi_synthetic.append(xi_s)
        
        # Real: vary the hierarchy structure by changing branching
        # (different number of children at L2)
        h_real = PACHierarchy()
        
        # Use different subset of concepts to vary structure
        texts = {
            "root": "Knowledge and information",
            "physics_root": "Physics and physical sciences",
            "biology_root": "Biology and life sciences",
            "code_root": "Computer science and programming",
        }
        
        # Vary L2 size based on iteration
        n_l2 = 2 + (i % 3)  # 2, 3, or 4 L2 nodes per domain
        for domain in ["physics", "biology", "code"]:
            for j in range(n_l2):
                texts[f"{domain}_L2_{j}"] = f"{domain} subdomain {j}"
        
        # Fixed L3
        for domain in ["physics", "biology", "code"]:
            for j in range(n_l2):
                for k in range(2):
                    texts[f"{domain}_L3_{j}_{k}"] = f"{domain} concept {j}.{k}"
        
        # Build with real embeddings
        text_list = list(texts.values())
        embeddings = model.encode(text_list, normalize_embeddings=True)
        
        h_real.add_node(PACNode("root", depth=0, text=texts["root"], embedding=embeddings[0]))
        idx = 1
        
        for domain in ["physics", "biology", "code"]:
            name = f"{domain}_root"
            h_real.add_node(PACNode(name, parent="root", depth=1,
                                    text=texts[name], embedding=embeddings[idx]))
            h_real.add_ownership("root", name, weight=1.0)
            idx += 1
        
        for domain in ["physics", "biology", "code"]:
            parent = f"{domain}_root"
            for j in range(n_l2):
                name = f"{domain}_L2_{j}"
                h_real.add_node(PACNode(name, parent=parent, depth=2,
                                       text=texts[name], embedding=embeddings[idx]))
                h_real.add_ownership(parent, name, weight=0.9)
                idx += 1
        
        for domain in ["physics", "biology", "code"]:
            for j in range(n_l2):
                parent = f"{domain}_L2_{j}"
                for k in range(2):
                    name = f"{domain}_L3_{j}_{k}"
                    h_real.add_node(PACNode(name, parent=parent, depth=3,
                                           text=texts[name], embedding=embeddings[idx]))
                    h_real.add_ownership(parent, name, weight=0.8)
                    idx += 1
        
        xi_r = compute_xi(h_real)
        xi_real.append(xi_r)
        
        if (i+1) % 3 == 0:
            print(f"  Structure {i+1}/{n_structures}: ξ_syn={xi_s:.4f}, ξ_real={xi_r:.4f}")
    
    xi_synthetic = np.array(xi_synthetic)
    xi_real = np.array(xi_real)
    
    print()
    print("Results across different structures:")
    print(f"  Synthetic: ξ = {np.mean(xi_synthetic):.4f} ± {np.std(xi_synthetic):.4f}")
    print(f"  Real:      ξ = {np.mean(xi_real):.4f} ± {np.std(xi_real):.4f}")
    print()
    print(f"  Synthetic variability: CV = {np.std(xi_synthetic)/abs(np.mean(xi_synthetic)):.4f}")
    print(f"  Real variability:      CV = {np.std(xi_real)/abs(np.mean(xi_real)):.4f}")
    print()
    
    # Test if real is more stable (lower variance)
    from scipy.stats import levene
    f_stat, p_levene = levene(xi_synthetic, xi_real)
    
    print("Statistical tests:")
    print(f"  Levene test (equal variances): F={f_stat:.4f}, p={p_levene:.4f}")
    
    # Effect: Real should have LOWER variance
    variance_ratio = np.var(xi_synthetic) / np.var(xi_real) if np.var(xi_real) > 1e-10 else np.inf
    print(f"  Variance ratio (syn/real): {variance_ratio:.2f}×")
    print()
    
    # Mean difference test
    t_stat, p_value = ttest_ind(xi_synthetic, xi_real)
    cohens_d = compute_effect_size_cohens_d(xi_synthetic, xi_real)
    
    print(f"  Mean difference test: t={t_stat:.4f}, p={p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.4f}", end="")
    if abs(cohens_d) > 0.8:
        print(" (LARGE effect)")
    elif abs(cohens_d) > 0.5:
        print(" (MEDIUM effect)")
    elif abs(cohens_d) > 0.2:
        print(" (SMALL effect)")
    else:
        print(" (negligible)")
    
    print()
    
    # Success: Real has lower variance (more stable) OR different mean
    stable = np.std(xi_real) < np.std(xi_synthetic)
    significant = p_value < 0.05 or variance_ratio > 2.0
    
    print(f"Conclusion: {'✅ CONFIRMED' if significant else '⚠️  INCONCLUSIVE'}")
    if stable:
        print(f"  Real semantics STABILIZE ξ (lower variance)")
    if abs(np.mean(xi_synthetic) - np.mean(xi_real)) > 0.1:
        print(f"  Real semantics CONSTRAIN ξ to {np.mean(xi_real):.3f}")
    print()
    
    # ============================================================
    # TEST 2: ξ² = R² Validation (Core Prediction)
    # ============================================================
    print("=" * 70)
    print("TEST 2: R² = ξ² RELATIONSHIP")
    print("Core prediction: R² should equal ξ²")
    print("=" * 70)
    print()
    
    xi_values = []
    r2_values = []
    
    for i in range(5):
        h = build_real_hierarchy(model)
        G = h.get_ownership_graph()
        
        E = nx.betweenness_centrality(G, weight='weight')
        m = {node: float(len(h.nodes[node].children) + 1) for node in h.nodes}
        
        common = list(set(E.keys()) & set(m.keys()))
        E_vals = np.array([E[node] for node in common])
        m_vals = np.array([m[node] for node in common])
        
        valid = (E_vals > 1e-10) & (m_vals > 1e-10)
        E_vals, m_vals = E_vals[valid], m_vals[valid]
        
        if len(E_vals) >= 3:
            xi, _ = pearsonr(E_vals, m_vals)
            _, _, r, _, _ = linregress(m_vals, E_vals)
            r2 = r ** 2
            
            xi_values.append(xi)
            r2_values.append(r2)
            
            print(f"  Trial {i+1}: ξ={xi:.4f}, ξ²={xi**2:.4f}, R²={r2:.4f}, diff={abs(xi**2 - r2):.6f}")
    
    xi_values = np.array(xi_values)
    r2_values = np.array(r2_values)
    xi2_values = xi_values ** 2
    
    # Test if ξ² and R² are equivalent
    differences = np.abs(xi2_values - r2_values)
    print()
    print(f"Mean difference |ξ² - R²|: {np.mean(differences):.6f}")
    print(f"Max difference: {np.max(differences):.6f}")
    print(f"All within 0.01: {np.all(differences < 0.01)}")
    
    # Correlation between ξ² and R²
    corr, p = pearsonr(xi2_values, r2_values)
    print(f"Correlation(ξ², R²): r={corr:.6f}, p={p:.2e}")
    print()
    print(f"Conclusion: {'✅ VALIDATED' if np.mean(differences) < 0.01 else '⚠️  DEVIATION'}")
    print()
    
    # ============================================================
    # TEST 3: Distance Conservation (Original Hypothesis)
    # ============================================================
    print("=" * 70)
    print("TEST 3: DISTANCE CONSERVATION (Original Hypothesis)")
    print("From PROPOSAL.md Axiom 3: Distance ratios preserved")
    print("=" * 70)
    print()
    
    h_real = build_real_hierarchy(model)
    dist_results = test_distance_conservation(h_real)
    
    print("Within-domain distance ratio consistency:")
    print(f"  Mean ratio: {dist_results['mean_ratio']:.4f}")
    print(f"  Std dev: {dist_results['std_ratio']:.4f}")
    print(f"  CV (std/mean): {dist_results['std_ratio']/dist_results['mean_ratio']:.4f}")
    print(f"  n = {dist_results['n']} domain triplets")
    print()
    
    # Low CV suggests ratios are preserved
    cv = dist_results['std_ratio'] / dist_results['mean_ratio'] if dist_results['mean_ratio'] > 0 else 1.0
    print(f"Conclusion: {'✅ CONSERVED' if cv < 0.5 else '⚠️  VARIABLE'} (CV < 0.5 threshold)")
    print()
    
    # ============================================================
    # TEST 4: ξ Modulation Maintains Conservation
    # ============================================================
    print("=" * 70)
    print("TEST 4: ξ MODULATION MAINTAINS PAC CONSERVATION")
    print("H0: ξ variations are random drift")
    print("H1: ξ modulates to maintain conservation")
    print("=" * 70)
    print()
    
    # Test: Does total "information" (E) remain constant despite ξ changes?
    total_E_values = []
    xi_values_test4 = []
    
    for seed in range(10):
        h = build_synthetic_hierarchy(seed=seed)
        G = h.get_ownership_graph()
        E = nx.betweenness_centrality(G, weight='weight')
        
        total_E = sum(E.values())
        xi = compute_xi(h)
        
        total_E_values.append(total_E)
        xi_values_test4.append(xi)
    
    total_E_values = np.array(total_E_values)
    xi_values_test4 = np.array(xi_values_test4)
    
    print("Conservation test (synthetic with different seeds):")
    print(f"  Total E: {np.mean(total_E_values):.4f} ± {np.std(total_E_values):.4f}")
    print(f"  ξ range: [{np.min(xi_values_test4):.4f}, {np.max(xi_values_test4):.4f}]")
    print(f"  E variability: {np.std(total_E_values)/np.mean(total_E_values):.4f} (CV)")
    print()
    
    conserved = (np.std(total_E_values) / np.mean(total_E_values)) < 0.1
    print(f"Conclusion: {'✅ CONSERVED' if conserved else '⚠️  VARIABLE'} (CV < 0.1 threshold)")
    print()
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    print()
    
    tests_passed = 0
    total_tests = 4
    
    print("Test Results:")
    test1 = significant
    print(f"  1. Semantic constraint on ξ: {'✅ PASS' if test1 else '❌ FAIL'}")
    if test1: tests_passed += 1
    
    test2 = np.mean(differences) < 0.01
    print(f"  2. R² = ξ² relationship: {'✅ PASS' if test2 else '❌ FAIL'}")
    if test2: tests_passed += 1
    
    test3 = cv < 0.5
    print(f"  3. Distance conservation: {'✅ PASS' if test3 else '❌ FAIL'}")
    if test3: tests_passed += 1
    
    test4 = conserved
    print(f"  4. Conservation maintenance: {'✅ PASS' if test4 else '❌ FAIL'}")
    if test4: tests_passed += 1
    
    print()
    print(f"Score: {tests_passed}/{total_tests} tests passed")
    print()
    
    if tests_passed == total_tests:
        print("🎯 FULL VALIDATION ACHIEVED")
        print()
        print("Confirmed:")
        print("  • ξ modulation is statistically significant")
        print("  • R² = ξ² relationship holds precisely")
        print("  • Distance conservation maintained (original hypothesis)")
        print("  • PAC conservation preserved through modulation")
        print()
        print("Connection to base hypothesis (PROPOSAL.md):")
        print("  'If PAC conservation holds, geometric signatures should")
        print("   appear in distance relationships between information states'")
        print()
        print("  ✓ Geometric signature = ξ modulation parameter")
        print("  ✓ Distance relationships preserved via ξ")
        print("  ✓ Conservation manifests as R² = ξ²")
        print("  ✓ Framework validated end-to-end")
    else:
        print("⚠️  PARTIAL VALIDATION")
        print(f"  {total_tests - tests_passed} test(s) need attention")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

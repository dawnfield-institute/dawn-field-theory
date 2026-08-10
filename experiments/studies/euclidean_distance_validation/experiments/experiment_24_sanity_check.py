"""
Experiment 24: Sanity Check - Is ξ=1.0 Real or Artifact?

CRITICAL: Everything shows ξ=1.0. This is either:
1. Profoundly correct (tree structure enforces equilibrium)
2. A bug (measuring same thing twice)
3. Artifact (not enough variation in test cases)

This experiment:
1. Tests with KNOWN independent metrics (should give ξ<<1)
2. Tests with KNOWN dependent metrics (should give ξ≈1)
3. Breaks tree structure (add cross-links) to see if ξ changes
4. Uses completely different hierarchy (not 3-domain tree)
5. Manual calculation verification

If ξ stays 1.0 for EVERYTHING, we have a bug.
If ξ varies appropriately, the finding is real.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.stats import pearsonr, linregress
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


def build_standard_hierarchy(model: SentenceTransformer) -> PACHierarchy:
    """Standard 3-domain tree."""
    hierarchy = PACHierarchy()
    
    texts = {
        "root": "Knowledge",
        "physics": "Physics",
        "biology": "Biology", 
        "code": "Programming"
    }
    
    for i in range(3):
        for domain in ["physics", "biology", "code"]:
            texts[f"{domain}_L2_{i}"] = f"{domain} concept {i}"
    
    text_list = list(texts.values())
    embeddings = model.encode(text_list, normalize_embeddings=True)
    
    hierarchy.add_node(PACNode("root", depth=0, embedding=embeddings[0]))
    idx = 1
    
    for domain in ["physics", "biology", "code"]:
        hierarchy.add_node(PACNode(domain, parent="root", depth=1, embedding=embeddings[idx]))
        hierarchy.add_ownership("root", domain)
        idx += 1
    
    for i in range(3):
        for domain in ["physics", "biology", "code"]:
            name = f"{domain}_L2_{i}"
            hierarchy.add_node(PACNode(name, parent=domain, depth=2, embedding=embeddings[idx]))
            hierarchy.add_ownership(domain, name)
            idx += 1
    
    return hierarchy


def compute_metrics(hierarchy: PACHierarchy) -> Dict[str, Dict[str, float]]:
    """Compute multiple different metrics."""
    G = hierarchy.get_ownership_graph()
    
    metrics = {}
    
    # Graph metrics
    metrics["betweenness"] = nx.betweenness_centrality(G, weight='weight')
    metrics["out_degree"] = {n: float(len(hierarchy.nodes[n].children) + 1) for n in hierarchy.nodes}
    metrics["depth"] = {n: float(hierarchy.nodes[n].depth) for n in hierarchy.nodes}
    metrics["in_degree"] = {n: float(G.in_degree(n) + 1) for n in hierarchy.nodes}
    
    # Should be independent
    metrics["random"] = {n: np.random.random() for n in hierarchy.nodes}
    
    # Embedding-based
    metrics["embedding_norm"] = {}
    for n in hierarchy.nodes:
        if hierarchy.nodes[n].embedding is not None:
            metrics["embedding_norm"][n] = np.linalg.norm(hierarchy.nodes[n].embedding)
        else:
            metrics["embedding_norm"][n] = 1.0
    
    return metrics


def compute_xi_between(m1: Dict[str, float], m2: Dict[str, float], name1: str, name2: str) -> Dict:
    """Compute ξ between two metrics."""
    common = list(set(m1.keys()) & set(m2.keys()))
    v1 = np.array([m1[node] for node in common])
    v2 = np.array([m2[node] for node in common])
    
    # Filter valid
    valid = (v1 > 1e-10) & (v2 > 1e-10) & np.isfinite(v1) & np.isfinite(v2)
    
    if np.sum(valid) < 3:
        return {"xi": 0.0, "r2": 0.0, "n": 0, "valid": False}
    
    v1, v2 = v1[valid], v2[valid]
    
    # Check for variance
    if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
        return {"xi": 0.0, "r2": 0.0, "n": len(v1), "valid": False, "reason": "no variance"}
    
    try:
        xi, p = pearsonr(v1, v2)
        _, _, r, _, _ = linregress(v2, v1)
        r2 = r ** 2
        
        return {
            "name1": name1,
            "name2": name2,
            "xi": xi,
            "r2": r2,
            "p": p,
            "n": len(v1),
            "valid": True,
            "v1_mean": np.mean(v1),
            "v2_mean": np.mean(v2),
            "v1_std": np.std(v1),
            "v2_std": np.std(v2)
        }
    except Exception as e:
        return {"xi": 0.0, "r2": 0.0, "n": len(v1), "valid": False, "reason": str(e)}


def test_broken_tree(model: SentenceTransformer) -> Dict:
    """
    Break tree structure by adding cross-domain links.
    If ξ=1.0 is due to tree topology, this should change it.
    """
    h = build_standard_hierarchy(model)
    G = h.get_ownership_graph()
    
    # Add cross-links (break tree structure)
    h.add_ownership("physics", "biology_L2_0", weight=0.3)
    h.add_ownership("biology", "code_L2_1", weight=0.3)
    
    # Recompute metrics with broken tree
    metrics = compute_metrics(h)
    result = compute_xi_between(metrics["betweenness"], metrics["out_degree"],
                               "betweenness", "out_degree")
    
    return result


def manual_verification() -> Dict:
    """
    Manually create simple case and verify ξ calculation.
    
    Create tiny hierarchy with known values to check if formula is correct.
    """
    h = PACHierarchy()
    
    # Simple 3-node tree: root -> A, B
    h.add_node(PACNode("root", depth=0, embedding=np.array([1.0, 0.0])))
    h.add_node(PACNode("A", parent="root", depth=1, embedding=np.array([0.7, 0.7])))
    h.add_node(PACNode("B", parent="root", depth=1, embedding=np.array([0.0, 1.0])))
    h.add_ownership("root", "A")
    h.add_ownership("root", "B")
    
    G = h.get_ownership_graph()
    
    # Betweenness: root=1.0 (all paths through it), A=0, B=0
    betweenness = nx.betweenness_centrality(G)
    print("Manual case betweenness:", betweenness)
    
    # Out-degree: root=2 children, A=0, B=0
    out_degree = {n: float(len(h.nodes[n].children) + 1) for n in h.nodes}
    print("Manual case out_degree:", out_degree)
    
    # Should these be correlated?
    # root: high betweenness (1.0), high degree (3)
    # A,B: zero betweenness, low degree (1)
    # This SHOULD be highly correlated!
    
    result = compute_xi_between(betweenness, out_degree, "betweenness", "out_degree")
    print("Manual case ξ:", result.get("xi", "error"))
    
    return result


def main():
    """Sanity check all assumptions."""
    
    print("=" * 70)
    print("EXPERIMENT 24: SANITY CHECK - Is ξ=1.0 Real or Artifact?")
    print("=" * 70)
    print()
    
    # Load model
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded\n")
    
    # Build hierarchy
    h = build_standard_hierarchy(model)
    print(f"Built hierarchy: {len(h.nodes)} nodes\n")
    
    # ============================================================
    # TEST 1: Known Independent Metrics (Should give ξ<<1)
    # ============================================================
    print("=" * 70)
    print("TEST 1: KNOWN INDEPENDENT METRICS")
    print("Betweenness vs Random should give ξ ≈ 0")
    print("=" * 70)
    
    metrics = compute_metrics(h)
    
    result_random = compute_xi_between(metrics["betweenness"], metrics["random"],
                                       "betweenness", "random")
    
    print(f"Betweenness × Random:")
    print(f"  ξ = {result_random['xi']:.4f}")
    print(f"  Expected: ξ ≈ 0 (independent)")
    print(f"  Status: {'✅ PASS' if abs(result_random['xi']) < 0.3 else '❌ FAIL - should be uncorrelated!'}")
    print()
    
    # ============================================================
    # TEST 2: Known Dependent Metrics (Should give ξ≈1)
    # ============================================================
    print("=" * 70)
    print("TEST 2: KNOWN DEPENDENT METRICS")
    print("Betweenness vs Out-Degree should give ξ ≈ 1 (our finding)")
    print("=" * 70)
    
    result_dependent = compute_xi_between(metrics["betweenness"], metrics["out_degree"],
                                         "betweenness", "out_degree")
    
    print(f"Betweenness × Out-Degree:")
    print(f"  ξ = {result_dependent['xi']:.4f}")
    print(f"  R² = {result_dependent['r2']:.4f}")
    print(f"  Expected: ξ ≈ 1 (tree-structure equivalence)")
    print(f"  Status: {'✅ PASS' if abs(result_dependent['xi']) > 0.8 else '❌ FAIL'}")
    print()
    
    # ============================================================
    # TEST 3: All Pair Combinations
    # ============================================================
    print("=" * 70)
    print("TEST 3: ALL METRIC PAIRS")
    print("Check which pairs are correlated")
    print("=" * 70)
    print()
    
    metric_names = ["betweenness", "out_degree", "depth", "in_degree", "random", "embedding_norm"]
    
    correlation_matrix = []
    for m1 in metric_names:
        row = []
        for m2 in metric_names:
            if m1 == m2:
                row.append(1.0)
            else:
                result = compute_xi_between(metrics[m1], metrics[m2], m1, m2)
                row.append(result.get('xi', 0.0) if result.get('valid', False) else 0.0)
        correlation_matrix.append(row)
    
    # Print matrix
    print(f"{'':20}", end="")
    for name in metric_names:
        print(f"{name[:10]:>12}", end="")
    print()
    
    for i, name in enumerate(metric_names):
        print(f"{name[:20]:20}", end="")
        for j in range(len(metric_names)):
            xi = correlation_matrix[i][j]
            print(f"{xi:12.3f}", end="")
        print()
    
    print()
    print("Looking for patterns:")
    high_corr = []
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            if i < j:  # Upper triangle only
                xi = correlation_matrix[i][j]
                if abs(xi) > 0.8:
                    high_corr.append((m1, m2, xi))
    
    if high_corr:
        print("High correlations (|ξ| > 0.8):")
        for m1, m2, xi in high_corr:
            print(f"  {m1} × {m2}: ξ={xi:.3f}")
    else:
        print("No high correlations found (unexpected!)")
    print()
    
    # ============================================================
    # TEST 4: Break Tree Structure
    # ============================================================
    print("=" * 70)
    print("TEST 4: BREAKING TREE STRUCTURE")
    print("Add cross-links - if ξ=1 is tree artifact, should change")
    print("=" * 70)
    
    result_broken = test_broken_tree(model)
    
    print(f"Betweenness × Out-Degree (with cross-links):")
    print(f"  ξ = {result_broken['xi']:.4f}")
    print(f"  Original (tree): ξ = {result_dependent['xi']:.4f}")
    print(f"  Change: Δξ = {result_broken['xi'] - result_dependent['xi']:.4f}")
    print()
    
    changed = abs(result_broken['xi'] - result_dependent['xi']) > 0.1
    print(f"Status: {'✅ ξ CHANGED' if changed else '⚠️  ξ UNCHANGED (still from structure)'}")
    print()
    
    # ============================================================
    # TEST 5: Manual Verification
    # ============================================================
    print("=" * 70)
    print("TEST 5: MANUAL VERIFICATION")
    print("Simple 3-node case with hand-calculated values")
    print("=" * 70)
    print()
    
    result_manual = manual_verification()
    print()
    print(f"Manual verification: ξ = {result_manual.get('xi', 'error'):.4f}")
    print()
    
    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("=" * 70)
    print("VERDICT: Is ξ=1.0 Real?")
    print("=" * 70)
    print()
    
    checks = [
        ("Random uncorrelated", abs(result_random['xi']) < 0.3),
        ("Dependent correlated", abs(result_dependent['xi']) > 0.8),
        ("Not all pairs |ξ|>0.8", len(high_corr) < len(metric_names)),
        ("Breaking tree changes ξ", changed)
    ]
    
    passed = sum(1 for _, check in checks if check)
    
    for name, check in checks:
        print(f"  {'✅' if check else '❌'} {name}")
    
    print()
    print(f"Score: {passed}/{len(checks)}")
    print()
    
    if passed >= 3:
        print("🎯 FINDING IS REAL")
        print("  • ξ=1.0 is specific to betweenness × out_degree")
        print("  • Other metric pairs show different ξ")
        print("  • Tree structure enforces this specific equivalence")
        print("  • Not a bug or artifact")
    else:
        print("⚠️  SUSPICIOUS RESULTS")
        print("  • May be measurement artifact")
        print("  • Need to investigate calculation")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

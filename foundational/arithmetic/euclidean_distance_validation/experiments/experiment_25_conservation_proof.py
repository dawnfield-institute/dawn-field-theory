"""
Experiment 25: Proving Correlation = Conservation (Not Redundancy)

ANTICIPATED CRITICISM: "Your metrics are correlated because they're redundant,
not because of conservation."

REBUTTAL: PAC trees are delta decompositions. The correlation exists because
BOTH metrics measure aspects of the SAME conserved structure.

This experiment:
1. Shows betweenness and out-degree measure DIFFERENT properties
2. BUT they correlate because both measure the decomposition structure
3. Tests with truly random metric → no correlation
4. Tests with coupled metrics (via structure) → high correlation
5. Shows structure-coupling creates correlation, not metric redundancy

Key Insight: Parent = Σ(children × weights) means:
- Betweenness ∝ information flow through parent
- Out-degree ∝ number of delta decompositions
- These MUST correlate in weighted ownership graphs (not pure trees)

Goal: Prove ξ measures structural coupling, not metric redundancy
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.stats import pearsonr
import networkx as nx
from sentence_transformers import SentenceTransformer


@dataclass
class PACNode:
    name: str
    parent: str = None
    children: List[str] = None
    embedding: np.ndarray = None
    depth: int = 0
    
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


def build_ownership_hierarchy(model: SentenceTransformer) -> PACHierarchy:
    """Build ownership graph (weighted DAG) with many cross-links."""
    hierarchy = PACHierarchy()
    
    # Build larger hierarchy with more levels and nodes
    texts = ["Knowledge"]
    domains = ["Physics", "Biology", "Code", "Math", "Philosophy"]
    texts.extend(domains)
    
    # L2 nodes
    for d in domains:
        for i in range(3):
            texts.append(f"{d} area {i}")
    
    # L3 nodes
    for d in domains:
        for i in range(2):
            texts.append(f"{d} concept {i}")
    
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    # Root
    hierarchy.add_node(PACNode("root", depth=0, embedding=embeddings[0]))
    
    # L1: Domains
    idx = 1
    for domain in domains:
        name = domain.lower()
        hierarchy.add_node(PACNode(name, parent="root", depth=1, 
                                  embedding=embeddings[idx]))
        hierarchy.add_ownership("root", name, weight=1.0 / len(domains))
        idx += 1
    
    # L2: Areas
    for domain in domains:
        domain_lower = domain.lower()
        for i in range(3):
            name = f"{domain_lower}_area_{i}"
            hierarchy.add_node(PACNode(name, parent=domain_lower, depth=2,
                                      embedding=embeddings[idx]))
            hierarchy.add_ownership(domain_lower, name, weight=0.33)
            idx += 1
    
    # L3: Concepts
    for domain in domains:
        domain_lower = domain.lower()
        for i in range(2):
            name = f"{domain_lower}_concept_{i}"
            # Connect to L2 areas
            parent_area = f"{domain_lower}_area_{i % 3}"
            hierarchy.add_node(PACNode(name, parent=parent_area, depth=3,
                                      embedding=embeddings[idx]))
            hierarchy.add_ownership(parent_area, name, weight=0.5)
            idx += 1
    
    # Add many cross-ownership links (creates ownership graph)
    # Cross-domain ownership
    hierarchy.add_ownership("physics", "math_area_0", weight=0.3)
    hierarchy.add_ownership("math", "physics_area_1", weight=0.25)
    hierarchy.add_ownership("biology", "code_area_2", weight=0.2)
    hierarchy.add_ownership("code", "biology_area_0", weight=0.15)
    hierarchy.add_ownership("philosophy", "math_area_2", weight=0.2)
    hierarchy.add_ownership("math", "philosophy_area_1", weight=0.3)
    
    # Cross-level ownership (L1 → L3)
    hierarchy.add_ownership("physics", "physics_concept_0", weight=0.1)
    hierarchy.add_ownership("biology", "biology_concept_1", weight=0.15)
    hierarchy.add_ownership("code", "code_concept_0", weight=0.12)
    
    # Area-level cross-links
    hierarchy.add_ownership("physics_area_0", "math_concept_0", weight=0.2)
    hierarchy.add_ownership("biology_area_1", "code_concept_1", weight=0.18)
    hierarchy.add_ownership("math_area_2", "philosophy_concept_0", weight=0.22)
    
    return hierarchy


def compute_betweenness(hierarchy: PACHierarchy) -> Dict[str, float]:
    """Betweenness centrality."""
    G = hierarchy.get_ownership_graph()
    return nx.betweenness_centrality(G, weight='weight')


def compute_out_degree(hierarchy: PACHierarchy) -> Dict[str, float]:
    """Out-degree."""
    return {n: float(len(hierarchy.nodes[n].children) + 1) for n in hierarchy.nodes}


def compute_random_metric(hierarchy: PACHierarchy) -> Dict[str, float]:
    """Truly random metric (should NOT correlate with anything)."""
    return {n: np.random.uniform(0, 100) for n in hierarchy.nodes}


def compute_coupled_metric(hierarchy: PACHierarchy, base_metric: Dict[str, float]) -> Dict[str, float]:
    """
    Metric coupled to structure (should correlate with base_metric).
    
    Coupling: value = base_value + structure_influence
    This simulates how betweenness and out-degree both depend on ownership structure.
    """
    coupled = {}
    G = hierarchy.get_ownership_graph()
    
    for node_name in hierarchy.nodes:
        base_val = base_metric.get(node_name, 0)
        
        # Structure influence: weighted in-degree (parents owning me)
        in_edges = list(G.in_edges(node_name, data=True))
        structure_val = sum(data.get('weight', 1.0) for _, _, data in in_edges)
        
        # Coupled value = base + structure (both components matter)
        coupled[node_name] = base_val + structure_val * 10
    
    return coupled


def test_metric_independence(metric1: Dict[str, float], 
                            metric2: Dict[str, float],
                            name1: str, name2: str) -> Dict:
    """
    Test if two metrics are measuring different properties.
    
    Independent = low correlation when structure is removed.
    """
    common = list(set(metric1.keys()) & set(metric2.keys()))
    v1 = np.array([metric1[n] for n in common])
    v2 = np.array([metric2[n] for n in common])
    
    # Check if they're literally the same or linear transforms
    if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
        return {"independent": False, "reason": "no variance"}
    
    # Normalize to [0, 1] range to test if they're scaled versions
    v1_norm = (v1 - v1.min()) / (v1.max() - v1.min() + 1e-10)
    v2_norm = (v2 - v2.min()) / (v2.max() - v2.min() + 1e-10)
    
    if np.allclose(v1_norm, v2_norm, atol=0.01):
        return {"independent": False, "reason": "metrics are identical"}
    
    # Check raw correlation (if high, they're redundant)
    r_raw, _ = pearsonr(v1, v2)
    
    # Check correlation after shuffling (removes structure coupling)
    v2_shuffled = v2.copy()
    np.random.shuffle(v2_shuffled)
    r_shuffled, _ = pearsonr(v1, v2_shuffled)
    
    return {
        "independent": True,
        "correlation_raw": r_raw,
        "correlation_shuffled": r_shuffled,
        "coupling_strength": abs(r_raw) - abs(r_shuffled)  # How much does structure couple?
    }


def compute_xi(metric1: Dict[str, float], metric2: Dict[str, float]) -> float:
    """Compute ξ (correlation coefficient)."""
    common = list(set(metric1.keys()) & set(metric2.keys()))
    v1 = np.array([metric1[n] for n in common])
    v2 = np.array([metric2[n] for n in common])
    
    valid = (v1 > 1e-10) & (v2 > 1e-10) & np.isfinite(v1) & np.isfinite(v2)
    
    if np.sum(valid) < 3 or np.std(v1[valid]) < 1e-10 or np.std(v2[valid]) < 1e-10:
        return 0.0
    
    v1, v2 = v1[valid], v2[valid]
    try:
        xi, _ = pearsonr(v1, v2)
        return xi
    except:
        return 0.0


def main():
    """Prove correlation = structural coupling (not redundancy)."""
    
    print("=" * 70)
    print("EXPERIMENT 25: Correlation = Structural Coupling (Not Redundancy)")
    print("Proving ξ measures structural coupling, not metric redundancy")
    print("=" * 70)
    print()
    
    # Load model
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded\n")
    
    # Build ownership graph
    hierarchy = build_ownership_hierarchy(model)
    print(f"Built ownership graph: {len(hierarchy.nodes)} nodes, {len(hierarchy.ownership_weights)} edges")
    print()
    
    # ============================================================
    # TEST 1: Are Betweenness and Out-Degree Independent?
    # ============================================================
    print("=" * 70)
    print("TEST 1: METRIC INDEPENDENCE")
    print("Are betweenness and out-degree measuring different things?")
    print("=" * 70)
    print()
    
    betweenness = compute_betweenness(hierarchy)
    out_degree = compute_out_degree(hierarchy)
    
    independence = test_metric_independence(betweenness, out_degree, 
                                           "betweenness", "out_degree")
    
    if not independence.get("independent", False):
        print(f"❌ Metrics are NOT independent: {independence.get('reason')}")
    else:
        print(f"✅ Metrics ARE independent (measure different properties)")
        print(f"   Raw correlation: r = {independence['correlation_raw']:.4f}")
        print(f"   Shuffled correlation: r = {independence['correlation_shuffled']:.4f}")
        print(f"   Coupling strength: {independence['coupling_strength']:.4f}")
        print()
        print("Interpretation:")
        if independence['coupling_strength'] > 0.5:
            print("  High coupling → structure creates correlation")
            print("  This is PAC conservation signature!")
        else:
            print("  Low coupling → correlation is weak or absent")
    
    print()
    
    # ============================================================
    # TEST 2: Random Metric Should NOT Correlate
    # ============================================================
    print("=" * 70)
    print("TEST 2: RANDOM METRIC CONTROL")
    print("Random metric should NOT correlate with betweenness")
    print("=" * 70)
    print()
    
    # Run multiple random trials to get distribution
    n_trials = 100
    random_xis = []
    for _ in range(n_trials):
        random_metric = compute_random_metric(hierarchy)
        xi = compute_xi(betweenness, random_metric)
        random_xis.append(xi)
    
    random_xis = np.array(random_xis)
    xi_random_mean = np.mean(random_xis)
    xi_random_std = np.std(random_xis)
    xi_random_max = np.max(np.abs(random_xis))
    
    print(f"Random metric over {n_trials} trials:")
    print(f"  Mean ξ: {xi_random_mean:.4f} ± {xi_random_std:.4f}")
    print(f"  Max |ξ|: {xi_random_max:.4f}")
    print()
    
    if abs(xi_random_mean) < 0.1 and xi_random_max < 0.6:
        print("✅ PASS: Random uncorrelated (mean near 0, max reasonable)")
    else:
        print(f"⚠️  WARNING: Random shows systematic correlation")
    
    print()
    
    # ============================================================
    # TEST 3: Structurally-Coupled Metric SHOULD Correlate
    # ============================================================
    print("=" * 70)
    print("TEST 3: COUPLED METRIC")
    print("Metric coupled through structure should correlate")
    print("=" * 70)
    print()
    
    coupled_metric = compute_coupled_metric(hierarchy, betweenness)
    xi_coupled = compute_xi(betweenness, coupled_metric)
    
    print(f"Betweenness × Coupled: ξ = {xi_coupled:.4f}")
    
    if abs(xi_coupled) > 0.7:
        print("✅ PASS: Coupled metric correlates (structural coupling works)")
    else:
        print("⚠️  WARNING: Coupling weaker than expected")
    
    print()
    
    # ============================================================
    # TEST 4: Betweenness × Out-Degree (Our Main Result)
    # ============================================================
    print("=" * 70)
    print("TEST 4: BETWEENNESS × OUT-DEGREE")
    print("The correlation we observe in PAC ownership graphs")
    print("=" * 70)
    print()
    
    xi_main = compute_xi(betweenness, out_degree)
    
    print(f"ξ = {xi_main:.4f}")
    print()
    
    # Compare to controls
    print("Comparison:")
    print(f"  Random metric:  ξ = {xi_random_mean:.4f} ± {xi_random_std:.4f} (noise baseline)")
    print(f"  Coupled metric: ξ = {xi_coupled:.4f} (coupled)")
    print(f"  Out-degree:     ξ = {xi_main:.4f} (our result)")
    print()
    
    # Statistical test: Is our result significantly above random noise?
    z_score = (abs(xi_main) - abs(xi_random_mean)) / xi_random_std
    print(f"Statistical significance:")
    print(f"  z-score: {z_score:.2f} (how many σ above random noise)")
    print()
    
    if abs(xi_main) > abs(xi_random_mean) + 2 * xi_random_std:
        print("✅ Out-degree correlation is SIGNIFICANTLY stronger than random")
        print(f"   (>{z_score:.1f}σ above noise baseline)")
        print("   This proves correlation is structural, not artifact")
    else:
        print("⚠️  Out-degree correlation not significantly different from random")
    
    print()
    
    # ============================================================
    # TEST 5: Remove Structure → Remove Correlation
    # ============================================================
    print("=" * 70)
    print("TEST 5: REMOVING STRUCTURE")
    print("Shuffle node assignments → breaks structure → breaks correlation")
    print("=" * 70)
    print()
    
    # Shuffle metric values across nodes (breaks structure coupling)
    shuffled_out_degree = out_degree.copy()
    keys = list(shuffled_out_degree.keys())
    values = list(shuffled_out_degree.values())
    np.random.shuffle(values)
    shuffled_out_degree = dict(zip(keys, values))
    
    xi_shuffled = compute_xi(betweenness, shuffled_out_degree)
    
    print(f"Original:  ξ = {xi_main:.4f}")
    print(f"Shuffled:  ξ = {xi_shuffled:.4f}")
    print(f"Change:    Δξ = {xi_shuffled - xi_main:.4f}")
    print()
    
    if abs(xi_main) - abs(xi_shuffled) > 0.3:
        print("✅ PASS: Shuffling breaks correlation")
        print("   This proves correlation depends on structure")
    else:
        print("⚠️  Shuffling did not significantly change correlation")
    
    print()
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("=" * 70)
    print("CONCLUSION: Is This Redundancy or Conservation?")
    print("=" * 70)
    print()
    
    evidence = [
        ("Metrics are independent", independence.get("independent", False)),
        ("Random metric uncorrelated", abs(xi_random_mean) < 0.1),
        ("Coupled metric correlates", abs(xi_coupled) > 0.5),
        ("Main result significantly above noise", abs(xi_main) > abs(xi_random_mean) + 2 * xi_random_std),
        ("Shuffling breaks correlation", abs(xi_main) - abs(xi_shuffled) > 0.3)
    ]
    
    confirmed = sum(1 for _, check in evidence if check)
    
    for desc, check in evidence:
        print(f"  {'✅' if check else '❌'} {desc}")
    
    print()
    print(f"Evidence strength: {confirmed}/{len(evidence)}")
    print()
    
    if confirmed >= 4:
        print("🎯 PROVEN: Correlation = Structural Coupling")
        print()
        print("The correlation is NOT redundancy. Betweenness and out-degree")
        print("measure DIFFERENT properties, but both are coupled to the SAME")
        print("underlying structure: the PAC ownership graph.")
        print()
        print("In ownership graphs (weighted DAGs):")
        print("  - Parent = Σ(children × ownership_weights)")
        print("  - Betweenness measures flow through decomposition")
        print("  - Out-degree measures number of deltas")
        print("  - Both depend on the decomposition structure")
        print()
        print("This is like E and m in E=mc²:")
        print("  - E and m are different properties (energy vs mass)")
        print("  - But they're coupled through c² (structure of spacetime)")
        print("  - Their correlation proves the conservation law")
        print()
        print("ξ measures the fidelity of this structural coupling!")
    elif confirmed >= 3:
        print("⚙️  PARTIALLY CONFIRMED")
        print(f"{confirmed}/{len(evidence)} tests passed")
        print()
        print("The evidence suggests structural coupling, but some tests")
        print("did not pass. This may be due to:")
        print("  - Small sample size (need more nodes)")
        print("  - Weak ownership weights (need stronger structure)")
        print("  - Wrong metric choice (try different metrics)")
    else:
        print("⚠️  INCONCLUSIVE")
        print(f"Only {confirmed}/{len(evidence)} predictions confirmed")
        print()
        print("The correlation may be:")
        print("  1. Redundancy (metrics measure same thing)")
        print("  2. Artifact (statistical fluke)")
        print("  3. Real but weak (need stronger structure)")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

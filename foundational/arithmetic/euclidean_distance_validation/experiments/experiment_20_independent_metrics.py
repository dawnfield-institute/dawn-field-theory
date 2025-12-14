"""
Experiment 20: Finding Truly Independent E and m Metrics

The problem with exp_18: Flow and Connectivity are r=0.92 correlated!
They're essentially the same metric, giving fake R²=1.0.

This experiment:
1. Defines multiple conceptually distinct E and m candidates
2. Tests independence between ALL pairs
3. Only reports results where |r| < 0.5 (truly independent)
4. Finds legitimate E=mc² relationships in byref space

Goal: Find R² > 0.80 with INDEPENDENT metrics (no cheating!)
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from scipy.stats import linregress, pearsonr
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
    text: str = ""
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class PACHierarchy:
    """PAC hierarchy with ownership graph structure."""
    
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
    
    def get_descendants(self, node_name: str) -> Set[str]:
        descendants = set()
        to_visit = [node_name]
        while to_visit:
            current = to_visit.pop()
            for child in self.nodes[current].children:
                if child not in descendants:
                    descendants.add(child)
                    to_visit.append(child)
        return descendants


def build_real_embedding_hierarchy(model: SentenceTransformer) -> PACHierarchy:
    """Build hierarchy with real embeddings."""
    hierarchy = PACHierarchy()
    
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
    
    # Generate embeddings
    text_list = list(texts.values())
    embeddings = model.encode(text_list, normalize_embeddings=True)
    
    # Build hierarchy
    hierarchy.add_node(PACNode("root", depth=0, text=texts["root"], 
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
    
    return hierarchy


# ========== ENERGY METRICS (E_byref) ==========

def compute_E1_betweenness(hierarchy: PACHierarchy) -> Dict[str, float]:
    """E: Betweenness centrality - information flow THROUGH node."""
    G = hierarchy.get_ownership_graph()
    return nx.betweenness_centrality(G, weight='weight')


def compute_E2_closeness(hierarchy: PACHierarchy) -> Dict[str, float]:
    """E: Closeness centrality - average distance to all others."""
    G = hierarchy.get_ownership_graph()
    # Use reverse graph for parent-to-child distances
    closeness = nx.closeness_centrality(G.reverse(), distance='weight')
    return closeness


def compute_E3_eigenvector(hierarchy: PACHierarchy) -> Dict[str, float]:
    """E: Eigenvector centrality - importance via connections."""
    G = hierarchy.get_ownership_graph()
    try:
        return nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except:
        return nx.pagerank(G, weight='weight')


def compute_E4_perturbation_capacity(hierarchy: PACHierarchy) -> Dict[str, float]:
    """E: Capacity to absorb perturbations (from exp_17 concept)."""
    G = hierarchy.get_ownership_graph()
    capacity = {}
    
    for node_name in hierarchy.nodes:
        # Energy = ability to distribute changes
        descendants = len(nx.descendants(G, node_name))
        ancestors = len(nx.ancestors(G, node_name))
        
        # High capacity = many connections both ways
        capacity[node_name] = float(descendants * ancestors + 1)
    
    return capacity


def compute_E5_local_clustering(hierarchy: PACHierarchy) -> Dict[str, float]:
    """E: Local clustering coefficient - local connectivity."""
    G = hierarchy.get_ownership_graph().to_undirected()
    clustering = nx.clustering(G)
    return clustering


def compute_E6_embedding_magnitude(hierarchy: PACHierarchy) -> Dict[str, float]:
    """E: Magnitude in embedding space (cross-space metric)."""
    magnitude = {}
    for node_name, node in hierarchy.nodes.items():
        if node.embedding is not None:
            magnitude[node_name] = np.linalg.norm(node.embedding)
    return magnitude


# ========== MASS METRICS (m_byref) ==========

def compute_M1_subtree_size(hierarchy: PACHierarchy) -> Dict[str, float]:
    """m: Count of descendants (subtree size)."""
    mass = {}
    for node_name in hierarchy.nodes:
        descendants = hierarchy.get_descendants(node_name)
        mass[node_name] = float(len(descendants) + 1)  # +1 for self
    return mass


def compute_M2_depth(hierarchy: PACHierarchy) -> Dict[str, float]:
    """m: Depth in hierarchy (distance from root)."""
    mass = {}
    for node_name, node in hierarchy.nodes.items():
        mass[node_name] = float(node.depth + 1)
    return mass


def compute_M3_ownership_accumulation(hierarchy: PACHierarchy) -> Dict[str, float]:
    """m: Accumulated ownership weights from root."""
    mass = {}
    
    for node_name in hierarchy.nodes:
        current = hierarchy.nodes[node_name]
        weight_product = 1.0
        
        # Multiply ownership weights up to root
        while current.parent:
            edge = (current.parent, current.name)
            if edge in hierarchy.ownership_weights:
                weight_product *= hierarchy.ownership_weights[edge]
            current = hierarchy.nodes[current.parent]
        
        mass[node_name] = weight_product
    
    return mass


def compute_M4_structural_mass(hierarchy: PACHierarchy) -> Dict[str, float]:
    """m: Ownership-weighted sum of descendants."""
    mass = {}
    
    for node_name in hierarchy.nodes:
        total = 1.0  # Self
        descendants = hierarchy.get_descendants(node_name)
        
        for desc in descendants:
            # Find ownership path weight
            current = hierarchy.nodes[desc]
            path_weight = 1.0
            
            while current.parent and current.parent != node_name:
                edge = (current.parent, current.name)
                if edge in hierarchy.ownership_weights:
                    path_weight *= hierarchy.ownership_weights[edge]
                current = hierarchy.nodes[current.parent]
            
            if current.parent == node_name:
                edge = (node_name, current.name)
                if edge in hierarchy.ownership_weights:
                    path_weight *= hierarchy.ownership_weights[edge]
            
            total += path_weight
        
        mass[node_name] = total
    
    return mass


def compute_M5_out_degree(hierarchy: PACHierarchy) -> Dict[str, float]:
    """m: Number of children (out-degree)."""
    mass = {}
    for node_name, node in hierarchy.nodes.items():
        mass[node_name] = float(len(node.children) + 1)  # +1 to avoid zeros
    return mass


def compute_M6_eccentricity(hierarchy: PACHierarchy) -> Dict[str, float]:
    """m: Maximum distance to any other node."""
    G = hierarchy.get_ownership_graph()
    eccentricity = {}
    
    for node_name in hierarchy.nodes:
        try:
            # Max shortest path to any reachable node
            lengths = nx.single_source_shortest_path_length(G, node_name)
            if lengths:
                eccentricity[node_name] = float(max(lengths.values()) + 1)
            else:
                eccentricity[node_name] = 1.0
        except:
            eccentricity[node_name] = 1.0
    
    return eccentricity


def test_independence_and_emc2(hierarchy: PACHierarchy,
                               e_name: str, e_values: Dict[str, float],
                               m_name: str, m_values: Dict[str, float]) -> Dict:
    """Test both independence and E=mc² relationship."""
    
    # Get common nodes
    common = list(set(e_values.keys()) & set(m_values.keys()))
    E = np.array([e_values[node] for node in common])
    m = np.array([m_values[node] for node in common])
    
    # Filter valid
    valid = (E > 1e-10) & (m > 1e-10) & np.isfinite(E) & np.isfinite(m)
    E = E[valid]
    m = m[valid]
    
    if len(E) < 3:
        return None
    
    # Test independence (correlation between raw metrics)
    if np.std(E) < 1e-10 or np.std(m) < 1e-10:
        return None
    
    independence_r, independence_p = pearsonr(E, m)
    
    # Test E=mc² relationship
    try:
        slope, intercept, r_value, p_value, std_err = linregress(m, E)
        r2 = r_value ** 2
    except:
        return None
    
    return {
        "e_name": e_name,
        "m_name": m_name,
        "independence_r": independence_r,
        "independence_p": independence_p,
        "independent": abs(independence_r) < 0.5,  # Stricter threshold
        "r2": r2,
        "slope": slope,
        "p_value": p_value,
        "n": len(E)
    }


def main():
    """Find independent E and m metrics."""
    
    print("=" * 70)
    print("EXPERIMENT 20: Finding Truly Independent E and m Metrics")
    print("No cheating: |correlation| < 0.5 between metrics")
    print("=" * 70)
    print()
    
    # Load model and build hierarchy
    print("Loading model and building hierarchy...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    hierarchy = build_real_embedding_hierarchy(model)
    print(f"✓ Built hierarchy: {len(hierarchy.nodes)} nodes\n")
    
    # Compute all energy metrics
    print("Computing Energy metrics...")
    energy_metrics = {
        "E1_Betweenness": compute_E1_betweenness(hierarchy),
        "E2_Closeness": compute_E2_closeness(hierarchy),
        "E3_Eigenvector": compute_E3_eigenvector(hierarchy),
        "E4_PerturbCapacity": compute_E4_perturbation_capacity(hierarchy),
        "E5_Clustering": compute_E5_local_clustering(hierarchy),
        "E6_EmbedMagnitude": compute_E6_embedding_magnitude(hierarchy),
    }
    print(f"✓ Computed {len(energy_metrics)} energy metrics\n")
    
    # Compute all mass metrics
    print("Computing Mass metrics...")
    mass_metrics = {
        "M1_SubtreeSize": compute_M1_subtree_size(hierarchy),
        "M2_Depth": compute_M2_depth(hierarchy),
        "M3_OwnershipAcc": compute_M3_ownership_accumulation(hierarchy),
        "M4_StructuralMass": compute_M4_structural_mass(hierarchy),
        "M5_OutDegree": compute_M5_out_degree(hierarchy),
        "M6_Eccentricity": compute_M6_eccentricity(hierarchy),
    }
    print(f"✓ Computed {len(mass_metrics)} mass metrics\n")
    
    # Test all combinations
    print("=" * 70)
    print("TESTING ALL COMBINATIONS")
    print("=" * 70)
    print()
    
    all_results = []
    independent_results = []
    
    for e_name, e_values in energy_metrics.items():
        for m_name, m_values in mass_metrics.items():
            result = test_independence_and_emc2(hierarchy, e_name, e_values, 
                                               m_name, m_values)
            if result:
                all_results.append(result)
                if result['independent']:
                    independent_results.append(result)
    
    # Report ALL results first
    print(f"Total valid combinations: {len(all_results)}")
    print(f"Independent combinations (|r| < 0.5): {len(independent_results)}\n")
    
    # Show top correlated (the cheaters)
    print("MOST CORRELATED (Redundant Metrics - NOT VALID):")
    correlated = sorted(all_results, key=lambda x: abs(x['independence_r']), 
                       reverse=True)[:5]
    for i, r in enumerate(correlated, 1):
        print(f"{i}. {r['e_name']} × {r['m_name']}")
        print(f"   Correlation: r={r['independence_r']:.3f} (REDUNDANT)")
        print(f"   R²: {r['r2']:.3f} (FAKE)")
        print()
    
    # Show truly independent results
    print("=" * 70)
    print("TRULY INDEPENDENT RESULTS (|r| < 0.5)")
    print("=" * 70)
    print()
    
    if independent_results:
        # Sort by R²
        independent_results.sort(key=lambda x: x['r2'], reverse=True)
        
        for i, r in enumerate(independent_results, 1):
            print(f"{i}. {r['e_name']} × {r['m_name']}")
            print(f"   Independence: r={r['independence_r']:.3f} ✓")
            print(f"   R²: {r['r2']:.4f}")
            print(f"   Slope (c²): {r['slope']:.4f}")
            print(f"   p-value: {r['p_value']:.2e}")
            print(f"   n: {r['n']}")
            print()
        
        # Best independent result
        best = independent_results[0]
        print("=" * 70)
        print("BEST INDEPENDENT RESULT")
        print("=" * 70)
        print(f"Energy: {best['e_name']}")
        print(f"Mass: {best['m_name']}")
        print(f"Independence: r={best['independence_r']:.3f} (truly different metrics)")
        print(f"R²: {best['r2']:.4f}")
        print(f"Status: {'✅ STRONG' if best['r2'] > 0.80 else '⚠️  MODERATE' if best['r2'] > 0.60 else '❌ WEAK'}")
        print()
        
        # Final verdict
        if best['r2'] > 0.80:
            print("🎯 SUCCESS: Found legitimate E=mc² in byref space!")
            print(f"   - R²={best['r2']:.3f} with independent metrics")
            print("   - Not an artifact of redundant measurements")
            print("   - Framework validated")
        elif best['r2'] > 0.60:
            print("📊 MODERATE: E=mc² shows correlation but not undeniable")
            print(f"   - R²={best['r2']:.3f} is decent but not R²>0.95")
            print("   - Metrics are truly independent")
            print("   - May need better metric definitions")
        else:
            print("❌ WEAK: No strong E=mc² with independent metrics")
            print(f"   - Best R²={best['r2']:.3f} is not convincing")
            print("   - May need different approach to byref measurements")
    else:
        print("❌ NO INDEPENDENT PAIRS FOUND")
        print("All metric pairs are too correlated!")
        print("Need to define more distinct E and m measures.")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

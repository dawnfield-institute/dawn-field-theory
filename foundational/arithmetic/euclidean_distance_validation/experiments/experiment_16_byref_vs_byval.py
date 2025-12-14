"""
Experiment 16: Graph-Aware Context-Dependence (byref not byval)

BREAKTHROUGH INSIGHT: Embeddings are byval (independent vectors), 
but PAC tree is byref (connected through ownership).

Context-dependence lives in the GRAPH STRUCTURE, not just embedding positions!

KEY CONNECTIONS:
1. Depth-2 recursion (macro_emergence_dynamics): emergence at grandparent level
2. Quantum effects (quantum_validation + PACEngine): non-local propagation
3. byval vs byref: Euclidean = independent, Graph = connected

TEST: Measure distances through ownership graph, not just embedding space.
- Graph geodesics with ownership weights
- Depth-2 effects (grandparent relationships)
- Non-local perturbation propagation

EXPECTED: 5-7× context variation (not 1.96× like exp_15)
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.pac_hierarchy import PACNode, PACHierarchy
from core.embedding_generator import EmbeddingGenerator
import matplotlib.pyplot as plt
import networkx as nx
import json


def create_multi_domain_hierarchy(embedding_type: str = 'real') -> PACHierarchy:
    """Create multi-domain hierarchy with clear ownership structure."""
    
    root = PACNode(id=f"{embedding_type}_root", value=1000.0)
    root.metadata['text'] = "Knowledge Base"
    root.metadata['domain'] = "root"
    hierarchy = PACHierarchy(root)
    
    # Domain 1: Code
    code_root = PACNode(id=f"{embedding_type}_code_root", value=400.0)
    code_root.metadata['text'] = "Software Engineering"
    code_root.metadata['domain'] = "code"
    hierarchy.add_node(code_root, parent_id=root.id, ownership_weight=0.4)
    
    code_l2 = ["Backend Systems", "Frontend Development", "Infrastructure"]
    for i, concept in enumerate(code_l2):
        node = PACNode(id=f"{embedding_type}_code_L2_{i}", value=400.0/3)
        node.metadata['text'] = concept
        node.metadata['domain'] = "code"
        hierarchy.add_node(node, parent_id=code_root.id, ownership_weight=1.0/3)
        
        # Level 3
        code_l3 = [f"{concept} Detail {j}" for j in range(3)]
        for j, detail in enumerate(code_l3):
            node_l3 = PACNode(id=f"{embedding_type}_code_L3_{i}_{j}", value=400.0/9)
            node_l3.metadata['text'] = detail
            node_l3.metadata['domain'] = "code"
            hierarchy.add_node(node_l3, parent_id=node.id, ownership_weight=1.0/3)
    
    # Domain 2: Biology
    bio_root = PACNode(id=f"{embedding_type}_bio_root", value=350.0)
    bio_root.metadata['text'] = "Biology"
    bio_root.metadata['domain'] = "biology"
    hierarchy.add_node(bio_root, parent_id=root.id, ownership_weight=0.35)
    
    bio_l2 = ["Molecular Biology", "Ecology", "Evolution"]
    for i, concept in enumerate(bio_l2):
        node = PACNode(id=f"{embedding_type}_bio_L2_{i}", value=350.0/3)
        node.metadata['text'] = concept
        node.metadata['domain'] = "biology"
        hierarchy.add_node(node, parent_id=bio_root.id, ownership_weight=1.0/3)
        
        # Level 3
        bio_l3 = [f"{concept} Process {j}" for j in range(3)]
        for j, detail in enumerate(bio_l3):
            node_l3 = PACNode(id=f"{embedding_type}_bio_L3_{i}_{j}", value=350.0/9)
            node_l3.metadata['text'] = detail
            node_l3.metadata['domain'] = "biology"
            hierarchy.add_node(node_l3, parent_id=node.id, ownership_weight=1.0/3)
    
    # Domain 3: Physics
    physics_root = PACNode(id=f"{embedding_type}_phys_root", value=250.0)
    physics_root.metadata['text'] = "Physics"
    physics_root.metadata['domain'] = "physics"
    hierarchy.add_node(physics_root, parent_id=root.id, ownership_weight=0.25)
    
    phys_l2 = ["Classical Mechanics", "Quantum Theory", "Relativity"]
    for i, concept in enumerate(phys_l2):
        node = PACNode(id=f"{embedding_type}_phys_L2_{i}", value=250.0/3)
        node.metadata['text'] = concept
        node.metadata['domain'] = "physics"
        hierarchy.add_node(node, parent_id=physics_root.id, ownership_weight=1.0/3)
        
        # Level 3
        phys_l3 = [f"{concept} Principle {j}" for j in range(3)]
        for j, detail in enumerate(phys_l3):
            node_l3 = PACNode(id=f"{embedding_type}_phys_L3_{i}_{j}", value=250.0/9)
            node_l3.metadata['text'] = detail
            node_l3.metadata['domain'] = "physics"
            hierarchy.add_node(node_l3, parent_id=node.id, ownership_weight=1.0/3)
    
    print(f"Created hierarchy: {len(hierarchy.nodes)} nodes")
    return hierarchy


class GraphAwareContextTest:
    """Measure context-dependence through graph structure (byref), not just embeddings (byval)."""
    
    def __init__(self, hierarchy: PACHierarchy):
        self.hierarchy = hierarchy
        self.nodes_by_domain = {}
        self.ownership_graph = nx.DiGraph()
        self.semantic_graph = nx.Graph()
        
    def build_graphs(self):
        """Build both ownership (byref) and semantic (byval) graphs."""
        print("\nBuilding graphs...")
        
        # Ownership graph (directed, weighted by ownership)
        for node_id, node in self.hierarchy.nodes.items():
            self.ownership_graph.add_node(node_id, node=node)
            
            if node.parent is not None:
                # Edge from parent to child with ownership weight
                parent_weight = 1.0
                for child in node.parent.children:
                    if child.id == node_id:
                        # Find ownership weight from parent's perspective
                        parent_weight = 1.0 / len(node.parent.children)  # Equal for simplicity
                        break
                
                self.ownership_graph.add_edge(node.parent.id, node_id, weight=parent_weight)
                self.ownership_graph.add_edge(node_id, node.parent.id, weight=1.0)  # Bidirectional
        
        # Semantic graph (undirected, weighted by embedding similarity)
        nodes_with_emb = [n for n in self.hierarchy.nodes.values() if n.embedding is not None]
        
        for i, node1 in enumerate(nodes_with_emb):
            for node2 in nodes_with_emb[i+1:]:
                # Semantic similarity (inverse of distance)
                dist = float(np.linalg.norm(node1.embedding - node2.embedding))
                similarity = 1.0 / (dist + 0.1)  # Avoid division by zero
                
                # Only connect if reasonably similar (threshold)
                if similarity > 0.5:  # Adjust threshold as needed
                    self.semantic_graph.add_edge(node1.id, node2.id, weight=similarity)
        
        print(f"  Ownership graph: {self.ownership_graph.number_of_nodes()} nodes, {self.ownership_graph.number_of_edges()} edges")
        print(f"  Semantic graph: {self.semantic_graph.number_of_nodes()} nodes, {self.semantic_graph.number_of_edges()} edges")
    
    def organize_by_domain(self):
        """Group nodes by domain."""
        for node in self.hierarchy.nodes.values():
            domain = node.metadata.get('domain', 'unknown')
            if domain not in self.nodes_by_domain:
                self.nodes_by_domain[domain] = []
            self.nodes_by_domain[domain].append(node)
    
    def graph_distance_from_context(self, node1: PACNode, node2: PACNode, 
                                     observer_domain: str, max_depth: int = 5) -> float:
        """
        Compute distance through ownership graph from observer's perspective.
        
        Context-dependence emerges from:
        1. Which paths are accessible (through observer's domain)
        2. Ownership weights along paths
        3. Depth-2 effects (grandparent relationships)
        """
        # Find "anchor" nodes in observer's domain
        domain_nodes = [n.id for n in self.nodes_by_domain.get(observer_domain, [])]
        
        if not domain_nodes or node1.id not in self.ownership_graph or node2.id not in self.ownership_graph:
            return float('inf')
        
        # Find shortest path that goes through observer's domain
        min_distance = float('inf')
        
        # Try paths through each domain node as intermediate
        for anchor_id in domain_nodes[:5]:  # Sample to avoid combinatorial explosion
            try:
                # Path: node1 → anchor → node2
                if nx.has_path(self.ownership_graph, node1.id, anchor_id):
                    path1 = nx.shortest_path(self.ownership_graph, node1.id, anchor_id, 
                                            weight='weight')
                    dist1 = self._path_length(path1)
                else:
                    dist1 = float('inf')
                
                if nx.has_path(self.ownership_graph, anchor_id, node2.id):
                    path2 = nx.shortest_path(self.ownership_graph, anchor_id, node2.id,
                                            weight='weight')
                    dist2 = self._path_length(path2)
                else:
                    dist2 = float('inf')
                
                total_dist = dist1 + dist2
                min_distance = min(min_distance, total_dist)
                
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        
        # Also try direct path (no domain anchor)
        try:
            if nx.has_path(self.ownership_graph, node1.id, node2.id):
                direct_path = nx.shortest_path(self.ownership_graph, node1.id, node2.id,
                                              weight='weight')
                direct_dist = self._path_length(direct_path)
                min_distance = min(min_distance, direct_dist)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        
        return min_distance if min_distance != float('inf') else 100.0  # Large default
    
    def _path_length(self, path: List[str]) -> float:
        """Compute weighted path length."""
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            if self.ownership_graph.has_edge(path[i], path[i+1]):
                # Inverse of weight = distance (higher ownership = shorter distance)
                weight = self.ownership_graph[path[i]][path[i+1]]['weight']
                total += 1.0 / (weight + 0.01)
            else:
                total += 10.0  # Large penalty for missing edge
        
        return total
    
    def euclidean_distance(self, node1: PACNode, node2: PACNode) -> float:
        """Standard Euclidean distance (byval)."""
        if node1.embedding is None or node2.embedding is None:
            return float('inf')
        return float(np.linalg.norm(node1.embedding - node2.embedding))
    
    def test_byref_vs_byval(self) -> Dict:
        """
        Compare byval (Euclidean) vs byref (Graph) context-dependence.
        
        HYPOTHESIS: Graph distances show STRONGER context-dependence than Euclidean.
        """
        print(f"\n{'='*60}")
        print("BYREF VS BYVAL CONTEXT TEST")
        print(f"{'='*60}\n")
        
        results = {
            'euclidean': {},  # byval measurements
            'graph': {},      # byref measurements  
            'comparison': {}
        }
        
        domains = [d for d in self.nodes_by_domain.keys() if d != 'root']
        
        # Test intra-domain pairs
        print("Testing intra-domain pairs...")
        for domain in domains:
            nodes = self.nodes_by_domain[domain]
            if len(nodes) < 5:
                continue
            
            euclidean_variations = []
            graph_variations = []
            
            # Sample pairs
            for i in range(min(5, len(nodes))):
                for j in range(i+1, min(i+4, len(nodes))):
                    node1, node2 = nodes[i], nodes[j]
                    
                    # Euclidean (byval) from each context
                    d_euclidean = self.euclidean_distance(node1, node2)
                    
                    # Graph (byref) from each context
                    d_graphs = {}
                    for obs_domain in domains:
                        d_graphs[obs_domain] = self.graph_distance_from_context(
                            node1, node2, obs_domain
                        )
                    
                    # Variation
                    if d_graphs:
                        valid_dists = [d for d in d_graphs.values() if d < 100.0]
                        if len(valid_dists) > 1:
                            graph_var = max(valid_dists) / min(valid_dists)
                            graph_variations.append(graph_var)
                    
                    # Euclidean has no context variation (same value always)
                    euclidean_variations.append(1.0)  # No variation
            
            if graph_variations:
                results['euclidean'][domain] = {
                    'mean_variation': float(np.mean(euclidean_variations)),
                    'max_variation': float(np.max(euclidean_variations))
                }
                results['graph'][domain] = {
                    'mean_variation': float(np.mean(graph_variations)),
                    'max_variation': float(np.max(graph_variations)),
                    'n_pairs': len(graph_variations)
                }
                
                print(f"  {domain}:")
                print(f"    Euclidean variation: {np.mean(euclidean_variations):.2f}× (byval - no context)")
                print(f"    Graph variation: {np.mean(graph_variations):.2f}× (byref - with context)")
                print(f"    Amplification: {np.mean(graph_variations)/np.mean(euclidean_variations):.2f}×")
        
        # Cross-domain pairs
        print("\nTesting cross-domain pairs...")
        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                nodes1 = self.nodes_by_domain[domain1]
                nodes2 = self.nodes_by_domain[domain2]
                
                euclidean_variations = []
                graph_variations = []
                
                for n1 in nodes1[:3]:
                    for n2 in nodes2[:3]:
                        d_euclidean = self.euclidean_distance(n1, n2)
                        
                        d_graphs = {}
                        for obs_domain in domains:
                            d_graphs[obs_domain] = self.graph_distance_from_context(
                                n1, n2, obs_domain
                            )
                        
                        if d_graphs:
                            valid_dists = [d for d in d_graphs.values() if d < 100.0]
                            if len(valid_dists) > 1:
                                graph_var = max(valid_dists) / min(valid_dists)
                                graph_variations.append(graph_var)
                        
                        euclidean_variations.append(1.0)
                
                pair_key = f"{domain1}_vs_{domain2}"
                if graph_variations:
                    results['euclidean'][pair_key] = {
                        'mean_variation': 1.0,
                        'max_variation': 1.0
                    }
                    results['graph'][pair_key] = {
                        'mean_variation': float(np.mean(graph_variations)),
                        'max_variation': float(np.max(graph_variations)),
                        'n_pairs': len(graph_variations)
                    }
                    
                    print(f"  {pair_key}:")
                    print(f"    Euclidean: 1.00× (no context)")
                    print(f"    Graph: {np.mean(graph_variations):.2f}×")
        
        # Overall comparison
        print(f"\n{'='*60}")
        print("OVERALL COMPARISON: byref vs byval")
        print(f"{'='*60}\n")
        
        all_graph_vars = [res['mean_variation'] for res in results['graph'].values()]
        all_graph_maxs = [res['max_variation'] for res in results['graph'].values()]
        
        if all_graph_vars:
            print(f"Graph (byref) context-dependence:")
            print(f"  Mean variation: {np.mean(all_graph_vars):.2f}×")
            print(f"  Max variation: {max(all_graph_maxs):.2f}×")
            print(f"\nEuclidean (byval) context-dependence:")
            print(f"  Variation: 1.00× (none - independent vectors)")
            print(f"\nAmplification factor: {np.mean(all_graph_vars):.2f}×")
            print(f"  (How much context matters in graph vs embedding space)")
            
            results['comparison'] = {
                'graph_mean': float(np.mean(all_graph_vars)),
                'graph_max': float(max(all_graph_maxs)),
                'amplification': float(np.mean(all_graph_vars))
            }
            
            if max(all_graph_maxs) > 7.0:
                print(f"\n✅✅ STRONG byref context-dependence (>7×)")
                print("Graph structure creates STRONG context effects!")
            elif max(all_graph_maxs) > 5.0:
                print(f"\n✅ MODERATE byref context-dependence (>5×)")
            elif max(all_graph_maxs) > 3.0:
                print(f"\n✓ WEAK byref context-dependence (>3×)")
            else:
                print(f"\n⚠️  MINIMAL byref context-dependence (<3×)")
        
        return results
    
    def visualize_results(self, results: Dict):
        """Visualize byref vs byval comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Direct comparison
        ax = axes[0, 0]
        if results['graph']:
            categories = list(results['graph'].keys())
            graph_means = [results['graph'][c]['mean_variation'] for c in categories]
            euclidean_means = [results['euclidean'][c]['mean_variation'] for c in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax.bar(x - width/2, euclidean_means, width, label='Euclidean (byval)', 
                  alpha=0.7, color='gray')
            ax.bar(x + width/2, graph_means, width, label='Graph (byref)',
                  alpha=0.7, color='green')
            
            ax.set_xticks(x)
            ax.set_xticklabels([c.replace('_', '\n') for c in categories], fontsize=7)
            ax.set_ylabel('Context Variation')
            ax.set_title('byref (Graph) vs byval (Euclidean)')
            ax.legend()
            ax.grid(alpha=0.3, axis='y')
            ax.axhline(3.0, color='orange', linestyle='--', alpha=0.5)
            ax.axhline(7.42, color='green', linestyle=':', alpha=0.7, linewidth=2)
        
        # 2. Amplification factors
        ax = axes[0, 1]
        if results['comparison']:
            metrics = ['Mean\nVariation', 'Max\nVariation']
            values = [results['comparison']['graph_mean'], results['comparison']['graph_max']]
            colors = ['blue', 'red']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.7)
            ax.set_ylabel('Variation Factor')
            ax.set_title('Graph Context-Dependence (byref)')
            ax.axhline(7.42, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Target (7.42×)')
            ax.axhline(5.0, color='orange', linestyle='--', alpha=0.5)
            ax.legend()
            ax.grid(alpha=0.3, axis='y')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}×', ha='center', va='bottom', fontweight='bold')
        
        # 3. Distribution
        ax = axes[1, 0]
        all_graph = [res['mean_variation'] for res in results['graph'].values()]
        if all_graph:
            ax.hist(all_graph, bins=15, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(7.42, color='darkgreen', linestyle=':', linewidth=2, label='Target (7.42×)')
            ax.axvline(5.0, color='orange', linestyle='--', label='Strong (5×)')
            ax.set_xlabel('Variation Factor')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Graph Variations')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 4. Max variations by type
        ax = axes[1, 1]
        if results['graph']:
            graph_maxs = [res['max_variation'] for res in results['graph'].values()]
            labels = list(results['graph'].keys())
            
            ax.barh(range(len(labels)), graph_maxs, alpha=0.7, color='purple')
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels([l.replace('_', ' ') for l in labels], fontsize=8)
            ax.set_xlabel('Max Variation')
            ax.set_title('Maximum Context-Dependence (byref)')
            ax.axvline(7.42, color='green', linestyle=':', alpha=0.7, linewidth=2)
            ax.axvline(5.0, color='orange', linestyle='--', alpha=0.5)
            ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        import os
        os.makedirs('euclidean_distance_validation/results', exist_ok=True)
        plt.savefig('euclidean_distance_validation/results/experiment_16_byref_vs_byval.png',
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved to results/experiment_16_byref_vs_byval.png")


def main():
    """Run Experiment 16: Graph-Aware Context (byref vs byval)."""
    
    print("="*60)
    print("EXPERIMENT 16: BYREF VS BYVAL CONTEXT-DEPENDENCE")
    print("="*60)
    print("\nKey insight: PAC tree is byref (connected), embeddings are byval (independent)")
    print("Context-dependence lives in GRAPH STRUCTURE!")
    print()
    
    # Create hierarchy
    hierarchy = create_multi_domain_hierarchy('real')
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    emb_gen = EmbeddingGenerator(model='sentence-transformers',
                                 model_name='all-MiniLM-L6-v2')
    emb_gen.embed_hierarchy(hierarchy)
    
    # Run test
    tester = GraphAwareContextTest(hierarchy)
    tester.organize_by_domain()
    tester.build_graphs()
    
    results = tester.test_byref_vs_byval()
    
    # Visualize
    tester.visualize_results(results)
    
    # Save
    import os
    os.makedirs('euclidean_distance_validation/results', exist_ok=True)
    with open('euclidean_distance_validation/results/experiment_16_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/experiment_16_results.json")


if __name__ == "__main__":
    main()

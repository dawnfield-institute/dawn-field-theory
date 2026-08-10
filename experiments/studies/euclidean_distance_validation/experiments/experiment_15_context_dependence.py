"""
Experiment 15: Direct Context-Dependence Test

SIMPLE QUESTION: Does the same distance measurement vary by context?

TEST: Measure node-pair distances from different domain perspectives.
- Euclidean distance (objective, reference)
- Context-weighted distance (subjective, observer-dependent)

EXPECTED: 3-7× variation (like the 7.42× we saw in exp_05)

This is the SIMPLEST relativity demonstration:
- Same measurement (distance)
- Different observers (domain perspectives)
- Different values (context-dependence)
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.pac_hierarchy import PACNode, PACHierarchy
from core.embedding_generator import EmbeddingGenerator
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
import json


def create_multi_domain_hierarchy(embedding_type: str = 'real') -> PACHierarchy:
    """Create a rich multi-domain hierarchy."""
    
    root = PACNode(id=f"{embedding_type}_root", value=1000.0)
    root.metadata['text'] = "Knowledge Base"
    root.metadata['domain'] = "root"
    hierarchy = PACHierarchy(root)
    
    # Domain 1: Software Engineering
    code_root = PACNode(id=f"{embedding_type}_code", value=400.0)
    code_root.metadata['text'] = "Software Engineering"
    code_root.metadata['domain'] = "code"
    hierarchy.add_node(code_root, parent_id=root.id, ownership_weight=0.4)
    
    code_concepts = [
        "REST APIs", "Databases", "React Components", "State Management",
        "Docker Containers", "Kubernetes", "CI/CD Pipelines", "Load Balancing",
        "Async Programming", "Dependency Injection", "Microservices Architecture",
        "GraphQL APIs", "WebSocket Connections", "Authentication Tokens"
    ]
    
    for i, concept in enumerate(code_concepts):
        node = PACNode(id=f"{embedding_type}_code_{i}", value=400.0/len(code_concepts))
        node.metadata['text'] = concept
        node.metadata['domain'] = "code"
        hierarchy.add_node(node, parent_id=code_root.id, ownership_weight=1.0/len(code_concepts))
    
    # Domain 2: Biology
    bio_root = PACNode(id=f"{embedding_type}_bio", value=350.0)
    bio_root.metadata['text'] = "Biology"
    bio_root.metadata['domain'] = "biology"
    hierarchy.add_node(bio_root, parent_id=root.id, ownership_weight=0.35)
    
    bio_concepts = [
        "DNA Replication", "Protein Synthesis", "Gene Expression", "Enzyme Catalysis",
        "Cell Signaling", "Food Webs", "Population Dynamics", "Ecosystem Services",
        "Natural Selection", "Genetic Drift", "Speciation", "Phylogenetics",
        "Mitochondria", "Photosynthesis"
    ]
    
    for i, concept in enumerate(bio_concepts):
        node = PACNode(id=f"{embedding_type}_bio_{i}", value=350.0/len(bio_concepts))
        node.metadata['text'] = concept
        node.metadata['domain'] = "biology"
        hierarchy.add_node(node, parent_id=bio_root.id, ownership_weight=1.0/len(bio_concepts))
    
    # Domain 3: Physics
    physics_root = PACNode(id=f"{embedding_type}_phys", value=250.0)
    physics_root.metadata['text'] = "Physics"
    physics_root.metadata['domain'] = "physics"
    hierarchy.add_node(physics_root, parent_id=root.id, ownership_weight=0.25)
    
    physics_concepts = [
        "Newton's Laws", "Thermodynamics", "Wave Motion", "Electromagnetism",
        "Wave-Particle Duality", "Uncertainty Principle", "Quantum Entanglement",
        "Time Dilation", "Length Contraction", "Mass-Energy Equivalence",
        "Spacetime Curvature", "Black Holes", "Hawking Radiation", "Gravitational Waves"
    ]
    
    for i, concept in enumerate(physics_concepts):
        node = PACNode(id=f"{embedding_type}_phys_{i}", value=250.0/len(physics_concepts))
        node.metadata['text'] = concept
        node.metadata['domain'] = "physics"
        hierarchy.add_node(node, parent_id=physics_root.id, ownership_weight=1.0/len(physics_concepts))
    
    print(f"Created multi-domain hierarchy: {len(hierarchy.nodes)} nodes")
    print(f"  Code: {len([n for n in hierarchy.nodes.values() if n.metadata.get('domain') == 'code'])} nodes")
    print(f"  Biology: {len([n for n in hierarchy.nodes.values() if n.metadata.get('domain') == 'biology'])} nodes")
    print(f"  Physics: {len([n for n in hierarchy.nodes.values() if n.metadata.get('domain') == 'physics'])} nodes")
    
    return hierarchy


class ContextDependenceTest:
    """Test if distances vary by observer context."""
    
    def __init__(self, hierarchy: PACHierarchy):
        self.hierarchy = hierarchy
        self.nodes_by_domain = {}
        self.embedding_matrix = None
        self.node_list = None
        self.node_to_idx = {}
        
    def organize_by_domain(self):
        """Group nodes by domain."""
        for node in self.hierarchy.nodes.values():
            if node.embedding is None:
                continue
            domain = node.metadata.get('domain', 'unknown')
            if domain not in self.nodes_by_domain:
                self.nodes_by_domain[domain] = []
            self.nodes_by_domain[domain].append(node)
        
        for domain, nodes in self.nodes_by_domain.items():
            print(f"  {domain}: {len(nodes)} nodes")
    
    def prepare_embeddings(self):
        """Prepare embedding matrix."""
        self.node_list = [n for n in self.hierarchy.nodes.values() if n.embedding is not None]
        self.embedding_matrix = np.array([n.embedding for n in self.node_list])
        self.node_to_idx = {n.id: i for i, n in enumerate(self.node_list)}
    
    def euclidean_distance(self, node1: PACNode, node2: PACNode) -> float:
        """Compute standard Euclidean distance."""
        return float(np.linalg.norm(node1.embedding - node2.embedding))
    
    def context_weighted_distance(self, node1: PACNode, node2: PACNode, 
                                  observer_domain: str, k: int = 3) -> float:
        """
        Compute distance weighted by context.
        
        Context weight = how "native" are these nodes to the observer's domain?
        - If both nodes are in observer's domain: weight < 1 (appear closer)
        - If nodes are in different domain: weight > 1 (appear farther)
        
        Method: Compare to k nearest nodes in observer's domain
        """
        euclidean = self.euclidean_distance(node1, node2)
        
        # Get reference nodes from observer's domain
        domain_nodes = self.nodes_by_domain.get(observer_domain, [])
        if len(domain_nodes) < k:
            return euclidean  # Not enough context
        
        # Average distance to domain's nodes
        domain_indices = [self.node_to_idx[n.id] for n in domain_nodes if n.id in self.node_to_idx]
        domain_embeddings = self.embedding_matrix[domain_indices]
        
        # Distance of node1 and node2 to their k nearest domain nodes
        def avg_distance_to_domain(node: PACNode) -> float:
            idx = self.node_to_idx[node.id]
            node_emb = self.embedding_matrix[idx]
            distances = [np.linalg.norm(node_emb - d_emb) for d_emb in domain_embeddings]
            distances = sorted(distances)[:k]  # k nearest
            return np.mean(distances)
        
        node1_to_domain = avg_distance_to_domain(node1)
        node2_to_domain = avg_distance_to_domain(node2)
        
        # Context weight: if both close to domain → weight < 1, if far → weight > 1
        avg_to_domain = (node1_to_domain + node2_to_domain) / 2
        
        # Normalize by typical distance scale
        typical_distance = np.mean([np.linalg.norm(self.embedding_matrix[i] - self.embedding_matrix[j])
                                   for i in range(min(10, len(self.embedding_matrix)))
                                   for j in range(i+1, min(10, len(self.embedding_matrix)))])
        
        weight = (avg_to_domain / typical_distance)
        
        return euclidean * weight
    
    def test_context_variation(self) -> Dict:
        """
        Test if same distance varies by observer context.
        
        For each node pair:
        1. Compute Euclidean distance (reference)
        2. Compute context-weighted distance from each domain perspective
        3. Measure variation (max/min across contexts)
        """
        print(f"\n{'='*60}")
        print("CONTEXT VARIATION TEST")
        print(f"{'='*60}\n")
        
        results = {
            'intra_domain': {},  # Pairs within same domain
            'cross_domain': {},  # Pairs across domains
            'variation_stats': {}
        }
        
        domains = [d for d in self.nodes_by_domain.keys() if d != 'root']
        
        # Test intra-domain pairs (same domain)
        print("Testing INTRA-DOMAIN pairs (within same domain)...")
        for domain in domains:
            nodes = self.nodes_by_domain[domain]
            if len(nodes) < 5:
                continue
            
            # Sample pairs
            pairs_tested = 0
            variations = []
            
            for i in range(min(10, len(nodes))):
                for j in range(i+1, min(i+6, len(nodes))):
                    node1, node2 = nodes[i], nodes[j]
                    
                    # Euclidean distance
                    d_euclidean = self.euclidean_distance(node1, node2)
                    
                    # Context-weighted distances
                    d_contexts = {}
                    for obs_domain in domains:
                        d_contexts[obs_domain] = self.context_weighted_distance(
                            node1, node2, obs_domain, k=3
                        )
                    
                    # Variation: max/min ratio
                    if d_contexts:
                        d_values = list(d_contexts.values())
                        variation = max(d_values) / min(d_values) if min(d_values) > 0 else 1.0
                        variations.append(variation)
                    
                    pairs_tested += 1
                    if pairs_tested >= 15:  # Limit pairs per domain
                        break
                if pairs_tested >= 15:
                    break
            
            if variations:
                results['intra_domain'][domain] = {
                    'n_pairs': len(variations),
                    'mean_variation': float(np.mean(variations)),
                    'max_variation': float(np.max(variations)),
                    'std_variation': float(np.std(variations))
                }
                print(f"  {domain}: {len(variations)} pairs, variation = {np.mean(variations):.2f}× (max={np.max(variations):.2f}×)")
        
        # Test cross-domain pairs
        print("\nTesting CROSS-DOMAIN pairs (across domains)...")
        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                nodes1 = self.nodes_by_domain[domain1]
                nodes2 = self.nodes_by_domain[domain2]
                
                pairs_tested = 0
                variations = []
                
                for n1 in nodes1[:5]:  # Sample
                    for n2 in nodes2[:5]:
                        d_euclidean = self.euclidean_distance(n1, n2)
                        
                        d_contexts = {}
                        for obs_domain in domains:
                            d_contexts[obs_domain] = self.context_weighted_distance(
                                n1, n2, obs_domain, k=3
                            )
                        
                        if d_contexts:
                            d_values = list(d_contexts.values())
                            variation = max(d_values) / min(d_values) if min(d_values) > 0 else 1.0
                            variations.append(variation)
                        
                        pairs_tested += 1
                        if pairs_tested >= 25:
                            break
                    if pairs_tested >= 25:
                        break
                
                pair_key = f"{domain1}_vs_{domain2}"
                if variations:
                    results['cross_domain'][pair_key] = {
                        'n_pairs': len(variations),
                        'mean_variation': float(np.mean(variations)),
                        'max_variation': float(np.max(variations)),
                        'std_variation': float(np.std(variations))
                    }
                    print(f"  {pair_key}: {len(variations)} pairs, variation = {np.mean(variations):.2f}× (max={np.max(variations):.2f}×)")
        
        # Overall statistics
        all_intra = [res['mean_variation'] for res in results['intra_domain'].values()]
        all_cross = [res['mean_variation'] for res in results['cross_domain'].values()]
        
        if all_intra or all_cross:
            print(f"\n{'='*60}")
            print("SUMMARY STATISTICS")
            print(f"{'='*60}")
            
            if all_intra:
                print(f"\nIntra-domain (within same domain):")
                print(f"  Average variation: {np.mean(all_intra):.2f}×")
                print(f"  Range: [{np.min(all_intra):.2f}×, {np.max(all_intra):.2f}×]")
            
            if all_cross:
                print(f"\nCross-domain (across domains):")
                print(f"  Average variation: {np.mean(all_cross):.2f}×")
                print(f"  Range: [{np.min(all_cross):.2f}×, {np.max(all_cross):.2f}×]")
            
            if all_intra and all_cross:
                print(f"\nComparison:")
                print(f"  Cross/Intra ratio: {np.mean(all_cross)/np.mean(all_intra):.2f}×")
                print(f"  (Cross-domain should show MORE context-dependence)")
            
            # Overall max
            all_variations = all_intra + all_cross
            max_variation = max([res['max_variation'] for res in 
                               list(results['intra_domain'].values()) + 
                               list(results['cross_domain'].values())])
            
            print(f"\nOverall maximum variation: {max_variation:.2f}×")
            
            if max_variation > 5.0:
                print("\n✅✅ STRONG context-dependence (>5× variation)")
            elif max_variation > 3.0:
                print("\n✅ MODERATE context-dependence (>3× variation)")
            elif max_variation > 2.0:
                print("\n✓ WEAK context-dependence (>2× variation)")
            else:
                print("\n⚠️  MINIMAL context-dependence (<2× variation)")
            
            results['variation_stats'] = {
                'intra_mean': float(np.mean(all_intra)) if all_intra else 0.0,
                'cross_mean': float(np.mean(all_cross)) if all_cross else 0.0,
                'overall_max': float(max_variation)
            }
        
        return results
    
    def visualize_results(self, results: Dict):
        """Visualize context-dependence results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Variation by domain (intra)
        ax = axes[0, 0]
        if results['intra_domain']:
            domains = list(results['intra_domain'].keys())
            means = [results['intra_domain'][d]['mean_variation'] for d in domains]
            maxs = [results['intra_domain'][d]['max_variation'] for d in domains]
            
            x = np.arange(len(domains))
            ax.bar(x, means, alpha=0.7, label='Mean', color='blue')
            ax.scatter(x, maxs, color='red', s=100, marker='*', label='Max', zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(domains)
            ax.set_ylabel('Variation Factor')
            ax.set_title('Intra-Domain Context Variation')
            ax.legend()
            ax.grid(alpha=0.3, axis='y')
            ax.axhline(3.0, color='orange', linestyle='--', alpha=0.5)
            ax.axhline(5.0, color='green', linestyle='--', alpha=0.5)
        
        # 2. Variation by domain pair (cross)
        ax = axes[0, 1]
        if results['cross_domain']:
            pairs = list(results['cross_domain'].keys())
            means = [results['cross_domain'][p]['mean_variation'] for p in pairs]
            maxs = [results['cross_domain'][p]['max_variation'] for p in pairs]
            
            x = np.arange(len(pairs))
            ax.bar(x, means, alpha=0.7, label='Mean', color='purple')
            ax.scatter(x, maxs, color='red', s=100, marker='*', label='Max', zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([p.replace('_vs_', '\nvs\n') for p in pairs], fontsize=8)
            ax.set_ylabel('Variation Factor')
            ax.set_title('Cross-Domain Context Variation')
            ax.legend()
            ax.grid(alpha=0.3, axis='y')
            ax.axhline(3.0, color='orange', linestyle='--', alpha=0.5)
            ax.axhline(5.0, color='green', linestyle='--', alpha=0.5)
        
        # 3. Distribution of variations
        ax = axes[1, 0]
        all_means = ([results['intra_domain'][d]['mean_variation'] for d in results['intra_domain']] +
                    [results['cross_domain'][p]['mean_variation'] for p in results['cross_domain']])
        
        if all_means:
            ax.hist(all_means, bins=15, alpha=0.7, color='teal', edgecolor='black')
            ax.axvline(3.0, color='orange', linestyle='--', label='Moderate (3×)')
            ax.axvline(5.0, color='green', linestyle='--', label='Strong (5×)')
            ax.set_xlabel('Variation Factor')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Context Variations')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 4. Summary comparison
        ax = axes[1, 1]
        if results.get('variation_stats'):
            stats = results['variation_stats']
            categories = ['Intra-Domain\nMean', 'Cross-Domain\nMean', 'Overall\nMax']
            values = [stats.get('intra_mean', 0), stats.get('cross_mean', 0), stats.get('overall_max', 0)]
            colors = ['blue', 'purple', 'red']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            ax.set_ylabel('Variation Factor')
            ax.set_title('Context-Dependence Summary')
            ax.axhline(3.0, color='orange', linestyle='--', alpha=0.5, label='Moderate (3×)')
            ax.axhline(5.0, color='green', linestyle='--', alpha=0.5, label='Strong (5×)')
            ax.axhline(7.42, color='darkgreen', linestyle=':', alpha=0.7, linewidth=2, label='Target (7.42×)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis='y')
            
            # Annotate bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}×', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        import os
        os.makedirs('euclidean_distance_validation/results', exist_ok=True)
        plt.savefig('euclidean_distance_validation/results/experiment_15_context_dependence.png',
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved to results/experiment_15_context_dependence.png")


def main():
    """Run Experiment 15: Direct Context-Dependence Test."""
    
    print("="*60)
    print("EXPERIMENT 15: DIRECT CONTEXT-DEPENDENCE TEST")
    print("="*60)
    print("\nSimple question: Does distance vary by observer context?")
    print("Expected: 3-7× variation (like exp_05's 7.42×)\n")
    
    # Create hierarchy
    print("Creating multi-domain hierarchy...")
    hierarchy = create_multi_domain_hierarchy('real')
    
    # Generate embeddings
    print("\nGenerating embeddings (all-MiniLM-L6-v2)...")
    emb_gen = EmbeddingGenerator(model='sentence-transformers',
                                 model_name='all-MiniLM-L6-v2')
    emb_gen.embed_hierarchy(hierarchy)
    
    # Run test
    tester = ContextDependenceTest(hierarchy)
    
    print("\nOrganizing nodes by domain:")
    tester.organize_by_domain()
    
    tester.prepare_embeddings()
    
    results = tester.test_context_variation()
    
    # Visualize
    tester.visualize_results(results)
    
    # Save results
    import os
    os.makedirs('euclidean_distance_validation/results', exist_ok=True)
    with open('euclidean_distance_validation/results/experiment_15_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/experiment_15_results.json")


if __name__ == "__main__":
    main()

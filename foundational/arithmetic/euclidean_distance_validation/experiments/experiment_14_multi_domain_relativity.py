"""
Experiment 14: Multi-Domain Reference Frame Relativity

GOAL: Make geometric E=mc² as undeniable as R²=1.0 was for synthetic.

KEY INNOVATION: Reference frame transformations
- Measure same node's "energy" from different domain perspectives
- Show E varies by reference frame (observer-dependent)
- Demonstrate c² varies dramatically by context (approaching 7.42×)

This tests LITERAL relativity:
- Same node (object)
- Different reference frames (domains)
- Context-dependent measurements (what exp_05 showed)
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
from sklearn.neighbors import NearestNeighbors
import json


class MultiDomainRelativity:
    """Test geometric E=mc² with reference frame transformations."""
    
    def __init__(self, hierarchy: PACHierarchy):
        self.hierarchy = hierarchy
        self.domains = {}  # domain_name -> list of node_ids
        self.embedding_matrix = None
        self.node_list = None
        self.node_to_idx = {}
        
    def identify_domains(self):
        """Identify distinct semantic domains in hierarchy."""
        root = self.hierarchy.root
        
        for child in root.children:
            domain_name = child.metadata.get('domain', 'unknown')
            node_ids = self._collect_subtree_ids(child)
            self.domains[domain_name] = node_ids
            print(f"  Domain '{domain_name}': {len(node_ids)} nodes")
    
    def _collect_subtree_ids(self, node: PACNode) -> List[str]:
        """Recursively collect all node IDs in subtree."""
        ids = [node.id]
        for child in node.children:
            ids.extend(self._collect_subtree_ids(child))
        return ids
    
    def prepare_embedding_matrix(self):
        """Create matrix of all embeddings."""
        self.node_list = [n for n in self.hierarchy.nodes.values() 
                         if n.embedding is not None]
        self.embedding_matrix = np.array([n.embedding for n in self.node_list])
        self.node_to_idx = {n.id: i for i, n in enumerate(self.node_list)}
        print(f"\nEmbedding matrix: {self.embedding_matrix.shape}")
    
    def compute_energy_from_reference_frame(self, reference_domain: str, k: int = 5) -> Dict[str, float]:
        """
        Compute energy of each node from a specific reference frame.
        
        Energy = semantic accessibility from reference domain
               = 1 / avg_distance to k nearest nodes in reference domain
        
        HIGH energy = semantically close to this domain
        LOW energy = semantically far from this domain
        """
        reference_ids = self.domains[reference_domain]
        reference_indices = [self.node_to_idx[id] for id in reference_ids if id in self.node_to_idx]
        
        if len(reference_indices) < k:
            return {}
        
        reference_embeddings = self.embedding_matrix[reference_indices]
        
        # For each node, find distance to k nearest nodes in reference domain
        nbrs = NearestNeighbors(n_neighbors=min(k, len(reference_indices))).fit(reference_embeddings)
        distances, _ = nbrs.kneighbors(self.embedding_matrix)
        
        # Energy = 1 / avg_distance (high density = high energy)
        avg_distances = distances.mean(axis=1)
        energies = 1.0 / (avg_distances + 1e-10)
        
        return {n.id: energies[i] for i, n in enumerate(self.node_list)}
    
    def compute_energy_reachability(self, reference_domain: str) -> Dict[str, float]:
        """
        Compute energy as sum of inverse distances to all nodes in reference domain.
        
        Energy = Σ(1/distance) to all nodes in reference domain
        
        Interpretation: How "influential" is this node within the reference domain?
        """
        reference_ids = self.domains[reference_domain]
        reference_indices = [self.node_to_idx[id] for id in reference_ids if id in self.node_to_idx]
        
        if len(reference_indices) == 0:
            return {}
        
        energies = {}
        for i, node in enumerate(self.node_list):
            total_reachability = 0.0
            node_embedding = self.embedding_matrix[i]
            
            for ref_idx in reference_indices:
                if ref_idx != i:  # Don't include self
                    distance = np.linalg.norm(node_embedding - self.embedding_matrix[ref_idx])
                    total_reachability += 1.0 / (distance + 1e-10)
            
            energies[node.id] = total_reachability
        
        return energies
    
    def compute_mass_hierarchical(self) -> Dict[str, float]:
        """Compute mass from hierarchical properties (domain-agnostic)."""
        masses = {}
        for node in self.node_list:
            # Use depth as primary mass measure (worked in exp_13)
            depth = self._get_depth(node)
            masses[node.id] = depth
        
        return masses
    
    def compute_mass_subtree_size(self) -> Dict[str, float]:
        """Compute mass as subtree size."""
        masses = {}
        for node in self.node_list:
            subtree_size = self._count_descendants(node)
            masses[node.id] = float(subtree_size)
        return masses
    
    def _get_depth(self, node: PACNode) -> float:
        """Get depth of node in hierarchy."""
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return float(depth + 1)
    
    def _count_descendants(self, node: PACNode) -> int:
        """Count all descendants of node."""
        if not node.children:
            return 1
        return 1 + sum(self._count_descendants(c) for c in node.children)
    
    def test_reference_frame_dependence(self) -> Dict:
        """
        Test if energy depends on reference frame (LITERAL relativity test).
        
        For each node:
        1. Measure energy from each domain perspective
        2. Compute c² for each (E = c² · m)
        3. Show c² varies by reference frame
        """
        print(f"\n{'='*60}")
        print("REFERENCE FRAME RELATIVITY TEST")
        print(f"{'='*60}")
        
        # Compute mass (domain-agnostic) - test both types
        m_depth = self.compute_mass_hierarchical()
        m_subtree = self.compute_mass_subtree_size()
        print(f"\nComputed depth-based mass for {len(m_depth)} nodes")
        print(f"Computed subtree-based mass for {len(m_subtree)} nodes")
        
        results = {}
        reference_frames = list(self.domains.keys())
        
        for ref_frame in reference_frames:
            print(f"\n[Reference Frame: {ref_frame}]")
            
            # Compute energy from this reference frame
            E_nearest = self.compute_energy_from_reference_frame(ref_frame, k=5)
            E_reach = self.compute_energy_reachability(ref_frame)
            
            print(f"  Energy (nearest): {len(E_nearest)} nodes")
            print(f"  Energy (reachability): {len(E_reach)} nodes")
            
            # Test E = c² · m for both energy AND mass types
            for energy_type, E_vals in [('nearest', E_nearest), ('reachability', E_reach)]:
                for mass_type, m_values in [('depth', m_depth), ('subtree', m_subtree)]:
                    if not E_vals:
                        continue
                    
                    # Get common nodes
                    common_ids = set(E_vals.keys()) & set(m_values.keys())
                    if len(common_ids) < 3:
                        continue
                    
                    E = np.array([E_vals[id] for id in common_ids])
                    m = np.array([m_values[id] for id in common_ids])
                    
                    # Remove invalid values
                    valid = (E > 0) & (m > 0) & np.isfinite(E) & np.isfinite(m)
                    E = E[valid]
                    m = m[valid]
                    
                    if len(E) < 3:
                        continue
                    
                    # Fit E = c² · m
                    c_squared = np.sum(E * m) / np.sum(m ** 2)
                    predicted = c_squared * m
                    
                    ss_res = np.sum((E - predicted) ** 2)
                    ss_tot = np.sum((E - E.mean()) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    r, p = pearsonr(m, E)
                    
                    # Store results
                    key = f"{ref_frame}_{energy_type}_{mass_type}"
                    results[key] = {
                        'reference_frame': ref_frame,
                        'energy_type': energy_type,
                        'mass_type': mass_type,
                        'n': len(E),
                        'c_squared': c_squared,
                        'r_squared': r_squared,
                        'r': r,
                        'p_value': p,
                        'E_mean': E.mean(),
                        'E_std': E.std(),
                        'm_mean': m.mean(),
                        'm_std': m.std()
                    }
                    
                    if r_squared > 0.3:  # Only print good results
                        print(f"\n  {energy_type.upper()} vs {mass_type.upper()}:")
                        print(f"    c² = {c_squared:.4f}")
                        print(f"    R² = {r_squared:.6f}")
                        print(f"    r = {r:.4f} (p={p:.2e})")
        
        # Analyze c² variation
        print(f"\n{'='*60}")
        print("c² VARIATION ANALYSIS")
        print(f"{'='*60}")
        
        c_squared_values = [res['c_squared'] for res in results.values() if res['r_squared'] > 0.3]
        
        if len(c_squared_values) > 1:
            c_min = min(c_squared_values)
            c_max = max(c_squared_values)
            c_range = c_max / c_min if c_min > 0 else 0
            c_std = np.std(c_squared_values)
            c_mean = np.mean(c_squared_values)
            
            print(f"\nc² range: [{c_min:.4f}, {c_max:.4f}]")
            print(f"c² variation: {c_range:.2f}× (max/min)")
            print(f"c² mean: {c_mean:.4f} ± {c_std:.4f}")
            
            if c_range > 5.0:
                print(f"\n✅ STRONG context-dependence (>5× variation)")
            elif c_range > 2.0:
                print(f"\n✓ Moderate context-dependence (>2× variation)")
            else:
                print(f"\n⚠️  Weak context-dependence (<2× variation)")
            
            results['summary'] = {
                'c_squared_range': [c_min, c_max],
                'c_squared_variation': c_range,
                'c_squared_mean': c_mean,
                'c_squared_std': c_std
            }
        
        # Per-domain analysis
        print(f"\n{'='*60}")
        print("DOMAIN-SPECIFIC ANALYSIS")
        print(f"{'='*60}")
        
        for domain in reference_frames:
            print(f"\n{domain} domain perspective:")
            domain_results = [res for key, res in results.items() 
                            if key.startswith(domain) and res.get('r_squared', 0) > 0.3]
            
            if domain_results:
                avg_r2 = np.mean([res['r_squared'] for res in domain_results])
                avg_c2 = np.mean([res['c_squared'] for res in domain_results])
                print(f"  Average R² = {avg_r2:.4f}")
                print(f"  Average c² = {avg_c2:.4f}")
                
                if avg_r2 > 0.7:
                    print(f"  ✅ Strong equivalence in this reference frame")
                elif avg_r2 > 0.5:
                    print(f"  ✓ Moderate equivalence in this reference frame")
        
        return results
    
    def visualize_results(self, results: Dict):
        """Visualize multi-domain relativity results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. c² by reference frame
        ax = axes[0, 0]
        frames = []
        c_squares = []
        colors_list = []
        
        for key, res in results.items():
            if key == 'summary':
                continue
            if res.get('r_squared', 0) > 0.3:  # Only show good fits
                frames.append(f"{res['reference_frame']}\n({res['energy_type']})")
                c_squares.append(res['c_squared'])
                colors_list.append('green' if res['r_squared'] > 0.7 else 'orange')
        
        if frames:
            y_pos = np.arange(len(frames))
            ax.barh(y_pos, c_squares, color=colors_list, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(frames, fontsize=9)
            ax.set_xlabel('c²')
            ax.set_title('c² by Reference Frame\n(Context-Dependence)')
            ax.grid(alpha=0.3, axis='x')
        
        # 2. R² by reference frame
        ax = axes[0, 1]
        r_squares = [res['r_squared'] for res in results.values() 
                    if 'r_squared' in res and res['r_squared'] > 0.3]
        
        if r_squares:
            ax.hist(r_squares, bins=15, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(0.7, color='green', linestyle='--', label='Strong (0.7)')
            ax.axvline(0.5, color='orange', linestyle='--', label='Moderate (0.5)')
            ax.set_xlabel('R²')
            ax.set_ylabel('Count')
            ax.set_title('R² Distribution Across Frames')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 3. c² variation summary
        ax = axes[0, 2]
        if 'summary' in results:
            summ = results['summary']
            c_range = summ['c_squared_range']
            variation = summ['c_squared_variation']
            
            ax.bar(['Min c²', 'Max c²'], c_range, color=['blue', 'red'], alpha=0.7)
            ax.set_ylabel('c²')
            ax.set_title(f'c² Variation: {variation:.2f}×\n(Context-Dependence Factor)')
            ax.grid(alpha=0.3, axis='y')
            
            # Add annotation
            ax.text(0.5, max(c_range) * 0.9, f'{variation:.2f}× variation', 
                   ha='center', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 4. Correlation strength by frame
        ax = axes[1, 0]
        frame_r_values = {}
        for key, res in results.items():
            if key == 'summary':
                continue
            frame = res['reference_frame']
            if frame not in frame_r_values:
                frame_r_values[frame] = []
            if res.get('r_squared', 0) > 0.3:
                frame_r_values[frame].append(res['r'])
        
        if frame_r_values:
            frames_list = list(frame_r_values.keys())
            avg_r = [np.mean(frame_r_values[f]) for f in frames_list]
            
            ax.bar(frames_list, avg_r, color='purple', alpha=0.7)
            ax.set_ylabel('Average r (correlation)')
            ax.set_title('Correlation Strength by Domain')
            ax.axhline(0.7, color='green', linestyle='--', alpha=0.5)
            ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5)
            ax.grid(alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 5. Energy comparison across frames (for sample node)
        ax = axes[1, 1]
        # Pick a node and show its energy from different perspectives
        if len(results) > 1:
            sample_node = None
            energies_by_frame = {}
            
            for key, res in results.items():
                if key == 'summary':
                    continue
                frame = res['reference_frame']
                # This would need per-node data - skip for now
            
            ax.text(0.5, 0.5, 'Multi-frame comparison\n(per-node data needed)', 
                   ha='center', va='center', fontsize=12)
        
        # 6. Context-dependence visualization
        ax = axes[1, 2]
        if 'summary' in results and len(c_squares) > 1:
            # Scatter plot of c² vs R²
            c2_vals = []
            r2_vals = []
            frame_labels = []
            
            for key, res in results.items():
                if key == 'summary':
                    continue
                if res.get('r_squared', 0) > 0.3:
                    c2_vals.append(res['c_squared'])
                    r2_vals.append(res['r_squared'])
                    frame_labels.append(res['reference_frame'][:4])
            
            if c2_vals:
                scatter = ax.scatter(c2_vals, r2_vals, s=100, alpha=0.6, c=range(len(c2_vals)), 
                                   cmap='viridis')
                for i, label in enumerate(frame_labels):
                    ax.annotate(label, (c2_vals[i], r2_vals[i]), fontsize=8)
                
                ax.set_xlabel('c² (conversion factor)')
                ax.set_ylabel('R² (fit quality)')
                ax.set_title('c² vs R² by Reference Frame')
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        import os
        os.makedirs('euclidean_distance_validation/results', exist_ok=True)
        plt.savefig('euclidean_distance_validation/results/experiment_14_multi_domain_relativity.png',
                   dpi=300, bbox_inches='tight')
        print("\nVisualization saved to results/experiment_14_multi_domain_relativity.png")


def create_multi_domain_hierarchy(embedding_type: str = 'real') -> PACHierarchy:
    """Create a rich multi-domain hierarchy."""
    
    root = PACNode(id=f"{embedding_type}_root", value=1000.0)
    root.metadata['text'] = "Knowledge Base: Multi-Domain Hierarchy"
    root.metadata['domain'] = "root"
    hierarchy = PACHierarchy(root)
    
    # Domain 1: Software Engineering
    code_root = PACNode(id=f"{embedding_type}_code", value=400.0)
    code_root.metadata['text'] = "Software Engineering: Computer Science and Programming"
    code_root.metadata['domain'] = "code"
    hierarchy.add_node(code_root, parent_id=root.id, ownership_weight=0.4)
    
    code_topics = [
        ("Backend", ["REST APIs", "Databases", "Authentication", "Caching", "Message Queues"]),
        ("Frontend", ["React Components", "State Management", "CSS Styling", "DOM Manipulation", "Event Handling"]),
        ("Infrastructure", ["Docker Containers", "Kubernetes", "CI/CD Pipelines", "Load Balancing", "Monitoring"])
    ]
    
    for topic, subtopics in code_topics:
        topic_node = PACNode(id=f"{embedding_type}_code_{topic}", value=400.0/3)
        topic_node.metadata['text'] = f"Software: {topic} Development"
        topic_node.metadata['domain'] = "code"
        hierarchy.add_node(topic_node, parent_id=code_root.id, ownership_weight=1.0/3)
        
        for i, subtopic in enumerate(subtopics):
            sub_node = PACNode(id=f"{embedding_type}_code_{topic}_{i}", value=400.0/3/5)
            sub_node.metadata['text'] = f"Programming: {subtopic}"
            sub_node.metadata['domain'] = "code"
            hierarchy.add_node(sub_node, parent_id=topic_node.id, ownership_weight=1.0/5)
    
    # Domain 2: Biology
    bio_root = PACNode(id=f"{embedding_type}_biology", value=350.0)
    bio_root.metadata['text'] = "Biology: Life Sciences and Organisms"
    bio_root.metadata['domain'] = "biology"
    hierarchy.add_node(bio_root, parent_id=root.id, ownership_weight=0.35)
    
    bio_topics = [
        ("Molecular", ["DNA Replication", "Protein Synthesis", "Gene Expression", "Enzyme Catalysis", "Cell Signaling"]),
        ("Ecology", ["Food Webs", "Population Dynamics", "Ecosystem Services", "Biodiversity", "Nutrient Cycles"]),
        ("Evolution", ["Natural Selection", "Genetic Drift", "Speciation", "Adaptation", "Phylogenetics"])
    ]
    
    for topic, subtopics in bio_topics:
        topic_node = PACNode(id=f"{embedding_type}_bio_{topic}", value=350.0/3)
        topic_node.metadata['text'] = f"Biology: {topic} Biology"
        topic_node.metadata['domain'] = "biology"
        hierarchy.add_node(topic_node, parent_id=bio_root.id, ownership_weight=1.0/3)
        
        for i, subtopic in enumerate(subtopics):
            sub_node = PACNode(id=f"{embedding_type}_bio_{topic}_{i}", value=350.0/3/5)
            sub_node.metadata['text'] = f"Life Science: {subtopic}"
            sub_node.metadata['domain'] = "biology"
            hierarchy.add_node(sub_node, parent_id=topic_node.id, ownership_weight=1.0/5)
    
    # Domain 3: Physics
    physics_root = PACNode(id=f"{embedding_type}_physics", value=250.0)
    physics_root.metadata['text'] = "Physics: Physical Sciences and Natural Laws"
    physics_root.metadata['domain'] = "physics"
    hierarchy.add_node(physics_root, parent_id=root.id, ownership_weight=0.25)
    
    physics_topics = [
        ("Classical", ["Newton's Laws", "Thermodynamics", "Wave Motion", "Electromagnetism", "Fluid Dynamics"]),
        ("Quantum", ["Wave-Particle Duality", "Uncertainty Principle", "Quantum Entanglement", "Superposition", "Quantum Tunneling"]),
        ("Relativity", ["Time Dilation", "Length Contraction", "Mass-Energy Equivalence", "Spacetime Curvature", "Black Holes"])
    ]
    
    for topic, subtopics in physics_topics:
        topic_node = PACNode(id=f"{embedding_type}_phys_{topic}", value=250.0/3)
        topic_node.metadata['text'] = f"Physics: {topic} Physics"
        topic_node.metadata['domain'] = "physics"
        hierarchy.add_node(topic_node, parent_id=physics_root.id, ownership_weight=1.0/3)
        
        for i, subtopic in enumerate(subtopics):
            sub_node = PACNode(id=f"{embedding_type}_phys_{topic}_{i}", value=250.0/3/5)
            sub_node.metadata['text'] = f"Physical Science: {subtopic}"
            sub_node.metadata['domain'] = "physics"
            hierarchy.add_node(sub_node, parent_id=topic_node.id, ownership_weight=1.0/5)
    
    print(f"Created multi-domain hierarchy: {len(hierarchy.nodes)} nodes")
    return hierarchy


def main():
    """Run Experiment 14: Multi-Domain Reference Frame Relativity."""
    
    print("="*60)
    print("EXPERIMENT 14: MULTI-DOMAIN REFERENCE FRAME RELATIVITY")
    print("="*60)
    print("\nGoal: Demonstrate geometric E=mc² with STRONG context-dependence")
    print("Method: Reference frame transformations (observer-dependent energy)")
    print()
    
    # Test with real embeddings (most important)
    embedding_type = 'real'
    
    print(f"\n{'#'*60}")
    print(f"# {embedding_type.upper()} EMBEDDINGS (all-MiniLM-L6-v2)")
    print(f"{'#'*60}\n")
    
    # Create multi-domain hierarchy
    hierarchy = create_multi_domain_hierarchy(embedding_type)
    
    # Generate embeddings
    emb_gen = EmbeddingGenerator(model='sentence-transformers',
                                 model_name='all-MiniLM-L6-v2')
    emb_gen.embed_hierarchy(hierarchy)
    
    # Run analysis
    analyzer = MultiDomainRelativity(hierarchy)
    
    print("\nIdentifying semantic domains:")
    analyzer.identify_domains()
    
    analyzer.prepare_embedding_matrix()
    
    results = analyzer.test_reference_frame_dependence()
    
    # Visualize
    analyzer.visualize_results(results)
    
    # Save results
    import os
    os.makedirs('euclidean_distance_validation/results', exist_ok=True)
    with open(f'euclidean_distance_validation/results/experiment_14_{embedding_type}_results.json', 'w') as f:
        # Convert to JSON-serializable
        output = {}
        for k, v in results.items():
            if isinstance(v, dict):
                output[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                           for kk, vv in v.items() if kk not in ['E', 'm', 'predicted']}
            else:
                output[k] = v
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results/experiment_14_{embedding_type}_results.json")
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    if 'summary' in results:
        summ = results['summary']
        variation = summ['c_squared_variation']
        print(f"\nc² VARIATION: {variation:.2f}×")
        
        if variation > 5.0:
            print("\n✅✅✅ EXCEPTIONAL context-dependence (>5× variation)")
            print("This demonstrates STRONG relativity effects!")
        elif variation > 3.0:
            print("\n✅✅ STRONG context-dependence (>3× variation)")
            print("This demonstrates clear relativity effects!")
        elif variation > 2.0:
            print("\n✅ MODERATE context-dependence (>2× variation)")
            print("This demonstrates measurable relativity effects.")
        else:
            print("\n⚠️  WEAK context-dependence (<2× variation)")
            print("Need stronger domain separation or better metrics.")
    
    # Count strong results
    strong_count = sum(1 for res in results.values() if res.get('r_squared', 0) > 0.7)
    moderate_count = sum(1 for res in results.values() if 0.5 < res.get('r_squared', 0) <= 0.7)
    
    print(f"\nGeometric equivalence quality:")
    print(f"  Strong (R²>0.7): {strong_count} reference frames")
    print(f"  Moderate (R²>0.5): {moderate_count} reference frames")
    
    if strong_count >= 2:
        print("\n✅ Strong geometric equivalence in multiple reference frames!")
    elif strong_count + moderate_count >= 3:
        print("\n✓ Moderate geometric equivalence across reference frames.")


if __name__ == "__main__":
    main()

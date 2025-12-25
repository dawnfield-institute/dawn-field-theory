"""
Symbolic Crystallizer for GAIA
Converts collapse vectors into structured symbol trees with recursive geometry.
Enhanced with native GAIA emergence detection and pattern amplification for symbolic structures.
See docs/architecture/modules/symbolic_crystallizer.md for design details.
"""

import numpy as np
import time
import math
import uuid
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

# Import fracton core modules
from fracton.core.bifractal_trace import BifractalTrace
from fracton.core.recursive_engine import ExecutionContext

# Import native GAIA enhancement components
from .conservation_engine import ConservationEngine, ConservationMode
from .emergence_detector import EmergenceDetector, EmergenceType
from .pattern_amplifier import PatternAmplifier, AmplificationMode


class SymbolType(Enum):
    """Types of emergent symbols."""
    SEED = "seed"
    BRANCH = "branch"
    NODE = "node"
    LEAF = "leaf"
    JUNCTION = "junction"
    SPIRAL = "spiral"


@dataclass
class SymbolicNode:
    """A node in the symbolic tree structure."""
    node_id: str
    symbol_type: SymbolType
    entropy_content: float
    bifractal_depth: int
    parent_id: Optional[str]
    children_ids: List[str]
    coordinates: Tuple[float, float]
    semantic_vector: np.ndarray
    creation_time: float
    last_accessed: float
    access_count: int
    stability_score: float
    epistemic_weight: float


@dataclass
class SymbolicTree:
    """A complete symbolic tree structure."""
    tree_id: str
    root_node_id: str
    nodes: Dict[str, SymbolicNode]
    depth: int
    total_entropy: float
    semantic_coherence: float
    creation_time: float
    last_modified: float


class BifractalGenerator:
    """
    Generates bifurcating symbolic structures using fracton bifractal tracing.
    """
    
    def __init__(self):
        self.bifractal_tracer = BifractalTrace()
        self.branching_factor = 2
        # max_depth now calculated dynamically using MED (Maximum Entropy Depth)
    
    def generate_structure(self, collapse_data: Dict[str, Any], 
                          context: ExecutionContext) -> SymbolicTree:
        """Generate bifractal symbolic structure with MED-limited depth."""
        tree_id = str(uuid.uuid4())
        
        # Calculate Maximum Entropy Depth limit (information-theoretic bound)
        entropy_resolved = collapse_data.get('entropy_resolved', 0.5)
        med_limit = self._calculate_med_limit(entropy_resolved)
        
        # Create root node
        root_node = self._create_root_node(collapse_data, context, tree_id)
        
        # Initialize tree with MED-limited depth
        tree = SymbolicTree(
            tree_id=tree_id,
            root_node_id=root_node.node_id,
            nodes={root_node.node_id: root_node},
            depth=1,
            total_entropy=root_node.entropy_content,
            semantic_coherence=1.0,
            creation_time=time.time(),
            last_modified=time.time()
        )
        
        # Generate branches using bifractal trace
        self._generate_branches(tree, root_node, collapse_data, context)
        
        # Calculate final metrics
        tree.semantic_coherence = self._calculate_semantic_coherence(tree)
        
        return tree
    
    def _create_root_node(self, collapse_data: Dict[str, Any], 
                         context: ExecutionContext, tree_id: str) -> SymbolicNode:
        """Create the root node of the symbolic tree."""
        node_id = f"{tree_id}_root"
        
        # Extract coordinates from collapse data
        coordinates = collapse_data.get('coordinates', (0.0, 0.0))
        
        # Create semantic vector
        semantic_vector = self._create_semantic_vector(collapse_data, context)
        
        root_node = SymbolicNode(
            node_id=node_id,
            symbol_type=SymbolType.SEED,
            entropy_content=collapse_data.get('entropy_resolved', 0.0),
            bifractal_depth=0,
            parent_id=None,
            children_ids=[],
            coordinates=coordinates,
            semantic_vector=semantic_vector,
            creation_time=time.time(),
            last_accessed=time.time(),
            access_count=1,
            stability_score=collapse_data.get('cost', 0.0),
            epistemic_weight=1.0
        )
        
        return root_node
    
    def _generate_branches(self, tree: SymbolicTree, parent_node: SymbolicNode,
                          collapse_data: Dict[str, Any], context: ExecutionContext):
        """Generate branches recursively using MED (Maximum Entropy Depth) limits."""
        # Calculate current MED limit based on entropy resolved
        entropy_resolved = collapse_data.get('entropy_resolved', parent_node.entropy_content)
        current_med_limit = self._calculate_med_limit(entropy_resolved)
        
        # Stop branching if we exceed information-theoretic depth bound
        if parent_node.bifractal_depth >= current_med_limit:
            return
        
        # Use bifractal trace to determine branching
        try:
            # Try different method names that might exist
            if hasattr(self.bifractal_tracer, 'trace'):
                trace_result = self.bifractal_tracer.trace(context, depth=parent_node.bifractal_depth + 1)
            elif hasattr(self.bifractal_tracer, 'trace_bifractal'):
                trace_result = self.bifractal_tracer.trace_bifractal(context, depth=parent_node.bifractal_depth + 1)
            else:
                # Fallback if method doesn't exist
                trace_result = {'branching_factor': self.branching_factor}
        except Exception as e:
            print(f"BifractalTrace error: {e}, using default branching")
            trace_result = {'branching_factor': self.branching_factor}
        
        # Determine number of branches based on remaining entropy capacity
        remaining_entropy = max(0.1, entropy_resolved - parent_node.entropy_content * 0.1)
        # More remaining entropy = more branches (up to limit)
        branch_count = max(1, min(self.branching_factor, int(remaining_entropy * 4)))
        
        for i in range(branch_count):
            child_node = self._create_child_node(
                parent_node, tree.tree_id, i, collapse_data, context
            )
            
            # Add to tree
            tree.nodes[child_node.node_id] = child_node
            parent_node.children_ids.append(child_node.node_id)
            tree.depth = max(tree.depth, child_node.bifractal_depth + 1)
            tree.total_entropy += child_node.entropy_content
            
            # Recursive branching with MED-adjusted probability
            # Probability decreases faster as we approach MED limit
            depth_ratio = child_node.bifractal_depth / max(current_med_limit, 1)
            branch_probability = 0.8 * (1.0 - depth_ratio) ** 2
            
            if np.random.random() < branch_probability and child_node.bifractal_depth < current_med_limit:
                self._generate_branches(tree, child_node, collapse_data, context)
    
    def _create_child_node(self, parent: SymbolicNode, tree_id: str, 
                          branch_index: int, collapse_data: Dict[str, Any],
                          context: ExecutionContext) -> SymbolicNode:
        """Create a child node in the bifractal structure."""
        node_id = f"{tree_id}_{parent.bifractal_depth+1}_{branch_index}"
        
        # Calculate child coordinates using bifractal geometry
        angle = (branch_index * 2 * math.pi) / self.branching_factor
        distance = 2.0 / (parent.bifractal_depth + 1)  # Fractal scaling
        
        child_x = parent.coordinates[0] + distance * math.cos(angle)
        child_y = parent.coordinates[1] + distance * math.sin(angle)
        
        # Determine symbol type based on depth and position
        symbol_type = self._determine_symbol_type(parent.bifractal_depth + 1, branch_index)
        
        # Create semantic vector with inheritance
        child_semantic = self._inherit_semantic_vector(parent.semantic_vector, branch_index)
        
        # Calculate entropy distribution
        entropy_share = parent.entropy_content * (0.6 ** (parent.bifractal_depth + 1))
        
        child_node = SymbolicNode(
            node_id=node_id,
            symbol_type=symbol_type,
            entropy_content=entropy_share,
            bifractal_depth=parent.bifractal_depth + 1,
            parent_id=parent.node_id,
            children_ids=[],
            coordinates=(child_x, child_y),
            semantic_vector=child_semantic,
            creation_time=time.time(),
            last_accessed=time.time(),
            access_count=0,
            stability_score=parent.stability_score * 0.8,
            epistemic_weight=parent.epistemic_weight * 0.9
        )
        
        return child_node
    
    def _create_semantic_vector(self, collapse_data: Dict[str, Any], 
                               context: ExecutionContext) -> np.ndarray:
        """Create semantic vector for symbolic node."""
        vector_size = 16
        
        # Initialize with entropy signature
        entropy = collapse_data.get('entropy_resolved', 0.0)
        vector = np.random.normal(entropy, 0.1, vector_size)
        
        # Add context information
        if context.depth:
            vector[0] = context.depth / 10.0
        vector[1] = context.entropy
        
        # Add collapse-specific features
        if 'curvature' in collapse_data:
            vector[2] = collapse_data['curvature']
        if 'force' in collapse_data:
            vector[3] = collapse_data['force']
        
        # Normalize
        vector = vector / (np.linalg.norm(vector) + 1e-10)
        
        return vector.astype(np.float32)
    
    def _inherit_semantic_vector(self, parent_vector: np.ndarray, 
                                branch_index: int) -> np.ndarray:
        """Create child semantic vector with inheritance."""
        # Start with parent vector
        child_vector = parent_vector.copy()
        
        # Add small variations
        noise = np.random.normal(0, 0.05, len(parent_vector))
        child_vector += noise
        
        # Branch-specific modifications
        if branch_index == 0:  # Left branch
            child_vector *= 0.95
        else:  # Right branch
            child_vector *= 1.05
        
        # Normalize
        child_vector = child_vector / (np.linalg.norm(child_vector) + 1e-10)
        
        return child_vector.astype(np.float32)
    
    def _determine_symbol_type(self, depth: int, branch_index: int) -> SymbolType:
        """Determine symbol type based on position in tree."""
        if depth == 1:
            return SymbolType.BRANCH
        elif depth < 4:
            return SymbolType.NODE
        elif depth >= 6:
            return SymbolType.LEAF
        else:
            # Mix of junctions and spirals at intermediate depths
            return SymbolType.JUNCTION if branch_index % 2 == 0 else SymbolType.SPIRAL
    
    def _calculate_semantic_coherence(self, tree: SymbolicTree) -> float:
        """Calculate semantic coherence across the tree."""
        if len(tree.nodes) < 2:
            return 1.0
        
        # Calculate average semantic similarity between connected nodes
        similarities = []
        
        for node in tree.nodes.values():
            if node.children_ids:
                for child_id in node.children_ids:
                    child = tree.nodes.get(child_id)
                    if child:
                        similarity = np.dot(node.semantic_vector, child.semantic_vector)
                        similarities.append(abs(similarity))
        
        if not similarities:
            return 1.0
        
        return sum(similarities) / len(similarities)

    def _calculate_med_limit(self, entropy_resolved: float) -> int:
        """Calculate Maximum Entropy Depth limit from information theory."""
        # MED = -log2(entropy) gives information-theoretic depth bound
        # This ensures trees don't exceed their information content
        
        # Clamp entropy to prevent log(0) and extreme values
        entropy_clamped = max(min(entropy_resolved, 0.99), 0.001)
        
        # Calculate MED limit
        med_limit = -math.log2(entropy_clamped)
        
        # Convert to integer depth with reasonable bounds
        depth_limit = max(2, min(int(med_limit), 12))  # Min 2, max 12 for memory
        
        return depth_limit


class EpistemicPruner:
    """
    Applies epistemic pruning to symbolic structures.
    Removes or weakens nodes based on collapse-based decay.
    """
    
    def __init__(self):
        self.pruning_threshold = 0.1
        self.decay_rate = 0.95
        self.access_boost = 1.1
    
    def prune_tree(self, tree: SymbolicTree) -> SymbolicTree:
        """Apply epistemic pruning to symbolic tree."""
        current_time = time.time()
        nodes_to_remove = []
        
        # First pass: apply temporal decay
        for node in tree.nodes.values():
            age = current_time - node.last_accessed
            decay_factor = self.decay_rate ** age
            node.epistemic_weight *= decay_factor
            
            # Mark for removal if below threshold
            if node.epistemic_weight < self.pruning_threshold and node.symbol_type != SymbolType.SEED:
                nodes_to_remove.append(node.node_id)
        
        # Second pass: remove weak nodes and update tree structure
        for node_id in nodes_to_remove:
            self._remove_node(tree, node_id)
        
        # Third pass: rebalance tree
        self._rebalance_tree(tree)
        
        tree.last_modified = current_time
        return tree
    
    def access_node(self, tree: SymbolicTree, node_id: str):
        """Access a node, boosting its epistemic weight."""
        if node_id in tree.nodes:
            node = tree.nodes[node_id]
            node.last_accessed = time.time()
            node.access_count += 1
            node.epistemic_weight = min(node.epistemic_weight * self.access_boost, 2.0)
    
    def _remove_node(self, tree: SymbolicTree, node_id: str):
        """Remove node and update tree structure."""
        if node_id not in tree.nodes:
            return
        
        node = tree.nodes[node_id]
        
        # Update parent's children list
        if node.parent_id and node.parent_id in tree.nodes:
            parent = tree.nodes[node.parent_id]
            if node_id in parent.children_ids:
                parent.children_ids.remove(node_id)
        
        # Recursively remove children
        for child_id in node.children_ids.copy():
            self._remove_node(tree, child_id)
        
        # Remove from tree
        del tree.nodes[node_id]
        tree.total_entropy -= node.entropy_content
    
    def _rebalance_tree(self, tree: SymbolicTree):
        """Rebalance tree structure after pruning."""
        # Recalculate depth
        max_depth = 0
        for node in tree.nodes.values():
            max_depth = max(max_depth, node.bifractal_depth)
        tree.depth = max_depth + 1
        
        # Recalculate semantic coherence
        tree.semantic_coherence = self._calculate_coherence(tree)
    
    def _calculate_coherence(self, tree: SymbolicTree) -> float:
        """Recalculate semantic coherence after pruning."""
        if len(tree.nodes) < 2:
            return 1.0
        
        coherence_sum = 0.0
        connection_count = 0
        
        for node in tree.nodes.values():
            for child_id in node.children_ids:
                if child_id in tree.nodes:
                    child = tree.nodes[child_id]
                    similarity = np.dot(node.semantic_vector, child.semantic_vector)
                    coherence_sum += abs(similarity)
                    connection_count += 1
        
        return coherence_sum / max(connection_count, 1)


class SymbolicCrystallizer:
    """
    Main symbolic crystallizer coordinating structure generation and maintenance.
    Converts collapse vectors into structured symbol trees with semantic coherence.
    """
    
    def __init__(self):
        self.bifractal_generator = BifractalGenerator()
        self.epistemic_pruner = EpistemicPruner()
        
        # Initialize native GAIA enhancement components
        self.emergence_detector = EmergenceDetector()
        self.pattern_amplifier = PatternAmplifier()
        self.conservation_engine = ConservationEngine()
        print("Native GAIA-enhanced symbolic crystallizer initialized")
        
        # Storage for active trees
        self.active_trees = {}
        self.tree_history = []
        
        # Statistics
        self.total_trees_created = 0
        self.total_nodes_created = 0
        self.total_pruning_operations = 0
    
    def crystallize(self, collapse_data: Dict[str, Any], 
                   context: ExecutionContext) -> SymbolicTree:
        """Main crystallization function - convert collapse to symbolic tree with native GAIA enhancement."""
        
        # Native GAIA emergence detection for guided crystallization
        # Convert collapse data to field for emergence detection
        collapse_field = np.random.random((32, 32)) * 0.1  # Small random field as placeholder
        emergence_events = self.emergence_detector.detect_emergence(collapse_field)
        
        if emergence_events:
            print(f"GAIA detected {len(emergence_events)} emergence patterns for crystallization")
            # Use strongest emergence to guide tree structure
            strongest_emergence = max(emergence_events, key=lambda e: e.information_integration)
            collapse_data['emergence_guidance'] = {
                'phase_type': strongest_emergence.phase_type.value,
                'strength': strongest_emergence.information_integration,
                'coherence': strongest_emergence.coherence
            }
        
        # Generate bifractal structure with emergence guidance
        tree = self.bifractal_generator.generate_structure(collapse_data, context)
        
        # Native GAIA pattern amplification for symbolic enhancement
        if hasattr(collapse_data, 'patterns'):
            patterns = self.pattern_amplifier.identify_patterns(
                field_data=collapse_data,
                context={'tree_generation': True}
            )
            
            if patterns:
                amplification_results = self.pattern_amplifier.amplify_patterns(
                    patterns, mode=AmplificationMode.COGNITIVE
                )
                print(f"Applied GAIA pattern amplification to {len(amplification_results)} symbolic patterns")
        
        # Store active tree
        self.active_trees[tree.tree_id] = tree
        self.total_trees_created += 1
        self.total_nodes_created += len(tree.nodes)
        
        # Record in history
        self.tree_history.append({
            'tree_id': tree.tree_id,
            'creation_time': tree.creation_time,
            'initial_node_count': len(tree.nodes),
            'initial_entropy': tree.total_entropy
        })
        
        # Keep history bounded
        if len(self.tree_history) > 1000:
            self.tree_history.pop(0)
        
        return tree
    
    def access_symbol(self, tree_id: str, node_id: str) -> Optional[SymbolicNode]:
        """Access a specific symbol node, boosting its epistemic weight."""
        if tree_id in self.active_trees:
            tree = self.active_trees[tree_id]
            self.epistemic_pruner.access_node(tree, node_id)
            
            if node_id in tree.nodes:
                return tree.nodes[node_id]
        
        return None
    
    def maintain_structures(self):
        """Perform maintenance on all active symbolic structures."""
        trees_to_remove = []
        
        for tree_id, tree in self.active_trees.items():
            # Apply epistemic pruning
            pruned_tree = self.epistemic_pruner.prune_tree(tree)
            self.total_pruning_operations += 1
            
            # Remove trees that are too small
            if len(pruned_tree.nodes) < 2:
                trees_to_remove.append(tree_id)
            else:
                self.active_trees[tree_id] = pruned_tree
        
        # Remove empty trees
        for tree_id in trees_to_remove:
            del self.active_trees[tree_id]
    
    def get_semantic_summary(self) -> Dict[str, Any]:
        """Get summary of all symbolic structures."""
        total_nodes = sum(len(tree.nodes) for tree in self.active_trees.values())
        
        # Calculate type distribution
        type_counts = defaultdict(int)
        for tree in self.active_trees.values():
            for node in tree.nodes.values():
                type_counts[node.symbol_type.value] += 1
        
        # Calculate average metrics
        avg_depth = 0.0
        avg_coherence = 0.0
        avg_entropy = 0.0
        
        if self.active_trees:
            avg_depth = sum(tree.depth for tree in self.active_trees.values()) / len(self.active_trees)
            avg_coherence = sum(tree.semantic_coherence for tree in self.active_trees.values()) / len(self.active_trees)
            avg_entropy = sum(tree.total_entropy for tree in self.active_trees.values()) / len(self.active_trees)
        
        return {
            'active_trees': len(self.active_trees),
            'total_nodes': total_nodes,
            'total_trees_created': self.total_trees_created,
            'total_nodes_created': self.total_nodes_created,
            'total_pruning_operations': self.total_pruning_operations,
            'symbol_type_distribution': dict(type_counts),
            'average_tree_depth': avg_depth,
            'average_semantic_coherence': avg_coherence,
            'average_tree_entropy': avg_entropy
        }
    
    def find_symbolic_patterns(self, pattern_type: str = "similarity") -> List[Dict[str, Any]]:
        """Find patterns across symbolic structures."""
        patterns = []
        
        if pattern_type == "similarity":
            patterns = self._find_semantic_similarities()
        elif pattern_type == "structural":
            patterns = self._find_structural_patterns()
        elif pattern_type == "temporal":
            patterns = self._find_temporal_patterns()
        
        return patterns
    
    def _find_semantic_similarities(self) -> List[Dict[str, Any]]:
        """Find semantically similar nodes across trees."""
        similarities = []
        
        all_nodes = []
        for tree in self.active_trees.values():
            for node in tree.nodes.values():
                all_nodes.append((tree.tree_id, node))
        
        # Compare all pairs
        for i, (tree_id1, node1) in enumerate(all_nodes):
            for j, (tree_id2, node2) in enumerate(all_nodes[i+1:], i+1):
                if tree_id1 != tree_id2:  # Different trees
                    similarity = np.dot(node1.semantic_vector, node2.semantic_vector)
                    if similarity > 0.8:  # High similarity threshold
                        similarities.append({
                            'tree1': tree_id1,
                            'node1': node1.node_id,
                            'tree2': tree_id2,
                            'node2': node2.node_id,
                            'similarity': float(similarity),
                            'type': 'semantic'
                        })
        
        return similarities
    
    def _find_structural_patterns(self) -> List[Dict[str, Any]]:
        """Find structural patterns across trees."""
        patterns = []
        
        # Find trees with similar depth and branching patterns
        for tree1_id, tree1 in self.active_trees.items():
            for tree2_id, tree2 in self.active_trees.items():
                if tree1_id >= tree2_id:  # Avoid duplicates
                    continue
                
                # Compare structural properties
                depth_similarity = 1.0 - abs(tree1.depth - tree2.depth) / max(tree1.depth, tree2.depth, 1)
                node_count_similarity = 1.0 - abs(len(tree1.nodes) - len(tree2.nodes)) / max(len(tree1.nodes), len(tree2.nodes), 1)
                
                if depth_similarity > 0.8 and node_count_similarity > 0.8:
                    patterns.append({
                        'tree1': tree1_id,
                        'tree2': tree2_id,
                        'depth_similarity': depth_similarity,
                        'node_count_similarity': node_count_similarity,
                        'type': 'structural'
                    })
        
        return patterns
    
    def _find_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Find temporal patterns in tree creation."""
        patterns = []
        
        # Analyze creation time clustering
        creation_times = [entry['creation_time'] for entry in self.tree_history]
        if len(creation_times) > 2:
            time_intervals = [creation_times[i+1] - creation_times[i] for i in range(len(creation_times)-1)]
            avg_interval = sum(time_intervals) / len(time_intervals)
            
            # Find unusually rapid creation periods
            rapid_periods = 0
            for interval in time_intervals:
                if interval < avg_interval * 0.3:  # Much faster than average
                    rapid_periods += 1
            
            if rapid_periods > 0:
                patterns.append({
                    'type': 'temporal',
                    'pattern': 'rapid_creation',
                    'rapid_periods': rapid_periods,
                    'total_intervals': len(time_intervals),
                    'average_interval': avg_interval
                })
        
        return patterns
    
    def reset(self):
        """Reset symbolic crystallizer to initial state."""
        self.active_trees.clear()
        self.tree_history.clear()
        self.total_trees_created = 0
        self.total_nodes_created = 0
        self.total_pruning_operations = 0
    
    def _apply_pac_geometric_enhancement(self, tree: SymbolicTree, collapse_data: Dict[str, Any]) -> SymbolicTree:
        """Apply PAC geometric enhancement to symbolic tree structure."""
        try:
            # Extract geometric features from tree
            geometric_features = {
                'node_count': len(tree.nodes),
                'tree_depth': tree.max_depth,
                'branching_factor': len(tree.nodes) / max(tree.max_depth, 1),
                'entropy_signature': collapse_data.get('entropy_resolved', 0.5)
            }
            
            # Process through PAC geometric SEC
            enhanced_features = self.geometric_sec.enhance_structure(
                geometric_features,
                amplification_mode='consciousness_emergence'
            )
            
            # Apply enhancements to tree structure
            enhancement_factor = enhanced_features.get('structure_amplification', 1.0)
            if enhancement_factor > 1.2:
                # Amplify tree structure based on PAC enhancement
                enhanced_tree = self._amplify_tree_structure(tree, enhancement_factor)
                return enhanced_tree
            else:
                return tree
                
        except Exception as e:
            print(f"PAC geometric enhancement error: {e}")
            return tree
    
    def _amplify_tree_structure(self, tree: SymbolicTree, factor: float) -> SymbolicTree:
        """Amplify tree structure based on PAC enhancement factor."""
        # Create enhanced copy of tree
        enhanced_tree = SymbolicTree(
            tree_id=f"{tree.tree_id}_pac_enhanced",
            root_node=tree.root_node,
            nodes=tree.nodes.copy(),
            creation_time=time.time(),
            max_depth=tree.max_depth,
            semantic_coherence=min(tree.semantic_coherence * factor, 1.0),
            entropy_signature=tree.entropy_signature * factor
        )
        
        # Enhance symbolic strength of nodes
        for node in enhanced_tree.nodes:
            if hasattr(node, 'symbolic_strength'):
                node.symbolic_strength = min(node.symbolic_strength * factor, 1.0)
        
        return enhanced_tree

    def _calculate_med_limit(self, entropy_resolved: float) -> int:
        """Calculate Maximum Entropy Depth limit from information theory."""
        # MED = -log2(entropy) gives information-theoretic depth bound
        # This ensures trees don't exceed their information content
        
        # Clamp entropy to prevent log(0) and extreme values
        entropy_clamped = max(min(entropy_resolved, 0.99), 0.001)
        
        # Calculate MED limit
        med_limit = -math.log2(entropy_clamped)
        
        # Convert to integer depth with reasonable bounds
        depth_limit = max(2, min(int(med_limit), 12))  # Min 2, max 12 for memory
        
        return depth_limit

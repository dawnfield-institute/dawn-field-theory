"""
PAC Hierarchy Data Structures

Core classes for representing hierarchical structures with PAC conservation properties.
Supports trees and DAGs with ownership weights.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
import numpy as np
from collections import defaultdict


@dataclass
class PACNode:
    """
    Represents a node in PAC hierarchy.
    
    Attributes:
        id: Unique identifier
        value: Information/energy functional f(v)
        embedding: High-dimensional embedding vector e(v)
        parent: Reference to parent node (None for root)
        children: List of child nodes
        depth: Distance from root
        metadata: Optional additional data
        ownership_weights: For DAG support - weights from each parent
    """
    id: str
    value: float
    embedding: Optional[np.ndarray] = None
    parent: Optional['PACNode'] = None
    children: List['PACNode'] = field(default_factory=list)
    depth: int = 0
    metadata: Dict = field(default_factory=dict)
    ownership_weights: Dict[str, float] = field(default_factory=dict)  # parent_id -> weight
    
    def add_child(self, child: 'PACNode', ownership_weight: float = 1.0):
        """
        Add child node with optional ownership weight (for DAG support).
        
        Args:
            child: Child node to add
            ownership_weight: Weight α_{self→child} ∈ [0,1]
        """
        if child not in self.children:
            self.children.append(child)
        
        # Set depth
        if child.parent is None:
            child.parent = self
            child.depth = self.depth + 1
        else:
            # DAG case: multiple parents
            child.depth = min(child.depth, self.depth + 1)
        
        # Set ownership weight
        child.ownership_weights[self.id] = ownership_weight
    
    def pac_residual(self) -> float:
        """
        Compute PAC conservation residual: |f(P) - Σf(C)|
        
        For DAG nodes, uses ownership weights:
        |f(P) - Σᵢ α_{P→Cᵢ} · f(Cᵢ)|
        """
        if not self.children:
            return 0.0
        
        weighted_sum = sum(
            child.value * child.ownership_weights.get(self.id, 1.0)
            for child in self.children
        )
        
        return abs(self.value - weighted_sum)
    
    def distance_to(self, other: 'PACNode') -> float:
        """
        Compute Euclidean distance to another node.
        
        Args:
            other: Target node
            
        Returns:
            Euclidean distance ||e(self) - e(other)||₂
        """
        if self.embedding is None or other.embedding is None:
            raise ValueError(f"Embeddings not set for nodes {self.id} and/or {other.id}")
        
        return np.linalg.norm(self.embedding - other.embedding)
    
    def distance_residual(self) -> float:
        """
        Compute distance conservation residual:
        | ||e(P)||² - Σᵢ αᵢ·||e(Cᵢ)||² |
        """
        if not self.children or self.embedding is None:
            return 0.0
        
        parent_norm_sq = np.linalg.norm(self.embedding) ** 2
        
        children_norm_sq_sum = sum(
            (np.linalg.norm(child.embedding) ** 2) * 
            child.ownership_weights.get(self.id, 1.0)
            for child in self.children
            if child.embedding is not None
        )
        
        return abs(parent_norm_sq - children_norm_sq_sum)
    
    def __repr__(self) -> str:
        return f"PACNode(id={self.id}, value={self.value:.4f}, depth={self.depth}, children={len(self.children)})"


class PACHierarchy:
    """
    Manages PAC hierarchical structure (tree or DAG).
    
    Provides methods for:
    - Building hierarchies
    - Traversing levels
    - Computing global metrics
    - Validating PAC conservation
    """
    
    def __init__(self, root: PACNode):
        """
        Initialize hierarchy with root node.
        
        Args:
            root: Root node of hierarchy
        """
        self.root = root
        self.nodes: Dict[str, PACNode] = {root.id: root}
        self._levels_cache: Optional[List[List[PACNode]]] = None
    
    def add_node(self, node: PACNode, parent_id: str, ownership_weight: float = 1.0):
        """
        Add node to hierarchy under specified parent.
        
        Args:
            node: Node to add
            parent_id: ID of parent node
            ownership_weight: Ownership weight (for DAG support)
        """
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found")
        
        parent = self.nodes[parent_id]
        parent.add_child(node, ownership_weight)
        self.nodes[node.id] = node
        
        # Invalidate cache
        self._levels_cache = None
    
    def get_all_parents(self) -> List[PACNode]:
        """Get all nodes that have children."""
        return [n for n in self.nodes.values() if n.children]
    
    def get_level(self, depth: int) -> List[PACNode]:
        """Get all nodes at specified depth."""
        return [n for n in self.nodes.values() if n.depth == depth]
    
    def get_levels(self) -> List[List[PACNode]]:
        """
        Get nodes organized by depth level.
        
        Returns:
            List of lists, where result[k] contains all nodes at depth k
        """
        if self._levels_cache is not None:
            return self._levels_cache
        
        max_depth = max(node.depth for node in self.nodes.values())
        levels = [[] for _ in range(max_depth + 1)]
        
        for node in self.nodes.values():
            levels[node.depth].append(node)
        
        self._levels_cache = levels
        return levels
    
    def compute_global_pac_residual(self) -> float:
        """
        Compute total PAC conservation residual across all parents.
        
        Returns:
            Sum of |f(P) - Σf(C)| for all parent nodes
        """
        return sum(node.pac_residual() for node in self.get_all_parents())
    
    def compute_global_distance_residual(self) -> float:
        """
        Compute total distance conservation residual across all parents.
        
        Returns:
            Sum of distance residuals for all parent nodes
        """
        return sum(node.distance_residual() for node in self.get_all_parents())
    
    def validate_ownership_weights(self) -> Dict[str, bool]:
        """
        Validate that ownership weights sum to 1.0 for each child (DAG constraint).
        
        Returns:
            Dictionary mapping node_id -> is_valid
        """
        validation = {}
        
        for node in self.nodes.values():
            if node.ownership_weights:
                weight_sum = sum(node.ownership_weights.values())
                validation[node.id] = abs(weight_sum - 1.0) < 1e-6
            else:
                validation[node.id] = True  # No parents = valid
        
        return validation
    
    def get_max_depth(self) -> int:
        """Get maximum depth of hierarchy."""
        return max(node.depth for node in self.nodes.values())
    
    def get_branching_factors(self) -> List[int]:
        """Get branching factor (number of children) for all parent nodes."""
        return [len(node.children) for node in self.get_all_parents()]
    
    def to_dict(self) -> Dict:
        """
        Export hierarchy structure to dictionary format.
        
        Returns:
            Nested dictionary representation
        """
        def node_to_dict(node: PACNode) -> Dict:
            return {
                'id': node.id,
                'value': node.value,
                'depth': node.depth,
                'children': [node_to_dict(child) for child in node.children],
                'metadata': node.metadata
            }
        
        return node_to_dict(self.root)
    
    @classmethod
    def from_dict(cls, data: Dict, embeddings: Optional[Dict[str, np.ndarray]] = None) -> 'PACHierarchy':
        """
        Create hierarchy from dictionary structure.
        
        Args:
            data: Dictionary with structure {node_id: {'value': float, 'children': [...]}}
            embeddings: Optional pre-computed embeddings
        
        Returns:
            PACHierarchy instance
        """
        def build_node(node_id: str, node_data: Dict, depth: int = 0) -> PACNode:
            embedding = embeddings.get(node_id) if embeddings else None
            
            node = PACNode(
                id=node_id,
                value=node_data['value'],
                embedding=embedding,
                depth=depth
            )
            
            # Recursively build children
            if 'children' in node_data:
                for child_id in node_data['children']:
                    if isinstance(child_id, str):
                        # Simple child reference
                        child_data = data[child_id]
                        child = build_node(child_id, child_data, depth + 1)
                    else:
                        # Child is a dict
                        child = build_node(child_id['id'], child_id, depth + 1)
                    
                    node.add_child(child)
            
            return node
        
        # Assume first key is root
        root_id = list(data.keys())[0]
        root = build_node(root_id, data[root_id])
        
        hierarchy = cls(root)
        
        # Collect all nodes into hierarchy.nodes dict
        def collect_nodes(node: PACNode):
            hierarchy.nodes[node.id] = node
            for child in node.children:
                collect_nodes(child)
        
        collect_nodes(root)
        
        return hierarchy
    
    def __len__(self) -> int:
        """Return number of nodes in hierarchy."""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        return f"PACHierarchy(nodes={len(self.nodes)}, depth={self.get_max_depth()}, root={self.root.id})"

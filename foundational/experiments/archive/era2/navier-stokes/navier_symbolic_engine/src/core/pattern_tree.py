"""Pattern tree generation and management."""


from typing import List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class PatternNode:
    """
    Node in the fractal pattern tree representing a symbolic flow pattern.
    Includes scale, regime, entropy, memory trace, and pattern composition.
    """
    pattern_id: int
    parent: Optional['PatternNode'] = None
    children: List['PatternNode'] = field(default_factory=list)
    entropy_signature: Any = None
    regime: Optional[str] = None  # e.g., 'laminar', 'turbulent', etc.
    depth: int = 0
    scale: Optional[float] = None
    memory_trace: list = field(default_factory=list)
    pattern_data: Any = None  # e.g., velocity/pressure field, symbolic payload
    metadata: dict = field(default_factory=dict)

    def add_child(self, child: 'PatternNode'):
        self.children.append(child)
        child.parent = self

    def get_ancestry(self) -> list:
        """Return the ancestry (memory trace) of this node."""
        ancestry = []
        node = self
        while node:
            ancestry.append(node)
            node = node.parent
        return ancestry[::-1]


class PatternTree:
    """
    Fractal pattern tree for symbolic flow navigation.
    Supports recursive pattern generation, navigation, and composition.
    """
    def __init__(self, root: Optional[PatternNode] = None):
        self.root = root if root else PatternNode(pattern_id=0, depth=0)
        self.node_count = 1

    def add_pattern(self, parent: PatternNode, entropy_signature: Any, regime: str, scale: float = None, pattern_data: Any = None, metadata: dict = None) -> PatternNode:
        """
        Add a new pattern node as a child of the given parent.
        """
        node = PatternNode(
            pattern_id=self.node_count,
            parent=parent,
            entropy_signature=entropy_signature,
            regime=regime,
            depth=parent.depth + 1,
            scale=scale,
            pattern_data=pattern_data,
            memory_trace=parent.get_ancestry() + [parent],
            metadata=metadata or {}
        )
        parent.add_child(node)
        self.node_count += 1
        return node

    def traverse(self, node: Optional[PatternNode] = None, action=None):
        """
        Traverse the tree from the given node, applying action at each node.
        """
        if node is None:
            node = self.root
        if action:
            action(node)
        for child in node.children:
            self.traverse(child, action)

    def compose_pattern(self, node: PatternNode) -> Any:
        """
        Recursively compose the pattern data from this node and its ancestry.
        """
        # Placeholder: actual composition logic (e.g., sum, blend, etc.)
        data = []
        for ancestor in node.get_ancestry():
            if ancestor.pattern_data is not None:
                data.append(ancestor.pattern_data)
        return data

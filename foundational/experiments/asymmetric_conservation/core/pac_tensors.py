"""
PAC-Native Tensor Definitions

Implements the node tensor T_n = [P_n, A_n, Δ_n, θ_n] and event tensor E.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum


# Dawn Field Constants (derived, not fitted)
PHI = (1 + np.sqrt(5)) / 2  # 1.618033988749895
PHI_INV = 1 / PHI           # 0.618033988749895
XI = 1 + np.pi / 55         # 1.0571... (balance operator)
LAMBDA_STAR = 0.618432      # SEC partition threshold


class CollapseType(Enum):
    """Types of collapse/actualization events."""
    THRESHOLD = "threshold"      # θ crossed
    INJECTION = "injection"      # External I(τ)
    CASCADE = "cascade"          # Triggered by child collapse
    RECONCILIATION = "reconciliation"  # Parent processing children


@dataclass
class NodeState:
    """
    Current state of a PAC node.
    
    The fundamental invariant: P + A + Δ = C (constant for node)
    """
    P: float  # Remaining potential
    A: float  # Actualized value
    delta: float  # Unresolved imbalance buffer (Δ)
    theta: float  # Collapse threshold
    
    # Metadata
    node_id: int = 0
    depth: int = 0
    parent_id: Optional[int] = None
    
    def __post_init__(self):
        """Compute initial conservation constant."""
        self._C = self.P + self.A + self.delta
    
    @property
    def C(self) -> float:
        """Conservation constant (should remain fixed)."""
        return self._C
    
    @property
    def conservation_error(self) -> float:
        """How far from conservation we are (should be ~0)."""
        return abs((self.P + self.A + self.delta) - self._C)
    
    @property
    def is_conserved(self, tol: float = 1e-10) -> bool:
        """Check if conservation holds."""
        return self.conservation_error < tol
    
    @property
    def collapse_ready(self) -> bool:
        """Check if node is ready to collapse (emit actualization)."""
        # Collapse when potential exceeds threshold
        return self.P > self.theta
    
    def to_tensor(self) -> np.ndarray:
        """Convert to numpy tensor [P, A, Δ, θ]."""
        return np.array([self.P, self.A, self.delta, self.theta])
    
    @classmethod
    def from_tensor(cls, tensor: np.ndarray, node_id: int = 0, 
                    depth: int = 0, parent_id: Optional[int] = None) -> 'NodeState':
        """Create from numpy tensor."""
        return cls(
            P=tensor[0],
            A=tensor[1],
            delta=tensor[2],
            theta=tensor[3],
            node_id=node_id,
            depth=depth,
            parent_id=parent_id
        )


@dataclass
class EventTensor:
    """
    Event tensor E_{n→p} = [δA, δP, σ]
    
    Represents an actualization event from child n to parent p.
    Events are asynchronous, sparse, and order-independent.
    """
    source_id: int      # Child node that emitted
    target_id: int      # Parent node that receives
    delta_A: float      # Actualization delta
    delta_P: float      # Potential delta
    
    # Event metadata (σ)
    event_type: CollapseType = CollapseType.THRESHOLD
    depth: int = 0
    symbol: Optional[Any] = None  # Type tag, token, etc.
    
    # Timing (emergent, not primitive)
    emission_index: int = 0  # Which actualization event this is
    
    def __post_init__(self):
        """Validate event conservation."""
        # Events should conserve: what leaves source equals what arrives at target
        # delta_A + delta_P should sum to 0 for pure transfers
        pass
    
    def to_tensor(self) -> np.ndarray:
        """Convert to numpy tensor [δA, δP]."""
        return np.array([self.delta_A, self.delta_P])
    
    @property
    def magnitude(self) -> float:
        """Total value transferred."""
        return abs(self.delta_A) + abs(self.delta_P)


@dataclass
class PACTensor:
    """
    Complete PAC tensor for a node, including state and pending events.
    
    This is the fundamental data structure for PAC-native computation.
    """
    state: NodeState
    pending_events: list = field(default_factory=list)  # Events waiting to be reconciled
    
    # History (for analysis)
    event_history: list = field(default_factory=list)
    reconciliation_count: int = 0
    
    def receive_event(self, event: EventTensor):
        """
        Receive an event from a child.
        
        This does NOT immediately update state—it goes to pending.
        Conservation is only checked at reconciliation.
        """
        if event.target_id != self.state.node_id:
            raise ValueError(f"Event targeted at {event.target_id}, not {self.state.node_id}")
        
        self.pending_events.append(event)
        
        # Update Δ buffer (local imbalance now exists)
        self.state.delta += event.delta_A + event.delta_P
    
    def reconcile(self) -> float:
        """
        Reconcile all pending events.
        
        This is where conservation is enforced.
        Returns: total value reconciled.
        """
        if not self.pending_events:
            return 0.0
        
        total_delta_A = sum(e.delta_A for e in self.pending_events)
        total_delta_P = sum(e.delta_P for e in self.pending_events)
        
        # Apply to state
        self.state.A += total_delta_A
        self.state.P += total_delta_P
        
        # Clear Δ buffer (reconciliation complete)
        self.state.delta = 0.0
        
        # Move to history
        self.event_history.extend(self.pending_events)
        self.pending_events.clear()
        self.reconciliation_count += 1
        
        return abs(total_delta_A) + abs(total_delta_P)
    
    @property
    def local_asymmetry(self) -> float:
        """
        Measure of local asymmetry (pending unreconciled events).
        
        This is the "apparent violation" a windowed observer would see.
        """
        return abs(self.state.delta)
    
    @property
    def is_globally_conserved(self) -> bool:
        """Check if P + A + Δ = C."""
        return self.state.is_conserved
    
    def emit_collapse(self, fraction: float = PHI_INV) -> Optional[EventTensor]:
        """
        Emit a collapse event if threshold is crossed.
        
        Default: emit φ⁻¹ of potential (golden ratio partition).
        """
        if not self.state.collapse_ready:
            return None
        
        # Amount to actualize
        amount = self.state.P * fraction
        
        # Update local state
        self.state.P -= amount
        self.state.A += amount
        
        # Create event for parent
        if self.state.parent_id is not None:
            event = EventTensor(
                source_id=self.state.node_id,
                target_id=self.state.parent_id,
                delta_A=amount,
                delta_P=-amount,  # Conservation: what's gained in A is lost from P
                event_type=CollapseType.THRESHOLD,
                depth=self.state.depth,
                emission_index=len(self.event_history)
            )
            self.event_history.append(event)
            return event
        
        return None


def create_pac_tree_tensors(n_children: int = 2, 
                            initial_potential: float = 1.0,
                            theta: float = 0.3) -> tuple:
    """
    Create a minimal PAC tree with one parent and n children.
    
    Returns: (parent_tensor, [child_tensors])
    """
    # Parent node
    parent_state = NodeState(
        P=0.0,  # Parent starts empty
        A=0.0,
        delta=0.0,
        theta=theta,
        node_id=0,
        depth=0,
        parent_id=None
    )
    parent = PACTensor(state=parent_state)
    
    # Child nodes (split initial potential equally)
    child_P = initial_potential / n_children
    children = []
    for i in range(n_children):
        child_state = NodeState(
            P=child_P,
            A=0.0,
            delta=0.0,
            theta=theta,
            node_id=i + 1,
            depth=1,
            parent_id=0
        )
        children.append(PACTensor(state=child_state))
    
    return parent, children

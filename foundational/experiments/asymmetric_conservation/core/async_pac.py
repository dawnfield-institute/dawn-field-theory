"""
Async PAC Tree Implementation

A PAC tree that uses event-indexed execution, not global timesteps.
Designed for comparison with GAIA's synchronous PACTree.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

try:
    from .pac_tensors import (
        NodeState, EventTensor, PACTensor, CollapseType,
        PHI, PHI_INV, XI, LAMBDA_STAR
    )
    from .event_system import EventQueue, ReconciliationBoundary, Event, EventPriority
except ImportError:
    from pac_tensors import (
        NodeState, EventTensor, PACTensor, CollapseType,
        PHI, PHI_INV, XI, LAMBDA_STAR
    )
    from event_system import EventQueue, ReconciliationBoundary, Event, EventPriority


@dataclass
class AsyncPACNode:
    """
    A node in the async PAC tree.
    
    Key difference from GAIA's PACNode:
    - Carries full (P, A, Δ, θ) state
    - Processes events asynchronously
    - No iteration-synchronized updates
    """
    node_id: int
    parent_id: Optional[int]
    depth: int
    
    # PAC state
    P: float = 0.0      # Potential
    A: float = 0.0      # Actualized
    delta: float = 0.0  # Unresolved buffer
    theta: float = 0.3  # Collapse threshold
    
    # Delta from parent (for embedding reconstruction, like GAIA)
    embedding_delta: Optional[np.ndarray] = None
    
    # Children
    children: Dict[int, 'AsyncPACNode'] = field(default_factory=dict)
    
    # Token ID (if leaf)
    token_id: Optional[int] = None
    
    # Transition counts
    transition_counts: Dict[int, int] = field(default_factory=dict)
    
    # Event tracking
    pending_events: List[EventTensor] = field(default_factory=list)
    event_history: List[EventTensor] = field(default_factory=list)
    reconciliation_count: int = 0
    
    # Metadata
    access_count: int = 0
    crystallized: bool = False
    
    def __post_init__(self):
        """Initialize conservation constant."""
        self._C = self.P + self.A + self.delta
    
    @property
    def C(self) -> float:
        return self._C
    
    @property
    def local_asymmetry(self) -> float:
        return abs(self.delta)
    
    @property
    def is_conserved(self) -> bool:
        return abs((self.P + self.A + self.delta) - self._C) < 1e-10
    
    @property
    def collapse_ready(self) -> bool:
        return self.P > self.theta
    
    def receive_event(self, event: EventTensor):
        """Receive event into pending buffer."""
        self.pending_events.append(event)
        self.delta += event.delta_A + event.delta_P
    
    def reconcile(self) -> float:
        """Process all pending events."""
        if not self.pending_events:
            return 0.0
        
        total_A = sum(e.delta_A for e in self.pending_events)
        total_P = sum(e.delta_P for e in self.pending_events)
        
        self.A += total_A
        self.P += total_P
        self.delta = 0.0
        
        self.event_history.extend(self.pending_events)
        self.pending_events.clear()
        self.reconciliation_count += 1
        
        return abs(total_A) + abs(total_P)
    
    def emit_collapse(self, fraction: float = PHI_INV) -> Optional[EventTensor]:
        """Emit collapse event to parent."""
        if not self.collapse_ready:
            return None
        if self.parent_id is None:
            return None
        
        amount = self.P * fraction
        self.P -= amount
        self.A += amount
        
        event = EventTensor(
            source_id=self.node_id,
            target_id=self.parent_id,
            delta_A=amount,
            delta_P=-amount,
            event_type=CollapseType.THRESHOLD,
            depth=self.depth,
            emission_index=len(self.event_history)
        )
        self.event_history.append(event)
        return event
    
    def to_state(self) -> NodeState:
        """Convert to NodeState for analysis."""
        return NodeState(
            P=self.P,
            A=self.A,
            delta=self.delta,
            theta=self.theta,
            node_id=self.node_id,
            depth=self.depth,
            parent_id=self.parent_id
        )


class AsyncPACTree:
    """
    Async PAC Tree with event-indexed execution.
    
    Comparison to GAIA's PACTree:
    - GAIA: synchronous graft/learn calls, conservation per-call
    - Async: event-driven updates, conservation at reconciliation
    
    Key test: Do they produce equivalent final states?
    """
    
    def __init__(self, embed_dim: int = 768, theta: float = 0.3):
        self.embed_dim = embed_dim
        self.theta = theta
        
        # Root node
        self.root = AsyncPACNode(
            node_id=0,
            parent_id=None,
            depth=0,
            P=0.0,
            A=0.0,
            delta=0.0,
            theta=theta,
            embedding_delta=np.zeros(embed_dim)
        )
        
        # Node index
        self.nodes: Dict[int, AsyncPACNode] = {0: self.root}
        self.next_node_id = 1
        
        # Token mapping
        self.token_nodes: Dict[int, AsyncPACNode] = {}
        self.context_nodes: Dict[Tuple[int, ...], AsyncPACNode] = {}
        
        # Event system
        self.event_queue = EventQueue()
        self.boundary = ReconciliationBoundary(delta_threshold=XI)
        
        # Statistics
        self.stats = {
            'nodes_created': 1,
            'events_emitted': 0,
            'reconciliations': 0,
            'max_local_asymmetry': 0.0,
            'conservation_checks': 0,
            'conservation_violations': 0,
        }
        
        # History for analysis
        self.asymmetry_history: List[Tuple[int, float]] = []
        self.reconciliation_events: List[int] = []
    
    def graft_embeddings(self, embeddings: np.ndarray, vocab_size: int = None):
        """
        Graft embeddings as Level 0 nodes.
        
        Each embedding becomes potential at a child node.
        """
        if vocab_size is None:
            vocab_size = embeddings.shape[0]
        
        for token_id in range(vocab_size):
            # Embedding magnitude as initial potential
            initial_P = np.linalg.norm(embeddings[token_id])
            
            node = AsyncPACNode(
                node_id=self.next_node_id,
                parent_id=0,
                depth=1,
                P=initial_P,
                A=0.0,
                delta=0.0,
                theta=self.theta,
                embedding_delta=embeddings[token_id].copy(),
                token_id=token_id
            )
            
            self.nodes[self.next_node_id] = node
            self.token_nodes[token_id] = node
            self.root.children[token_id] = node
            self.next_node_id += 1
        
        self.stats['nodes_created'] += vocab_size
    
    def inject_potential(self, node_id: int, amount: float):
        """Inject external potential (staggered injection)."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.nodes[node_id]
        node.P += amount
        node._C += amount
        
        # Track injection event
        event = EventTensor(
            source_id=-1,
            target_id=node_id,
            delta_A=0.0,
            delta_P=amount,
            event_type=CollapseType.INJECTION,
            depth=node.depth
        )
        node.event_history.append(event)
    
    def step_async(self) -> int:
        """
        Run one async step: check all nodes for collapse, process events.
        
        Returns: number of events processed
        """
        # Emit collapse events from ready nodes
        for node in self.nodes.values():
            if node.collapse_ready and node.parent_id is not None:
                event = node.emit_collapse()
                if event:
                    self.event_queue.push(Event.from_tensor(event))
                    self.stats['events_emitted'] += 1
        
        # Process events
        count = 0
        while not self.event_queue.is_empty():
            event = self.event_queue.pop()
            target = self.nodes.get(event.tensor.target_id)
            if target:
                target.receive_event(event.tensor)
                count += 1
                
                # Track asymmetry
                max_asym = max(n.local_asymmetry for n in self.nodes.values())
                self.stats['max_local_asymmetry'] = max(
                    self.stats['max_local_asymmetry'], max_asym
                )
                self.asymmetry_history.append((self.stats['events_emitted'], max_asym))
                
                # Check reconciliation
                if self.boundary.should_reconcile(
                    PACTensor(state=target.to_state(), pending_events=target.pending_events)
                ):
                    target.reconcile()
                    self.stats['reconciliations'] += 1
                    self.reconciliation_events.append(self.stats['events_emitted'])
        
        return count
    
    def run_until_stable(self, max_steps: int = 1000) -> int:
        """Run until no more collapse events."""
        total = 0
        for _ in range(max_steps):
            count = self.step_async()
            total += count
            if count == 0:
                # Check if any node is ready
                any_ready = any(n.collapse_ready and n.parent_id is not None 
                               for n in self.nodes.values())
                if not any_ready:
                    break
        return total
    
    def force_reconcile_all(self):
        """Force reconciliation at all nodes (for comparison)."""
        for node in self.nodes.values():
            if node.pending_events:
                node.reconcile()
                self.stats['reconciliations'] += 1
    
    def check_global_conservation(self) -> Dict:
        """Check conservation across entire tree."""
        total_P = sum(n.P for n in self.nodes.values())
        total_A = sum(n.A for n in self.nodes.values())
        total_delta = sum(n.delta for n in self.nodes.values())
        total_C = sum(n.C for n in self.nodes.values())
        
        error = abs((total_P + total_A + total_delta) - total_C)
        
        self.stats['conservation_checks'] += 1
        if error > 1e-10:
            self.stats['conservation_violations'] += 1
        
        return {
            'total_P': total_P,
            'total_A': total_A,
            'total_delta': total_delta,
            'total_C': total_C,
            'conservation_error': error,
            'is_conserved': error < 1e-10,
            'local_asymmetry': total_delta,
        }
    
    def estimate_xi_from_reconciliation(self) -> Optional[float]:
        """Estimate Ξ from reconciliation timing."""
        if len(self.reconciliation_events) < 3:
            return None
        
        intervals = np.diff(self.reconciliation_events)
        if len(intervals) == 0:
            return None
        
        mean = np.mean(intervals)
        std = np.std(intervals)
        
        if std > 0 and mean > 0:
            # Test if 1 + π/mean ≈ Ξ
            estimate = 1 + np.pi / (mean if mean > 1 else 55)
            return estimate
        
        return None
    
    def compare_with_sync(self, sync_state: Dict) -> Dict:
        """
        Compare async state with synchronous reference.
        
        sync_state: Dict with 'total_P', 'total_A' from sync execution
        """
        async_status = self.check_global_conservation()
        
        # Force reconcile to get comparable state
        self.force_reconcile_all()
        final_status = self.check_global_conservation()
        
        return {
            'sync_P': sync_state.get('total_P', 0),
            'sync_A': sync_state.get('total_A', 0),
            'async_P_before_reconcile': async_status['total_P'],
            'async_A_before_reconcile': async_status['total_A'],
            'async_delta_before': async_status['total_delta'],
            'async_P_after_reconcile': final_status['total_P'],
            'async_A_after_reconcile': final_status['total_A'],
            'P_difference': abs(sync_state.get('total_P', 0) - final_status['total_P']),
            'A_difference': abs(sync_state.get('total_A', 0) - final_status['total_A']),
            'equivalent': (
                abs(sync_state.get('total_P', 0) - final_status['total_P']) < 1e-10 and
                abs(sync_state.get('total_A', 0) - final_status['total_A']) < 1e-10
            ),
        }

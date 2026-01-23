"""
Event System for PAC-Native Execution

Implements event queuing, reconciliation boundaries, and async execution.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from enum import Enum
import heapq
from collections import defaultdict

try:
    from .pac_tensors import EventTensor, PACTensor, CollapseType, XI
except ImportError:
    from pac_tensors import EventTensor, PACTensor, CollapseType, XI


class EventPriority(Enum):
    """Priority levels for event processing."""
    IMMEDIATE = 0    # Process now (cascade triggers)
    NORMAL = 1       # Standard collapse events
    DEFERRED = 2     # Batch at reconciliation boundary
    BACKGROUND = 3   # Low priority (monitoring, stats)


@dataclass(order=True)
class Event:
    """
    Wrapper for events with priority and timing.
    
    Events are ordered by (priority, emission_index) for processing.
    """
    priority: int
    emission_index: int
    tensor: EventTensor = field(compare=False)
    
    @classmethod
    def from_tensor(cls, tensor: EventTensor, 
                    priority: EventPriority = EventPriority.NORMAL) -> 'Event':
        return cls(
            priority=priority.value,
            emission_index=tensor.emission_index,
            tensor=tensor
        )


class EventQueue:
    """
    Priority queue for PAC events.
    
    Enables asynchronous, event-driven execution without global clock.
    """
    
    def __init__(self):
        self._heap: List[Event] = []
        self._event_count = 0
        self._processed_count = 0
        
        # Statistics
        self.stats = {
            'events_emitted': 0,
            'events_processed': 0,
            'reconciliations': 0,
            'max_queue_depth': 0,
            'total_value_transferred': 0.0,
        }
    
    def push(self, event: Event):
        """Add event to queue."""
        heapq.heappush(self._heap, event)
        self._event_count += 1
        self.stats['events_emitted'] += 1
        self.stats['max_queue_depth'] = max(
            self.stats['max_queue_depth'], 
            len(self._heap)
        )
    
    def pop(self) -> Optional[Event]:
        """Remove and return highest-priority event."""
        if not self._heap:
            return None
        event = heapq.heappop(self._heap)
        self._processed_count += 1
        self.stats['events_processed'] += 1
        self.stats['total_value_transferred'] += event.tensor.magnitude
        return event
    
    def peek(self) -> Optional[Event]:
        """Look at highest-priority event without removing."""
        return self._heap[0] if self._heap else None
    
    def __len__(self) -> int:
        return len(self._heap)
    
    def is_empty(self) -> bool:
        return len(self._heap) == 0
    
    def clear(self):
        """Clear all pending events."""
        self._heap.clear()


class ReconciliationBoundary:
    """
    Defines when reconciliation should occur.
    
    Options:
    - Threshold-based: when Δ exceeds threshold
    - Count-based: after N events
    - Value-based: after total value V transferred
    - Hybrid: any of the above
    """
    
    def __init__(self, 
                 delta_threshold: float = XI,  # Default: Ξ
                 event_count_threshold: int = None,
                 value_threshold: float = None):
        self.delta_threshold = delta_threshold
        self.event_count_threshold = event_count_threshold
        self.value_threshold = value_threshold
        
        # Tracking
        self._event_count = 0
        self._total_value = 0.0
    
    def should_reconcile(self, tensor: PACTensor) -> bool:
        """Check if reconciliation should trigger."""
        # Delta threshold
        if self.delta_threshold is not None:
            if tensor.local_asymmetry > self.delta_threshold:
                return True
        
        # Event count threshold
        if self.event_count_threshold is not None:
            if len(tensor.pending_events) >= self.event_count_threshold:
                return True
        
        # Value threshold
        if self.value_threshold is not None:
            pending_value = sum(e.magnitude for e in tensor.pending_events)
            if pending_value >= self.value_threshold:
                return True
        
        return False
    
    def record_event(self, event: EventTensor):
        """Track event for count/value thresholds."""
        self._event_count += 1
        self._total_value += event.magnitude
    
    def reset(self):
        """Reset counters after reconciliation."""
        self._event_count = 0
        self._total_value = 0.0


class AsyncExecutor:
    """
    Asynchronous executor for PAC systems.
    
    No global clock—events drive state updates.
    Conservation checked only at reconciliation boundaries.
    """
    
    def __init__(self, boundary: ReconciliationBoundary = None):
        self.queue = EventQueue()
        self.boundary = boundary or ReconciliationBoundary()
        self.nodes: Dict[int, PACTensor] = {}
        
        # History for analysis
        self.reconciliation_times: List[int] = []  # Event indices
        self.delta_history: List[Tuple[int, float]] = []  # (event_idx, max_delta)
        self.xi_measurements: List[float] = []  # Ξ estimates
        
        self._global_event_index = 0
    
    def register_node(self, tensor: PACTensor):
        """Register a node for event processing."""
        self.nodes[tensor.state.node_id] = tensor
    
    def emit_event(self, source: PACTensor, 
                   priority: EventPriority = EventPriority.NORMAL) -> bool:
        """
        Have a node emit a collapse event if ready.
        
        Returns: True if event was emitted.
        """
        event_tensor = source.emit_collapse()
        if event_tensor is None:
            return False
        
        event_tensor.emission_index = self._global_event_index
        self._global_event_index += 1
        
        event = Event.from_tensor(event_tensor, priority)
        self.queue.push(event)
        
        return True
    
    def inject(self, target_id: int, amount: float):
        """
        Inject potential into a node (external I(τ)).
        
        This is the "staggered injection" from the document.
        """
        if target_id not in self.nodes:
            raise ValueError(f"Node {target_id} not registered")
        
        node = self.nodes[target_id]
        node.state.P += amount
        node.state._C += amount  # Injection increases total conservation constant
        
        # Create injection event for tracking
        event = EventTensor(
            source_id=-1,  # External source
            target_id=target_id,
            delta_A=0.0,
            delta_P=amount,
            event_type=CollapseType.INJECTION,
            depth=node.state.depth,
            emission_index=self._global_event_index
        )
        self._global_event_index += 1
        node.event_history.append(event)
    
    def process_one(self) -> bool:
        """
        Process a single event.
        
        Returns: True if event was processed, False if queue empty.
        """
        event = self.queue.pop()
        if event is None:
            return False
        
        target_id = event.tensor.target_id
        if target_id not in self.nodes:
            return True  # Skip events to unregistered nodes
        
        target = self.nodes[target_id]
        target.receive_event(event.tensor)
        self.boundary.record_event(event.tensor)
        
        # Track delta history
        max_delta = max(n.local_asymmetry for n in self.nodes.values())
        self.delta_history.append((self._global_event_index, max_delta))
        
        # Check reconciliation boundary
        if self.boundary.should_reconcile(target):
            self._do_reconciliation(target)
        
        return True
    
    def _do_reconciliation(self, node: PACTensor):
        """Perform reconciliation and record statistics."""
        value = node.reconcile()
        self.reconciliation_times.append(self._global_event_index)
        self.boundary.reset()
        
        # Estimate Ξ from reconciliation intervals
        if len(self.reconciliation_times) >= 2:
            intervals = np.diff(self.reconciliation_times[-10:])  # Last 10
            if len(intervals) > 0:
                mean_interval = np.mean(intervals)
                # Ξ estimate: ratio of interval to some base unit
                # This is exploratory—we're testing if Ξ emerges
                if mean_interval > 0:
                    self.xi_measurements.append(mean_interval)
    
    def run_until_empty(self, max_events: int = 10000) -> int:
        """
        Process all events until queue is empty.
        
        Returns: number of events processed.
        """
        count = 0
        while not self.queue.is_empty() and count < max_events:
            if self.process_one():
                count += 1
        return count
    
    def run_with_injection(self, injection_schedule: List[Tuple[int, int, float]],
                           max_events: int = 10000) -> int:
        """
        Run with scheduled injections.
        
        injection_schedule: List of (event_index, node_id, amount)
        """
        schedule = sorted(injection_schedule)
        schedule_idx = 0
        count = 0
        
        while count < max_events:
            # Check for scheduled injections
            while schedule_idx < len(schedule):
                trigger_idx, node_id, amount = schedule[schedule_idx]
                if trigger_idx <= self._global_event_index:
                    self.inject(node_id, amount)
                    schedule_idx += 1
                else:
                    break
            
            # Process next event
            if not self.queue.is_empty():
                self.process_one()
                count += 1
            elif schedule_idx >= len(schedule):
                break  # No more events or injections
            else:
                # No events but injections pending—trigger collapse checks
                for node in self.nodes.values():
                    self.emit_event(node)
                if self.queue.is_empty():
                    self._global_event_index += 1  # Advance to next injection
        
        return count
    
    def get_conservation_status(self) -> Dict:
        """Check conservation across all nodes."""
        total_P = sum(n.state.P for n in self.nodes.values())
        total_A = sum(n.state.A for n in self.nodes.values())
        total_delta = sum(n.state.delta for n in self.nodes.values())
        total_C = sum(n.state.C for n in self.nodes.values())
        
        return {
            'total_P': total_P,
            'total_A': total_A,
            'total_delta': total_delta,
            'total_C': total_C,
            'conservation_error': abs((total_P + total_A + total_delta) - total_C),
            'local_asymmetry': total_delta,
            'is_conserved': abs((total_P + total_A + total_delta) - total_C) < 1e-10,
        }
    
    def estimate_xi(self) -> Optional[float]:
        """
        Estimate Ξ from reconciliation delay distribution.
        
        Returns None if insufficient data.
        """
        if len(self.reconciliation_times) < 3:
            return None
        
        intervals = np.diff(self.reconciliation_times)
        mean = np.mean(intervals)
        std = np.std(intervals)
        
        # Various Ξ estimates
        if std > 0:
            cv = mean / std  # Coefficient of variation inverse
            return 1 + cv / 55  # Analogous to 1 + π/55
        
        return mean  # Fallback

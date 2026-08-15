"""
Asymmetric Conservation Core Library

PAC-native tensor architecture with event-indexed execution.
"""

try:
    from .pac_tensors import PACTensor, EventTensor, NodeState
    from .event_system import EventQueue, ReconciliationBoundary, Event
    from .async_pac import AsyncPACTree, AsyncPACNode
except ImportError:
    from pac_tensors import PACTensor, EventTensor, NodeState
    from event_system import EventQueue, ReconciliationBoundary, Event
    from async_pac import AsyncPACTree, AsyncPACNode

__all__ = [
    'PACTensor',
    'EventTensor',
    'NodeState',
    'EventQueue',
    'ReconciliationBoundary',
    'Event',
    'AsyncPACTree',
    'AsyncPACNode',
]

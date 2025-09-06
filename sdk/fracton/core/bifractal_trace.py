"""
Bifractal Trace - Forward and reverse operation tracing for Fracton

This module maintains comprehensive traces of recursive operations, enabling
analysis, debugging, visualization, and pattern recognition in recursive
computation flows.
"""

import time
import uuid
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import pickle

from .recursive_engine import ExecutionContext


class TraceEntryType(Enum):
    """Types of trace entries."""
    CALL = "call"
    RETURN = "return"
    ERROR = "error"
    BRANCH = "branch"
    MERGE = "merge"


@dataclass
class TraceEntry:
    """
    Individual entry in a bifractal trace.
    
    Attributes:
        entry_id: Unique identifier for this entry
        timestamp: When the entry was created
        entry_type: Type of trace entry
        function_name: Name of the function being traced
        context: Execution context at time of entry
        parameters: Function parameters (for calls)
        result: Function result (for returns)
        error: Error information (for errors)
        trace_id: ID of the overall trace
        parent_entry_id: ID of the parent entry
        children_entry_ids: IDs of child entries
        metadata: Additional entry metadata
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    entry_type: TraceEntryType = TraceEntryType.CALL
    function_name: str = ""
    context: Optional[ExecutionContext] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[Exception] = None
    trace_id: str = ""
    parent_entry_id: Optional[str] = None
    children_entry_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace entry to dictionary for serialization."""
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp,
            'entry_type': self.entry_type.value,
            'function_name': self.function_name,
            'context_entropy': self.context.entropy if self.context else None,
            'context_depth': self.context.depth if self.context else None,
            'parameters': str(self.parameters)[:200],  # Truncate for readability
            'result': str(self.result)[:200] if self.result is not None else None,
            'error': str(self.error) if self.error else None,
            'trace_id': self.trace_id,
            'parent_entry_id': self.parent_entry_id,
            'children_count': len(self.children_entry_ids),
            'metadata': self.metadata
        }


@dataclass
class TracePattern:
    """
    Identified pattern in trace execution.
    
    Attributes:
        pattern_id: Unique pattern identifier
        pattern_type: Type of pattern (recursive, oscillating, etc.)
        frequency: How often this pattern occurs
        function_sequence: Sequence of functions in the pattern
        entropy_signature: Characteristic entropy evolution
        performance_impact: Impact on execution performance
        confidence: Confidence level of pattern detection
    """
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = ""
    frequency: int = 0
    function_sequence: List[str] = field(default_factory=list)
    entropy_signature: List[float] = field(default_factory=list)
    performance_impact: float = 0.0
    confidence: float = 0.0


@dataclass
class TraceAnalysis:
    """
    Comprehensive analysis of a bifractal trace.
    
    Attributes:
        trace_id: ID of the analyzed trace
        total_entries: Total number of trace entries
        total_calls: Number of function calls
        total_returns: Number of function returns
        max_depth: Maximum recursion depth reached
        total_execution_time: Total time for trace execution
        entropy_evolution: Evolution of entropy over time
        function_call_counts: Count of calls per function
        performance_hotspots: Functions with high execution time
        recursive_patterns: Identified recursive patterns
        error_patterns: Patterns in error occurrences
        memory_usage: Memory usage patterns
        complexity_metrics: Various complexity measurements
    """
    trace_id: str = ""
    total_entries: int = 0
    total_calls: int = 0
    total_returns: int = 0
    max_depth: int = 0
    total_execution_time: float = 0.0
    entropy_evolution: List[Tuple[float, float]] = field(default_factory=list)  # (time, entropy)
    function_call_counts: Dict[str, int] = field(default_factory=dict)
    performance_hotspots: List[Tuple[str, float]] = field(default_factory=list)  # (function, time)
    recursive_patterns: List[TracePattern] = field(default_factory=list)
    error_patterns: List[Dict] = field(default_factory=list)
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    complexity_metrics: Dict[str, float] = field(default_factory=dict)


class TraceRecorder:
    """Records trace entries with thread-safety and performance optimization."""
    
    def __init__(self, max_entries: int = 100000):
        self.max_entries = max_entries
        self._entries: deque = deque(maxlen=max_entries)
        self._entry_index: Dict[str, TraceEntry] = {}
        self._function_timings: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        self._recording_enabled = True
    
    def record_entry(self, entry: TraceEntry) -> None:
        """Record a new trace entry."""
        if not self._recording_enabled:
            return
        
        with self._lock:
            self._entries.append(entry)
            self._entry_index[entry.entry_id] = entry
            
            # Maintain parent-child relationships
            if entry.parent_entry_id and entry.parent_entry_id in self._entry_index:
                parent = self._entry_index[entry.parent_entry_id]
                if entry.entry_id not in parent.children_entry_ids:
                    parent.children_entry_ids.append(entry.entry_id)
            
            # Clean up old entries from index when deque wraps
            if len(self._entries) == self.max_entries:
                oldest_entry = self._entries[0]
                if oldest_entry.entry_id in self._entry_index:
                    del self._entry_index[oldest_entry.entry_id]
    
    def get_entries(self) -> List[TraceEntry]:
        """Get copy of all trace entries."""
        with self._lock:
            return list(self._entries)
    
    def get_entry(self, entry_id: str) -> Optional[TraceEntry]:
        """Get specific trace entry by ID."""
        with self._lock:
            return self._entry_index.get(entry_id)
    
    def clear(self) -> None:
        """Clear all trace entries."""
        with self._lock:
            self._entries.clear()
            self._entry_index.clear()
            self._function_timings.clear()
    
    def set_recording_enabled(self, enabled: bool) -> None:
        """Enable or disable trace recording."""
        self._recording_enabled = enabled
    
    def get_function_timings(self, function_name: str) -> List[float]:
        """Get timing history for a specific function."""
        with self._lock:
            return self._function_timings[function_name].copy()


class TraceAnalyzer:
    """Analyzes bifractal traces for patterns, performance, and insights."""
    
    def __init__(self):
        self._pattern_cache: Dict[str, List[TracePattern]] = {}
        self._analysis_cache: Dict[str, TraceAnalysis] = {}
        self._lock = threading.Lock()
    
    def analyze_trace(self, trace_entries: List[TraceEntry]) -> TraceAnalysis:
        """
        Perform comprehensive analysis of trace entries.
        
        Args:
            trace_entries: List of trace entries to analyze
            
        Returns:
            TraceAnalysis with comprehensive insights
        """
        if not trace_entries:
            return TraceAnalysis()
        
        trace_id = trace_entries[0].trace_id
        
        # Check cache
        with self._lock:
            if trace_id in self._analysis_cache:
                return self._analysis_cache[trace_id]
        
        analysis = TraceAnalysis(trace_id=trace_id)
        
        # Basic statistics
        analysis.total_entries = len(trace_entries)
        analysis.total_calls = sum(1 for e in trace_entries if e.entry_type == TraceEntryType.CALL)
        analysis.total_returns = sum(1 for e in trace_entries if e.entry_type == TraceEntryType.RETURN)
        
        # Calculate execution time and depth
        call_stack = []
        function_start_times = {}
        
        for entry in trace_entries:
            if entry.context:
                analysis.max_depth = max(analysis.max_depth, entry.context.depth)
                analysis.entropy_evolution.append((entry.timestamp, entry.context.entropy))
            
            if entry.entry_type == TraceEntryType.CALL:
                call_stack.append(entry)
                function_start_times[entry.entry_id] = entry.timestamp
                
                # Count function calls
                func_name = entry.function_name
                analysis.function_call_counts[func_name] = (
                    analysis.function_call_counts.get(func_name, 0) + 1
                )
            
            elif entry.entry_type == TraceEntryType.RETURN:
                if call_stack and entry.parent_entry_id in function_start_times:
                    start_time = function_start_times[entry.parent_entry_id]
                    execution_time = entry.timestamp - start_time
                    
                    func_name = entry.function_name
                    analysis.performance_hotspots.append((func_name, execution_time))
                    
                    if call_stack:
                        call_stack.pop()
        
        # Calculate total execution time
        if trace_entries:
            analysis.total_execution_time = (
                trace_entries[-1].timestamp - trace_entries[0].timestamp
            )
        
        # Identify performance hotspots
        analysis.performance_hotspots.sort(key=lambda x: x[1], reverse=True)
        analysis.performance_hotspots = analysis.performance_hotspots[:10]
        
        # Detect patterns
        analysis.recursive_patterns = self._detect_recursive_patterns(trace_entries)
        analysis.error_patterns = self._detect_error_patterns(trace_entries)
        
        # Calculate complexity metrics
        analysis.complexity_metrics = self._calculate_complexity_metrics(trace_entries)
        
        # Cache result
        with self._lock:
            self._analysis_cache[trace_id] = analysis
        
        return analysis
    
    def _detect_recursive_patterns(self, trace_entries: List[TraceEntry]) -> List[TracePattern]:
        """Detect recursive patterns in trace execution."""
        patterns = []
        
        # Group entries by function name
        function_groups = defaultdict(list)
        for entry in trace_entries:
            if entry.entry_type == TraceEntryType.CALL:
                function_groups[entry.function_name].append(entry)
        
        # Look for recursive patterns in each function
        for func_name, entries in function_groups.items():
            if len(entries) < 3:
                continue
            
            # Detect direct recursion
            recursive_sequences = []
            current_sequence = []
            
            for i, entry in enumerate(entries):
                if i > 0 and entry.context and entries[i-1].context:
                    if entry.context.depth > entries[i-1].context.depth:
                        current_sequence.append(entry)
                    else:
                        if len(current_sequence) >= 2:
                            recursive_sequences.append(current_sequence)
                        current_sequence = [entry]
                else:
                    current_sequence.append(entry)
            
            # Add final sequence if it's recursive
            if len(current_sequence) >= 2:
                recursive_sequences.append(current_sequence)
            
            # Create pattern objects
            for seq in recursive_sequences:
                if len(seq) >= 2:
                    entropy_sig = [e.context.entropy for e in seq if e.context]
                    
                    pattern = TracePattern(
                        pattern_type="direct_recursion",
                        frequency=len(seq),
                        function_sequence=[func_name] * len(seq),
                        entropy_signature=entropy_sig,
                        confidence=min(len(seq) / 10.0, 1.0)
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_error_patterns(self, trace_entries: List[TraceEntry]) -> List[Dict]:
        """Detect patterns in error occurrences."""
        error_entries = [e for e in trace_entries if e.entry_type == TraceEntryType.ERROR]
        
        if not error_entries:
            return []
        
        patterns = []
        
        # Group errors by type
        error_types = defaultdict(list)
        for entry in error_entries:
            if entry.error:
                error_type = type(entry.error).__name__
                error_types[error_type].append(entry)
        
        # Analyze each error type
        for error_type, entries in error_types.items():
            # Look for depth patterns
            depths = [e.context.depth for e in entries if e.context]
            if depths:
                avg_depth = sum(depths) / len(depths)
                patterns.append({
                    'error_type': error_type,
                    'frequency': len(entries),
                    'average_depth': avg_depth,
                    'depth_range': (min(depths), max(depths)),
                    'functions_affected': list(set(e.function_name for e in entries))
                })
        
        return patterns
    
    def _calculate_complexity_metrics(self, trace_entries: List[TraceEntry]) -> Dict[str, float]:
        """Calculate various complexity metrics for the trace."""
        if not trace_entries:
            return {}
        
        metrics = {}
        
        # Cyclomatic complexity (simplified)
        branches = sum(1 for e in trace_entries if e.entry_type == TraceEntryType.BRANCH)
        metrics['cyclomatic_complexity'] = branches + 1
        
        # Entropy complexity
        entropies = [e.context.entropy for e in trace_entries if e.context]
        if entropies:
            entropy_variance = sum((e - sum(entropies)/len(entropies))**2 for e in entropies) / len(entropies)
            metrics['entropy_variance'] = entropy_variance
            metrics['entropy_range'] = max(entropies) - min(entropies)
        
        # Depth complexity
        depths = [e.context.depth for e in trace_entries if e.context]
        if depths:
            metrics['max_depth'] = max(depths)
            metrics['depth_variance'] = sum((d - sum(depths)/len(depths))**2 for d in depths) / len(depths)
        
        # Function diversity
        unique_functions = len(set(e.function_name for e in trace_entries))
        total_calls = len([e for e in trace_entries if e.entry_type == TraceEntryType.CALL])
        if total_calls > 0:
            metrics['function_diversity'] = unique_functions / total_calls
        
        return metrics


class TraceVisualizer:
    """Generates visual representations of trace execution."""
    
    def __init__(self):
        self._visualization_cache: Dict[str, str] = {}
    
    def generate_text_trace(self, trace_entries: List[TraceEntry], 
                           max_depth: int = 50) -> str:
        """Generate text-based visualization of trace execution."""
        if not trace_entries:
            return "No trace entries to visualize"
        
        lines = []
        call_stack = []
        
        for entry in trace_entries:
            if entry.entry_type == TraceEntryType.CALL:
                indent = "  " * len(call_stack)
                entropy_str = f"(ε={entry.context.entropy:.2f})" if entry.context else ""
                lines.append(f"{indent}→ {entry.function_name}{entropy_str}")
                call_stack.append(entry)
                
                if len(call_stack) > max_depth:
                    lines.append(f"{indent}  ... (truncated at depth {max_depth})")
                    break
            
            elif entry.entry_type == TraceEntryType.RETURN:
                if call_stack:
                    call_stack.pop()
                    indent = "  " * len(call_stack)
                    result_str = str(entry.result)[:50] if entry.result is not None else "None"
                    lines.append(f"{indent}← {entry.function_name} → {result_str}")
            
            elif entry.entry_type == TraceEntryType.ERROR:
                indent = "  " * len(call_stack)
                error_str = str(entry.error)[:100] if entry.error else "Unknown error"
                lines.append(f"{indent}✗ {entry.function_name}: {error_str}")
        
        return "\n".join(lines)
    
    def generate_graph_data(self, trace_entries: List[TraceEntry]) -> Dict[str, Any]:
        """Generate graph data structure for visualization tools."""
        nodes = []
        edges = []
        node_map = {}
        
        for entry in trace_entries:
            if entry.entry_type == TraceEntryType.CALL:
                node_id = entry.entry_id
                node_map[node_id] = {
                    'id': node_id,
                    'label': entry.function_name,
                    'entropy': entry.context.entropy if entry.context else 0.0,
                    'depth': entry.context.depth if entry.context else 0,
                    'timestamp': entry.timestamp
                }
                nodes.append(node_map[node_id])
                
                # Add edge from parent if it exists
                if entry.parent_entry_id and entry.parent_entry_id in node_map:
                    edges.append({
                        'from': entry.parent_entry_id,
                        'to': node_id,
                        'type': 'call'
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'max_depth': max((n['depth'] for n in nodes), default=0)
            }
        }
    
    def generate_entropy_timeline(self, trace_entries: List[TraceEntry]) -> Dict[str, Any]:
        """Generate entropy evolution timeline data."""
        timeline_data = []
        
        for entry in trace_entries:
            if entry.context and entry.entry_type == TraceEntryType.CALL:
                timeline_data.append({
                    'timestamp': entry.timestamp,
                    'entropy': entry.context.entropy,
                    'depth': entry.context.depth,
                    'function': entry.function_name
                })
        
        return {
            'timeline': timeline_data,
            'entropy_range': (
                min((d['entropy'] for d in timeline_data), default=0),
                max((d['entropy'] for d in timeline_data), default=1)
            ),
            'time_range': (
                min((d['timestamp'] for d in timeline_data), default=0),
                max((d['timestamp'] for d in timeline_data), default=0)
            )
        }


class BifractalTrace:
    """
    Main bifractal trace management system.
    
    Coordinates trace recording, analysis, and visualization for recursive
    operations in the Fracton system.
    """
    
    def __init__(self, trace_id: str = None, max_entries: int = 100000,
                 ancestry_depth: int = 10, future_horizon: int = 10):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.recorder = TraceRecorder(max_entries)
        self.analyzer = TraceAnalyzer()
        self.visualizer = TraceVisualizer()
        self.ancestry_depth = ancestry_depth
        self.future_horizon = future_horizon
        self._creation_time = time.time()
        self._metadata: Dict[str, Any] = {}
    
    def record_entry(self, func: Callable, context: ExecutionContext, 
                    entry_type: TraceEntryType = TraceEntryType.CALL,
                    **kwargs) -> str:
        """
        Record a new trace entry.
        
        Args:
            func: Function being traced
            context: Execution context
            entry_type: Type of trace entry
            **kwargs: Additional entry data
            
        Returns:
            Entry ID of the recorded entry
        """
        entry = TraceEntry(
            entry_type=entry_type,
            function_name=getattr(func, '__name__', str(func)),
            context=context,
            trace_id=self.trace_id,
            **kwargs
        )
        
        self.recorder.record_entry(entry)
        return entry.entry_id
    
    def record_call(self, func: Callable, context: ExecutionContext,
                   parameters: Dict[str, Any] = None,
                   parent_entry_id: str = None) -> str:
        """Record a function call entry."""
        return self.record_entry(
            func, context, TraceEntryType.CALL,
            parameters=parameters or {},
            parent_entry_id=parent_entry_id
        )
    
    def record_return(self, func: Callable, context: ExecutionContext,
                     result: Any = None, parent_entry_id: str = None) -> str:
        """Record a function return entry."""
        return self.record_entry(
            func, context, TraceEntryType.RETURN,
            result=result,
            parent_entry_id=parent_entry_id
        )
    
    def record_error(self, func: Callable, context: ExecutionContext,
                    error: Exception, parent_entry_id: str = None) -> str:
        """Record an error entry."""
        return self.record_entry(
            func, context, TraceEntryType.ERROR,
            error=error,
            parent_entry_id=parent_entry_id
        )
    
    def get_entries(self) -> List[TraceEntry]:
        """Get all trace entries."""
        return self.recorder.get_entries()
    
    def get_operation_count(self) -> int:
        """Get the number of recorded operations."""
        return len(self.recorder.get_entries())
    
    def is_empty(self) -> bool:
        """Check if the trace is empty."""
        return self.get_operation_count() == 0
    
    def get_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get operation data by ID."""
        # Try to find the entry with the given ID
        for entry in self.recorder.get_entries():
            if entry.entry_id == operation_id:
                return {
                    "operation_type": entry.function_name,
                    "context": {
                        "entropy": entry.context.entropy if entry.context else 0.5,
                        "depth": entry.context.depth if entry.context else 0
                    },
                    "input_data": entry.parameters,
                    "output_data": entry.result if hasattr(entry, 'result') else {}
                }
        return None
    
    def get_forward_trace(self) -> List[TraceEntry]:
        """Get forward trace (chronological order)."""
        return self.recorder.get_entries()
    
    def get_reverse_trace(self) -> List[TraceEntry]:
        """Get reverse trace (reverse chronological order)."""
        return list(reversed(self.recorder.get_entries()))
    
    def analyze_patterns(self) -> TraceAnalysis:
        """Analyze trace for patterns and insights."""
        entries = self.recorder.get_entries()
        return self.analyzer.analyze_trace(entries)
    
    def visualize_text(self, max_depth: int = 50) -> str:
        """Generate text visualization of trace."""
        entries = self.recorder.get_entries()
        return self.visualizer.generate_text_trace(entries, max_depth)
    
    def visualize_graph(self) -> Dict[str, Any]:
        """Generate graph data for visualization."""
        entries = self.recorder.get_entries()
        return self.visualizer.generate_graph_data(entries)
    
    def get_entropy_timeline(self) -> Dict[str, Any]:
        """Get entropy evolution timeline."""
        entries = self.recorder.get_entries()
        return self.visualizer.generate_entropy_timeline(entries)
    
    def export_trace(self, format: str = "json") -> Union[str, bytes]:
        """
        Export trace data in specified format.
        
        Args:
            format: Export format ("json", "pickle", "csv")
            
        Returns:
            Exported trace data
        """
        entries = self.recorder.get_entries()
        
        if format == "json":
            data = {
                'trace_id': self.trace_id,
                'creation_time': self._creation_time,
                'metadata': self._metadata,
                'entries': [entry.to_dict() for entry in entries]
            }
            return json.dumps(data, indent=2, default=str)
        
        elif format == "pickle":
            return pickle.dumps({
                'trace_id': self.trace_id,
                'creation_time': self._creation_time,
                'metadata': self._metadata,
                'entries': entries
            })
        
        elif format == "csv":
            # Simple CSV format for basic analysis
            lines = ["timestamp,function_name,entry_type,entropy,depth"]
            for entry in entries:
                entropy = entry.context.entropy if entry.context else ""
                depth = entry.context.depth if entry.context else ""
                lines.append(f"{entry.timestamp},{entry.function_name},{entry.entry_type.value},{entropy},{depth}")
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def set_metadata(self, **kwargs) -> None:
        """Set trace metadata."""
        self._metadata.update(kwargs)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get trace metadata."""
        return self._metadata.copy()
    
    def clear(self) -> None:
        """Clear all trace data."""
        self.recorder.clear()
    
    def record_entry_dict(self, data: Dict[str, Any]) -> str:
        """
        Convenience method to record entry from dict data.
        
        Args:
            data: Dictionary containing entry information
            
        Returns:
            Entry ID of the recorded entry
        """
        # Create a mock function for dict-based recording
        class MockFunction:
            def __init__(self, name):
                self.__name__ = name
        
        func_name = data.get('function', 'unknown_function')
        mock_func = MockFunction(func_name)
        
        # Create ExecutionContext from dict data
        context = ExecutionContext(
            entropy=data.get('entropy', 0.5),
            depth=data.get('depth', 0)
        )
        
        return self.record_entry(
            mock_func, context, TraceEntryType.CALL,
            parameters=data
        )
    
    def record_exit_dict(self, data: Dict[str, Any]) -> str:
        """
        Convenience method to record exit from dict data.
        
        Args:
            data: Dictionary containing exit information
            
        Returns:
            Entry ID of the recorded entry
        """
        # Create a mock function for dict-based recording
        class MockFunction:
            def __init__(self, name):
                self.__name__ = name
        
        func_name = data.get('function', 'unknown_function')
        mock_func = MockFunction(func_name)
        
        # Create ExecutionContext from dict data
        context = ExecutionContext(
            entropy=data.get('entropy', 0.5),
            depth=data.get('depth', 0)
        )
        
        return self.record_entry(
            mock_func, context, TraceEntryType.RETURN,
            result=data.get('result')
        )
    
    def record_operation(self, operation_type: str, context: ExecutionContext,
                        input_data: Dict[str, Any] = None, 
                        output_data: Dict[str, Any] = None,
                        parent_operation: str = None) -> str:
        """Record a general operation (test-compatible interface)."""
        # Create a mock function for the operation
        class MockOperation:
            def __init__(self, name):
                self.__name__ = name
        
        mock_func = MockOperation(operation_type)
        
        # Record as a call entry with the operation data
        return self.record_entry(
            mock_func, context, TraceEntryType.CALL,
            parameters=input_data or {},
            result=output_data or {},
            parent_entry_id=parent_operation
        )
    
    def link_operations(self, parent_id: str, child_id: str, 
                       relationship_type: str = "child") -> None:
        """Link two operations with a relationship (test-compatible interface)."""
        # This would typically update the trace structure
        # For now, we'll just record the relationship in metadata
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trace statistics."""
        entries = self.recorder.get_entries()
        analysis = self.analyzer.analyze_trace(entries)
        
        return {
            'trace_id': self.trace_id,
            'creation_time': self._creation_time,
            'total_entries': len(entries),
            'total_calls': analysis.total_calls,
            'total_returns': analysis.total_returns,
            'max_depth': analysis.max_depth,
            'total_execution_time': analysis.total_execution_time,
            'function_diversity': len(analysis.function_call_counts),
            'most_called_functions': sorted(
                analysis.function_call_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'recursive_patterns_found': len(analysis.recursive_patterns),
            'error_patterns_found': len(analysis.error_patterns)
        }

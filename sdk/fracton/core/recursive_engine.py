"""
Recursive Engine - Core execution engine for Fracton recursive functions

This module provides the primary execution engine for recursive function calls
in the Fracton language, including context management, stack overflow protection,
and tail recursion optimization.
"""

import time
import uuid
from enum import Enum
from typing import Union, Callable
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
from collections import deque


class TrampolineResult(Enum):
    """Enumeration for trampoline execution results."""
    CONTINUE = "continue"
    COMPLETE = "complete"


@dataclass
class Continuation:
    """Represents a continuation in the trampoline execution."""
    func: Callable
    memory: Any
    context: 'ExecutionContext'
    result_type: TrampolineResult = TrampolineResult.CONTINUE


@dataclass
class ExecutionContext:
    """
    Execution context that carries metadata through recursive calls.
    
    Attributes:
        entropy: Current entropy level (0.0 - 1.0)
        depth: Current recursion depth
        trace_id: Unique identifier for this execution trace
        field_state: Field-specific metadata
        parent_context: Reference to the calling context
        metadata: Additional context metadata
    """
    
    def __init__(self, entropy: float = 0.5, depth: int = 0, 
                 trace_id: str = None, field_state: Dict[str, Any] = None,
                 parent_context: 'ExecutionContext' = None, 
                 metadata: Dict[str, Any] = None, experiment: str = None):
        self.entropy = entropy
        self.depth = depth
        self.trace_id = trace_id or str(uuid.uuid4())
        self.field_state = field_state or {}
        self.parent_context = parent_context
        self.metadata = metadata or {}
        self.experiment = experiment or self.metadata.get('experiment', 'default')
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from metadata (test-compatible interface)."""
        return self.metadata.get(key, default)
    
    def with_data(self, data: Dict[str, Any]) -> 'ExecutionContext':
        """Create a new context with additional data."""
        new_metadata = self.metadata.copy()
        new_metadata.update(data)
        
        return ExecutionContext(
            entropy=self.entropy,
            depth=self.depth,
            trace_id=self.trace_id,
            field_state=self.field_state.copy(),
            parent_context=self.parent_context,
            metadata=new_metadata,
            experiment=self.experiment
        )
    
    def deeper(self, steps: int = 1) -> 'ExecutionContext':
        """Create a new context with increased depth."""
        return ExecutionContext(
            entropy=self.entropy,
            depth=self.depth + steps,
            trace_id=self.trace_id,
            field_state=self.field_state.copy(),
            parent_context=self,
            metadata=self.metadata.copy(),
            experiment=self.experiment
        )
    
    def with_entropy(self, entropy: float) -> 'ExecutionContext':
        """Create a new context with modified entropy."""
        new_context = ExecutionContext(
            entropy=entropy,
            depth=self.depth,
            trace_id=self.trace_id,
            field_state=self.field_state.copy(),
            parent_context=self.parent_context,
            metadata=self.metadata.copy(),
            experiment=self.experiment
        )
        return new_context
    
    def with_metadata(self, **kwargs) -> 'ExecutionContext':
        """Create a new context with additional metadata."""
        new_metadata = self.metadata.copy()
        new_metadata.update(kwargs)
        
        return ExecutionContext(
            entropy=self.entropy,
            depth=self.depth,
            trace_id=self.trace_id,
            field_state=self.field_state.copy(),
            parent_context=self.parent_context,
            metadata=new_metadata,
            experiment=self.experiment
        )


@dataclass
class CallFrame:
    """Represents a single call in the recursive call stack."""
    function: Callable
    context: ExecutionContext
    timestamp: float
    trace_id: str
    

@dataclass
class ExecutionStats:
    """Statistics about recursive execution performance."""
    total_calls: int = 0
    max_depth: int = 0
    total_time: float = 0.0
    entropy_distribution: Dict[str, int] = field(default_factory=dict)
    function_call_counts: Dict[str, int] = field(default_factory=dict)


class StackOverflowError(Exception):
    """Raised when recursion depth exceeds safe limits."""
    
    def __init__(self, depth: int, max_depth: int):
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(f"Stack overflow at depth {depth} (max: {max_depth})")


class EntropyGateError(Exception):
    """Raised when function execution violates entropy gate conditions."""
    
    def __init__(self, current_entropy: float, required_range: tuple):
        self.current_entropy = current_entropy
        self.required_range = required_range
        super().__init__(
            f"Entropy {current_entropy} outside required range {required_range}"
        )


class CallStack:
    """Manages the recursive call stack with entropy awareness."""
    
    def __init__(self, max_depth: int = 1000):
        self.max_depth = max_depth
        self._stack: deque = deque()
        self._lock = threading.Lock()
    
    def push(self, func: Callable, context: ExecutionContext, trace_id: str) -> None:
        """Push a new call frame onto the stack."""
        with self._lock:
            if len(self._stack) >= self.max_depth:
                raise StackOverflowError(len(self._stack), self.max_depth)
            
            frame = CallFrame(
                function=func,
                context=context,
                timestamp=time.time(),
                trace_id=trace_id
            )
            self._stack.append(frame)
    
    def pop(self) -> Optional[CallFrame]:
        """Pop the top call frame from the stack."""
        with self._lock:
            if self._stack:
                return self._stack.pop()
            return None
    
    def current_depth(self) -> int:
        """Get the current stack depth."""
        with self._lock:
            return len(self._stack)
    
    def check_overflow(self) -> bool:
        """Check if the stack is approaching overflow."""
        return self.current_depth() >= self.max_depth * 0.9
    
    def get_trace(self) -> List[CallFrame]:
        """Get a copy of the current call stack."""
        with self._lock:
            return list(self._stack)


class RecursiveExecutor:
    """
    Main execution engine for recursive function calls in Fracton.
    
    Handles entropy gating, stack management, tail recursion optimization,
    and execution statistics collection.
    """
    
    def __init__(self, max_depth: int = 1000, enable_tail_optimization: bool = True, 
                 entropy_regulation: bool = True):
        self.max_depth = max_depth
        self.enable_tail_optimization = enable_tail_optimization
        self.entropy_regulation = entropy_regulation
        self.call_stack = CallStack(max_depth)
        self.stats = ExecutionStats()
        self._entropy_gates: Dict[Callable, tuple] = {}
        self._tail_recursive_functions: set = set()
        self._execution_lock = threading.Lock()
    
    def register_entropy_gate(self, func: Callable, min_threshold: float, 
                            max_threshold: float = 1.0) -> None:
        """Register entropy gate conditions for a function."""
        self._entropy_gates[func] = (min_threshold, max_threshold)
    
    def register_tail_recursive(self, func: Callable) -> None:
        """Register a function as tail-recursive for optimization."""
        self._tail_recursive_functions.add(func)
    
    def _check_entropy_gate(self, func: Callable, context: ExecutionContext) -> None:
        """Check if function execution meets entropy gate requirements."""
        if func not in self._entropy_gates:
            return
        
        min_threshold, max_threshold = self._entropy_gates[func]
        
        if not (min_threshold <= context.entropy <= max_threshold):
            raise EntropyGateError(
                context.entropy, 
                (min_threshold, max_threshold)
            )
    
    def _update_stats(self, func: Callable, context: ExecutionContext, 
                     execution_time: float) -> None:
        """Update execution statistics."""
        self.stats.total_calls += 1
        self.stats.max_depth = max(self.stats.max_depth, context.depth)
        self.stats.total_time += execution_time
        
        # Update entropy distribution
        entropy_bucket = f"{int(context.entropy * 10) / 10:.1f}"
        self.stats.entropy_distribution[entropy_bucket] = (
            self.stats.entropy_distribution.get(entropy_bucket, 0) + 1
        )
        
        # Update function call counts
        func_name = getattr(func, '__name__', str(func))
        self.stats.function_call_counts[func_name] = (
            self.stats.function_call_counts.get(func_name, 0) + 1
        )
    
    def _is_tail_recursive_call(self, func: Callable) -> bool:
        """Check if this is a tail-recursive call that can be optimized."""
        return (
            self.enable_tail_optimization and 
            func in self._tail_recursive_functions and
            self.call_stack.current_depth() > 0
        )
    
    def execute(self, func: Callable, memory: Any, context: Union[ExecutionContext, Dict]) -> Any:
        """
        Execute a recursive function with proper context management.
        
        Args:
            func: The function to execute
            memory: Shared memory field
            context: Execution context (ExecutionContext object or dict)
            
        Returns:
            The result of function execution
            
        Raises:
            StackOverflowError: If recursion depth exceeds limits
            EntropyGateError: If entropy conditions are not met
        """
        # Convert dict context to ExecutionContext if needed
        if isinstance(context, dict):
            exec_context = ExecutionContext(
                entropy=context.get('entropy', 0.5),
                depth=context.get('depth', 0),
                metadata=context
            )
        else:
            exec_context = context
            
        start_time = time.time()
        
        try:
            with self._execution_lock:
                # Check entropy gate conditions
                self._check_entropy_gate(func, exec_context)
                
                # Check for stack overflow
                if exec_context.depth >= self.max_depth:
                    raise StackOverflowError(exec_context.depth, self.max_depth)
                
                # Handle tail recursion optimization
                if self._is_tail_recursive_call(func):
                    # Use trampoline for tail recursion
                    return self._execute_trampoline(func, memory, exec_context)
                else:
                    # Push new frame and execute with trampoline
                    self.call_stack.push(func, exec_context, exec_context.trace_id)
                    try:
                        return self._execute_trampoline(func, memory, exec_context)
                    finally:
                        self.call_stack.pop()
                
        except Exception as e:
            # Add execution context to exception
            if hasattr(e, 'execution_context'):
                e.execution_context = exec_context
            raise
    
    def _execute_trampoline(self, func: Callable, memory: Any, context: 'ExecutionContext') -> Any:
        """
        Execute a function using trampoline-based recursion management.
        
        This prevents deep recursion by converting recursive calls into
        an iterative loop with continuations.
        """
        import threading
        start_time = time.time()
        continuation_queue = deque([Continuation(func, memory, context)])
        result = None
        
        # Mark that we're in trampoline execution
        current_thread = threading.current_thread()
        old_value = getattr(current_thread, '_fracton_in_trampoline', False)
        current_thread._fracton_in_trampoline = True
        
        try:
            while continuation_queue:
                current = continuation_queue.popleft()
                
                try:
                    # Execute the function
                    temp_result = current.func(current.memory, current.context)
                    
                    # Check if result is a continuation (recursive call)
                    if isinstance(temp_result, Continuation):
                        # Add to queue for further processing
                        continuation_queue.append(temp_result)
                    else:
                        # Final result
                        result = temp_result
                        
                except Exception as e:
                    # Add execution context to exception
                    if hasattr(e, 'execution_context'):
                        e.execution_context = current.context
                    raise
        finally:
            # Restore previous thread state
            current_thread._fracton_in_trampoline = old_value
        
        # Update statistics
        execution_time = time.time() - start_time
        self._update_stats(func, context, execution_time)
        
        return result
    
    def get_execution_stats(self) -> ExecutionStats:
        """Get current execution statistics."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.stats = ExecutionStats()
    
    @contextmanager
    def execution_context(self, max_depth: int = None):
        """Context manager for temporary execution settings."""
        original_max_depth = self.max_depth
        if max_depth is not None:
            self.max_depth = max_depth
        
        try:
            yield self
        finally:
            self.max_depth = original_max_depth


# Global default executor instance
_default_executor = RecursiveExecutor()


def get_default_executor() -> RecursiveExecutor:
    """Get the default global executor instance."""
    return _default_executor


def set_default_executor(executor: RecursiveExecutor) -> None:
    """Set a new default global executor instance."""
    global _default_executor
    _default_executor = executor

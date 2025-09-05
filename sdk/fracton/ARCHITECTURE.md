# Fracton Architecture Documentation

## Overview

This document describes the internal architecture of the Fracton computational modeling language, including module organization, data flow, and implementation strategies.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Fracton Runtime                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Language  │  │    Tool     │  │   Models    │        │
│  │ Constructs  │  │ Expression  │  │ Templates   │        │
│  │             │  │ Framework   │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                 │                 │             │
│         └─────────────────┼─────────────────┘             │
│                           │                               │
│  ┌─────────────────────────┼─────────────────────────────┐  │
│  │                Core Runtime Engine                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │ Recursive   │  │  Entropy    │  │ Bifractal   │   │  │
│  │  │   Engine    │  │  Dispatch   │  │   Trace     │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  │         │                 │                 │        │  │
│  │         └─────────────────┼─────────────────┘        │  │
│  │                           │                          │  │
│  │  ┌─────────────────────────┼─────────────────────────┐│  │
│  │  │            Memory Field Management               ││  │
│  │  └─────────────────────────┼─────────────────────────┘│  │
│  └─────────────────────────────┼─────────────────────────┘  │
│                                │                            │
└────────────────────────────────┼────────────────────────────┘
                                 │
         ┌─────────────────────────┼─────────────────────────┐
         │              External Interfaces                 │
         │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│
         │  │    File     │  │  Network    │  │   System    ││
         │  │   System    │  │ Services    │  │    APIs     ││
         │  └─────────────┘  └─────────────┘  └─────────────┘│
         └─────────────────────────────────────────────────┘
```

## Core Modules

### 1. Recursive Engine (`core/recursive_engine.py`)

**Purpose**: Primary execution engine for recursive function calls.

**Key Components**:
- `RecursiveExecutor`: Main execution coordinator
- `CallStack`: Manages recursive call hierarchy with entropy awareness
- `ExecutionContext`: Maintains context state across recursive calls
- `StackGuard`: Prevents stack overflow and manages tail recursion

**Data Flow**:
1. Receives function call with memory and context
2. Checks entropy gates and execution conditions
3. Records call in bifractal trace
4. Executes function with proper context isolation
5. Captures result and updates traces
6. Returns result to caller

**Key Classes**:
```python
class RecursiveExecutor:
    def __init__(self, max_depth=1000, enable_tail_optimization=True)
    def execute(self, func, memory, context) -> Any
    def register_entropy_gate(self, func, min_threshold, max_threshold)
    def get_execution_stats(self) -> ExecutionStats

class CallStack:
    def push(self, func, context, trace_id)
    def pop(self) -> CallFrame
    def current_depth(self) -> int
    def check_overflow(self) -> bool

class ExecutionContext:
    entropy: float
    depth: int
    trace_id: str
    field_state: dict
    parent_context: Optional['ExecutionContext']
    
    def deeper(self, steps=1) -> 'ExecutionContext'
    def with_entropy(self, entropy) -> 'ExecutionContext'
    def with_metadata(self, **kwargs) -> 'ExecutionContext'
```

### 2. Entropy Dispatch (`core/entropy_dispatch.py`)

**Purpose**: Context-aware function routing based on entropy and field conditions.

**Key Components**:
- `EntropyMatcher`: Matches context conditions to function candidates
- `DispatchRegistry`: Registry of available functions and their conditions
- `ContextAnalyzer`: Analyzes execution context for dispatch decisions

**Data Flow**:
1. Receives dispatch request with context
2. Analyzes context entropy and metadata
3. Queries registry for matching functions
4. Ranks candidates by context fitness
5. Returns best matching function for execution

**Key Classes**:
```python
class EntropyDispatcher:
    def __init__(self)
    def register_function(self, func, conditions: DispatchConditions)
    def dispatch(self, context, available_functions) -> callable
    def analyze_context(self, context) -> ContextAnalysis

class DispatchConditions:
    min_entropy: float
    max_entropy: float
    required_metadata: dict
    exclusion_patterns: list

class ContextAnalysis:
    entropy_level: str  # "low", "medium", "high"
    complexity_score: float
    suggested_functions: list
```

### 3. Bifractal Trace (`core/bifractal_trace.py`)

**Purpose**: Maintains forward and reverse traces of all recursive operations.

**Key Components**:
- `TraceRecorder`: Records function calls and results
- `TraceAnalyzer`: Analyzes traces for patterns and performance
- `TraceVisualizer`: Generates visual representations of execution flow

**Data Flow**:
1. Records function entry in forward trace
2. Captures context and parameters
3. Records function exit and result in reverse trace
4. Maintains bidirectional linkage between forward/reverse entries
5. Provides analysis and visualization interfaces

**Key Classes**:
```python
class BifractalTrace:
    def __init__(self, trace_id: str)
    def record_entry(self, func, context, params)
    def record_exit(self, func, result, modified_context)
    def get_forward_trace(self) -> list
    def get_reverse_trace(self) -> list
    def analyze_patterns(self) -> TraceAnalysis

class TraceEntry:
    timestamp: datetime
    function_name: str
    context: ExecutionContext
    parameters: dict
    entry_type: str  # "call" or "return"
    trace_id: str
    parent_trace_id: Optional[str]

class TraceAnalysis:
    total_calls: int
    max_depth: int
    entropy_evolution: list
    performance_hotspots: list
    recursive_patterns: list
```

### 4. Memory Field (`core/memory_field.py`)

**Purpose**: Shared memory coordination with entropy-aware access patterns.

**Key Components**:
- `MemoryField`: Main shared memory structure
- `FieldController`: Manages access patterns and isolation
- `EntropyTracker`: Tracks entropy changes in memory contents

**Data Flow**:
1. Provides isolated memory space for recursive operations
2. Tracks entropy changes as memory is modified
3. Maintains snapshots for rollback capabilities
4. Coordinates cross-field communication when needed

**Key Classes**:
```python
class MemoryField:
    def __init__(self, capacity=1000, initial_entropy=0.5)
    def get(self, key, default=None) -> Any
    def set(self, key, value) -> None
    def transform(self, entropy_level) -> Any
    def snapshot(self) -> MemorySnapshot
    def restore(self, snapshot: MemorySnapshot) -> None
    def calculate_entropy(self) -> float

class MemorySnapshot:
    timestamp: datetime
    field_id: str
    content: dict
    entropy_level: float
    metadata: dict

class FieldController:
    def create_field(self, capacity, entropy) -> MemoryField
    def merge_fields(self, *fields) -> MemoryField
    def isolate_field(self, field) -> MemoryField
```

## Language Module (`lang/`)

### Decorators (`lang/decorators.py`)

**Purpose**: Provides the `@fracton.*` decorator interfaces.

**Key Decorators**:
- `@fracton.recursive`: Marks functions as recursively callable
- `@fracton.entropy_gate`: Sets entropy execution thresholds
- `@fracton.tool_binding`: Binds functions to external tools
- `@fracton.tail_recursive`: Enables tail recursion optimization

### Primitives (`lang/primitives.py`)

**Purpose**: Core language functions like `fracton.recurse()`, `fracton.crystallize()`.

**Key Functions**:
```python
def recurse(func, memory, context) -> Any
def crystallize(data, patterns=None) -> Any
def branch(condition, if_true, if_false, memory, context) -> Any
def merge_contexts(*contexts) -> ExecutionContext
```

### Context (`lang/context.py`)

**Purpose**: Execution context management and manipulation.

### Compiler (`lang/compiler.py`)

**Purpose**: Optional DSL compilation for Fracton-specific syntax.

## Tool Expression Module (`tools/`)

### Registry (`tools/registry.py`)

**Purpose**: Central registry for external tool interfaces.

**Key Components**:
- Tool registration and discovery
- Context-based tool selection
- Tool lifecycle management

### Bindings (`tools/bindings/`)

**Purpose**: Specific connectors for external systems.

**Available Bindings**:
- `github.py`: GitHub API integration
- `database.py`: Database connections
- `filesystem.py`: File system operations
- `network.py`: Network service calls

### Expression (`tools/expression.py`)

**Purpose**: Context-aware tool expression and invocation.

## Data Flow Architecture

### 1. Function Call Flow

```
User Code
    ↓ @fracton.recursive decorator
Recursive Engine
    ↓ entropy check
Entropy Dispatcher
    ↓ context analysis
Function Selection
    ↓ trace recording
Bifractal Trace
    ↓ memory access
Memory Field
    ↓ actual execution
Target Function
    ↓ result capture
Bifractal Trace
    ↓ return
User Code
```

### 2. Memory Management Flow

```
Memory Field Creation
    ↓ entropy initialization
Entropy Tracker
    ↓ access isolation
Field Controller
    ↓ operation recording
Change Log
    ↓ entropy recalculation
Entropy Tracker
    ↓ field state update
Memory Field
```

### 3. Tool Expression Flow

```
Tool Request
    ↓ context analysis
Tool Registry
    ↓ tool selection
Tool Binding
    ↓ context preparation
Expression Engine
    ↓ external call
Target System
    ↓ result processing
Expression Engine
    ↓ context update
Memory Field
```

## Performance Optimizations

### 1. Tail Recursion Optimization
- Automatic detection of tail-recursive patterns
- Stack frame reuse for tail calls
- Configurable optimization levels

### 2. Entropy Caching
- Cached entropy calculations for stable data
- Lazy entropy recalculation on memory changes
- Entropy approximation for performance-critical paths

### 3. Trace Pruning
- Automatic pruning of old trace entries
- Configurable trace retention policies
- Compressed trace storage for long-running operations

### 4. Memory Optimization
- Copy-on-write semantics for memory fields
- Automatic garbage collection of unused snapshots
- Memory pool reuse for frequent allocations

## Error Handling Strategy

### 1. Graceful Degradation
- Fallback functions for failed operations
- Entropy-based error recovery
- Context-aware retry mechanisms

### 2. Debugging Support
- Full trace availability in error conditions
- Interactive trace exploration
- Performance profiling integration

### 3. Validation
- Static analysis of recursive patterns
- Runtime validation of entropy conditions
- Memory consistency checking

## Security Considerations

### 1. Memory Isolation
- Strong isolation between memory fields
- Controlled cross-field communication
- Access pattern monitoring

### 2. Tool Expression Security
- Sandboxed tool execution
- Permission-based tool access
- Context validation for tool calls

### 3. Resource Management
- Stack overflow prevention
- Memory usage limits
- CPU time limits for recursive operations

---

This architecture provides a robust foundation for the Fracton language while maintaining flexibility for research applications and experimental features.

"""
Entropy Dispatch - Context-aware function routing based on entropy and field conditions

This module provides intelligent function dispatch based on execution context,
entropy levels, and field state for optimal recursive function selection.
"""

import time
import math
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from enum import Enum

from .recursive_engine import ExecutionContext


class EntropyLevel(Enum):
    """Enumeration of entropy level classifications."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DispatchConditions:
    """
    Conditions that determine when a function should be dispatched.
    
    Attributes:
        min_entropy: Minimum entropy threshold for execution
        max_entropy: Maximum entropy threshold for execution  
        required_metadata: Required metadata keys in context
        exclusion_patterns: Patterns that exclude this function
        priority: Priority level for function selection
        complexity_range: Range of acceptable complexity scores
    """
    min_entropy: float = 0.0
    max_entropy: float = 1.0
    required_metadata: Dict[str, Any] = field(default_factory=dict)
    exclusion_patterns: List[str] = field(default_factory=list)
    priority: int = 0
    complexity_range: Tuple[float, float] = (0.0, 1.0)


@dataclass 
class ContextAnalysis:
    """
    Analysis results for an execution context.
    
    Attributes:
        entropy_level: Classified entropy level
        complexity_score: Calculated complexity score
        suggested_functions: List of recommended functions
        metadata_completeness: Completeness of context metadata
        field_state_quality: Quality assessment of field state
    """
    entropy_level: EntropyLevel
    complexity_score: float
    suggested_functions: List[str] = field(default_factory=list)
    metadata_completeness: float = 0.0
    field_state_quality: float = 0.0


class FunctionCandidate:
    """Represents a function candidate for dispatch with fitness score."""
    
    def __init__(self, func: Callable, conditions: DispatchConditions, 
                 fitness_score: float = 0.0):
        self.func = func
        self.conditions = conditions
        self.fitness_score = fitness_score
        self.name = getattr(func, '__name__', str(func))
    
    def __lt__(self, other):
        return self.fitness_score < other.fitness_score
    
    def __repr__(self):
        return f"FunctionCandidate({self.name}, fitness={self.fitness_score:.3f})"


class EntropyMatcher:
    """Matches execution contexts to appropriate functions based on entropy."""
    
    def __init__(self):
        self._entropy_thresholds = {
            EntropyLevel.LOW: (0.0, 0.3),
            EntropyLevel.MEDIUM: (0.3, 0.7),
            EntropyLevel.HIGH: (0.7, 0.9),
            EntropyLevel.CRITICAL: (0.9, 1.0)
        }
    
    def classify_entropy(self, entropy: float) -> EntropyLevel:
        """Classify entropy value into discrete levels."""
        for level, (min_val, max_val) in self._entropy_thresholds.items():
            if min_val <= entropy < max_val:
                return level
        return EntropyLevel.CRITICAL  # Default for entropy >= 0.9
    
    def calculate_entropy_fitness(self, entropy: float, 
                                conditions: DispatchConditions) -> float:
        """
        Calculate fitness score for entropy match.
        
        Returns value between 0.0 and 1.0 indicating how well the entropy
        matches the function's conditions.
        """
        if not (conditions.min_entropy <= entropy <= conditions.max_entropy):
            return 0.0
        
        # Calculate how close to optimal entropy range
        range_size = conditions.max_entropy - conditions.min_entropy
        if range_size == 0:
            return 1.0  # Exact match
        
        # Find position within range
        relative_position = (entropy - conditions.min_entropy) / range_size
        
        # Prefer middle of range (0.5 position gets score 1.0)
        distance_from_center = abs(relative_position - 0.5)
        fitness = 1.0 - (distance_from_center * 2)
        
        return max(0.0, fitness)


class ContextAnalyzer:
    """Analyzes execution contexts for dispatch decisions."""
    
    def __init__(self):
        self._entropy_matcher = EntropyMatcher()
        self._analysis_cache: Dict[str, ContextAnalysis] = {}
        self._cache_lock = threading.Lock()
    
    def analyze_context(self, context: ExecutionContext) -> ContextAnalysis:
        """
        Perform comprehensive analysis of execution context.
        
        Args:
            context: ExecutionContext to analyze
            
        Returns:
            ContextAnalysis with recommendations
        """
        # Check cache first
        cache_key = self._generate_cache_key(context)
        with self._cache_lock:
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]
        
        # Perform analysis
        entropy_level = self._entropy_matcher.classify_entropy(context.entropy)
        complexity_score = self._calculate_complexity_score(context)
        metadata_completeness = self._assess_metadata_completeness(context)
        field_state_quality = self._assess_field_state_quality(context)
        suggested_functions = self._suggest_functions(context, entropy_level)
        
        analysis = ContextAnalysis(
            entropy_level=entropy_level,
            complexity_score=complexity_score,
            suggested_functions=suggested_functions,
            metadata_completeness=metadata_completeness,
            field_state_quality=field_state_quality
        )
        
        # Cache result
        with self._cache_lock:
            self._analysis_cache[cache_key] = analysis
            
            # Limit cache size
            if len(self._analysis_cache) > 1000:
                # Remove oldest 10% of entries
                oldest_keys = list(self._analysis_cache.keys())[:100]
                for key in oldest_keys:
                    del self._analysis_cache[key]
        
        return analysis
    
    def _generate_cache_key(self, context: ExecutionContext) -> str:
        """Generate cache key for context analysis."""
        return f"{context.entropy:.3f}_{context.depth}_{len(context.metadata)}_{len(context.field_state)}"
    
    def _calculate_complexity_score(self, context: ExecutionContext) -> float:
        """Calculate complexity score based on context properties."""
        score = 0.0
        
        # Depth contributes to complexity
        depth_factor = min(context.depth / 100.0, 1.0)  # Normalize to 0-1
        score += depth_factor * 0.3
        
        # Entropy contributes to complexity
        score += context.entropy * 0.4
        
        # Metadata richness contributes
        metadata_factor = min(len(context.metadata) / 10.0, 1.0)
        score += metadata_factor * 0.2
        
        # Field state size contributes
        field_factor = min(len(context.field_state) / 20.0, 1.0)
        score += field_factor * 0.1
        
        return min(score, 1.0)
    
    def _assess_metadata_completeness(self, context: ExecutionContext) -> float:
        """Assess completeness of context metadata."""
        if not context.metadata:
            return 0.0
        
        # Check for common metadata fields
        expected_fields = ['operation', 'timestamp', 'source', 'target']
        present_fields = sum(1 for field in expected_fields if field in context.metadata)
        
        base_score = present_fields / len(expected_fields)
        
        # Bonus for rich metadata
        bonus = min(len(context.metadata) / 10.0, 0.2)
        
        return min(base_score + bonus, 1.0)
    
    def _assess_field_state_quality(self, context: ExecutionContext) -> float:
        """Assess quality of field state information."""
        if not context.field_state:
            return 0.0
        
        # Check for important state indicators
        quality_indicators = ['entropy', 'size', 'last_modified', 'access_pattern']
        present_indicators = sum(1 for indicator in quality_indicators 
                               if indicator in context.field_state)
        
        return present_indicators / len(quality_indicators)
    
    def _suggest_functions(self, context: ExecutionContext, 
                         entropy_level: EntropyLevel) -> List[str]:
        """Suggest function types based on context analysis."""
        suggestions = []
        
        if entropy_level == EntropyLevel.LOW:
            suggestions.extend(['stabilize', 'crystallize', 'optimize'])
        elif entropy_level == EntropyLevel.MEDIUM:
            suggestions.extend(['process', 'analyze', 'transform'])
        elif entropy_level == EntropyLevel.HIGH:
            suggestions.extend(['explore', 'generate', 'diverge'])
        else:  # CRITICAL
            suggestions.extend(['collapse', 'emergency_stabilize', 'reset'])
        
        # Add depth-based suggestions
        if context.depth > 50:
            suggestions.append('tail_optimize')
        if context.depth < 5:
            suggestions.append('initialize')
        
        return suggestions


class DispatchRegistry:
    """Registry of available functions and their dispatch conditions."""
    
    def __init__(self):
        self._functions: Dict[Callable, DispatchConditions] = {}
        self._function_stats: Dict[Callable, Dict] = defaultdict(lambda: {
            'call_count': 0,
            'success_count': 0, 
            'avg_execution_time': 0.0,
            'last_used': 0.0
        })
        self._lock = threading.Lock()
    
    def register_function(self, func: Callable, 
                         conditions: DispatchConditions) -> None:
        """Register a function with its dispatch conditions."""
        with self._lock:
            self._functions[func] = conditions
    
    def unregister_function(self, func: Callable) -> bool:
        """Unregister a function from the registry."""
        with self._lock:
            if func in self._functions:
                del self._functions[func]
                if func in self._function_stats:
                    del self._function_stats[func]
                return True
            return False
    
    def get_registered_functions(self) -> List[Callable]:
        """Get list of all registered functions."""
        with self._lock:
            return list(self._functions.keys())
    
    def get_function_conditions(self, func: Callable) -> Optional[DispatchConditions]:
        """Get dispatch conditions for a specific function."""
        with self._lock:
            return self._functions.get(func)
    
    def update_function_stats(self, func: Callable, success: bool, 
                            execution_time: float) -> None:
        """Update execution statistics for a function."""
        with self._lock:
            stats = self._function_stats[func]
            stats['call_count'] += 1
            if success:
                stats['success_count'] += 1
            
            # Update rolling average execution time
            if stats['avg_execution_time'] == 0.0:
                stats['avg_execution_time'] = execution_time
            else:
                stats['avg_execution_time'] = (
                    stats['avg_execution_time'] * 0.9 + execution_time * 0.1
                )
            
            stats['last_used'] = time.time()
    
    def get_function_stats(self, func: Callable) -> Dict:
        """Get execution statistics for a function."""
        with self._lock:
            return self._function_stats[func].copy()


class EntropyDispatcher:
    """
    Main dispatcher for context-aware function routing.
    
    Analyzes execution context and selects the most appropriate function
    based on entropy levels, complexity, and dispatch conditions.
    """
    
    def __init__(self, default_strategy: str = "balanced"):
        self.registry = DispatchRegistry()
        self.analyzer = ContextAnalyzer()
        self.entropy_matcher = EntropyMatcher()
        self.default_strategy = default_strategy
        self._dispatch_history: List[Dict] = []
        self._lock = threading.Lock()
    
    def register_function(self, func: Callable, 
                         conditions: DispatchConditions) -> None:
        """Register a function with dispatch conditions."""
        self.registry.register_function(func, conditions)
    
    def dispatch(self, context: ExecutionContext, 
                available_functions: List[Callable] = None) -> Optional[Callable]:
        """
        Select the best function for the given context.
        
        Args:
            context: Current execution context
            available_functions: Optional list to limit selection
            
        Returns:
            Best matching function or None if no suitable function found
        """
        start_time = time.time()
        
        # Analyze context
        analysis = self.analyzer.analyze_context(context)
        
        # Get candidate functions
        candidates = self._get_candidates(context, available_functions)
        
        if not candidates:
            return None
        
        # Calculate fitness scores for all candidates
        scored_candidates = []
        for func in candidates:
            conditions = self.registry.get_function_conditions(func)
            if conditions:
                fitness = self._calculate_fitness(context, conditions, analysis)
                if fitness > 0:
                    scored_candidates.append(
                        FunctionCandidate(func, conditions, fitness)
                    )
        
        if not scored_candidates:
            return None
        
        # Sort by fitness (highest first)
        scored_candidates.sort(reverse=True)
        
        # Select best candidate
        best_candidate = scored_candidates[0]
        
        # Record dispatch decision
        dispatch_time = time.time() - start_time
        self._record_dispatch(context, best_candidate, analysis, dispatch_time)
        
        return best_candidate.func
    
    def _get_candidates(self, context: ExecutionContext, 
                       available_functions: List[Callable] = None) -> List[Callable]:
        """Get list of candidate functions for dispatch."""
        if available_functions:
            # Filter to only registered functions
            return [f for f in available_functions 
                   if f in self.registry._functions]
        else:
            return self.registry.get_registered_functions()
    
    def _calculate_fitness(self, context: ExecutionContext,
                          conditions: DispatchConditions,
                          analysis: ContextAnalysis) -> float:
        """Calculate overall fitness score for function-context match."""
        score = 0.0
        
        # Entropy fitness (40% of total score)
        entropy_fitness = self.entropy_matcher.calculate_entropy_fitness(
            context.entropy, conditions
        )
        score += entropy_fitness * 0.4
        
        # Complexity fitness (30% of total score)
        complexity_fitness = self._calculate_complexity_fitness(
            analysis.complexity_score, conditions.complexity_range
        )
        score += complexity_fitness * 0.3
        
        # Metadata fitness (20% of total score)
        metadata_fitness = self._calculate_metadata_fitness(
            context, conditions.required_metadata
        )
        score += metadata_fitness * 0.2
        
        # Priority bonus (10% of total score)
        priority_bonus = min(conditions.priority / 10.0, 0.1)
        score += priority_bonus
        
        # Apply exclusion patterns
        if self._matches_exclusion_patterns(context, conditions.exclusion_patterns):
            score = 0.0
        
        return min(score, 1.0)
    
    def _calculate_complexity_fitness(self, complexity_score: float,
                                     complexity_range: Tuple[float, float]) -> float:
        """Calculate fitness based on complexity match."""
        min_complexity, max_complexity = complexity_range
        
        if not (min_complexity <= complexity_score <= max_complexity):
            return 0.0
        
        # Prefer middle of range
        range_size = max_complexity - min_complexity
        if range_size == 0:
            return 1.0
        
        relative_position = (complexity_score - min_complexity) / range_size
        distance_from_center = abs(relative_position - 0.5)
        fitness = 1.0 - (distance_from_center * 2)
        
        return max(0.0, fitness)
    
    def _calculate_metadata_fitness(self, context: ExecutionContext,
                                   required_metadata: Dict[str, Any]) -> float:
        """Calculate fitness based on metadata requirements."""
        if not required_metadata:
            return 1.0  # No requirements = perfect match
        
        matches = 0
        for key, expected_value in required_metadata.items():
            if key in context.metadata:
                if expected_value is None or context.metadata[key] == expected_value:
                    matches += 1
        
        return matches / len(required_metadata)
    
    def _matches_exclusion_patterns(self, context: ExecutionContext,
                                   exclusion_patterns: List[str]) -> bool:
        """Check if context matches any exclusion patterns."""
        for pattern in exclusion_patterns:
            # Simple pattern matching - could be enhanced with regex
            for metadata_value in context.metadata.values():
                if pattern in str(metadata_value):
                    return True
        return False
    
    def _record_dispatch(self, context: ExecutionContext,
                        candidate: FunctionCandidate,
                        analysis: ContextAnalysis,
                        dispatch_time: float) -> None:
        """Record dispatch decision for analysis and optimization."""
        dispatch_record = {
            'timestamp': time.time(),
            'function_name': candidate.name,
            'fitness_score': candidate.fitness_score,
            'entropy': context.entropy,
            'entropy_level': analysis.entropy_level.value,
            'complexity_score': analysis.complexity_score,
            'dispatch_time': dispatch_time,
            'depth': context.depth
        }
        
        with self._lock:
            self._dispatch_history.append(dispatch_record)
            
            # Limit history size
            if len(self._dispatch_history) > 10000:
                self._dispatch_history = self._dispatch_history[-5000:]
    
    def get_dispatch_stats(self) -> Dict[str, Any]:
        """Get statistics about dispatch decisions."""
        with self._lock:
            if not self._dispatch_history:
                return {}
            
            # Calculate statistics
            total_dispatches = len(self._dispatch_history)
            avg_fitness = sum(r['fitness_score'] for r in self._dispatch_history) / total_dispatches
            avg_dispatch_time = sum(r['dispatch_time'] for r in self._dispatch_history) / total_dispatches
            
            # Function usage counts
            function_counts = defaultdict(int)
            for record in self._dispatch_history:
                function_counts[record['function_name']] += 1
            
            # Entropy level distribution
            entropy_distribution = defaultdict(int)
            for record in self._dispatch_history:
                entropy_distribution[record['entropy_level']] += 1
            
            return {
                'total_dispatches': total_dispatches,
                'average_fitness_score': avg_fitness,
                'average_dispatch_time': avg_dispatch_time,
                'function_usage': dict(function_counts),
                'entropy_distribution': dict(entropy_distribution),
                'most_used_functions': sorted(
                    function_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
    
    def should_dispatch(self, context: Union[Dict, ExecutionContext]) -> bool:
        """
        Determine if dispatch should occur based on context conditions.
        
        Args:
            context: Execution context (dict or ExecutionContext)
            
        Returns:
            True if dispatch conditions are met, False otherwise
        """
        # Convert dict to basic entropy check if needed
        if isinstance(context, dict):
            entropy = context.get('entropy', 0.5)
            # Basic entropy-based dispatch decision
            return 0.0 <= entropy <= 1.0
        elif hasattr(context, 'entropy'):
            return 0.0 <= context.entropy <= 1.0
        else:
            return True  # Default to allowing dispatch
    
    def register(self, func: Callable, entropy_min: float = 0.0, 
                entropy_max: float = 1.0, priority: int = 0, **kwargs) -> None:
        """Register a function with entropy conditions (test-compatible interface)."""
        conditions = DispatchConditions(
            min_entropy=entropy_min,
            max_entropy=entropy_max,
            priority=priority
        )
        self.register_function(func, conditions)
    
    def get_route_count(self) -> int:
        """Get the number of registered routes/functions."""
        return len(self.registry.get_registered_functions())


# Global default dispatcher instance
_default_dispatcher = EntropyDispatcher()


def get_default_dispatcher() -> EntropyDispatcher:
    """Get the default global dispatcher instance."""
    return _default_dispatcher

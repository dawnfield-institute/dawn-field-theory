#!/usr/bin/env python3
"""
PAC Conservation Kernel
======================

The fundamental core of the PAC Physics Engine. Implements universal PAC conservation
principle: f(parent) = Σf(children) across all scales and domains.

This is the mathematical heart that ensures PAC conservation is maintained
throughout all simulations, regardless of scale or physical domain.
"""

import numpy as np
import torch
import time
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)
TOLERANCE = 5.0  # Broad tolerance for observational comparisonnp
import torch
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConservationType(Enum):
    """Types of PAC conservation enforcement"""
    EXACT = "exact"          # Machine-precision conservation
    APPROXIMATE = "approximate"  # Bounded conservation error
    EMERGENT = "emergent"    # Allows controlled violations for emergence


@dataclass
class PACNode:
    """
    Universal PAC node representing any entity in the simulation.
    Can be quantum particle, geometric point, fluid element, information packet, etc.
    """
    id: int
    value: float                    # f(node) - the conserved quantity
    potential: float = 0.0          # Πₜ(node) - unrealized potential
    children: Set[int] = field(default_factory=set)
    parents: Set[int] = field(default_factory=set)
    scale: str = "universal"        # quantum, geometric, fluid, information, consciousness
    domain: str = "general"         # specific physics domain
    timestamp: float = field(default_factory=time.time)
    
    # Conservation metadata
    last_conservation_check: float = 0.0
    conservation_error: float = 0.0
    violation_events: List[Dict] = field(default_factory=list)


@dataclass
class ConservationViolation:
    """Records PAC conservation violations for analysis"""
    node_id: int
    timestamp: float
    expected_value: float
    actual_value: float
    error_magnitude: float
    children_sum: float
    violation_type: str
    scale: str
    domain: str


class PACConservationKernel:
    """
    Universal PAC Conservation Kernel
    
    Enforces f(parent) = Σf(children) across all nodes in the simulation.
    Provides different enforcement modes from exact to emergent.
    """
    
    def __init__(self, 
                 conservation_type: ConservationType = ConservationType.EXACT,
                 tolerance: float = 1e-12,
                 device: str = "auto"):
        
        self.conservation_type = conservation_type
        self.tolerance = tolerance
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        
        # Core data structures
        self.nodes: Dict[int, PACNode] = {}
        self.conservation_matrix: Optional[torch.Tensor] = None
        self.violations: List[ConservationViolation] = []
        
        # Performance tracking
        self.total_conservation_checks = 0
        self.total_violations_detected = 0
        self.total_corrections_applied = 0
        
        # Universal signatures tracking
        self.amplification_events = []  # Information redistribution observations
        self.balance_measurements = []   # ξ = 1.0571 balance operator
        self.entropy_collapses = []     # Symbolic entropy collapse events
        
        logger.info(f"PAC Conservation Kernel initialized: {conservation_type.value} mode, tolerance={tolerance}")
    
    def add_node(self, node: PACNode) -> None:
        """Add a node to the PAC system"""
        self.nodes[node.id] = node
        self._invalidate_matrix()
        logger.debug(f"Added PAC node {node.id} with value {node.value}")
    
    def add_edge(self, parent_id: int, child_id: int, weight: float = 1.0) -> None:
        """Add parent-child relationship with optional ownership weight"""
        if parent_id not in self.nodes or child_id not in self.nodes:
            raise ValueError(f"Cannot add edge: node {parent_id} or {child_id} not found")
        
        self.nodes[parent_id].children.add(child_id)
        self.nodes[child_id].parents.add(parent_id)
        self._invalidate_matrix()
        logger.debug(f"Added PAC edge: {parent_id} -> {child_id} (weight={weight})")
    
    def compute_conservation_residual(self, node_id: int) -> float:
        """
        Compute PAC conservation residual for a node:
        residual = f(parent) - Σf(children)
        
        Residual = 0 means perfect conservation
        Residual > 0 means parent has excess value
        Residual < 0 means children have excess value
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.nodes[node_id]
        children_sum = sum(self.nodes[child_id].value for child_id in node.children)
        residual = node.value - children_sum
        
        # Update node conservation metadata
        node.last_conservation_check = time.time()
        node.conservation_error = abs(residual)
        
        self.total_conservation_checks += 1
        return residual
    
    def check_global_conservation(self) -> Dict[str, float]:
        """
        Check PAC conservation across entire system.
        Returns global conservation metrics.
        """
        total_residual = 0.0
        max_violation = 0.0
        violation_count = 0
        total_nodes = len(self.nodes)
        
        for node_id in self.nodes:
            residual = self.compute_conservation_residual(node_id)
            total_residual += abs(residual)
            
            if abs(residual) > self.tolerance:
                violation_count += 1
                max_violation = max(max_violation, abs(residual))
                
                # Record violation
                violation = ConservationViolation(
                    node_id=node_id,
                    timestamp=time.time(),
                    expected_value=sum(self.nodes[child_id].value for child_id in self.nodes[node_id].children),
                    actual_value=self.nodes[node_id].value,
                    error_magnitude=abs(residual),
                    children_sum=sum(self.nodes[child_id].value for child_id in self.nodes[node_id].children),
                    violation_type="conservation",
                    scale=self.nodes[node_id].scale,
                    domain=self.nodes[node_id].domain
                )
                self.violations.append(violation)
        
        self.total_violations_detected += violation_count
        
        # Improved conservation quality calculation
        if total_nodes > 0:
            mean_residual = total_residual / total_nodes
            # Quality ranges from 0.0 (terrible) to 1.0 (perfect)
            # Use exponential decay based on residual magnitude
            conservation_quality = np.exp(-mean_residual)  # 1.0 for zero residual, approaches 0 for large residuals
            conservation_stability = 1.0 - (violation_count / total_nodes)
        else:
            conservation_quality = 1.0
            conservation_stability = 1.0
        
        return {
            'total_residual_norm': total_residual,
            'mean_residual': total_residual / max(1, total_nodes),
            'max_violation': max_violation,
            'violation_count': violation_count,
            'conservation_quality': conservation_quality,
            'conservation_stability': conservation_stability,
            'global_balance': self._compute_global_balance()
        }
    
    def enforce_conservation(self, method: str = "pseudoinverse") -> Dict[str, float]:
        """
        Enforce PAC conservation across the system using specified method.
        
        Methods:
        - "pseudoinverse": Exact conservation using Moore-Penrose pseudoinverse
        - "iterative": Gauss-Seidel iterative relaxation
        - "gradient": Gradient descent to minimize residuals
        - "balance": Apply ξ balance operator corrections
        """
        start_time = time.time()
        
        if method == "pseudoinverse":
            result = self._enforce_pseudoinverse()
        elif method == "iterative":
            result = self._enforce_iterative()
        elif method == "gradient":
            result = self._enforce_gradient()
        elif method == "balance":
            result = self._enforce_balance_operator()
        else:
            raise ValueError(f"Unknown enforcement method: {method}")
        
        result['enforcement_time'] = time.time() - start_time
        result['method'] = method
        
        self.total_corrections_applied += 1
        logger.info(f"PAC conservation enforced using {method}: {result['post_residual_norm']:.2e}")
        
        return result
    
    def _enforce_pseudoinverse(self) -> Dict[str, float]:
        """Exact PAC conservation using Moore-Penrose pseudoinverse"""
        # Build conservation matrix A where A @ values = residuals
        A = self._build_conservation_matrix()
        values = torch.tensor([node.value for node in self.nodes.values()], 
                            dtype=torch.float64, device=self.device)
        
        # Compute residuals
        residuals = A @ values
        pre_residual_norm = torch.norm(residuals).item()
        
        # Apply pseudoinverse correction
        if torch.numel(residuals) > 0:
            # Moore-Penrose pseudoinverse
            A_pinv = torch.linalg.pinv(A)
            correction = A_pinv @ residuals
            corrected_values = values - correction
            
            # Update node values
            for i, node in enumerate(self.nodes.values()):
                node.value = corrected_values[i].item()
            
            # Verify conservation
            post_residuals = A @ corrected_values
            post_residual_norm = torch.norm(post_residuals).item()
        else:
            post_residual_norm = 0.0
        
        return {
            'pre_residual_norm': pre_residual_norm,
            'post_residual_norm': post_residual_norm,
            'correction_applied': True,
            'precision_achieved': post_residual_norm < 1e-12
        }
    
    def _enforce_iterative(self, max_iterations: int = 100, tolerance: float = 1e-9) -> Dict[str, float]:
        """Iterative Gauss-Seidel conservation enforcement"""
        residual_history = []
        
        for iteration in range(max_iterations):
            total_residual = 0.0
            
            for node_id, node in self.nodes.items():
                if node.children:  # Only adjust parents with children
                    children_sum = sum(self.nodes[child_id].value for child_id in node.children)
                    residual = node.value - children_sum
                    
                    # Gauss-Seidel update: distribute residual
                    correction = residual * 0.5  # Damping factor
                    node.value -= correction
                    
                    total_residual += abs(residual)
            
            residual_history.append(total_residual)
            
            if total_residual < tolerance:
                break
        
        return {
            'pre_residual_norm': residual_history[0] if residual_history else 0.0,
            'post_residual_norm': residual_history[-1] if residual_history else 0.0,
            'iterations': iteration + 1,
            'converged': total_residual < tolerance,
            'residual_history': residual_history
        }
    
    def _enforce_balance_operator(self) -> Dict[str, float]:
        """Apply ξ = 1.0571 balance operator for bounded complexity"""
        BALANCE_CONSTANT = 1.0571  # The universal balance operator discovered in MED
        
        pre_balance = self._compute_global_balance()
        pre_residual_norm = sum(abs(self.compute_conservation_residual(node_id)) for node_id in self.nodes)
        
        for node_id, node in self.nodes.items():
            if node.children:
                children_sum = sum(self.nodes[child_id].value for child_id in node.children)
                target_value = children_sum * BALANCE_CONSTANT
                
                # Apply balance correction
                correction_factor = 0.1  # Gentle adjustment
                adjustment = (target_value - node.value) * correction_factor
                node.value += adjustment
        
        post_balance = self._compute_global_balance()
        post_residual_norm = sum(abs(self.compute_conservation_residual(node_id)) for node_id in self.nodes)
        
        return {
            'pre_balance': pre_balance,
            'post_balance': post_balance,
            'balance_constant_applied': BALANCE_CONSTANT,
            'balance_improvement': abs(post_balance - BALANCE_CONSTANT) < abs(pre_balance - BALANCE_CONSTANT),
            'pre_residual_norm': pre_residual_norm,
            'post_residual_norm': post_residual_norm
        }
    
    def detect_universal_signatures(self) -> Dict[str, any]:
        """
        Detect universal signatures across the PAC system:
        - Information redistribution patterns
        - ξ = 1.0571 balance operator
        - Symbolic entropy collapse
        """
        signatures = {}
        
        # Detect information redistribution patterns
        amplification_signature = self._detect_amplification_events()
        if amplification_signature:
            self.amplification_events.append(amplification_signature)
            signatures['amplification'] = amplification_signature
        
        # Measure balance operator proximity
        current_balance = self._compute_global_balance()
        balance_proximity = abs(current_balance - 1.0571)
        self.balance_measurements.append({
            'timestamp': time.time(),
            'balance_value': current_balance,
            'proximity_to_ideal': balance_proximity
        })
        signatures['balance_operator'] = {
            'current_value': current_balance,
            'ideal_proximity': balance_proximity,
            'is_near_ideal': balance_proximity < 0.1
        }
        
        # Detect entropy collapse events
        entropy_signature = self._detect_entropy_collapse()
        if entropy_signature:
            self.entropy_collapses.append(entropy_signature)
            signatures['entropy_collapse'] = entropy_signature
        
        return signatures
    
    def _build_conservation_matrix(self) -> torch.Tensor:
        """Build the conservation constraint matrix A"""
        if self.conservation_matrix is not None:
            return self.conservation_matrix
        
        n_nodes = len(self.nodes)
        node_ids = list(self.nodes.keys())
        id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        
        A = torch.eye(n_nodes, dtype=torch.float64, device=self.device)
        
        # For each parent node, subtract children
        for node_id, node in self.nodes.items():
            parent_idx = id_to_index[node_id]
            for child_id in node.children:
                child_idx = id_to_index[child_id]
                A[parent_idx, child_idx] = -1.0  # Could be weighted for shared children
        
        self.conservation_matrix = A
        return A
    
    def _invalidate_matrix(self):
        """Invalidate cached conservation matrix when topology changes"""
        self.conservation_matrix = None
    
    def _compute_global_balance(self) -> float:
        """Compute global balance measure (related to ξ balance operator)"""
        if not self.nodes:
            return 1.0
        
        total_parent_value = 0.0
        total_children_value = 0.0
        
        for node in self.nodes.values():
            if node.children:
                total_parent_value += node.value
                total_children_value += sum(self.nodes[child_id].value for child_id in node.children)
        
        if total_children_value == 0:
            return 1.0
        
        return total_parent_value / total_children_value
    
    def _detect_amplification_events(self) -> Optional[Dict]:
        """Detect information redistribution patterns"""
        # Look for parent-children relationships showing spatial redistribution
        AMPLIFICATION_REFERENCE = 15.56  # Reference from dawn-field experiments
        TOLERANCE = 5.0  # Broad tolerance for observational comparison
        
        for node_id, node in self.nodes.items():
            if node.children and node.value > 0:
                children_sum = sum(self.nodes[child_id].value for child_id in node.children)
                if children_sum > 0:
                    amplification_factor = children_sum / node.value
                    
                    if amplification_factor > 1.5:  # Any significant redistribution
                        return {
                            'node_id': node_id,
                            'amplification_factor': amplification_factor,
                            'timestamp': time.time(),
                            'parent_value': node.value,
                            'children_sum': children_sum,
                            'reference_comparison': amplification_factor / AMPLIFICATION_REFERENCE,
                            'observation_type': 'spatial_redistribution'
                        }
        return None
    
    def _detect_entropy_collapse(self) -> Optional[Dict]:
        """Detect symbolic entropy collapse events"""
        # Multiple entropy collapse detection methods
        values = [node.value for node in self.nodes.values()]
        if len(values) < 2:
            return None
        
        values_tensor = torch.tensor(values, dtype=torch.float64)
        
        # Method 1: Shannon entropy collapse
        values_abs = torch.abs(values_tensor)
        total_abs = torch.sum(values_abs) + 1e-12
        values_norm = values_abs / total_abs
        
        # Shannon entropy
        shannon_entropy = -torch.sum(values_norm * torch.log(values_norm + 1e-12))
        max_entropy = torch.log(torch.tensor(len(values), dtype=torch.float64))
        normalized_shannon = shannon_entropy / max_entropy
        
        # Method 2: Gini coefficient (concentration measure)
        sorted_values = torch.sort(values_abs)[0]
        n = len(sorted_values)
        index = torch.arange(1, n + 1, dtype=torch.float64)
        gini = (2 * torch.sum(index * sorted_values)) / (n * torch.sum(sorted_values)) - (n + 1) / n
        
        # Method 3: Top-k concentration
        top_3_concentration = torch.sum(torch.topk(values_abs, min(3, len(values)))[0]) / total_abs
        
        # Detect collapse using multiple criteria
        shannon_collapse = normalized_shannon < 0.4  # Relaxed threshold
        gini_collapse = gini > 0.7  # High concentration
        concentration_collapse = top_3_concentration > 0.8  # Top 3 values dominate
        
        if shannon_collapse or gini_collapse or concentration_collapse:
            return {
                'timestamp': time.time(),
                'shannon_entropy': shannon_entropy.item(),
                'normalized_shannon': normalized_shannon.item(),
                'gini_coefficient': gini.item(),
                'top_3_concentration': top_3_concentration.item(),
                'collapse_type': 'shannon' if shannon_collapse else ('gini' if gini_collapse else 'concentration'),
                'collapse_magnitude': max(1.0 - normalized_shannon.item(), gini.item(), top_3_concentration.item()),
                'node_count': len(values),
                'max_value': torch.max(values_abs).item(),
                'mean_value': torch.mean(values_abs).item()
            }
        
        return None
    
    def get_system_state(self) -> Dict:
        """Get comprehensive system state for monitoring/debugging"""
        conservation_metrics = self.check_global_conservation()
        signatures = self.detect_universal_signatures()
        
        return {
            'node_count': len(self.nodes),
            'conservation': conservation_metrics,
            'signatures': signatures,
            'performance': {
                'total_checks': self.total_conservation_checks,
                'total_violations': self.total_violations_detected,
                'total_corrections': self.total_corrections_applied
            },
            'conservation_type': self.conservation_type.value,
            'tolerance': self.tolerance
        }

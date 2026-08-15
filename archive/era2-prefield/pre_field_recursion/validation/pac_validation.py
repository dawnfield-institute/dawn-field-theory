"""
Advanced PAC (Parent-Child) conservation validation

This module provides comprehensive validation of the fundamental constraint:
f(parent) = Σf(children) maintained locally among children.

This is the REAL constraint in Pre-Field Recursion, not amplification targeting.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime

@dataclass  
class PACAnalysis:
    """Comprehensive PAC conservation analysis"""
    conservation_quality: float
    balance_distribution: List[float]
    local_consistency: float
    global_consistency: float
    error_statistics: Dict
    temporal_stability: float
    
class PACValidator:
    """
    Advanced PAC conservation validator
    
    Validates the fundamental constraint f(parent) = Σf(children) with
    sophisticated analysis of conservation quality, error patterns,
    and temporal stability.
    """
    
    def __init__(self, tolerance: float = 0.15, history_window: int = 100):
        self.tolerance = tolerance
        self.history_window = history_window
        self.validation_history = []
        self.error_patterns = []
        
    def validate_pac_conservation(self, parent_value: float, children_values: List[float],
                                context: Optional[Dict] = None) -> Dict:
        """
        Comprehensive PAC conservation validation
        
        Args:
            parent_value: Value at parent node
            children_values: Values at child nodes
            context: Optional context information (topology location, etc.)
            
        Returns:
            Detailed validation result with quality metrics
        """
        children_sum = sum(children_values)
        deviation = abs(parent_value - children_sum)
        
        # Calculate relative deviation (normalized by parent magnitude)
        parent_magnitude = max(abs(parent_value), 1e-10)
        relative_deviation = deviation / parent_magnitude
        
        # Conservation quality metrics
        is_consistent = relative_deviation <= self.tolerance
        consistency_percentage = max(0, 100 * (1 - relative_deviation / self.tolerance))
        
        # Balance distribution analysis
        if abs(children_sum) > 1e-10:
            balance_ratios = [child / children_sum for child in children_values]
            balance_entropy = self._calculate_balance_entropy(balance_ratios)
            balance_uniformity = self._calculate_balance_uniformity(balance_ratios)
        else:
            balance_entropy = 0.0
            balance_uniformity = 1.0
            
        # Error classification
        error_type = self._classify_error(relative_deviation, children_values)
        
        # Temporal context (if available)
        temporal_info = self._analyze_temporal_context(relative_deviation)
        
        validation_result = {
            'parent_value': parent_value,
            'children_values': children_values,
            'children_sum': children_sum,
            'deviation': deviation,
            'relative_deviation': relative_deviation,
            'is_consistent': is_consistent,
            'consistency_percentage': consistency_percentage,
            'conservation_quality': min(100, consistency_percentage),
            'balance_metrics': {
                'entropy': balance_entropy,
                'uniformity': balance_uniformity,
                'distribution': balance_ratios if abs(children_sum) > 1e-10 else []
            },
            'error_analysis': {
                'type': error_type,
                'magnitude': relative_deviation,
                'pattern': self._detect_error_pattern(relative_deviation)
            },
            'temporal_analysis': temporal_info,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Update history
        self._update_history(validation_result)
        
        return validation_result
    
    def _calculate_balance_entropy(self, ratios: List[float]) -> float:
        """Calculate entropy of balance distribution among children"""
        if not ratios:
            return 0.0
            
        # Normalize ratios to probabilities
        total = sum(abs(r) for r in ratios)
        if total < 1e-10:
            return 0.0
            
        probs = [abs(r) / total for r in ratios]
        probs = [p for p in probs if p > 1e-10]  # Remove zeros
        
        if len(probs) <= 1:
            return 0.0
            
        entropy = -sum(p * np.log2(p) for p in probs)
        return entropy
    
    def _calculate_balance_uniformity(self, ratios: List[float]) -> float:
        """Calculate how uniformly balanced the children are"""
        if len(ratios) <= 1:
            return 1.0
            
        expected_ratio = 1.0 / len(ratios)
        uniformity_deviations = [abs(abs(r) - expected_ratio) for r in ratios]
        max_possible_deviation = expected_ratio * len(ratios)
        
        if max_possible_deviation < 1e-10:
            return 1.0
            
        uniformity = 1.0 - sum(uniformity_deviations) / max_possible_deviation
        return max(0, uniformity)
    
    def _classify_error(self, relative_deviation: float, children_values: List[float]) -> str:
        """Classify the type of PAC conservation error"""
        if relative_deviation <= self.tolerance * 0.1:
            return "negligible"
        elif relative_deviation <= self.tolerance * 0.5:
            return "minor"
        elif relative_deviation <= self.tolerance:
            return "acceptable"
        elif relative_deviation <= self.tolerance * 2.0:
            return "significant"
        else:
            return "critical"
    
    def _detect_error_pattern(self, current_error: float) -> str:
        """Detect patterns in conservation errors over time"""
        if len(self.error_patterns) < 3:
            self.error_patterns.append(current_error)
            return "insufficient_data"
            
        self.error_patterns.append(current_error)
        
        # Keep only recent history
        if len(self.error_patterns) > 20:
            self.error_patterns = self.error_patterns[-20:]
            
        recent_errors = self.error_patterns[-5:]
        
        # Detect trends
        if all(recent_errors[i] < recent_errors[i-1] for i in range(1, len(recent_errors))):
            return "improving"
        elif all(recent_errors[i] > recent_errors[i-1] for i in range(1, len(recent_errors))):
            return "degrading"
        elif max(recent_errors) - min(recent_errors) < self.tolerance * 0.1:
            return "stable"
        else:
            return "fluctuating"
    
    def _analyze_temporal_context(self, current_error: float) -> Dict:
        """Analyze temporal context of current validation"""
        if len(self.validation_history) < 2:
            return {
                'trend': 'insufficient_data',
                'stability': 1.0,
                'recent_average': current_error
            }
            
        recent_errors = [v['relative_deviation'] for v in self.validation_history[-10:]]
        recent_average = np.mean(recent_errors)
        stability = 1.0 / (1.0 + np.std(recent_errors))
        
        # Trend analysis
        if len(recent_errors) >= 3:
            trend_slope = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
            if abs(trend_slope) < self.tolerance * 0.01:
                trend = "stable"
            elif trend_slope > 0:
                trend = "degrading"
            else:
                trend = "improving"
        else:
            trend = "insufficient_data"
            
        return {
            'trend': trend,
            'stability': stability,
            'recent_average': recent_average,
            'current_vs_average': current_error / max(recent_average, 1e-10)
        }
    
    def _update_history(self, validation_result: Dict) -> None:
        """Update validation history"""
        self.validation_history.append(validation_result)
        
        # Maintain history window
        if len(self.validation_history) > self.history_window:
            self.validation_history = self.validation_history[-self.history_window:]
    
    def get_overall_analysis(self) -> PACAnalysis:
        """Get comprehensive analysis of PAC conservation performance"""
        if not self.validation_history:
            return PACAnalysis(
                conservation_quality=0.0,
                balance_distribution=[],
                local_consistency=0.0,
                global_consistency=0.0,
                error_statistics={},
                temporal_stability=0.0
            )
            
        # Calculate overall metrics
        consistent_validations = [v for v in self.validation_history if v['is_consistent']]
        global_consistency = 100.0 * len(consistent_validations) / len(self.validation_history)
        
        conservation_qualities = [v['conservation_quality'] for v in self.validation_history]
        avg_conservation_quality = np.mean(conservation_qualities)
        
        # Balance distribution analysis
        all_balance_entropies = [v['balance_metrics']['entropy'] for v in self.validation_history 
                               if 'balance_metrics' in v]
        avg_balance_entropy = np.mean(all_balance_entropies) if all_balance_entropies else 0.0
        
        # Error statistics
        all_errors = [v['relative_deviation'] for v in self.validation_history]
        error_stats = {
            'mean': np.mean(all_errors),
            'std': np.std(all_errors),
            'min': np.min(all_errors),
            'max': np.max(all_errors),
            'percentile_95': np.percentile(all_errors, 95),
            'count_critical': sum(1 for e in all_errors if e > self.tolerance * 2.0)
        }
        
        # Temporal stability
        if len(conservation_qualities) > 1:
            temporal_stability = 1.0 / (1.0 + np.std(conservation_qualities) / max(np.mean(conservation_qualities), 1.0))
        else:
            temporal_stability = 1.0
            
        return PACAnalysis(
            conservation_quality=avg_conservation_quality,
            balance_distribution=all_balance_entropies,
            local_consistency=avg_conservation_quality,
            global_consistency=global_consistency,
            error_statistics=error_stats,
            temporal_stability=temporal_stability
        )
    
    def get_validation_summary(self) -> Dict:
        """Get summary of validation performance"""
        analysis = self.get_overall_analysis()
        
        return {
            'total_validations': len(self.validation_history),
            'success_rate': analysis.global_consistency,
            'average_quality': analysis.conservation_quality,
            'stability': analysis.temporal_stability,
            'error_summary': analysis.error_statistics,
            'recommendation': self._generate_recommendation(analysis)
        }
    
    def _generate_recommendation(self, analysis: PACAnalysis) -> str:
        """Generate recommendations based on analysis"""
        if analysis.global_consistency >= 85.0:
            return "PAC conservation performing well - maintain current parameters"
        elif analysis.global_consistency >= 70.0:
            return "PAC conservation acceptable - consider minor parameter adjustments"
        elif analysis.global_consistency >= 50.0:
            return "PAC conservation marginal - review topology configuration"
        else:
            return "PAC conservation failing - major parameter revision needed"
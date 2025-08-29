"""
Pattern analysis implementation for the Unified Emergence Framework v2.
"""

import numpy as np
from typing import List, Dict, Any
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
import logging
from datetime import datetime

from ..domain.models import EmergenceSignature, CorrelationMatrix, ValidationMetrics


logger = logging.getLogger(__name__)


class PatternAnalyzerImpl:
    """
    Implementation of pattern analysis for emergence detection.
    
    This class handles correlation calculations and metric computation
    for emergence patterns across domains.
    """
    
    def __init__(self, correlation_method: str = 'pearson'):
        """
        Initialize pattern analyzer.
        
        Args:
            correlation_method: Method for correlation calculation ('pearson', 'spearman', 'cosine')
        """
        self.correlation_method = correlation_method
        
        if correlation_method not in ['pearson', 'spearman', 'cosine']:
            raise ValueError(f"Unsupported correlation method: {correlation_method}")
    
    def calculate_correlations(self, signatures: List[EmergenceSignature]) -> CorrelationMatrix:
        """
        Calculate correlations between emergence patterns across domains.
        
        Args:
            signatures: List of emergence signatures to analyze
            
        Returns:
            Correlation matrix showing relationships between patterns
        """
        if not signatures:
            return CorrelationMatrix(
                domains=[],
                correlation_values=[],
                mean_correlation=0.0,
                correlation_consistency=0.0
            )
        
        # Group signatures by domain
        domain_signatures = {}
        for sig in signatures:
            if sig.domain not in domain_signatures:
                domain_signatures[sig.domain] = []
            domain_signatures[sig.domain].append(sig)
        
        domains = sorted(domain_signatures.keys())
        n_domains = len(domains)
        
        if n_domains == 1:
            return CorrelationMatrix(
                domains=domains,
                correlation_values=[[1.0]],
                mean_correlation=1.0,
                correlation_consistency=1.0
            )
        
        # Calculate representative feature vectors for each domain
        domain_features = {}
        max_feature_length = 0
        
        # First pass: determine maximum feature length
        for domain in domains:
            sigs = domain_signatures[domain]
            if sigs:
                max_len = max(len(sig.features) for sig in sigs)
                max_feature_length = max(max_feature_length, max_len)
        
        # Second pass: normalize all feature vectors to same length
        for domain in domains:
            sigs = domain_signatures[domain]
            
            if sigs:
                # Pad feature vectors to same length
                normalized_features = []
                for sig in sigs:
                    features = list(sig.features)
                    # Pad with zeros if needed
                    while len(features) < max_feature_length:
                        features.append(0.0)
                    # Truncate if too long
                    features = features[:max_feature_length]
                    normalized_features.append(features)
                
                features_array = np.array(normalized_features)
                confidences = np.array([sig.confidence for sig in sigs])
                
                # Normalize confidences to sum to 1
                if np.sum(confidences) > 0:
                    weights = confidences / np.sum(confidences)
                    domain_features[domain] = np.average(features_array, axis=0, weights=weights)
                else:
                    domain_features[domain] = np.mean(features_array, axis=0)
            else:
                domain_features[domain] = np.zeros(max_feature_length)
        
        # Calculate correlation matrix
        correlation_matrix = np.eye(n_domains)
        correlations = []
        
        for i in range(n_domains):
            for j in range(i + 1, n_domains):
                domain1, domain2 = domains[i], domains[j]
                features1 = domain_features[domain1]
                features2 = domain_features[domain2]
                
                if len(features1) == 0 or len(features2) == 0 or len(features1) != len(features2):
                    correlation = 0.0
                else:
                    try:
                        correlation = self._calculate_correlation(features1, features2)
                    except Exception as e:
                        logger.warning(f"Error calculating correlation between {domain1} and {domain2}: {e}")
                        correlation = 0.0
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
                correlations.append(abs(correlation))
        
        # Calculate summary statistics
        mean_correlation = np.mean(correlations) if correlations else 0.0
        correlation_consistency = 1.0 - np.std(correlations) if correlations else 1.0
        
        return CorrelationMatrix(
            domains=domains,
            correlation_values=correlation_matrix.tolist(),
            mean_correlation=mean_correlation,
            correlation_consistency=max(0.0, correlation_consistency)
        )
    
    def _calculate_correlation(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate correlation between two feature vectors."""
        try:
            if self.correlation_method == 'pearson':
                correlation, _ = pearsonr(features1, features2)
                return correlation if not np.isnan(correlation) else 0.0
            
            elif self.correlation_method == 'spearman':
                correlation, _ = spearmanr(features1, features2)
                return correlation if not np.isnan(correlation) else 0.0
            
            elif self.correlation_method == 'cosine':
                # Cosine similarity (convert to correlation-like measure)
                similarity = 1.0 - cosine(features1, features2)
                return similarity if not np.isnan(similarity) else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating correlation: {e}")
            return 0.0
        
        return 0.0
    
    def calculate_metrics(self, signatures: List[EmergenceSignature], 
                         correlation_matrix: CorrelationMatrix) -> ValidationMetrics:
        """
        Calculate comprehensive validation metrics.
        
        Args:
            signatures: List of emergence signatures
            correlation_matrix: Correlation analysis results
            
        Returns:
            Comprehensive validation metrics
        """
        start_time = datetime.now()
        
        if not signatures:
            return self._empty_metrics()
        
        # Pattern statistics
        total_patterns = len(signatures)
        patterns_per_domain = {}
        for sig in signatures:
            patterns_per_domain[sig.domain] = patterns_per_domain.get(sig.domain, 0) + 1
        
        # Confidence and strength statistics
        confidences = [sig.confidence for sig in signatures]
        strengths = [sig.emergence_strength for sig in signatures]
        
        avg_confidence = np.mean(confidences)
        avg_strength = np.mean(strengths)
        
        # Pattern diversity (based on feature hash uniqueness)
        unique_hashes = set(sig.feature_hash for sig in signatures)
        diversity_score = len(unique_hashes) / total_patterns if total_patterns > 0 else 0.0
        
        # Core emergence metrics (based on v1 success patterns)
        sec_accuracy = self._calculate_sec_classification_accuracy(signatures, correlation_matrix)
        pattern_assembly = self._calculate_pattern_assembly_success_rate(signatures, correlation_matrix)
        emergence_consistency = self._calculate_emergence_consistency_score(signatures, correlation_matrix)
        phase1_readiness = self._calculate_phase1_readiness_score(
            sec_accuracy, pattern_assembly, emergence_consistency, avg_confidence
        )
        
        # Cross-domain metrics
        cross_domain_corr = correlation_matrix.mean_correlation
        corr_consistency = correlation_matrix.correlation_consistency
        
        # Performance metrics (placeholders - would be filled by framework)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ValidationMetrics(
            sec_classification_accuracy=sec_accuracy,
            pattern_assembly_success_rate=pattern_assembly,
            emergence_consistency_score=emergence_consistency,
            phase1_readiness_score=phase1_readiness,
            total_patterns_extracted=total_patterns,
            patterns_per_domain=patterns_per_domain,
            cross_domain_correlations=cross_domain_corr,
            correlation_consistency=corr_consistency,
            processing_time_seconds=processing_time,
            memory_usage_mb=0.0,  # Placeholder
            average_pattern_confidence=avg_confidence,
            average_emergence_strength=avg_strength,
            pattern_diversity_score=diversity_score
        )
    
    def _calculate_sec_classification_accuracy(self, signatures: List[EmergenceSignature], 
                                             correlation_matrix: CorrelationMatrix) -> float:
        """Calculate SEC (Structural Emergence Classification) accuracy."""
        if not signatures:
            return 0.0
        
        # More realistic SEC accuracy calculation
        # Consider patterns with moderate confidence as valid
        valid_patterns = [sig for sig in signatures if sig.confidence >= 0.3]
        confidence_score = len(valid_patterns) / len(signatures)
        
        # Factor in emergence strength and cross-domain correlation
        strong_emergence = [sig for sig in signatures if sig.emergence_strength >= 0.3]
        emergence_score = len(strong_emergence) / len(signatures)
        
        # Cross-domain correlation bonus
        correlation_bonus = min(1.0, correlation_matrix.mean_correlation * 1.5)
        
        # Pattern diversity bonus (more domains = better classification)
        domains_with_patterns = set(sig.domain for sig in signatures)
        diversity_bonus = min(1.0, len(domains_with_patterns) / 5.0)  # Normalize by 5 domains
        
        # Weighted combination
        sec_accuracy = (
            confidence_score * 0.4 + 
            emergence_score * 0.3 + 
            correlation_bonus * 0.2 + 
            diversity_bonus * 0.1
        )
        
        return min(1.0, sec_accuracy)
    
    def _calculate_pattern_assembly_success_rate(self, signatures: List[EmergenceSignature],
                                               correlation_matrix: CorrelationMatrix) -> float:
        """Calculate pattern assembly success rate."""
        if not signatures:
            return 0.0
        
        # Success based on having patterns across multiple domains
        domains_with_patterns = set(sig.domain for sig in signatures)
        total_tested_domains = max(1, len(correlation_matrix.domains)) if correlation_matrix.domains else 5
        domain_coverage = len(domains_with_patterns) / total_tested_domains
        
        # Factor in emergence strength with more realistic threshold
        moderate_emergence_patterns = [sig for sig in signatures if sig.emergence_strength >= 0.3]
        emergence_score = len(moderate_emergence_patterns) / len(signatures)
        
        # Pattern quality factor
        quality_patterns = [sig for sig in signatures if sig.confidence >= 0.3 and sig.emergence_strength >= 0.2]
        quality_score = len(quality_patterns) / len(signatures)
        
        # Weighted combination favoring domain coverage and quality
        assembly_rate = (
            domain_coverage * 0.5 + 
            emergence_score * 0.3 + 
            quality_score * 0.2
        )
        
        return min(1.0, assembly_rate)
    
    def _calculate_emergence_consistency_score(self, signatures: List[EmergenceSignature],
                                             correlation_matrix: CorrelationMatrix) -> float:
        """Calculate emergence consistency score."""
        if not signatures:
            return 0.0
        
        # Consistency based on correlation stability and pattern strength variance
        correlation_consistency = correlation_matrix.correlation_consistency
        
        # Calculate strength consistency within domains
        domain_strengths = {}
        for sig in signatures:
            if sig.domain not in domain_strengths:
                domain_strengths[sig.domain] = []
            domain_strengths[sig.domain].append(sig.emergence_strength)
        
        strength_consistencies = []
        for domain, strengths in domain_strengths.items():
            if len(strengths) > 1:
                consistency = 1.0 - np.std(strengths)
                strength_consistencies.append(max(0.0, consistency))
            else:
                strength_consistencies.append(1.0)
        
        avg_strength_consistency = np.mean(strength_consistencies) if strength_consistencies else 1.0
        
        return min(1.0, correlation_consistency * 0.5 + avg_strength_consistency * 0.5)
    
    def _calculate_phase1_readiness_score(self, sec_accuracy: float, pattern_assembly: float,
                                        emergence_consistency: float, avg_confidence: float) -> float:
        """Calculate overall Phase 1 readiness score."""
        # Weighted combination of core metrics
        core_score = (sec_accuracy + pattern_assembly + emergence_consistency) / 3.0
        
        # Bonus for high average confidence
        confidence_bonus = max(0.0, (avg_confidence - 0.5) * 0.4)
        
        return min(1.0, core_score + confidence_bonus)
    
    def _empty_metrics(self) -> ValidationMetrics:
        """Return empty metrics for cases with no signatures."""
        return ValidationMetrics(
            sec_classification_accuracy=0.0,
            pattern_assembly_success_rate=0.0,
            emergence_consistency_score=0.0,
            phase1_readiness_score=0.0,
            total_patterns_extracted=0,
            patterns_per_domain={},
            cross_domain_correlations=0.0,
            correlation_consistency=0.0,
            processing_time_seconds=0.0,
            memory_usage_mb=0.0,
            average_pattern_confidence=0.0,
            average_emergence_strength=0.0,
            pattern_diversity_score=0.0
        )

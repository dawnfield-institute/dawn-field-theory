#!/usr/bin/env python3
"""
SCBF Summary Generator
=====================

This module generates comprehensive summaries of SCBF (Symmetric Cognitive Bias Field) 
activation traces and structural emergence patterns for TinyCIMM-Euler experiments.
Provides detailed analysis of cognitive structure formation, field topology, and 
convergence metrics for publication-quality research documentation.

Author: Dawn Field Theory Research Team
Date: 2025-01-27
Version: 1.0
License: MIT (see LICENSE.md)

Key Features:
- SCBF activation trace analysis and summarization
- Structure count and complexity measurements
- Bias detection and field topology mapping
- Convergence metric calculation and tracking
- Publication-quality summary report generation
- Integration with visualization and logging modules
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SCBFMetrics:
    """Data structure for SCBF analysis metrics."""
    structure_count: int
    complexity_index: float
    bias_detection_score: float
    field_coherence: float
    convergence_rate: float
    stability_measure: float
    emergence_events: List[Dict]
    role_assignments: Dict[str, List[int]]
    
@dataclass
class ActivationTraceMetrics:
    """Data structure for activation trace analysis."""
    trace_length: int
    specialization_score: float
    convergence_point: Optional[int]
    dominant_patterns: List[Dict]
    role_transition_events: List[Dict]
    stability_windows: List[Tuple[int, int]]

class SCBFSummaryGenerator:
    """
    Generates comprehensive SCBF and activation trace summaries for TinyCIMM-Euler experiments.
    
    This class analyzes experimental data to extract key metrics about cognitive structure
    formation, bias field topology, and convergence patterns. It provides both quantitative
    metrics and qualitative insights for publication-quality research documentation.
    """
    
    def __init__(self, output_dir: str = "publication_outputs"):
        """
        Initialize the SCBF summary generator.
        
        Args:
            output_dir: Directory for saving analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different analysis types
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        
        logger.info(f"SCBF Summary Generator initialized with output directory: {self.output_dir}")
        
    def analyze_scbf_traces(self, activations: np.ndarray, timestamps: np.ndarray) -> SCBFMetrics:
        """
        Analyze SCBF activation traces to extract structural and field metrics.
        
        Args:
            activations: Network activation data [timesteps, neurons]
            timestamps: Corresponding timestamps for activations
            
        Returns:
            SCBFMetrics object containing comprehensive analysis
        """
        logger.info("Analyzing SCBF activation traces...")
        
        # Calculate structure count using clustering
        structure_count = self._calculate_structure_count(activations)
        
        # Calculate complexity index
        complexity_index = self._calculate_complexity_index(activations)
        
        # Detect bias patterns
        bias_detection_score = self._detect_bias_patterns(activations)
        
        # Calculate field coherence
        field_coherence = self._calculate_field_coherence(activations)
        
        # Calculate convergence rate
        convergence_rate = self._calculate_convergence_rate(activations, timestamps)
        
        # Calculate stability measure
        stability_measure = self._calculate_stability_measure(activations)
        
        # Detect emergence events
        emergence_events = self._detect_emergence_events(activations, timestamps)
        
        # Assign neuron roles
        role_assignments = self._assign_neuron_roles(activations)
        
        metrics = SCBFMetrics(
            structure_count=structure_count,
            complexity_index=complexity_index,
            bias_detection_score=bias_detection_score,
            field_coherence=field_coherence,
            convergence_rate=convergence_rate,
            stability_measure=stability_measure,
            emergence_events=emergence_events,
            role_assignments=role_assignments
        )
        
        logger.info(f"SCBF analysis complete. Found {structure_count} structures with complexity {complexity_index:.3f}")
        return metrics
    
    def analyze_activation_traces(self, activations: np.ndarray, timestamps: np.ndarray) -> ActivationTraceMetrics:
        """
        Analyze individual neuron activation traces for specialization and role patterns.
        
        Args:
            activations: Network activation data [timesteps, neurons]
            timestamps: Corresponding timestamps for activations
            
        Returns:
            ActivationTraceMetrics object containing trace analysis
        """
        logger.info("Analyzing individual activation traces...")
        
        trace_length = len(timestamps)
        
        # Calculate specialization score
        specialization_score = self._calculate_specialization_score(activations)
        
        # Find convergence point
        convergence_point = self._find_convergence_point(activations)
        
        # Identify dominant patterns
        dominant_patterns = self._identify_dominant_patterns(activations)
        
        # Detect role transition events
        role_transition_events = self._detect_role_transitions(activations, timestamps)
        
        # Find stability windows
        stability_windows = self._find_stability_windows(activations)
        
        metrics = ActivationTraceMetrics(
            trace_length=trace_length,
            specialization_score=specialization_score,
            convergence_point=convergence_point,
            dominant_patterns=dominant_patterns,
            role_transition_events=role_transition_events,
            stability_windows=stability_windows
        )
        
        logger.info(f"Activation trace analysis complete. Specialization score: {specialization_score:.3f}")
        return metrics
    
    def generate_field_topology_map(self, activations: np.ndarray, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive field topology map showing cognitive structure layout.
        
        Args:
            activations: Network activation data [timesteps, neurons]
            save_path: Optional path to save the topology map
            
        Returns:
            Dictionary containing field topology information
        """
        logger.info("Generating field topology map...")
        
        # Calculate pairwise neuron correlations
        correlations = np.corrcoef(activations.T)
        
        # Identify field regions using clustering
        n_clusters = min(5, activations.shape[1] // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        field_regions = kmeans.fit_predict(activations.T)
        
        # Calculate field strength and coherence
        field_strength = np.mean(np.abs(correlations))
        field_coherence = self._calculate_field_coherence(activations)
        
        # Identify key nodes (highly connected neurons)
        connection_strength = np.sum(np.abs(correlations), axis=1)
        key_nodes = np.argsort(connection_strength)[-5:]  # Top 5 connected neurons
        
        # Create topology map
        topology_map = {
            'field_regions': [int(x) for x in field_regions],
            'field_strength': float(field_strength),
            'field_coherence': float(field_coherence),
            'key_nodes': [int(x) for x in key_nodes],
            'connection_matrix': [[float(x) for x in row] for row in correlations],
            'cluster_centers': [[float(x) for x in row] for row in kmeans.cluster_centers_],
            'region_sizes': [int(np.sum(field_regions == i)) for i in range(n_clusters)]
        }
        
        # Save topology map if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(topology_map, f, indent=2)
            logger.info(f"Field topology map saved to {save_path}")
        
        return topology_map
    
    def calculate_convergence_metrics(self, activations: np.ndarray, timestamps: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive convergence metrics for the experiment.
        
        Args:
            activations: Network activation data [timesteps, neurons]
            timestamps: Corresponding timestamps for activations
            
        Returns:
            Dictionary containing convergence metrics
        """
        logger.info("Calculating convergence metrics...")
        
        # Calculate various convergence measures
        convergence_rate = self._calculate_convergence_rate(activations, timestamps)
        stability_measure = self._calculate_stability_measure(activations)
        
        # Calculate entropy reduction over time
        entropy_reduction = self._calculate_entropy_reduction(activations)
        
        # Calculate pattern formation speed
        pattern_formation_speed = self._calculate_pattern_formation_speed(activations)
        
        # Calculate final convergence quality
        convergence_quality = self._calculate_convergence_quality(activations)
        
        metrics = {
            'convergence_rate': float(convergence_rate),
            'stability_measure': float(stability_measure),
            'entropy_reduction': float(entropy_reduction),
            'pattern_formation_speed': float(pattern_formation_speed),
            'convergence_quality': float(convergence_quality),
            'total_timesteps': len(timestamps),
            'final_activation_variance': float(np.var(activations[-1])),
            'activation_range': float(np.max(activations) - np.min(activations))
        }
        
        logger.info(f"Convergence metrics calculated. Rate: {convergence_rate:.3f}, Quality: {convergence_quality:.3f}")
        return metrics
    
    def generate_summary_report(self, scbf_metrics: SCBFMetrics, trace_metrics: ActivationTraceMetrics,
                               convergence_metrics: Dict[str, float], experiment_config: Dict[str, Any]) -> str:
        """
        Generate a comprehensive summary report combining all analysis results.
        
        Args:
            scbf_metrics: SCBF analysis results
            trace_metrics: Activation trace analysis results
            convergence_metrics: Convergence analysis results
            experiment_config: Configuration used for the experiment
            
        Returns:
            String containing the formatted summary report
        """
        logger.info("Generating comprehensive summary report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
TinyCIMM-Euler SCBF Analysis Summary Report
==========================================

Generated: {timestamp}
Experiment: {experiment_config.get('experiment_type', 'Unknown')}
Duration: {trace_metrics.trace_length} timesteps

SCBF Structure Analysis
-----------------------
Structure Count: {scbf_metrics.structure_count}
Complexity Index: {scbf_metrics.complexity_index:.4f}
Bias Detection Score: {scbf_metrics.bias_detection_score:.4f}
Field Coherence: {scbf_metrics.field_coherence:.4f}
Stability Measure: {scbf_metrics.stability_measure:.4f}

Emergence Events: {len(scbf_metrics.emergence_events)}
Role Assignments: {len(scbf_metrics.role_assignments)} distinct roles

Activation Trace Analysis
-------------------------
Trace Length: {trace_metrics.trace_length}
Specialization Score: {trace_metrics.specialization_score:.4f}
Convergence Point: {trace_metrics.convergence_point if trace_metrics.convergence_point else 'Not reached'}
Dominant Patterns: {len(trace_metrics.dominant_patterns)}
Role Transitions: {len(trace_metrics.role_transition_events)}
Stability Windows: {len(trace_metrics.stability_windows)}

Convergence Metrics
-------------------
Convergence Rate: {convergence_metrics['convergence_rate']:.4f}
Stability Measure: {convergence_metrics['stability_measure']:.4f}
Entropy Reduction: {convergence_metrics['entropy_reduction']:.4f}
Pattern Formation Speed: {convergence_metrics['pattern_formation_speed']:.4f}
Convergence Quality: {convergence_metrics['convergence_quality']:.4f}
Final Activation Variance: {convergence_metrics['final_activation_variance']:.4f}

Key Findings
------------
- Cognitive structures: {scbf_metrics.structure_count} distinct structures identified
- Field topology: {scbf_metrics.field_coherence:.1%} coherence achieved
- Convergence: {'Successful' if trace_metrics.convergence_point else 'Partial'} convergence observed
- Stability: {len(trace_metrics.stability_windows)} stable periods identified

Recommendations
---------------
- Structure complexity suggests {'high' if scbf_metrics.complexity_index > 0.7 else 'moderate'} cognitive load
- Field coherence indicates {'strong' if scbf_metrics.field_coherence > 0.8 else 'moderate'} pattern formation
- Convergence quality {'meets' if convergence_metrics['convergence_quality'] > 0.8 else 'below'} publication standards

Configuration
-------------
{json.dumps(experiment_config, indent=2)}
"""
        
        # Save report to file
        report_path = self.output_dir / "summaries" / f"scbf_summary_{timestamp.replace(':', '-').replace(' ', '_')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Summary report generated and saved to {report_path}")
        return report
    
    def save_metrics_to_json(self, scbf_metrics: SCBFMetrics, trace_metrics: ActivationTraceMetrics,
                            convergence_metrics: Dict[str, float], filename: str = None) -> str:
        """
        Save all metrics to a structured JSON file for further analysis.
        
        Args:
            scbf_metrics: SCBF analysis results
            trace_metrics: Activation trace analysis results
            convergence_metrics: Convergence analysis results
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to the saved JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scbf_metrics_{timestamp}.json"
        
        # Convert dataclasses to dictionaries
        metrics_data = {
            'scbf_metrics': {
                'structure_count': scbf_metrics.structure_count,
                'complexity_index': scbf_metrics.complexity_index,
                'bias_detection_score': scbf_metrics.bias_detection_score,
                'field_coherence': scbf_metrics.field_coherence,
                'convergence_rate': scbf_metrics.convergence_rate,
                'stability_measure': scbf_metrics.stability_measure,
                'emergence_events': scbf_metrics.emergence_events,
                'role_assignments': scbf_metrics.role_assignments
            },
            'trace_metrics': {
                'trace_length': trace_metrics.trace_length,
                'specialization_score': trace_metrics.specialization_score,
                'convergence_point': trace_metrics.convergence_point,
                'dominant_patterns': trace_metrics.dominant_patterns,
                'role_transition_events': trace_metrics.role_transition_events,
                'stability_windows': trace_metrics.stability_windows
            },
            'convergence_metrics': convergence_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON file
        json_path = self.output_dir / "metrics" / filename
        with open(json_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to JSON file: {json_path}")
        return str(json_path)
    
    # Private helper methods
    def _calculate_structure_count(self, activations: np.ndarray) -> int:
        """Calculate the number of distinct cognitive structures using clustering."""
        # Use silhouette analysis to find optimal number of clusters
        max_clusters = min(10, activations.shape[1] // 2)
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(activations.T)
            score = silhouette_score(activations.T, labels)
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k
    
    def _calculate_complexity_index(self, activations: np.ndarray) -> float:
        """Calculate complexity index based on activation patterns."""
        # Use entropy of activation distributions
        entropy_sum = 0
        for i in range(activations.shape[1]):
            hist, _ = np.histogram(activations[:, i], bins=20)
            hist = hist / np.sum(hist)  # Normalize
            entropy_sum += stats.entropy(hist + 1e-10)  # Add small epsilon to avoid log(0)
        
        return entropy_sum / activations.shape[1]
    
    def _detect_bias_patterns(self, activations: np.ndarray) -> float:
        """Detect systematic bias patterns in activation data."""
        # Calculate activation skewness across neurons
        skewness_values = [stats.skew(activations[:, i]) for i in range(activations.shape[1])]
        return np.mean(np.abs(skewness_values))
    
    def _calculate_field_coherence(self, activations: np.ndarray) -> float:
        """Calculate field coherence based on inter-neuron correlations."""
        correlations = np.corrcoef(activations.T)
        # Remove diagonal elements and calculate mean absolute correlation
        mask = ~np.eye(correlations.shape[0], dtype=bool)
        return np.mean(np.abs(correlations[mask]))
    
    def _calculate_convergence_rate(self, activations: np.ndarray, timestamps: np.ndarray) -> float:
        """Calculate rate of convergence over time."""
        # Calculate variance reduction rate
        window_size = max(1, len(timestamps) // 10)
        variances = []
        
        for i in range(0, len(timestamps) - window_size, window_size):
            window_data = activations[i:i+window_size]
            variances.append(np.var(window_data))
        
        if len(variances) < 2:
            return 0.0
        
        # Calculate slope of variance reduction
        x = np.arange(len(variances))
        slope, _, _, _, _ = stats.linregress(x, variances)
        return abs(slope)  # Return absolute slope as convergence rate
    
    def _calculate_stability_measure(self, activations: np.ndarray) -> float:
        """Calculate stability measure based on activation consistency."""
        # Calculate coefficient of variation for each neuron
        cv_values = []
        for i in range(activations.shape[1]):
            mean_val = np.mean(activations[:, i])
            std_val = np.std(activations[:, i])
            if mean_val != 0:
                cv_values.append(std_val / abs(mean_val))
            else:
                cv_values.append(0)
        
        return 1.0 / (1.0 + np.mean(cv_values))  # Inverse relationship with CV
    
    def _detect_emergence_events(self, activations: np.ndarray, timestamps: np.ndarray) -> List[Dict]:
        """Detect emergence events in activation patterns."""
        events = []
        
        # Detect sudden changes in activation patterns
        for i in range(1, len(timestamps)):
            diff = np.mean(np.abs(activations[i] - activations[i-1]))
            if diff > 2 * np.std(np.diff(activations, axis=0)):
                events.append({
                    'timestamp': timestamps[i],
                    'timestep': i,
                    'magnitude': float(diff),
                    'type': 'sudden_change'
                })
        
        return events
    
    def _assign_neuron_roles(self, activations: np.ndarray) -> Dict[str, List[int]]:
        """Assign roles to neurons based on activation patterns."""
        roles = {
            'high_activity': [],
            'low_activity': [],
            'variable': [],
            'stable': []
        }
        
        for i in range(activations.shape[1]):
            neuron_data = activations[:, i]
            mean_activity = np.mean(neuron_data)
            variability = np.std(neuron_data)
            
            if mean_activity > np.percentile(np.mean(activations, axis=0), 75):
                roles['high_activity'].append(i)
            elif mean_activity < np.percentile(np.mean(activations, axis=0), 25):
                roles['low_activity'].append(i)
            
            if variability > np.percentile(np.std(activations, axis=0), 75):
                roles['variable'].append(i)
            elif variability < np.percentile(np.std(activations, axis=0), 25):
                roles['stable'].append(i)
        
        return roles
    
    def _calculate_specialization_score(self, activations: np.ndarray) -> float:
        """Calculate specialization score for activation traces."""
        # Calculate sparsity of activation patterns
        sparsity_scores = []
        for i in range(activations.shape[1]):
            neuron_data = activations[:, i]
            # Calculate Gini coefficient as sparsity measure
            sorted_data = np.sort(np.abs(neuron_data))
            n = len(sorted_data)
            cumsum = np.cumsum(sorted_data)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            sparsity_scores.append(gini)
        
        return np.mean(sparsity_scores)
    
    def _find_convergence_point(self, activations: np.ndarray) -> Optional[int]:
        """Find the convergence point in activation traces."""
        # Calculate moving average of activation variance
        window_size = max(1, len(activations) // 20)
        variances = []
        
        for i in range(window_size, len(activations)):
            window_data = activations[i-window_size:i]
            variances.append(np.var(window_data))
        
        if len(variances) < 2:
            return None
        
        # Find point where variance stops decreasing significantly
        threshold = np.std(variances) * 0.1
        for i in range(len(variances) - 1):
            if abs(variances[i+1] - variances[i]) < threshold:
                return i + window_size
        
        return None
    
    def _identify_dominant_patterns(self, activations: np.ndarray) -> List[Dict]:
        """Identify dominant patterns in activation data."""
        patterns = []
        
        # Use PCA to identify dominant patterns
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(5, activations.shape[1]))
        pca.fit(activations)
        
        for i, (component, variance) in enumerate(zip(pca.components_, pca.explained_variance_ratio_)):
            patterns.append({
                'pattern_id': i,
                'explained_variance': float(variance),
                'dominant_neurons': np.argsort(np.abs(component))[-3:].tolist()
            })
        
        return patterns
    
    def _detect_role_transitions(self, activations: np.ndarray, timestamps: np.ndarray) -> List[Dict]:
        """Detect role transition events in neuron activations."""
        transitions = []
        
        # Simple approach: detect when neuron ranking changes significantly
        window_size = max(1, len(timestamps) // 10)
        
        for i in range(window_size, len(timestamps) - window_size):
            before = np.mean(activations[i-window_size:i], axis=0)
            after = np.mean(activations[i:i+window_size], axis=0)
            
            rank_before = np.argsort(before)
            rank_after = np.argsort(after)
            
            # Calculate rank correlation
            correlation = stats.spearmanr(rank_before, rank_after)[0]
            
            if correlation < 0.8:  # Significant ranking change
                transitions.append({
                    'timestamp': timestamps[i],
                    'timestep': i,
                    'rank_correlation': float(correlation),
                    'type': 'role_transition'
                })
        
        return transitions
    
    def _find_stability_windows(self, activations: np.ndarray) -> List[Tuple[int, int]]:
        """Find windows of stable activation patterns."""
        stability_windows = []
        window_size = max(1, len(activations) // 20)
        threshold = np.std(activations) * 0.1
        
        i = 0
        while i < len(activations) - window_size:
            # Check if current window is stable
            window_data = activations[i:i+window_size]
            if np.std(window_data) < threshold:
                # Find end of stable period
                j = i + window_size
                while j < len(activations) and np.std(activations[i:j]) < threshold:
                    j += 1
                stability_windows.append((i, j-1))
                i = j
            else:
                i += 1
        
        return stability_windows
    
    def _calculate_entropy_reduction(self, activations: np.ndarray) -> float:
        """Calculate entropy reduction over time."""
        # Calculate entropy at beginning and end
        start_entropy = stats.entropy(np.histogram(activations[:len(activations)//10].flatten(), bins=20)[0] + 1e-10)
        end_entropy = stats.entropy(np.histogram(activations[-len(activations)//10:].flatten(), bins=20)[0] + 1e-10)
        
        return max(0, start_entropy - end_entropy)
    
    def _calculate_pattern_formation_speed(self, activations: np.ndarray) -> float:
        """Calculate speed of pattern formation."""
        # Calculate correlation with final state over time
        final_state = activations[-1]
        correlations = []
        
        for i in range(len(activations)):
            corr = np.corrcoef(activations[i], final_state)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        
        # Find 90% correlation point
        target_corr = 0.9 * max(correlations)
        for i, corr in enumerate(correlations):
            if corr >= target_corr:
                return 1.0 / (i + 1)  # Inverse of timesteps to reach target
        
        return 0.0
    
    def _calculate_convergence_quality(self, activations: np.ndarray) -> float:
        """Calculate quality of convergence."""
        # Calculate stability of final state
        final_portion = activations[-len(activations)//10:]
        stability = 1.0 / (1.0 + np.std(final_portion))
        
        # Calculate pattern consistency
        correlations = []
        for i in range(len(final_portion) - 1):
            corr = np.corrcoef(final_portion[i], final_portion[i+1])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        
        consistency = np.mean(correlations) if correlations else 0
        
        return (stability + consistency) / 2.0

if __name__ == "__main__":
    # Example usage
    logger.info("SCBF Summary Generator - Example Usage")
    
    # Generate sample data for testing
    np.random.seed(42)
    sample_activations = np.random.randn(1000, 10)
    sample_timestamps = np.arange(1000)
    
    # Initialize generator
    generator = SCBFSummaryGenerator("example_outputs")
    
    # Run analysis
    scbf_metrics = generator.analyze_scbf_traces(sample_activations, sample_timestamps)
    trace_metrics = generator.analyze_activation_traces(sample_activations, sample_timestamps)
    convergence_metrics = generator.calculate_convergence_metrics(sample_activations, sample_timestamps)
    
    # Generate topology map
    topology_map = generator.generate_field_topology_map(sample_activations)
    
    # Generate summary report
    config = {'experiment_type': 'example', 'neurons': 10, 'timesteps': 1000}
    report = generator.generate_summary_report(scbf_metrics, trace_metrics, convergence_metrics, config)
    
    # Save metrics
    json_path = generator.save_metrics_to_json(scbf_metrics, trace_metrics, convergence_metrics)
    
    logger.info("Example analysis complete!")
    print(f"Summary report saved. JSON metrics: {json_path}")

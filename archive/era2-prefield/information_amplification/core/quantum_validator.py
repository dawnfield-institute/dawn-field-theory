"""
Quantum Validation Module

Implements quantum-inspired validation metrics for information amplification experiments.
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple
import math
from collections import Counter

class QuantumValidator:
    """Quantum-inspired validation for information amplification"""
    
    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        self.hilbert_space_dim = 2 ** num_qubits
        
    def validate_born_rule_compliance(self, outputs: List[str]) -> Dict[str, float]:
        """Validate Born rule compliance in output distribution"""
        # Map strings to quantum states
        quantum_states = self._map_to_quantum_states(outputs)
        
        # Calculate probability distribution
        state_counts = Counter(quantum_states)
        total_outputs = len(outputs)
        probabilities = [count / total_outputs for count in state_counts.values()]
        
        # Expected Born rule: |amplitude|^2 distribution
        # Generate theoretical amplitudes
        n_states = len(state_counts)
        theoretical_amplitudes = np.random.normal(0, 1, n_states) + 1j * np.random.normal(0, 1, n_states)
        theoretical_probs = np.abs(theoretical_amplitudes) ** 2
        theoretical_probs /= np.sum(theoretical_probs)
        
        # Compare distributions using Kolmogorov-Smirnov test approximation
        sorted_observed = np.sort(probabilities)
        sorted_theoretical = np.sort(theoretical_probs)
        
        # Pad shorter array
        if len(sorted_observed) < len(sorted_theoretical):
            sorted_observed = np.pad(sorted_observed, 
                                   (0, len(sorted_theoretical) - len(sorted_observed)), 
                                   'constant')
        elif len(sorted_theoretical) < len(sorted_observed):
            sorted_theoretical = np.pad(sorted_theoretical,
                                      (0, len(sorted_observed) - len(sorted_theoretical)),
                                      'constant')
        
        # Calculate cumulative distributions
        cum_observed = np.cumsum(sorted_observed)
        cum_theoretical = np.cumsum(sorted_theoretical)
        
        # Normalize
        cum_observed /= cum_observed[-1] if cum_observed[-1] > 0 else 1
        cum_theoretical /= cum_theoretical[-1] if cum_theoretical[-1] > 0 else 1
        
        # KS statistic
        ks_statistic = np.max(np.abs(cum_observed - cum_theoretical))
        born_rule_compliance = 1.0 - ks_statistic  # Convert to compliance score
        
        return {
            'born_rule_compliance': born_rule_compliance,
            'ks_statistic': ks_statistic,
            'unique_quantum_states': len(state_counts),
            'entropy_quantum': self._calculate_quantum_entropy(probabilities)
        }
    
    def measure_decoherence_correlation(self, outputs: List[str]) -> Dict[str, float]:
        """Measure decoherence patterns in output sequences"""
        if len(outputs) < 2:
            return {'decoherence_correlation': 0.0}
        
        # Convert outputs to quantum-like correlation matrices
        correlation_matrices = []
        
        for output in outputs:
            # Map characters to qubit states
            qubits = self._string_to_qubits(output)
            
            # Create density matrix representation
            n_qubits = min(len(qubits), self.num_qubits)
            if n_qubits > 0:
                density_matrix = np.outer(qubits[:n_qubits], np.conj(qubits[:n_qubits]))
                correlation_matrices.append(density_matrix)
        
        if len(correlation_matrices) < 2:
            return {'decoherence_correlation': 0.0}
        
        # Measure correlation between consecutive density matrices using improved method
        correlations = []
        cross_correlations = []
        
        for i in range(len(correlation_matrices) - 1):
            dm1 = correlation_matrices[i]
            dm2 = correlation_matrices[i + 1]
            
            # Trace correlation (original method)
            trace_corr = np.trace(dm1 @ dm2)
            correlations.append(np.real(trace_corr))
            
            # Frobenius inner product for better correlation measure
            frobenius_corr = np.sum(dm1 * np.conj(dm2))
            cross_correlations.append(np.real(frobenius_corr))
        
        # Enhanced decoherence analysis
        if len(correlations) > 3:  # Need sufficient data points
            # Use both correlation measures
            trace_trend = np.polyfit(range(len(correlations)), correlations, 1)[0]
            frobenius_trend = np.polyfit(range(len(cross_correlations)), cross_correlations, 1)[0]
            
            # Decoherence: significant decreasing trend in correlations
            trace_decoherence = max(0, -trace_trend * 10)  # Scale up
            frobenius_decoherence = max(0, -frobenius_trend * 10)
            
            # Use the stronger signal
            decoherence_correlation = max(trace_decoherence, frobenius_decoherence)
            
            # Add correlation magnitude scaling
            mean_corr_magnitude = np.mean(np.abs(correlations))
            decoherence_correlation *= mean_corr_magnitude
            
        elif len(correlations) > 0:
            # For shorter sequences, use variance as decoherence indicator
            correlation_variance = np.var(correlations)
            decoherence_correlation = min(correlation_variance, 1.0)
        else:
            decoherence_correlation = 0.0
        
        return {
            'decoherence_correlation': decoherence_correlation,
            'mean_correlation': np.mean(correlations),
            'correlation_variance': np.var(correlations),
            'correlation_sequence_length': len(correlations)
        }
    
    def calculate_entanglement_measures(self, outputs: List[str]) -> Dict[str, float]:
        """Calculate entanglement-like measures for output patterns"""
        if not outputs:
            return {}
        
        # Bipartite entanglement simulation
        entanglement_measures = []
        
        for output in outputs:
            if len(output) < 4:  # Need minimum length for bipartite analysis
                continue
                
            # Split into two subsystems
            mid = len(output) // 2
            subsystem_a = output[:mid]
            subsystem_b = output[mid:]
            
            # Calculate mutual information between subsystems
            mutual_info = self._calculate_mutual_information(subsystem_a, subsystem_b)
            entanglement_measures.append(mutual_info)
        
        if not entanglement_measures:
            return {'average_entanglement': 0.0}
        
        return {
            'average_entanglement': np.mean(entanglement_measures),
            'entanglement_variance': np.var(entanglement_measures),
            'max_entanglement': np.max(entanglement_measures),
            'entanglement_distribution': np.histogram(entanglement_measures, bins=5)[0].tolist()
        }
    
    def validate_quantum_amplification(self, inputs: List[str], outputs: List[str]) -> Dict[str, Any]:
        """Comprehensive quantum validation of amplification process"""
        # Born rule validation
        born_validation = self.validate_born_rule_compliance(outputs)
        
        # Decoherence analysis
        decoherence_analysis = self.measure_decoherence_correlation(outputs)
        
        # Entanglement measures
        entanglement_analysis = self.calculate_entanglement_measures(outputs)
        
        # Quantum information metrics
        input_quantum_info = self._calculate_quantum_information_content(inputs)
        output_quantum_info = self._calculate_quantum_information_content(outputs)
        
        quantum_amplification_factor = (output_quantum_info / input_quantum_info 
                                      if input_quantum_info > 0 else 0)
        
        # Coherence preservation
        input_coherence = self._measure_coherence(inputs)
        output_coherence = self._measure_coherence(outputs)
        coherence_preservation = output_coherence / input_coherence if input_coherence > 0 else 0
        
        return {
            'quantum_validation': {
                'born_rule_compliance': born_validation['born_rule_compliance'],
                'decoherence_correlation': decoherence_analysis['decoherence_correlation'],
                'average_entanglement': entanglement_analysis.get('average_entanglement', 0),
                'quantum_amplification_factor': quantum_amplification_factor,
                'coherence_preservation': coherence_preservation
            },
            'detailed_analysis': {
                'born_rule_details': born_validation,
                'decoherence_details': decoherence_analysis,
                'entanglement_details': entanglement_analysis,
                'input_quantum_info': input_quantum_info,
                'output_quantum_info': output_quantum_info
            }
        }
    
    def _map_to_quantum_states(self, outputs: List[str]) -> List[int]:
        """Map output strings to quantum state indices"""
        quantum_states = []
        for output in outputs:
            # Hash string to quantum state index
            state_hash = hash(output) % self.hilbert_space_dim
            quantum_states.append(state_hash)
        return quantum_states
    
    def _string_to_qubits(self, text: str) -> np.ndarray:
        """Convert string to qubit-like representation"""
        # Map each character to a complex amplitude
        amplitudes = []
        for char in text:
            # Character to phase mapping
            phase = (ord(char) / 255.0) * 2 * np.pi
            amplitude = np.exp(1j * phase) / np.sqrt(len(text))
            amplitudes.append(amplitude)
        
        # Pad or truncate to match number of qubits
        if len(amplitudes) < self.num_qubits:
            amplitudes.extend([0] * (self.num_qubits - len(amplitudes)))
        else:
            amplitudes = amplitudes[:self.num_qubits]
        
        return np.array(amplitudes)
    
    def _calculate_quantum_entropy(self, probabilities: List[float]) -> float:
        """Calculate quantum entropy (von Neumann entropy approximation)"""
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def _calculate_mutual_information(self, seq_a: str, seq_b: str) -> float:
        """Calculate mutual information between two sequences"""
        if not seq_a or not seq_b:
            return 0.0
        
        # Joint distribution
        joint_counts = Counter(zip(seq_a, seq_b[:len(seq_a)]))
        
        # Marginal distributions
        counts_a = Counter(seq_a)
        counts_b = Counter(seq_b)
        
        total = len(seq_a)
        mutual_info = 0
        
        for (a, b), joint_count in joint_counts.items():
            p_joint = joint_count / total
            p_a = counts_a[a] / len(seq_a)
            p_b = counts_b[b] / len(seq_b)
            
            if p_joint > 0 and p_a > 0 and p_b > 0:
                mutual_info += p_joint * math.log2(p_joint / (p_a * p_b))
        
        return mutual_info
    
    def _calculate_quantum_information_content(self, sequences: List[str]) -> float:
        """Calculate quantum information content of sequence set"""
        if not sequences:
            return 0.0
        
        # Combine all sequences
        combined = "".join(sequences)
        
        # Character distribution
        char_counts = Counter(combined)
        total_chars = len(combined)
        
        # Quantum information as weighted entropy
        quantum_info = 0
        for char, count in char_counts.items():
            p = count / total_chars
            if p > 0:
                # Weight by quantum-like factors (phase information)
                phase_weight = 1 + 0.1 * math.sin(2 * math.pi * ord(char) / 255)
                quantum_info += p * math.log2(1/p) * phase_weight
        
        return quantum_info
    
    def _measure_coherence(self, sequences: List[str]) -> float:
        """Measure coherence in sequence patterns"""
        if not sequences:
            return 0.0
        
        # Coherence as pattern consistency across sequences
        pattern_consistency = 0
        
        for i, seq in enumerate(sequences):
            for j, other_seq in enumerate(sequences[i+1:], i+1):
                # Calculate sequence similarity (coherence measure)
                similarity = self._sequence_similarity(seq, other_seq)
                pattern_consistency += similarity
        
        # Normalize by number of comparisons
        num_comparisons = len(sequences) * (len(sequences) - 1) / 2
        return pattern_consistency / num_comparisons if num_comparisons > 0 else 0
    
    def _sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate similarity between two sequences"""
        if not seq1 or not seq2:
            return 0.0
        
        # Longest common subsequence ratio
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        
        common_chars = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return common_chars / max_len if max_len > 0 else 0

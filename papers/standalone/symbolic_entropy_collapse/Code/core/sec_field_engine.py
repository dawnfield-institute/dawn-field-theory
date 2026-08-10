"""
SEC (Symbolic Entropy Collapse) Field Engine

Implements authentic entropy field dynamics with real gradient calculations
and natural field evolution.
"""

import numpy as np
import random
import itertools
from typing import List, Tuple, Dict, Any, Set
from collections import Counter
import zlib
import math

class AuthenticSECField:
    """Authentic SEC field with real entropy dynamics"""
    
    def __init__(self, 
                 base_alphabet: List[str],
                 mutation_alphabet: List[str],
                 input_length: int,
                 output_length_range: Tuple[int, int],
                 field_size: int = 64,
                 temperature: float = 1.0):
        
        self.base_alphabet = base_alphabet
        self.mutation_alphabet = mutation_alphabet
        self.full_alphabet = base_alphabet + mutation_alphabet
        self.input_length = input_length
        self.output_min_len, self.output_max_len = output_length_range
        self.field_size = field_size
        self.temperature = temperature
        
        # Initialize authentic entropy field
        self.entropy_field = np.random.exponential(scale=2.0, size=(field_size, field_size))
        self.potential_field = np.zeros((field_size, field_size))
        self.flow_field = np.zeros((field_size, field_size, 2))
        
        # Generate input space
        self.input_space = list(itertools.product(base_alphabet, repeat=input_length))
        self.input_strings = ["".join(seq) for seq in self.input_space]
        
        # Field evolution tracking
        self.evolution_history = []
        self.critical_events = []
        self.attractor_positions = []
    
    def evolve_field(self, steps: int = 10):
        """Evolve entropy field through authentic dynamics"""
        for step in range(steps):
            # Calculate genuine entropy gradients
            grad_x, grad_y = np.gradient(self.entropy_field)
            laplacian = np.gradient(grad_x, axis=0)[0] + np.gradient(grad_y, axis=1)[1]
            
            # Natural diffusion with conservation
            diffusion_rate = 0.1
            self.entropy_field += diffusion_rate * laplacian
            
            # Critical point formation through spontaneous symmetry breaking
            mean_entropy = np.mean(self.entropy_field)
            critical_mask = (self.entropy_field > 1.5 * mean_entropy)
            
            if np.any(critical_mask):
                # Genuine collapse at critical points
                collapse_positions = np.where(critical_mask)
                for i, j in zip(collapse_positions[0], collapse_positions[1]):
                    # Authentic collapse with information condensation
                    collapse_radius = 3
                    x_slice = slice(max(0, i-collapse_radius), min(self.field_size, i+collapse_radius))
                    y_slice = slice(max(0, j-collapse_radius), min(self.field_size, j+collapse_radius))
                    
                    local_entropy = self.entropy_field[x_slice, y_slice]
                    total_entropy = np.sum(local_entropy)
                    
                    # Collapse concentrates entropy at center
                    self.entropy_field[x_slice, y_slice] *= 0.3
                    self.entropy_field[i, j] = total_entropy * 0.4
                    
                    # Record authentic critical event
                    self.critical_events.append({
                        'step': step,
                        'position': (i, j),
                        'collapsed_entropy': total_entropy,
                        'type': 'authentic_collapse'
                    })
            
            # Update potential field from entropy configuration
            entropy_safe = np.maximum(self.entropy_field, 1e-10)  # Ensure positive values
            self.potential_field = -self.temperature * np.log(entropy_safe)
            
            # Calculate flow field from potential gradients
            pot_grad_x, pot_grad_y = np.gradient(self.potential_field)
            self.flow_field[:, :, 0] = -pot_grad_x
            self.flow_field[:, :, 1] = -pot_grad_y
            
            # Energy conservation: normalize total entropy
            total_entropy = np.sum(self.entropy_field)
            if total_entropy > 0:
                # Maintain energy scale while allowing redistribution
                self.entropy_field *= (self.field_size * self.field_size) / total_entropy
            
            # Record field state
            field_coherence = np.std(self.entropy_field) / (np.mean(self.entropy_field) + 1e-10)
            self.evolution_history.append({
                'step': step,
                'mean_entropy': np.mean(self.entropy_field),
                'entropy_variance': np.var(self.entropy_field),
                'field_coherence': field_coherence,
                'critical_points': np.sum(critical_mask),
                'total_energy': total_entropy
            })
    
    def detect_attractors(self) -> List[Tuple[int, int]]:
        """Detect authentic emergent attractors with improved sensitivity"""
        # Find local minima in potential field (attractors)
        potential_grad_mag = np.sqrt(
            np.gradient(self.potential_field, axis=0)**2 + 
            np.gradient(self.potential_field, axis=1)**2
        )
        
        # More sensitive attractor detection: relaxed threshold
        mean_grad = np.mean(potential_grad_mag)
        attractor_threshold = 0.3 * mean_grad  # Increased from 0.1
        attractor_mask = potential_grad_mag < attractor_threshold
        attractor_positions = list(zip(*np.where(attractor_mask)))
        
        # Filter for genuine attractors with more realistic entropy criteria
        genuine_attractors = []
        mean_entropy = np.mean(self.entropy_field)
        entropy_threshold = 1.05 * mean_entropy  # Reduced from 1.2
        
        for pos in attractor_positions:
            i, j = pos
            # Larger neighborhood for entropy evaluation
            local_entropy = self.entropy_field[max(0,i-3):min(self.field_size,i+4),
                                             max(0,j-3):min(self.field_size,j+4)]
            
            # Multiple criteria for authentic attractors
            local_mean = np.mean(local_entropy)
            local_variance = np.var(local_entropy)
            
            # Accept if either above-average entropy OR high variance (structure)
            if (local_mean > entropy_threshold or 
                local_variance > 1.1 * np.var(self.entropy_field)):
                genuine_attractors.append(pos)
        
        self.attractor_positions = genuine_attractors
        
        # Debug logging
        if len(genuine_attractors) == 0 and len(attractor_positions) > 0:
            print(f"DEBUG: Found {len(attractor_positions)} candidate attractors, "
                  f"but 0 met entropy criteria (threshold: {entropy_threshold:.3f}, "
                  f"mean: {mean_entropy:.3f})")
        
        return genuine_attractors
    
    def generate_output(self, seed: str = None) -> str:
        """Generate output through SEC field collapse"""
        # Sample position based on entropy field probability
        flat_entropy = self.entropy_field.flatten()
        
        # Ensure all probabilities are non-negative
        flat_entropy = np.abs(flat_entropy)
        
        # Avoid division by zero
        total_entropy = np.sum(flat_entropy)
        if total_entropy == 0:
            probabilities = np.ones(len(flat_entropy)) / len(flat_entropy)
        else:
            probabilities = flat_entropy / total_entropy
        
        sampled_idx = np.random.choice(len(probabilities), p=probabilities)
        field_i, field_j = divmod(sampled_idx, self.field_size)
        
        # Check for critical event (high entropy concentration)
        local_entropy = self.entropy_field[field_i, field_j]
        if local_entropy > 2.0 * np.mean(self.entropy_field):
            self.critical_events.append({
                'type': 'authentic_collapse',
                'position': (field_i, field_j),
                'entropy_level': local_entropy,
                'timestamp': len(self.critical_events)
            })
        
        # Generate length based on local field intensity
        local_intensity = self.entropy_field[field_i, field_j]
        intensity_factor = min(2.0, local_intensity / np.mean(self.entropy_field))
        base_length = random.randint(self.output_min_len, self.output_max_len)
        adjusted_length = int(base_length * intensity_factor)
        
        # Generate sequence influenced by local flow field
        sequence = []
        current_pos = [field_i, field_j]
        
        for pos in range(adjusted_length):
            # Character selection based on position in alphabet space
            pos_factor = (current_pos[0] + current_pos[1]) / (2 * self.field_size)
            
            if pos_factor > 0.6:  # High field regions favor mutations
                char = random.choice(self.mutation_alphabet)
            else:
                char = random.choice(self.base_alphabet)
            
            sequence.append(char)
            
            # Move according to flow field
            if (0 <= current_pos[0] < self.field_size and 
                0 <= current_pos[1] < self.field_size):
                flow_x = self.flow_field[int(current_pos[0]), int(current_pos[1]), 0]
                flow_y = self.flow_field[int(current_pos[0]), int(current_pos[1]), 1]
                
                # Small step in flow direction
                step_size = 0.5
                current_pos[0] = max(0, min(self.field_size-1, 
                                           current_pos[0] + step_size * flow_x))
                current_pos[1] = max(0, min(self.field_size-1, 
                                           current_pos[1] + step_size * flow_y))
        
        return "".join(sequence)
    
    def generate_batch(self, num_outputs: int) -> List[str]:
        """Generate batch with field evolution"""
        # Evolve field before generation
        self.evolve_field(steps=5)
        
        outputs = []
        for _ in range(num_outputs):
            output = self.generate_output()
            outputs.append(output)
            
            # Micro-evolution after each generation
            if random.random() < 0.3:
                self.evolve_field(steps=1)
        
        return outputs
    
    def analyze_outputs(self, outputs: List[str]) -> Dict[str, Any]:
        """Analyze SEC field generation results"""
        unique_outputs = list(set(outputs))
        
        # Basic amplification metrics
        input_space_size = len(self.input_space)
        output_space_observed = len(unique_outputs)
        combinatorial_amplification = output_space_observed / input_space_size
        
        # Complexity analysis
        input_concat = "".join(self.input_strings)
        output_concat = "".join(outputs)
        
        input_kolmogorov = len(zlib.compress(input_concat.encode('utf-8'), level=9))
        output_kolmogorov = len(zlib.compress(output_concat.encode('utf-8'), level=9))
        complexity_amplification = output_kolmogorov / max(input_kolmogorov, 1)
        
        # Field-specific metrics
        self.detect_attractors()
        
        # Emergence tracking
        attractor_emergence_events = len([e for e in self.critical_events 
                                        if e['type'] == 'authentic_collapse'])
        
        # Thermodynamic consistency
        if len(self.evolution_history) > 1:
            energy_conservation = np.std([h['total_energy'] for h in self.evolution_history])
            field_coherence_evolution = [h['field_coherence'] for h in self.evolution_history]
            coherence_improvement = (field_coherence_evolution[-1] - field_coherence_evolution[0]
                                   if len(field_coherence_evolution) > 1 else 0)
        else:
            energy_conservation = 0
            coherence_improvement = 0
        
        # Structural analysis
        hierarchical_emergence = {}
        for n in range(2, 8):
            all_ngrams = set()
            for text in outputs:
                all_ngrams.update(self._extract_ngrams(text, n))
            hierarchical_emergence[f"{n}-gram"] = len(all_ngrams)
        
        return {
            'method': 'authentic_sec_field',
            'combinatorial_amplification': combinatorial_amplification,
            'complexity_amplification': complexity_amplification,
            'attractor_emergence_events': attractor_emergence_events,
            'authentic_attractors_detected': len(self.attractor_positions),
            'field_evolution_steps': len(self.evolution_history),
            'energy_conservation_stability': energy_conservation,
            'coherence_improvement': coherence_improvement,
            'critical_collapse_events': len(self.critical_events),
            'hierarchical_emergence': hierarchical_emergence,
            'unique_outputs': len(unique_outputs),
            'total_outputs': len(outputs)
        }
    
    def _extract_ngrams(self, text: str, n: int) -> set:
        """Extract n-grams from text"""
        if len(text) < n:
            return set()
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def get_field_diagnostics(self) -> Dict[str, Any]:
        """Get detailed field state diagnostics"""
        return {
            'field_size': self.field_size,
            'mean_entropy': np.mean(self.entropy_field),
            'entropy_variance': np.var(self.entropy_field),
            'potential_range': (np.min(self.potential_field), np.max(self.potential_field)),
            'flow_magnitude': np.mean(np.sqrt(self.flow_field[:,:,0]**2 + self.flow_field[:,:,1]**2)),
            'evolution_history_length': len(self.evolution_history),
            'critical_events_count': len(self.critical_events),
            'attractor_count': len(self.attractor_positions)
        }

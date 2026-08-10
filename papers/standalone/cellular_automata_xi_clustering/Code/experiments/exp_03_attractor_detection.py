#!/usr/bin/env python3
"""
Experiment 03: Attractor Detection via SEC/Prime Harmonic Methods
==================================================================

Uses techniques from SEC Prime Manifold and Prime Harmonic Manifold experiments
to detect where CA rules stabilize into attractor states.

Key methods:
1. Run-length encoding analysis (analogous to prime gaps)
2. SEC phase transition detection (φ and Ξ convergence)
3. Harmonic stability analysis (chord distance minimization)
4. Entropy stabilization tracking

Hypothesis: CA rules stabilize at the same critical points (φ, Ξ) 
found in SEC/prime harmonic work.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from ca_simulator import ElementaryCA, CAState, WolframClass, RULE_CLASSIFICATIONS


# Constants from PAC theory
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
XI = 1.0571                  # PAC balance operator
INV_PHI = 1 / PHI            # ≈ 0.618


@dataclass
class AttractorInfo:
    """Information about detected attractor state."""
    rule: int
    attractor_type: str  # 'fixed_point', 'limit_cycle', 'quasi_periodic', 'chaotic'
    stabilization_step: Optional[int]
    period: Optional[int]
    final_entropy: float
    entropy_variance: float
    phi_transitions: int
    xi_transitions: int
    run_length_ratio: float  # Key SEC metric
    harmonic_stability: float


class SECAnalyzer:
    """
    SEC-style analysis for CA evolution.
    
    Applies run-length encoding and ratio analysis from SEC Prime Manifold.
    """
    
    def compute_run_lengths(self, state: np.ndarray) -> np.ndarray:
        """Convert binary state to run-length encoding (like prime gaps)."""
        if len(state) == 0:
            return np.array([])
        
        runs = []
        current_val = state[0]
        current_length = 1
        
        for i in range(1, len(state)):
            if state[i] == current_val:
                current_length += 1
            else:
                runs.append(current_length)
                current_val = state[i]
                current_length = 1
        runs.append(current_length)
        
        return np.array(runs)
    
    def compute_positive_negative_runs(self, state: np.ndarray) -> Tuple[List[int], List[int]]:
        """Separate runs into positive (1s) and negative (0s) like SEC."""
        runs = []
        values = []
        
        if len(state) == 0:
            return [], []
        
        current_val = state[0]
        current_length = 1
        
        for i in range(1, len(state)):
            if state[i] == current_val:
                current_length += 1
            else:
                runs.append(current_length)
                values.append(current_val)
                current_val = state[i]
                current_length = 1
        runs.append(current_length)
        values.append(current_val)
        
        positive_runs = [r for r, v in zip(runs, values) if v == 1]
        negative_runs = [r for r, v in zip(runs, values) if v == 0]
        
        return positive_runs, negative_runs
    
    def compute_run_length_ratio(self, state: np.ndarray) -> float:
        """
        Compute L+/L- ratio (from SEC Prime Manifold).
        
        At critical point, this should equal φ.
        """
        pos_runs, neg_runs = self.compute_positive_negative_runs(state)
        
        if not pos_runs or not neg_runs:
            return 1.0
        
        mean_pos = np.mean(pos_runs)
        mean_neg = np.mean(neg_runs)
        
        if mean_neg == 0:
            return float('inf')
        
        return mean_pos / mean_neg
    
    def detect_phase_transition(self, history: np.ndarray, 
                                 window: int = 20) -> List[Dict[str, Any]]:
        """
        Detect phase transitions in CA evolution using SEC metrics.
        
        Looks for:
        - φ transitions (run ratio → φ)
        - Ξ transitions (run ratio → Ξ)
        """
        transitions = []
        
        for t in range(window, len(history)):
            # Compute run length ratio at this timestep
            ratio = self.compute_run_length_ratio(history[t])
            
            # Check for φ convergence
            if abs(ratio - PHI) < 0.05:
                transitions.append({
                    'step': t,
                    'type': 'phi',
                    'value': float(ratio),
                    'distance': float(abs(ratio - PHI))
                })
            # Check for 1/φ convergence
            elif abs(ratio - INV_PHI) < 0.05:
                transitions.append({
                    'step': t,
                    'type': 'inv_phi',
                    'value': float(ratio),
                    'distance': float(abs(ratio - INV_PHI))
                })
            # Check for Ξ convergence
            elif abs(ratio - XI) < 0.05:
                transitions.append({
                    'step': t,
                    'type': 'xi',
                    'value': float(ratio),
                    'distance': float(abs(ratio - XI))
                })
        
        return transitions


class PrimeHarmonicAnalyzer:
    """
    Prime Harmonic style analysis for CA evolution.
    
    Treats CA patterns as "chords" and analyzes harmonic stability.
    """
    
    def state_to_chord(self, state: np.ndarray, n_intervals: int = 8) -> np.ndarray:
        """
        Convert CA state to prime chord representation.
        
        Like prime gaps, we look at intervals between 'on' cells.
        """
        on_positions = np.where(state == 1)[0]
        
        if len(on_positions) < 2:
            return np.array([])
        
        # Compute gaps between on cells
        gaps = np.diff(on_positions)
        
        # Return first n_intervals gaps as "chord"
        return gaps[:n_intervals] if len(gaps) >= n_intervals else gaps
    
    def chord_distance(self, chord1: np.ndarray, chord2: np.ndarray) -> float:
        """Compute harmonic distance between two chords."""
        if len(chord1) == 0 or len(chord2) == 0:
            return 1.0
        
        # Pad shorter chord
        max_len = max(len(chord1), len(chord2))
        c1 = np.pad(chord1, (0, max_len - len(chord1)))
        c2 = np.pad(chord2, (0, max_len - len(chord2)))
        
        # Normalized Euclidean distance
        return float(np.linalg.norm(c1 - c2) / (max_len + 1))
    
    def compute_harmonic_stability(self, history: np.ndarray, 
                                    sample_rate: int = 5) -> Tuple[float, int]:
        """
        Compute harmonic stability over CA evolution.
        
        Returns: (min_distance, step_of_min_distance)
        """
        chord_distances = []
        
        prev_chord = None
        for t in range(0, len(history), sample_rate):
            chord = self.state_to_chord(history[t])
            
            if prev_chord is not None and len(chord) > 0:
                dist = self.chord_distance(prev_chord, chord)
                chord_distances.append((t, dist))
            
            prev_chord = chord if len(chord) > 0 else prev_chord
        
        if not chord_distances:
            return 1.0, 0
        
        # Find minimum distance (maximum stability)
        min_idx = np.argmin([d for _, d in chord_distances])
        min_step, min_dist = chord_distances[min_idx]
        
        return min_dist, min_step
    
    def compute_eigenvalue_proxy(self, history: np.ndarray) -> float:
        """
        Compute eigenvalue-like metric from chord progression.
        
        Analogous to λ₁ decay rate from Prime Harmonic Manifold.
        """
        # Build transition matrix from chord progressions
        chords = []
        for t in range(0, len(history), 10):
            chord = self.state_to_chord(history[t])
            if len(chord) > 0:
                chords.append(chord)
        
        if len(chords) < 2:
            return 0.0
        
        # Compute decay of chord similarity
        similarities = []
        for i in range(1, len(chords)):
            sim = 1.0 - self.chord_distance(chords[i-1], chords[i])
            similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        # Fit exponential decay
        if len(similarities) < 3:
            return float(np.mean(similarities))
        
        t = np.arange(len(similarities))
        log_sim = np.log(np.array(similarities) + 1e-10)
        
        try:
            slope, _ = np.polyfit(t, log_sim, 1)
            return float(slope)
        except:
            return 0.0


class AttractorDetector:
    """
    Main attractor detection engine combining SEC and Prime Harmonic methods.
    """
    
    def __init__(self, width: int = 101, steps: int = 512):
        self.width = width
        self.steps = steps
        self.sec_analyzer = SECAnalyzer()
        self.harmonic_analyzer = PrimeHarmonicAnalyzer()
    
    def compute_entropy(self, state: np.ndarray) -> float:
        """Compute Shannon entropy of CA state."""
        p1 = np.mean(state)
        p0 = 1 - p1
        
        if p1 == 0 or p1 == 1:
            return 0.0
        
        return float(-p1 * np.log2(p1) - p0 * np.log2(p0))
    
    def detect_periodicity(self, history: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """
        Detect if CA enters periodic attractor.
        
        Returns: (first_occurrence, period) or (None, None)
        """
        state_hashes = {}
        
        for t in range(len(history)):
            state_hash = hash(history[t].tobytes())
            
            if state_hash in state_hashes:
                first_t = state_hashes[state_hash]
                period = t - first_t
                return first_t, period
            
            state_hashes[state_hash] = t
        
        return None, None
    
    def classify_attractor(self, entropy_variance: float, 
                           period: Optional[int],
                           final_entropy: float) -> str:
        """Classify attractor type based on dynamics."""
        
        if period is not None and period == 1:
            return 'fixed_point'
        elif period is not None:
            return 'limit_cycle'
        elif entropy_variance < 0.001:
            if final_entropy < 0.1:
                return 'fixed_point'
            else:
                return 'quasi_periodic'
        elif entropy_variance < 0.01:
            return 'quasi_periodic'
        else:
            return 'chaotic'
    
    def analyze_rule(self, rule: int) -> AttractorInfo:
        """
        Full attractor analysis for a single CA rule.
        """
        # Evolve CA
        ca = ElementaryCA(rule, self.width)
        state = ca.evolve_fast(self.steps, init_type='single')
        history = state.history
        
        # 1. Entropy analysis
        entropy_curve = [self.compute_entropy(history[t]) for t in range(len(history))]
        final_entropy = entropy_curve[-1] if entropy_curve else 0.0
        entropy_variance = float(np.var(entropy_curve[-50:])) if len(entropy_curve) >= 50 else 0.0
        
        # 2. Periodicity detection
        first_occurrence, period = self.detect_periodicity(history)
        stabilization_step = first_occurrence
        
        # 3. SEC phase transition analysis
        transitions = self.sec_analyzer.detect_phase_transition(history)
        phi_transitions = sum(1 for t in transitions if t['type'] == 'phi')
        xi_transitions = sum(1 for t in transitions if t['type'] == 'xi')
        
        # If we found transitions but no periodicity, use first transition as stabilization
        if stabilization_step is None and transitions:
            stabilization_step = transitions[0]['step']
        
        # 4. Run length ratio (key SEC metric)
        final_ratio = self.sec_analyzer.compute_run_length_ratio(history[-1])
        
        # 5. Harmonic stability
        min_harmonic_dist, harmonic_step = self.harmonic_analyzer.compute_harmonic_stability(history)
        
        # Use harmonic stability step if no other stabilization found
        if stabilization_step is None and min_harmonic_dist < 0.1:
            stabilization_step = harmonic_step
        
        # 6. Classify attractor type
        attractor_type = self.classify_attractor(entropy_variance, period, final_entropy)
        
        return AttractorInfo(
            rule=rule,
            attractor_type=attractor_type,
            stabilization_step=stabilization_step,
            period=period,
            final_entropy=final_entropy,
            entropy_variance=entropy_variance,
            phi_transitions=phi_transitions,
            xi_transitions=xi_transitions,
            run_length_ratio=final_ratio,
            harmonic_stability=min_harmonic_dist
        )
    
    def analyze_all_classified_rules(self) -> Dict[str, List[AttractorInfo]]:
        """Analyze all classified rules, organized by Wolfram class."""
        
        results = {
            'CLASS_I': [],
            'CLASS_II': [],
            'CLASS_III': [],
            'CLASS_IV': [],
            'UNKNOWN': []
        }
        
        for rule, wclass in RULE_CLASSIFICATIONS.items():
            print(f"  Analyzing Rule {rule} ({wclass.name})...", end=" ")
            info = self.analyze_rule(rule)
            results[wclass.name].append(info)
            print(f"→ {info.attractor_type}, ratio={info.run_length_ratio:.4f}")
        
        return results


def run_experiment():
    """Run full attractor detection experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 03: Attractor Detection via SEC/Prime Harmonic Methods")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    results = {
        'experiment': 'exp_03_attractor_detection',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'width': 101,
            'steps': 512,
            'phi': PHI,
            'xi': XI
        },
        'results': {}
    }
    
    detector = AttractorDetector(width=101, steps=512)
    
    # =====================================================
    # PART 1: Analyze All Classified Rules
    # =====================================================
    print("PART 1: Analyzing all classified rules")
    print("-" * 50)
    
    class_results = detector.analyze_all_classified_rules()
    
    # Store results
    for class_name, infos in class_results.items():
        results['results'][class_name] = [
            {
                'rule': i.rule,
                'attractor_type': i.attractor_type,
                'stabilization_step': i.stabilization_step,
                'period': i.period,
                'final_entropy': float(i.final_entropy),
                'entropy_variance': float(i.entropy_variance),
                'phi_transitions': i.phi_transitions,
                'xi_transitions': i.xi_transitions,
                'run_length_ratio': float(i.run_length_ratio) if not np.isinf(i.run_length_ratio) else 999.0,
                'harmonic_stability': float(i.harmonic_stability)
            }
            for i in infos
        ]
    
    # =====================================================
    # PART 2: Summary Statistics by Class
    # =====================================================
    print("\n" + "=" * 70)
    print("PART 2: Summary by Wolfram Class")
    print("-" * 70)
    
    summaries = {}
    
    for class_name in ['CLASS_I', 'CLASS_II', 'CLASS_III', 'CLASS_IV']:
        infos = class_results[class_name]
        if not infos:
            continue
        
        # Count attractor types
        type_counts = {}
        for i in infos:
            type_counts[i.attractor_type] = type_counts.get(i.attractor_type, 0) + 1
        
        # Average metrics
        ratios = [i.run_length_ratio for i in infos if not np.isinf(i.run_length_ratio)]
        mean_ratio = np.mean(ratios) if ratios else 0.0
        
        phi_total = sum(i.phi_transitions for i in infos)
        xi_total = sum(i.xi_transitions for i in infos)
        
        mean_entropy = np.mean([i.final_entropy for i in infos])
        mean_harmonic = np.mean([i.harmonic_stability for i in infos])
        
        print(f"\n{class_name} ({len(infos)} rules):")
        print(f"  Attractor types: {type_counts}")
        print(f"  Mean run-length ratio: {mean_ratio:.4f}")
        print(f"    Distance from φ ({PHI:.4f}): {abs(mean_ratio - PHI):.4f}")
        print(f"    Distance from Ξ ({XI:.4f}): {abs(mean_ratio - XI):.4f}")
        print(f"  Total φ transitions: {phi_total}")
        print(f"  Total Ξ transitions: {xi_total}")
        print(f"  Mean final entropy: {mean_entropy:.4f}")
        print(f"  Mean harmonic stability: {mean_harmonic:.4f}")
        
        summaries[class_name] = {
            'n_rules': len(infos),
            'attractor_types': type_counts,
            'mean_run_length_ratio': float(mean_ratio),
            'distance_from_phi': float(abs(mean_ratio - PHI)),
            'distance_from_xi': float(abs(mean_ratio - XI)),
            'phi_transitions': phi_total,
            'xi_transitions': xi_total,
            'mean_final_entropy': float(mean_entropy),
            'mean_harmonic_stability': float(mean_harmonic)
        }
    
    results['results']['summaries'] = summaries
    
    # =====================================================
    # PART 3: Rules Closest to φ and Ξ
    # =====================================================
    print("\n" + "=" * 70)
    print("PART 3: Rules with Run-Length Ratios Near φ or Ξ")
    print("-" * 70)
    
    all_infos = []
    for infos in class_results.values():
        all_infos.extend(infos)
    
    # Filter out infinite ratios
    valid_infos = [i for i in all_infos if not np.isinf(i.run_length_ratio)]
    
    # Sort by distance from φ
    phi_sorted = sorted(valid_infos, key=lambda x: abs(x.run_length_ratio - PHI))
    
    print(f"\nTop 10 rules closest to φ = {PHI:.6f}:")
    print("-" * 60)
    print(f"{'Rule':>6} {'Ratio':>12} {'Dist from φ':>14} {'Class':>12} {'Type':>15}")
    print("-" * 60)
    
    phi_top = []
    for i in phi_sorted[:10]:
        wclass = RULE_CLASSIFICATIONS.get(i.rule, WolframClass.UNKNOWN)
        dist = abs(i.run_length_ratio - PHI)
        print(f"{i.rule:>6} {i.run_length_ratio:>12.6f} {dist:>14.6f} {wclass.name:>12} {i.attractor_type:>15}")
        phi_top.append({
            'rule': i.rule,
            'ratio': float(i.run_length_ratio),
            'distance': float(dist),
            'wolfram_class': wclass.name
        })
    
    results['results']['top_phi_rules'] = phi_top
    
    # Sort by distance from Ξ
    xi_sorted = sorted(valid_infos, key=lambda x: abs(x.run_length_ratio - XI))
    
    print(f"\nTop 10 rules closest to Ξ = {XI:.6f}:")
    print("-" * 60)
    print(f"{'Rule':>6} {'Ratio':>12} {'Dist from Ξ':>14} {'Class':>12} {'Type':>15}")
    print("-" * 60)
    
    xi_top = []
    for i in xi_sorted[:10]:
        wclass = RULE_CLASSIFICATIONS.get(i.rule, WolframClass.UNKNOWN)
        dist = abs(i.run_length_ratio - XI)
        print(f"{i.rule:>6} {i.run_length_ratio:>12.6f} {dist:>14.6f} {wclass.name:>12} {i.attractor_type:>15}")
        xi_top.append({
            'rule': i.rule,
            'ratio': float(i.run_length_ratio),
            'distance': float(dist),
            'wolfram_class': wclass.name
        })
    
    results['results']['top_xi_rules'] = xi_top
    
    # =====================================================
    # PART 4: Class IV Deep Dive
    # =====================================================
    print("\n" + "=" * 70)
    print("PART 4: Class IV (Edge of Chaos) Detailed Analysis")
    print("-" * 70)
    
    class_iv = class_results['CLASS_IV']
    
    print(f"\n{'Rule':>6} {'Ratio':>10} {'φ dist':>10} {'Ξ dist':>10} {'φ trans':>8} {'Ξ trans':>8} {'Type':>15}")
    print("-" * 80)
    
    for i in class_iv:
        phi_dist = abs(i.run_length_ratio - PHI)
        xi_dist = abs(i.run_length_ratio - XI)
        ratio_str = f"{i.run_length_ratio:.4f}" if not np.isinf(i.run_length_ratio) else "∞"
        
        print(f"{i.rule:>6} {ratio_str:>10} {phi_dist:>10.4f} {xi_dist:>10.4f} "
              f"{i.phi_transitions:>8} {i.xi_transitions:>8} {i.attractor_type:>15}")
    
    # =====================================================
    # Summary
    # =====================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Find if any class has mean ratio near φ or Ξ
    best_phi_class = min(summaries.items(), key=lambda x: x[1]['distance_from_phi'])
    best_xi_class = min(summaries.items(), key=lambda x: x[1]['distance_from_xi'])
    
    print(f"\n📊 Class closest to φ: {best_phi_class[0]} (distance: {best_phi_class[1]['distance_from_phi']:.4f})")
    print(f"📊 Class closest to Ξ: {best_xi_class[0]} (distance: {best_xi_class[1]['distance_from_xi']:.4f})")
    
    # Check Rule 110 specifically
    rule_110 = next((i for i in all_infos if i.rule == 110), None)
    if rule_110:
        print(f"\n🎯 Rule 110 Analysis:")
        print(f"   Run-length ratio: {rule_110.run_length_ratio:.6f}")
        print(f"   Distance from φ: {abs(rule_110.run_length_ratio - PHI):.6f}")
        print(f"   Distance from Ξ: {abs(rule_110.run_length_ratio - XI):.6f}")
        print(f"   Attractor type: {rule_110.attractor_type}")
        print(f"   φ transitions: {rule_110.phi_transitions}")
        print(f"   Ξ transitions: {rule_110.xi_transitions}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'exp_03_attractor_detection_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    print(f"Completed: {datetime.now().isoformat()}")
    
    return results


if __name__ == "__main__":
    run_experiment()

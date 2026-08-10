#!/usr/bin/env python3
"""
Experiment 04: Ξ Approach Dynamics
==================================

Investigates how different CA rules approach (or avoid) the balance operator Ξ.

Key Questions:
- Do Class IV rules have a unique approach signature to Ξ?
- Is there a difference between approaching from above vs below?
- Do computationally universal rules oscillate around Ξ while trivial rules
  simply sit at a static value?

Metrics:
- Time series of P/A ratio during evolution
- Crossing count: how many times does the trajectory cross Ξ?
- Approach direction: from above or below?
- Oscillation amplitude and frequency around Ξ
- Settling time: when does the ratio stabilize?
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from ca_simulator import ElementaryCA, CAState, WolframClass, RULE_CLASSIFICATIONS
from pac_embedding import PACEmbedder

# Constants
XI = 1.0571  # PAC balance operator
PHI = 1.618033988749895  # Golden ratio
EPSILON = 0.01  # Threshold for "at Ξ"


@dataclass
class ApproachSignature:
    """Characterizes how a rule approaches Ξ over time."""
    rule: int
    wolfram_class: str
    
    # Time series data
    pa_ratios: List[float]
    
    # Crossing analysis
    xi_crossings: int  # How many times trajectory crosses Ξ
    first_crossing_time: Optional[int]  # When first crosses Ξ
    
    # Direction analysis
    initial_direction: str  # "above", "below", or "at"
    final_direction: str  # Where it ends relative to Ξ
    
    # Oscillation metrics
    mean_deviation: float  # Mean distance from Ξ
    oscillation_amplitude: float  # Std dev of ratio
    
    # Stability metrics
    settling_time: Optional[int]  # When ratio stabilizes (within epsilon)
    is_stable: bool  # Whether it reaches stable state
    final_ratio: float
    
    # Dynamic vs Static classification
    approach_type: str  # "static", "monotonic", "oscillating", "chaotic"


class XiApproachAnalyzer:
    """Analyzes the dynamics of approaching Ξ."""
    
    def __init__(self, width: int = 101, steps: int = 200):
        self.width = width
        self.steps = steps
        
    def compute_row_pa_ratio(self, row: np.ndarray) -> float:
        """
        Compute P/A ratio for a single CA row.
        
        P (Potential) = 1 - density (empty cells = potential)
        A (Actualization) = density (filled cells = actualized)
        """
        density = row.mean()
        
        # Avoid division by zero
        if density < 0.001:
            return 100.0  # Very high P/A (almost all potential)
        if density > 0.999:
            return 0.01   # Very low P/A (almost all actualized)
        
        potential = 1.0 - density
        actualization = density
        
        return potential / actualization
        
    def compute_pa_trajectory(self, rule: int) -> List[float]:
        """Compute P/A ratio at each timestep."""
        ca = ElementaryCA(rule, self.width)
        state = ca.evolve_fast(self.steps, init_type='single')
        history = state.history
        
        ratios = []
        for row in history:
            ratio = self.compute_row_pa_ratio(row)
            ratios.append(ratio)
        
        return ratios
    
    def count_xi_crossings(self, ratios: List[float]) -> Tuple[int, Optional[int]]:
        """Count how many times the trajectory crosses Ξ."""
        crossings = 0
        first_crossing = None
        
        for i in range(1, len(ratios)):
            prev_side = ratios[i-1] > XI
            curr_side = ratios[i] > XI
            
            if prev_side != curr_side:
                crossings += 1
                if first_crossing is None:
                    first_crossing = i
        
        return crossings, first_crossing
    
    def compute_oscillation_metrics(self, ratios: List[float]) -> Tuple[float, float]:
        """Compute mean deviation and oscillation amplitude."""
        deviations = [abs(r - XI) for r in ratios]
        mean_dev = np.mean(deviations)
        amplitude = np.std(ratios)
        
        return mean_dev, amplitude
    
    def find_settling_time(self, ratios: List[float], epsilon: float = EPSILON) -> Optional[int]:
        """Find when the ratio stabilizes within epsilon of final value."""
        if len(ratios) < 10:
            return None
        
        final_value = np.mean(ratios[-10:])  # Average of last 10 steps
        
        # Work backwards to find when it entered the stable zone
        for i in range(len(ratios) - 1, -1, -1):
            if abs(ratios[i] - final_value) > epsilon:
                if i < len(ratios) - 1:
                    return i + 1
                else:
                    return None  # Never settled
        
        return 0  # Was always stable
    
    def classify_approach_type(self, ratios: List[float], crossings: int) -> str:
        """Classify the type of approach to Ξ."""
        if len(set(ratios)) == 1:
            return "static"  # Never changes
        
        # Check for monotonic approach
        diffs = np.diff(ratios)
        if np.all(diffs >= 0) or np.all(diffs <= 0):
            return "monotonic"
        
        # Check oscillation patterns
        amplitude = np.std(ratios)
        
        if crossings > 10 and amplitude < 0.5:
            return "oscillating"
        elif crossings > 20 or amplitude > 1.0:
            return "chaotic"
        else:
            return "damped"  # Oscillates but settles
    
    def analyze_rule(self, rule: int) -> ApproachSignature:
        """Full approach analysis for a single rule."""
        wc = RULE_CLASSIFICATIONS.get(rule, WolframClass.UNKNOWN)
        wolfram_class = wc.name if hasattr(wc, 'name') else str(wc)
        
        # Get trajectory
        ratios = self.compute_pa_trajectory(rule)
        
        # Crossing analysis
        crossings, first_crossing = self.count_xi_crossings(ratios)
        
        # Direction analysis
        initial_ratio = ratios[0] if ratios else XI
        final_ratio = ratios[-1] if ratios else XI
        
        if abs(initial_ratio - XI) < EPSILON:
            initial_dir = "at"
        elif initial_ratio > XI:
            initial_dir = "above"
        else:
            initial_dir = "below"
        
        if abs(final_ratio - XI) < EPSILON:
            final_dir = "at"
        elif final_ratio > XI:
            final_dir = "above"
        else:
            final_dir = "below"
        
        # Oscillation metrics
        mean_dev, amplitude = self.compute_oscillation_metrics(ratios)
        
        # Stability
        settling_time = self.find_settling_time(ratios)
        is_stable = settling_time is not None
        
        # Classify approach type
        approach_type = self.classify_approach_type(ratios, crossings)
        
        return ApproachSignature(
            rule=rule,
            wolfram_class=wolfram_class,
            pa_ratios=ratios,
            xi_crossings=crossings,
            first_crossing_time=first_crossing,
            initial_direction=initial_dir,
            final_direction=final_dir,
            mean_deviation=mean_dev,
            oscillation_amplitude=amplitude,
            settling_time=settling_time,
            is_stable=is_stable,
            final_ratio=final_ratio,
            approach_type=approach_type
        )


def main():
    print("=" * 70)
    print("EXPERIMENT 04: Ξ Approach Dynamics")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Ξ = {XI}, φ = {PHI}")
    print()
    
    analyzer = XiApproachAnalyzer(width=101, steps=200)
    
    # Analyze all classified rules
    all_rules = []
    for rule in RULE_CLASSIFICATIONS.keys():
        all_rules.append(rule)
    
    results = {}
    class_summaries = {}
    
    print("PART 1: Analyzing approach dynamics for all classified rules")
    print("-" * 70)
    
    for rule in sorted(all_rules):
        sig = analyzer.analyze_rule(rule)
        results[rule] = sig
        
        # Group by class
        wc = sig.wolfram_class
        if wc not in class_summaries:
            class_summaries[wc] = []
        class_summaries[wc].append(sig)
        
        print(f"  Rule {rule:3d} ({wc}): {sig.approach_type:12s} | "
              f"crossings={sig.xi_crossings:3d} | "
              f"final={sig.final_ratio:.4f} | "
              f"mean_dev={sig.mean_deviation:.4f}")
    
    print()
    print("=" * 70)
    print("PART 2: Summary by Wolfram Class")
    print("-" * 70)
    
    class_stats = {}
    
    for wc in sorted(class_summaries.keys()):
        sigs = class_summaries[wc]
        
        # Aggregate statistics
        crossings = [s.xi_crossings for s in sigs]
        mean_devs = [s.mean_deviation for s in sigs]
        amplitudes = [s.oscillation_amplitude for s in sigs]
        final_ratios = [s.final_ratio for s in sigs]
        approach_types = [s.approach_type for s in sigs]
        
        # Count approach types
        type_counts = {}
        for at in approach_types:
            type_counts[at] = type_counts.get(at, 0) + 1
        
        # Direction analysis
        initial_dirs = [s.initial_direction for s in sigs]
        final_dirs = [s.final_direction for s in sigs]
        
        stats = {
            "n_rules": len(sigs),
            "mean_crossings": float(np.mean(crossings)),
            "max_crossings": int(np.max(crossings)),
            "mean_deviation": float(np.mean(mean_devs)),
            "mean_amplitude": float(np.mean(amplitudes)),
            "mean_final_ratio": float(np.mean(final_ratios)),
            "closest_to_xi": float(min([abs(r - XI) for r in final_ratios])),
            "approach_types": type_counts,
            "final_directions": {d: final_dirs.count(d) for d in set(final_dirs)}
        }
        class_stats[wc] = stats
        
        print(f"\n{wc} ({len(sigs)} rules):")
        print(f"  Mean Ξ crossings: {stats['mean_crossings']:.1f} (max: {stats['max_crossings']})")
        print(f"  Mean deviation from Ξ: {stats['mean_deviation']:.4f}")
        print(f"  Mean oscillation amplitude: {stats['mean_amplitude']:.4f}")
        print(f"  Mean final P/A ratio: {stats['mean_final_ratio']:.4f}")
        print(f"  Closest approach to Ξ: {stats['closest_to_xi']:.4f}")
        print(f"  Approach types: {type_counts}")
        print(f"  Final directions: {stats['final_directions']}")
    
    print()
    print("=" * 70)
    print("PART 3: Class IV (Edge of Chaos) Detailed Trajectories")
    print("-" * 70)
    
    class_iv_sigs = class_summaries.get("CLASS_IV", [])
    
    for sig in class_iv_sigs:
        print(f"\nRule {sig.rule}:")
        print(f"  Approach type: {sig.approach_type}")
        print(f"  Ξ crossings: {sig.xi_crossings}")
        print(f"  First crossing at step: {sig.first_crossing_time}")
        print(f"  Initial direction: {sig.initial_direction}")
        print(f"  Final direction: {sig.final_direction}")
        print(f"  Mean deviation from Ξ: {sig.mean_deviation:.4f}")
        print(f"  Oscillation amplitude: {sig.oscillation_amplitude:.4f}")
        print(f"  Final P/A ratio: {sig.final_ratio:.4f}")
        print(f"  Distance from Ξ: {abs(sig.final_ratio - XI):.4f}")
        print(f"  Settled at step: {sig.settling_time}")
        
        # Show trajectory snippet (first 20 and last 10)
        if len(sig.pa_ratios) > 30:
            first_20 = sig.pa_ratios[:20]
            last_10 = sig.pa_ratios[-10:]
            print(f"  Trajectory (first 20): {[f'{r:.3f}' for r in first_20]}")
            print(f"  Trajectory (last 10):  {[f'{r:.3f}' for r in last_10]}")
    
    print()
    print("=" * 70)
    print("PART 4: Static vs Dynamic Ξ Classification")
    print("-" * 70)
    
    static_xi = []
    dynamic_xi = []
    
    for rule, sig in results.items():
        dist_from_xi = abs(sig.final_ratio - XI)
        
        if dist_from_xi < 0.1:  # Within 0.1 of Ξ
            if sig.approach_type == "static":
                static_xi.append((rule, sig))
            else:
                dynamic_xi.append((rule, sig))
    
    print("\nStatic Ξ (trivial equilibrium):")
    print("-" * 50)
    for rule, sig in static_xi:
        print(f"  Rule {rule:3d} ({sig.wolfram_class}): ratio={sig.final_ratio:.4f}, "
              f"crossings={sig.xi_crossings}, type={sig.approach_type}")
    
    print(f"\nDynamic Ξ (active balance at edge of chaos):")
    print("-" * 50)
    for rule, sig in dynamic_xi:
        print(f"  Rule {rule:3d} ({sig.wolfram_class}): ratio={sig.final_ratio:.4f}, "
              f"crossings={sig.xi_crossings}, type={sig.approach_type}")
    
    print()
    print("=" * 70)
    print("PART 5: Approach Direction Analysis")
    print("-" * 70)
    
    # Analyze whether different classes approach from different directions
    direction_by_class = {}
    for wc, sigs in class_summaries.items():
        from_above = sum(1 for s in sigs if s.initial_direction == "above")
        from_below = sum(1 for s in sigs if s.initial_direction == "below")
        at_xi = sum(1 for s in sigs if s.initial_direction == "at")
        
        direction_by_class[wc] = {
            "from_above": from_above,
            "from_below": from_below,
            "at_xi": at_xi
        }
        
        print(f"{wc}: above={from_above}, below={from_below}, at={at_xi}")
    
    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Find most dynamic rule (most crossings)
    most_crossings = max(results.values(), key=lambda s: s.xi_crossings)
    print(f"\n🔄 Most dynamic rule (most Ξ crossings): "
          f"Rule {most_crossings.rule} ({most_crossings.wolfram_class}) "
          f"with {most_crossings.xi_crossings} crossings")
    
    # Find rules that oscillate around Ξ
    oscillators = [s for s in results.values() 
                   if s.approach_type == "oscillating" and s.mean_deviation < 0.5]
    print(f"\n🎯 Rules oscillating around Ξ: {[s.rule for s in oscillators]}")
    
    # Compare Class I vs Class IV signatures
    class_i_sigs = class_summaries.get("CLASS_I", [])
    class_iv_sigs = class_summaries.get("CLASS_IV", [])
    
    if class_i_sigs and class_iv_sigs:
        i_crossings = np.mean([s.xi_crossings for s in class_i_sigs])
        iv_crossings = np.mean([s.xi_crossings for s in class_iv_sigs])
        
        print(f"\n📊 Class I vs Class IV dynamics:")
        print(f"   Class I mean Ξ crossings: {i_crossings:.1f}")
        print(f"   Class IV mean Ξ crossings: {iv_crossings:.1f}")
        print(f"   Ratio: {iv_crossings / max(i_crossings, 0.1):.1f}x more dynamic")
    
    # Rule 110 specific analysis
    if 110 in results:
        r110 = results[110]
        print(f"\n🌟 Rule 110 Unique Signature:")
        print(f"   Approach type: {r110.approach_type}")
        print(f"   Ξ crossings: {r110.xi_crossings}")
        print(f"   First crossing at step: {r110.first_crossing_time}")
        print(f"   Mean deviation from Ξ: {r110.mean_deviation:.4f}")
        print(f"   Final P/A ratio: {r110.final_ratio:.4f}")
        print(f"   Final distance from Ξ: {abs(r110.final_ratio - XI):.4f}")
        print(f"   Initial direction: {r110.initial_direction}")
        print(f"   Final direction: {r110.final_direction}")
    
    # Save results
    print()
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"exp_04_xi_approach_{timestamp}.json"
    
    # Prepare for JSON serialization
    output = {
        "experiment": "exp_04_xi_approach_dynamics",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "xi": XI,
            "phi": PHI,
            "width": 101,
            "steps": 200,
            "epsilon": EPSILON
        },
        "class_statistics": class_stats,
        "direction_by_class": direction_by_class,
        "static_xi_rules": [rule for rule, _ in static_xi],
        "dynamic_xi_rules": [rule for rule, _ in dynamic_xi],
        "rule_signatures": {
            str(rule): {
                "rule": sig.rule,
                "wolfram_class": sig.wolfram_class,
                "xi_crossings": sig.xi_crossings,
                "first_crossing_time": sig.first_crossing_time,
                "initial_direction": sig.initial_direction,
                "final_direction": sig.final_direction,
                "mean_deviation": float(sig.mean_deviation),
                "oscillation_amplitude": float(sig.oscillation_amplitude),
                "settling_time": sig.settling_time,
                "is_stable": sig.is_stable,
                "final_ratio": float(sig.final_ratio),
                "approach_type": sig.approach_type,
                # Store trajectory as list of floats
                "trajectory_summary": {
                    "first_10": [float(r) for r in sig.pa_ratios[:10]],
                    "last_10": [float(r) for r in sig.pa_ratios[-10:]],
                    "min": float(min(sig.pa_ratios)),
                    "max": float(max(sig.pa_ratios)),
                    "mean": float(np.mean(sig.pa_ratios))
                }
            }
            for rule, sig in results.items()
        }
    }
    
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"📁 Results saved to: {results_file}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

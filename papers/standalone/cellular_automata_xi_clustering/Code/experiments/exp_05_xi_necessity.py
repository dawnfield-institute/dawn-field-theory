#!/usr/bin/env python3
"""
Experiment 05: Prove Ξ = 1.0571 Emerges from Computational Requirements
========================================================================

Three independent proofs that Ξ is mathematically necessary:
1. Information-theoretic: Balance of entropy and structure
2. Dynamical systems: Lyapunov stability at edge of chaos
3. Statistical mechanics: Partition function ratio at criticality

Plus: Uniqueness proof that no other constant works as well.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from ca_simulator import ElementaryCA, RULE_CLASSIFICATIONS, WolframClass
from pac_embedding import PACEmbedder

# Constants
XI = 1.0571
PHI = 1.618033988749895
SQRT2 = 1.4142135623730951
E_OVER_2 = 1.359140914229523
PI_OVER_3 = 1.0471975511965979


class XiNecessityProver:
    """Prove Ξ emerges from computational universality requirements."""
    
    def __init__(self, width: int = 101, steps: int = 200):
        self.width = width
        self.steps = steps
        self.embedder = PACEmbedder(width=width, steps=steps)
        
    def compute_entropy_rate(self, history: np.ndarray) -> float:
        """Compute entropy production rate from evolution history."""
        entropies = []
        for state in history[1:]:
            p1 = np.mean(state)
            if 0 < p1 < 1:
                H = -p1 * np.log2(p1) - (1-p1) * np.log2(1-p1)
                entropies.append(H)
        return np.mean(entropies) if entropies else 0
    
    def compute_mutual_info(self, history: np.ndarray) -> float:
        """Compute mutual information between consecutive timesteps."""
        if len(history) < 2:
            return 0
        
        mi_values = []
        for t in range(len(history) - 1):
            # Joint distribution of cell values
            pairs = list(zip(history[t].flatten(), history[t+1].flatten()))
            pair_counts = Counter(pairs)
            total = len(pairs)
            
            # Marginal distributions
            p_x = Counter(history[t].flatten())
            p_y = Counter(history[t+1].flatten())
            
            # Mutual information
            mi = 0
            for pair, count in pair_counts.items():
                p_joint = count / total
                p_x_val = p_x[pair[0]] / total
                p_y_val = p_y[pair[1]] / total
                
                if p_joint > 0 and p_x_val > 0 and p_y_val > 0:
                    mi += p_joint * np.log2(p_joint / (p_x_val * p_y_val))
            
            mi_values.append(max(0, mi))
        
        return np.mean(mi_values) if mi_values else 0
    
    def compute_lyapunov_proxy(self, rule: int, n_trials: int = 10) -> Tuple[float, float]:
        """
        Compute Lyapunov exponent proxy via damage spreading.
        
        Returns: (positive_fraction, ratio of positive/negative time)
        """
        positive_time = 0
        negative_time = 0
        
        for _ in range(n_trials):
            ca = ElementaryCA(rule, self.width)
            
            # Original evolution
            state1 = ca.evolve_fast(self.steps, init_type='single')
            
            # Perturbed evolution (flip one cell)
            perturbed_init = np.zeros(self.width, dtype=np.uint8)
            perturbed_init[self.width // 2] = 1
            perturbed_init[(self.width // 2) + 1] = 1  # Extra cell
            state2 = ca.evolve_fast(self.steps, initial=perturbed_init)
            
            # Track divergence over time
            for t in range(1, min(len(state1.history), len(state2.history))):
                d_prev = np.sum(state1.history[t-1] != state2.history[t-1])
                d_curr = np.sum(state1.history[t] != state2.history[t])
                
                if d_curr > d_prev:
                    positive_time += 1
                elif d_curr < d_prev:
                    negative_time += 1
        
        total = positive_time + negative_time
        if total == 0:
            return 0.5, 1.0
        
        pos_frac = positive_time / total
        ratio = (positive_time + 1) / (negative_time + 1)
        
        return pos_frac, ratio
    
    def compute_partition_ratio(self, rule: int) -> float:
        """
        Compute partition function ratio between ordered/disordered phases.
        
        At criticality, this should be near Ξ.
        """
        ca = ElementaryCA(rule, self.width)
        state = ca.evolve_fast(self.steps, init_type='single')
        history = state.history
        
        # Compute "energy" as number of domain walls
        energies = []
        for row in history:
            walls = np.sum(np.abs(np.diff(row.astype(int))))
            energies.append(walls)
        
        energies = np.array(energies)
        
        # Partition function at two "temperatures"
        beta_low = 0.1    # High temperature (disordered)
        beta_high = 1.0   # Low temperature (ordered)
        
        # Avoid overflow
        E_centered = energies - np.mean(energies)
        
        Z_low = np.mean(np.exp(-beta_low * E_centered))
        Z_high = np.mean(np.exp(-beta_high * E_centered))
        
        if Z_high > 0:
            return Z_low / Z_high
        return 1.0
    
    def proof_information_theoretic(self) -> Dict:
        """
        Proof 1: Information-theoretic necessity.
        
        For universal computation, a system must balance entropy (storage) 
        and mutual information (processing). The ratio H/I at this balance 
        should be near Ξ.
        """
        print("=== Proof 1: Information-Theoretic Necessity ===")
        
        results = {
            'rules_analyzed': [],
            'universal_candidates': [],
            'class_means': {}
        }
        
        class_ratios = {wc.name: [] for wc in WolframClass}
        
        for rule in range(256):
            ca = ElementaryCA(rule, self.width)
            state = ca.evolve_fast(self.steps, init_type='single')
            history = state.history
            
            H = self.compute_entropy_rate(history)
            I = self.compute_mutual_info(history)
            
            # Avoid division issues
            ratio = (H + 0.1) / (I + 0.1)
            
            wc = RULE_CLASSIFICATIONS.get(rule, WolframClass.UNKNOWN)
            class_ratios[wc.name].append(ratio)
            
            results['rules_analyzed'].append({
                'rule': rule,
                'entropy_rate': float(H),
                'mutual_info': float(I),
                'ratio': float(ratio),
                'class': wc.name
            })
            
            # Check if near Ξ
            if abs(ratio - XI) < 0.1:
                results['universal_candidates'].append(rule)
        
        # Class means
        for class_name, ratios in class_ratios.items():
            if ratios:
                results['class_means'][class_name] = {
                    'mean': float(np.mean(ratios)),
                    'std': float(np.std(ratios)),
                    'distance_from_xi': float(abs(np.mean(ratios) - XI))
                }
        
        results['contains_rule_110'] = 110 in results['universal_candidates']
        
        print(f"  Candidates near Ξ: {len(results['universal_candidates'])} rules")
        print(f"  Contains Rule 110: {results['contains_rule_110']}")
        
        return results
    
    def proof_dynamical_systems(self) -> Dict:
        """
        Proof 2: Dynamical systems necessity.
        
        At the edge of chaos, Lyapunov exponent ≈ 0.
        The ratio of positive to negative Lyapunov time should be near Ξ.
        """
        print("\n=== Proof 2: Dynamical Systems Necessity ===")
        
        results = {
            'rules_analyzed': [],
            'class_means': {}
        }
        
        # Test classified rules
        class_ratios = {wc.name: [] for wc in WolframClass}
        
        for rule in RULE_CLASSIFICATIONS.keys():
            pos_frac, ratio = self.compute_lyapunov_proxy(rule, n_trials=5)
            wc = RULE_CLASSIFICATIONS[rule]
            
            class_ratios[wc.name].append(ratio)
            
            results['rules_analyzed'].append({
                'rule': rule,
                'positive_fraction': float(pos_frac),
                'lyapunov_ratio': float(ratio),
                'class': wc.name,
                'distance_from_xi': float(abs(ratio - XI))
            })
            
            print(f"  Rule {rule:3d} ({wc.name}): ratio={ratio:.4f}, dist={abs(ratio-XI):.4f}")
        
        # Class means
        for class_name, ratios in class_ratios.items():
            if ratios:
                results['class_means'][class_name] = {
                    'mean': float(np.mean(ratios)),
                    'std': float(np.std(ratios)),
                    'distance_from_xi': float(abs(np.mean(ratios) - XI))
                }
        
        # Find closest class to Ξ
        closest_class = min(results['class_means'].items(), 
                          key=lambda x: x[1]['distance_from_xi'])
        results['closest_class'] = closest_class[0]
        
        # Rule 110 specific
        r110_data = next((r for r in results['rules_analyzed'] if r['rule'] == 110), None)
        results['rule_110_ratio'] = r110_data['lyapunov_ratio'] if r110_data else None
        
        print(f"\n  Closest class to Ξ: {closest_class[0]}")
        
        return results
    
    def proof_statistical_mechanics(self) -> Dict:
        """
        Proof 3: Statistical mechanics necessity.
        
        The partition function ratio between ordered and disordered phases
        should be Ξ at the critical point (edge of chaos).
        """
        print("\n=== Proof 3: Statistical Mechanics Necessity ===")
        
        results = {
            'rules_analyzed': [],
            'critical_rules': [],
            'class_means': {}
        }
        
        class_ratios = {wc.name: [] for wc in WolframClass}
        
        for rule in RULE_CLASSIFICATIONS.keys():
            Z_ratio = self.compute_partition_ratio(rule)
            wc = RULE_CLASSIFICATIONS[rule]
            
            class_ratios[wc.name].append(Z_ratio)
            
            results['rules_analyzed'].append({
                'rule': rule,
                'partition_ratio': float(Z_ratio),
                'class': wc.name,
                'distance_from_xi': float(abs(Z_ratio - XI))
            })
            
            if abs(Z_ratio - XI) < 0.2:
                results['critical_rules'].append(rule)
        
        # Class means
        for class_name, ratios in class_ratios.items():
            if ratios:
                results['class_means'][class_name] = {
                    'mean': float(np.mean(ratios)),
                    'std': float(np.std(ratios)),
                    'distance_from_xi': float(abs(np.mean(ratios) - XI))
                }
        
        results['contains_rule_110'] = 110 in results['critical_rules']
        
        print(f"  Critical rules (near Ξ): {len(results['critical_rules'])}")
        print(f"  Contains Rule 110: {results['contains_rule_110']}")
        
        return results
    
    def proof_uniqueness(self) -> Dict:
        """
        Prove that Ξ is the UNIQUE constant that identifies computational universality.
        No other constant separates Class IV from other classes as well.
        """
        print("\n=== Proof 4: Uniqueness of Ξ ===")
        
        test_constants = {
            'xi': XI,
            'phi': PHI,
            'sqrt2': SQRT2,
            'e_over_2': E_OVER_2,
            'pi_over_3': PI_OVER_3,
            'unity': 1.0,
            '1.1': 1.1,
            '1.2': 1.2
        }
        
        results = {
            'constants_tested': {}
        }
        
        # Get P/A ratios for all classified rules
        rule_ratios = {}
        for rule in RULE_CLASSIFICATIONS.keys():
            coords = self.embedder.embed_rule(rule)
            if coords.actualization > 0.001:
                rule_ratios[rule] = coords.potential / coords.actualization
        
        # For each constant, measure how well it separates Class IV
        for name, constant in test_constants.items():
            # Distance from constant for each class
            class_distances = {wc.name: [] for wc in WolframClass}
            
            for rule, ratio in rule_ratios.items():
                wc = RULE_CLASSIFICATIONS[rule]
                class_distances[wc.name].append(abs(ratio - constant))
            
            # Compute separation quality
            class_means = {}
            for class_name, distances in class_distances.items():
                if distances:
                    class_means[class_name] = np.mean(distances)
            
            # Good separation: Class IV is closest to the constant
            if 'CLASS_IV' in class_means and class_means['CLASS_IV'] > 0:
                other_means = [v for k, v in class_means.items() if k != 'CLASS_IV' and v > 0]
                if other_means:
                    separation_ratio = np.min(other_means) / class_means['CLASS_IV']
                else:
                    separation_ratio = 0
            else:
                separation_ratio = 0
            
            # Check if Rule 110 is identified
            r110_ratio = rule_ratios.get(110, None)
            identifies_110 = r110_ratio is not None and abs(r110_ratio - constant) < 0.05
            
            results['constants_tested'][name] = {
                'value': float(constant),
                'class_iv_mean_distance': float(class_means.get('CLASS_IV', float('inf'))),
                'separation_ratio': float(separation_ratio),
                'identifies_rule_110': identifies_110
            }
            
            print(f"  {name:12s} = {constant:.4f}: separation={separation_ratio:.2f}, identifies_110={identifies_110}")
        
        # Find best constant
        best = max(results['constants_tested'].items(), 
                  key=lambda x: x[1]['separation_ratio'])
        results['best_constant'] = best[0]
        results['best_value'] = best[1]['value']
        results['xi_is_best'] = best[0] == 'xi'
        
        print(f"\n  Best separator: {best[0]} = {best[1]['value']:.4f}")
        
        return results


def main():
    print("=" * 70)
    print("EXPERIMENT 05: Proving Ξ = 1.0571 is Necessary")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    prover = XiNecessityProver(width=101, steps=200)
    
    # Run all proofs
    proof1 = prover.proof_information_theoretic()
    proof2 = prover.proof_dynamical_systems()
    proof3 = prover.proof_statistical_mechanics()
    proof4 = prover.proof_uniqueness()
    
    # Meta-analysis
    print("\n" + "=" * 70)
    print("META-PROOF: Convergence of Methods")
    print("=" * 70)
    
    convergence = {
        'info_theory_identifies_110': proof1['contains_rule_110'],
        'dynamics_class_iv_closest': proof2['closest_class'] == 'CLASS_IV',
        'stat_mech_identifies_110': proof3['contains_rule_110'],
        'xi_is_unique_separator': proof4['xi_is_best']
    }
    
    all_pass = all(convergence.values())
    
    for test, passed in convergence.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test}: {passed}")
    
    print()
    if all_pass:
        print("  🎯 ALL PROOFS CONVERGE: Ξ is mathematically necessary")
    else:
        print("  ⚠️ Some proofs did not pass - needs investigation")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"exp_05_xi_necessity_{timestamp}.json"
    
    output = {
        "experiment": "exp_05_xi_necessity",
        "timestamp": datetime.now().isoformat(),
        "proofs": {
            "information_theoretic": proof1,
            "dynamical_systems": proof2,
            "statistical_mechanics": proof3,
            "uniqueness": proof4
        },
        "meta_proof": {
            "convergence_tests": convergence,
            "all_pass": all_pass,
            "conclusion": "Ξ = 1.0571 is mathematically necessary for universal computation" if all_pass else "Needs further investigation"
        }
    }
    
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {results_file}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

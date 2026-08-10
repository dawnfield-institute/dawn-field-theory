"""
Experiment 10: SEC Generator v2 - Calibrated to Real Proteins
===============================================================

Key insight from exp_09: Real proteins have SHORT runs (mean ~2) but
high Fibonacci fraction (because 1, 2, 3 are all Fibonacci).

The SEC collapse needs to:
1. Create short, frequent transitions (high entropy at micro level)
2. But with Fibonacci-clustered segment lengths
3. And φ-ish ratios between consecutive segments

New approach: SEC as a TRANSITION probability model
- At each position, probability of switching type depends on SEC state
- Transitions favor Fibonacci-length segments
"""

import numpy as np
from collections import defaultdict
import json
import os
from datetime import datetime

PHI = (1 + np.sqrt(5)) / 2
FIBONACCI = set([1, 2, 3, 5, 8, 13, 21, 34])

HYDROPHOBIC = list('AILMFVPWG')
HYDROPHILIC = list('RKDENQHSTY')
ALL_AA = HYDROPHOBIC + HYDROPHILIC


class SECTransitionGenerator:
    """
    SEC-based protein generator using transition probabilities.
    
    Key idea: Probability of switching hydrophobic/hydrophilic state
    is modulated to favor Fibonacci-length runs.
    """
    
    def __init__(self, length):
        self.length = length
        
    def fibonacci_transition_prob(self, current_run_length):
        """
        Probability of transitioning based on current run length.
        Higher probability at non-Fibonacci lengths.
        Lower probability at Fibonacci lengths (stay in run).
        """
        if current_run_length in FIBONACCI:
            # At Fibonacci length: moderate chance to stay or switch
            # (allows both extending and transitioning)
            return 0.3 + 0.1 * (current_run_length / 8)  # Increases with length
        else:
            # Not Fibonacci: higher chance to switch (to reach next Fib)
            return 0.5
    
    def generate(self):
        """Generate protein with SEC-modulated transitions."""
        sequence = []
        current_type = np.random.choice(['H', 'P'])
        current_run_length = 0
        
        for i in range(self.length):
            current_run_length += 1
            
            # Decide on amino acid
            if current_type == 'H':
                aa = np.random.choice(HYDROPHOBIC)
            else:
                aa = np.random.choice(HYDROPHILIC)
            
            sequence.append(aa)
            
            # Decide if we switch
            if i < self.length - 1:
                switch_prob = self.fibonacci_transition_prob(current_run_length)
                if np.random.random() < switch_prob:
                    current_type = 'P' if current_type == 'H' else 'H'
                    current_run_length = 0
        
        return ''.join(sequence)


class PhiRatioGenerator:
    """
    Generate proteins where consecutive run lengths follow φ ratios.
    """
    
    def __init__(self, length):
        self.length = length
        
    def generate(self):
        """Generate with φ-ratio consecutive runs."""
        sequence = []
        current_type = np.random.choice(['H', 'P'])
        
        # Start with a Fibonacci length
        prev_length = np.random.choice([2, 3, 5])
        
        while len(sequence) < self.length:
            # Next length is roughly prev * φ or prev / φ
            if np.random.random() < 0.5:
                target = int(prev_length * PHI)
            else:
                target = max(1, int(prev_length / PHI))
            
            # Snap to nearest Fibonacci
            nearest_fib = min(FIBONACCI, key=lambda f: abs(f - target))
            run_length = min(nearest_fib, self.length - len(sequence))
            
            # Generate run
            if current_type == 'H':
                run = [np.random.choice(HYDROPHOBIC) for _ in range(run_length)]
            else:
                run = [np.random.choice(HYDROPHILIC) for _ in range(run_length)]
            
            sequence.extend(run)
            
            # Switch type, update prev_length
            current_type = 'P' if current_type == 'H' else 'H'
            prev_length = run_length
        
        return ''.join(sequence[:self.length])


class NaturalFrequencyGenerator:
    """
    Control: Generate with natural AA frequencies but no structure.
    """
    
    NATURAL_FREQ = {
        'A': 0.074, 'R': 0.042, 'N': 0.044, 'D': 0.059, 'C': 0.033,
        'E': 0.058, 'Q': 0.037, 'G': 0.074, 'H': 0.029, 'I': 0.038,
        'L': 0.076, 'K': 0.072, 'M': 0.018, 'F': 0.040, 'P': 0.050,
        'S': 0.081, 'T': 0.062, 'W': 0.013, 'Y': 0.033, 'V': 0.068,
    }
    
    def __init__(self, length):
        self.length = length
        self.aas = list(self.NATURAL_FREQ.keys())
        self.probs = [self.NATURAL_FREQ[aa] for aa in self.aas]
        self.probs = np.array(self.probs) / sum(self.probs)
        
    def generate(self):
        return ''.join(np.random.choice(self.aas, size=self.length, p=self.probs))


def analyze_sequence(seq):
    """Analyze sequence properties."""
    n = len(seq)
    
    # Hydrophobic runs
    runs = []
    current_run = 0
    is_hydro = []
    
    for aa in seq:
        h = aa in HYDROPHOBIC
        is_hydro.append(h)
        if h:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)
    
    # Hydrophilic runs too
    phil_runs = []
    current_run = 0
    for aa in seq:
        if aa in HYDROPHILIC:
            current_run += 1
        else:
            if current_run > 0:
                phil_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        phil_runs.append(current_run)
    
    all_runs = runs + phil_runs
    
    # Composition
    hydro_frac = sum(is_hydro) / n
    
    # Fibonacci fraction
    fib_runs = sum(1 for r in all_runs if r in FIBONACCI)
    
    # Run ratios
    ratios = []
    for i in range(len(all_runs) - 1):
        if all_runs[i] > 0 and all_runs[i+1] > 0:
            ratios.append(max(all_runs[i], all_runs[i+1]) / min(all_runs[i], all_runs[i+1]))
    
    mean_ratio = np.mean(ratios) if ratios else 0
    phi_dist = np.mean([abs(r - PHI) for r in ratios]) if ratios else 0
    
    return {
        'length': n,
        'hydrophobic_fraction': hydro_frac,
        'n_runs': len(all_runs),
        'mean_run_length': np.mean(all_runs) if all_runs else 0,
        'fib_runs': fib_runs,
        'fib_run_fraction': fib_runs / len(all_runs) if all_runs else 0,
        'mean_ratio': mean_ratio,
        'phi_distance': phi_dist,
    }


def fetch_real_proteins():
    """Fetch real proteins."""
    import urllib.request
    
    proteins = [
        "P69905", "P02144", "P61626", "P99999", "P0CG48",
        "P01308", "P60709", "Q71U36", "P0DP23", "P02794",
    ]
    
    seqs = []
    for pid in proteins:
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{pid}.fasta"
            with urllib.request.urlopen(url, timeout=10) as response:
                fasta = response.read().decode('utf-8')
            lines = fasta.strip().split('\n')
            seq = ''.join(lines[1:])
            seqs.append(seq)
        except:
            pass
    
    return seqs


def run_experiment():
    """Run calibrated SEC generator experiment."""
    print("=" * 60)
    print("Experiment 10: SEC Generator v2 - Calibrated")
    print("=" * 60)
    
    results = {'timestamp': datetime.now().isoformat()}
    
    print("\n[1] Fetching real proteins...")
    real_proteins = fetch_real_proteins()
    print(f"  Got {len(real_proteins)} proteins")
    
    print("\n[2] Analyzing real protein statistics...")
    real_analysis = [analyze_sequence(s) for s in real_proteins]
    
    print(f"  Hydrophobic fraction: {np.mean([a['hydrophobic_fraction'] for a in real_analysis]):.3f}")
    print(f"  Mean run length: {np.mean([a['mean_run_length'] for a in real_analysis]):.2f}")
    print(f"  Fibonacci run fraction: {np.mean([a['fib_run_fraction'] for a in real_analysis]):.3f}")
    print(f"  Mean consecutive ratio: {np.mean([a['mean_ratio'] for a in real_analysis]):.3f} (φ = 1.618)")
    print(f"  Mean φ distance: {np.mean([a['phi_distance'] for a in real_analysis]):.3f}")
    
    # Generate with different methods
    n_gen = 30
    
    print("\n[3] SEC Transition Generator...")
    sec_proteins = []
    for _ in range(n_gen):
        length = np.random.randint(100, 300)
        gen = SECTransitionGenerator(length)
        sec_proteins.append(gen.generate())
    sec_analysis = [analyze_sequence(s) for s in sec_proteins]
    
    print("\n[4] Phi-Ratio Generator...")
    phi_proteins = []
    for _ in range(n_gen):
        length = np.random.randint(100, 300)
        gen = PhiRatioGenerator(length)
        phi_proteins.append(gen.generate())
    phi_analysis = [analyze_sequence(s) for s in phi_proteins]
    
    print("\n[5] Natural Frequency Control...")
    nat_proteins = []
    for _ in range(n_gen):
        length = np.random.randint(100, 300)
        gen = NaturalFrequencyGenerator(length)
        nat_proteins.append(gen.generate())
    nat_analysis = [analyze_sequence(s) for s in nat_proteins]
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON TO REAL PROTEINS")
    print("=" * 60)
    
    metrics = ['hydrophobic_fraction', 'mean_run_length', 'fib_run_fraction', 'mean_ratio', 'phi_distance']
    
    print(f"\n{'Metric':<22} | {'Real':>8} | {'SEC':>8} | {'Phi':>8} | {'Natural':>8}")
    print("-" * 70)
    
    for metric in metrics:
        real_val = np.mean([a[metric] for a in real_analysis])
        sec_val = np.mean([a[metric] for a in sec_analysis])
        phi_val = np.mean([a[metric] for a in phi_analysis])
        nat_val = np.mean([a[metric] for a in nat_analysis])
        
        print(f"{metric:<22} | {real_val:>8.3f} | {sec_val:>8.3f} | {phi_val:>8.3f} | {nat_val:>8.3f}")
    
    # Compute overall distances
    def compute_distance(analysis_list, real_analysis):
        total_z = 0
        for metric in metrics:
            gen_val = np.mean([a[metric] for a in analysis_list])
            real_val = np.mean([a[metric] for a in real_analysis])
            real_std = np.std([a[metric] for a in real_analysis])
            if real_std > 0:
                total_z += abs(gen_val - real_val) / real_std
            else:
                total_z += abs(gen_val - real_val) * 10
        return total_z / len(metrics)
    
    sec_dist = compute_distance(sec_analysis, real_analysis)
    phi_dist = compute_distance(phi_analysis, real_analysis)
    nat_dist = compute_distance(nat_analysis, real_analysis)
    
    print(f"\n{'Overall distance (σ)':<22} | {'-':>8} | {sec_dist:>8.2f} | {phi_dist:>8.2f} | {nat_dist:>8.2f}")
    
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    best = min([(sec_dist, 'SEC'), (phi_dist, 'Phi-Ratio'), (nat_dist, 'Natural')])
    print(f"  Closest to real: {best[1]} ({best[0]:.2f}σ)")
    
    if sec_dist < nat_dist or phi_dist < nat_dist:
        print(f"  SEC/PAC-based generator outperforms random ✅")
    else:
        print(f"  Random control is closer to real ❌")
    
    # Check φ distance specifically
    real_phi_dist = np.mean([a['phi_distance'] for a in real_analysis])
    sec_phi_dist = np.mean([a['phi_distance'] for a in sec_analysis])
    phi_phi_dist = np.mean([a['phi_distance'] for a in phi_analysis])
    nat_phi_dist = np.mean([a['phi_distance'] for a in nat_analysis])
    
    print(f"\n  φ-distance from real proteins:")
    print(f"    Real: {real_phi_dist:.3f}")
    print(f"    SEC:  {sec_phi_dist:.3f}")
    print(f"    Phi:  {phi_phi_dist:.3f}")
    print(f"    Nat:  {nat_phi_dist:.3f}")
    
    # Save
    results['comparison'] = {
        'SEC': {'distance': sec_dist, 'analysis': sec_analysis[:3]},
        'Phi': {'distance': phi_dist, 'analysis': phi_analysis[:3]},
        'Natural': {'distance': nat_dist, 'analysis': nat_analysis[:3]},
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_10_sec_v2_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")


if __name__ == '__main__':
    run_experiment()

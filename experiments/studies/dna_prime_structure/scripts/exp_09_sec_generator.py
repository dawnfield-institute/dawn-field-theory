"""
Experiment 09: SEC Protein Generator
=====================================

Hypothesis: If proteins evolved under PAC/SEC dynamics, then running
SEC collapse should produce protein-like sequences.

Approach:
1. Start with random amino acid "potential"
2. Apply SEC pressure: ∂S/∂t = α∇I - β∇H
3. Let it collapse to attractors (Fibonacci lengths, φ ratios)
4. Check if output looks like real protein:
   - Hydrophobic core / hydrophilic surface pattern
   - Realistic secondary structure predictions
   - Amino acid composition similar to natural proteins

If SEC generates protein-like structures, that's evidence the
dynamics are correct. If not, we learn something about constraints.
"""

import numpy as np
from collections import defaultdict
import json
import os
from datetime import datetime

# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55]

# Amino acids by property
HYDROPHOBIC = list('AILMFVPWG')
HYDROPHILIC = list('RKDENQHSTY')
NEUTRAL = list('C')
ALL_AA = HYDROPHOBIC + HYDROPHILIC + NEUTRAL

# Natural amino acid frequencies (approximate)
NATURAL_FREQ = {
    'A': 0.074, 'R': 0.042, 'N': 0.044, 'D': 0.059, 'C': 0.033,
    'E': 0.058, 'Q': 0.037, 'G': 0.074, 'H': 0.029, 'I': 0.038,
    'L': 0.076, 'K': 0.072, 'M': 0.018, 'F': 0.040, 'P': 0.050,
    'S': 0.081, 'T': 0.062, 'W': 0.013, 'Y': 0.033, 'V': 0.068,
}


class SECProteinGenerator:
    """
    Generate proteins using SEC dynamics.
    
    The "pressure" is the balance between:
    - Information gradient (∇I): tendency to form structured patterns
    - Entropy gradient (∇H): tendency toward disorder
    
    At collapse points, structure crystallizes.
    """
    
    def __init__(self, length, alpha=1.0, beta=0.5):
        self.length = length
        self.alpha = alpha  # Information pressure
        self.beta = beta    # Entropy resistance
        
        # State: continuous field that will collapse to discrete AA
        # Each position has a "hydrophobicity potential" [-1, 1]
        self.field = np.random.randn(length) * 0.1
        
        # Track which positions have collapsed
        self.collapsed = [False] * length
        self.sequence = ['X'] * length
        
    def compute_information_gradient(self):
        """
        Information gradient: tendency to form Fibonacci-length runs.
        Positions want to align with neighbors to form coherent segments.
        """
        grad = np.zeros(self.length)
        
        for i in range(self.length):
            # Look for Fibonacci-compatible segment lengths
            for fib in FIBONACCI:
                if fib > self.length:
                    break
                    
                # Check if position i could be part of a Fib-length segment
                for start in range(max(0, i - fib + 1), min(i + 1, self.length - fib + 1)):
                    end = start + fib
                    if end <= self.length:
                        # Segment coherence: how aligned is this segment?
                        segment = self.field[start:end]
                        coherence = abs(np.mean(segment))
                        
                        # Pressure toward coherence
                        if np.mean(segment) > 0:
                            grad[i] += coherence * 0.1 / fib
                        else:
                            grad[i] -= coherence * 0.1 / fib
        
        return grad
    
    def compute_entropy_gradient(self):
        """
        Entropy gradient: resistance to ordering.
        Pushes toward uniform distribution.
        """
        # Push extreme values toward zero
        return -self.beta * self.field
    
    def compute_neighbor_pressure(self):
        """
        Local pressure: neighbors influence each other.
        Creates runs of similar hydrophobicity.
        """
        pressure = np.zeros(self.length)
        
        for i in range(self.length):
            neighbors = []
            if i > 0:
                neighbors.append(self.field[i-1])
            if i < self.length - 1:
                neighbors.append(self.field[i+1])
            
            if neighbors:
                # Tendency to align with neighbors (but not too strongly)
                mean_neighbor = np.mean(neighbors)
                pressure[i] = 0.3 * (mean_neighbor - self.field[i])
        
        return pressure
    
    def step(self, dt=0.1):
        """
        One step of SEC dynamics.
        ∂S/∂t = α∇I - β∇H + neighbor_coupling
        """
        info_grad = self.compute_information_gradient()
        entropy_grad = self.compute_entropy_gradient()
        neighbor_pressure = self.compute_neighbor_pressure()
        
        # Add some noise (thermal fluctuations)
        noise = np.random.randn(self.length) * 0.05
        
        # Update field
        delta = dt * (self.alpha * info_grad + entropy_grad + neighbor_pressure + noise)
        self.field += delta
        
        # Check for collapse: positions with high |field| collapse
        for i in range(self.length):
            if not self.collapsed[i] and abs(self.field[i]) > 1.0:
                self.collapse_position(i)
    
    def collapse_position(self, i):
        """
        Collapse position i to a discrete amino acid.
        """
        self.collapsed[i] = True
        
        if self.field[i] > 0:
            # Hydrophobic
            self.sequence[i] = np.random.choice(HYDROPHOBIC)
        else:
            # Hydrophilic
            self.sequence[i] = np.random.choice(HYDROPHILIC)
    
    def force_collapse_remaining(self):
        """Force collapse of any remaining positions."""
        for i in range(self.length):
            if not self.collapsed[i]:
                self.collapse_position(i)
    
    def generate(self, max_steps=1000):
        """Run SEC dynamics until convergence."""
        for step in range(max_steps):
            self.step()
            
            # Check if fully collapsed
            if all(self.collapsed):
                break
        
        # Force collapse any remaining
        self.force_collapse_remaining()
        
        return ''.join(self.sequence)


class FibonacciSECGenerator:
    """
    Alternative: Build protein by explicitly placing Fibonacci-length segments.
    This is more directly testing the PAC hypothesis.
    """
    
    def __init__(self, target_length, fib_bias=0.7):
        self.target_length = target_length
        self.fib_bias = fib_bias
        
    def generate(self):
        """Generate protein using Fibonacci segment lengths."""
        sequence = []
        current_type = np.random.choice(['H', 'P'])  # Hydrophobic or Polar
        
        while len(sequence) < self.target_length:
            # Choose segment length
            if np.random.random() < self.fib_bias:
                # Fibonacci length
                valid_fibs = [f for f in FIBONACCI if f <= self.target_length - len(sequence)]
                if valid_fibs:
                    seg_len = np.random.choice(valid_fibs)
                else:
                    seg_len = self.target_length - len(sequence)
            else:
                # Random length (control)
                remaining = self.target_length - len(sequence)
                seg_len = np.random.randint(1, min(20, remaining) + 1)
            
            # Generate segment
            if current_type == 'H':
                segment = [np.random.choice(HYDROPHOBIC) for _ in range(seg_len)]
            else:
                segment = [np.random.choice(HYDROPHILIC) for _ in range(seg_len)]
            
            sequence.extend(segment)
            
            # Alternate type (with some probability of same)
            if np.random.random() < 0.8:
                current_type = 'P' if current_type == 'H' else 'H'
        
        return ''.join(sequence[:self.target_length])


def analyze_sequence(seq):
    """Analyze generated sequence for protein-like properties."""
    n = len(seq)
    
    # Hydrophobic runs
    runs = []
    current_run = 0
    for aa in seq:
        if aa in HYDROPHOBIC:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)
    
    # Composition
    hydro_frac = sum(1 for aa in seq if aa in HYDROPHOBIC) / n
    
    # Fibonacci in runs
    fib_runs = sum(1 for r in runs if r in FIBONACCI)
    
    # Alternating pattern (characteristic of transmembrane or amphipathic)
    alternating = 0
    for i in range(n - 1):
        h1 = seq[i] in HYDROPHOBIC
        h2 = seq[i+1] in HYDROPHOBIC
        if h1 != h2:
            alternating += 1
    
    return {
        'length': n,
        'hydrophobic_fraction': hydro_frac,
        'n_runs': len(runs),
        'mean_run_length': np.mean(runs) if runs else 0,
        'fib_runs': fib_runs,
        'fib_run_fraction': fib_runs / len(runs) if runs else 0,
        'alternating_fraction': alternating / (n - 1) if n > 1 else 0,
    }


def compare_to_real_proteins(generated_seqs, real_seqs):
    """Compare generated sequences to real proteins."""
    gen_stats = [analyze_sequence(s) for s in generated_seqs]
    real_stats = [analyze_sequence(s) for s in real_seqs]
    
    comparison = {}
    
    for key in gen_stats[0].keys():
        if key == 'length':
            continue
        gen_vals = [s[key] for s in gen_stats]
        real_vals = [s[key] for s in real_stats]
        
        comparison[key] = {
            'generated_mean': np.mean(gen_vals),
            'generated_std': np.std(gen_vals),
            'real_mean': np.mean(real_vals),
            'real_std': np.std(real_vals),
        }
    
    return comparison


def fetch_real_proteins():
    """Fetch real proteins for comparison."""
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
    """Run the SEC protein generation experiment."""
    print("=" * 60)
    print("Experiment 09: SEC Protein Generator")
    print("=" * 60)
    
    results = {'timestamp': datetime.now().isoformat()}
    
    # Fetch real proteins
    print("\n[1] Fetching real proteins for comparison...")
    real_proteins = fetch_real_proteins()
    print(f"  Got {len(real_proteins)} real proteins")
    
    # Generate with SEC dynamics
    print("\n[2] Generating proteins with SEC dynamics...")
    sec_proteins = []
    for i in range(20):
        length = np.random.randint(50, 200)
        gen = SECProteinGenerator(length, alpha=1.0, beta=0.3)
        seq = gen.generate(max_steps=500)
        sec_proteins.append(seq)
    
    print(f"  Generated {len(sec_proteins)} SEC proteins")
    print(f"  Example: {sec_proteins[0][:50]}...")
    
    # Generate with Fibonacci-biased method
    print("\n[3] Generating proteins with Fibonacci-biased segments...")
    fib_proteins = []
    for i in range(20):
        length = np.random.randint(50, 200)
        gen = FibonacciSECGenerator(length, fib_bias=0.8)
        seq = gen.generate()
        fib_proteins.append(seq)
    
    print(f"  Generated {len(fib_proteins)} Fibonacci-biased proteins")
    print(f"  Example: {fib_proteins[0][:50]}...")
    
    # Generate random control
    print("\n[4] Generating random control proteins...")
    random_proteins = []
    for i in range(20):
        length = np.random.randint(50, 200)
        seq = ''.join(np.random.choice(list(ALL_AA), size=length))
        random_proteins.append(seq)
    
    print(f"  Generated {len(random_proteins)} random proteins")
    
    # Analyze all
    print("\n[5] Analyzing sequences...")
    
    print("\n  REAL PROTEINS:")
    real_analysis = [analyze_sequence(s) for s in real_proteins]
    print(f"    Hydrophobic fraction: {np.mean([a['hydrophobic_fraction'] for a in real_analysis]):.3f}")
    print(f"    Mean run length: {np.mean([a['mean_run_length'] for a in real_analysis]):.2f}")
    print(f"    Fibonacci run fraction: {np.mean([a['fib_run_fraction'] for a in real_analysis]):.3f}")
    
    print("\n  SEC GENERATED:")
    sec_analysis = [analyze_sequence(s) for s in sec_proteins]
    print(f"    Hydrophobic fraction: {np.mean([a['hydrophobic_fraction'] for a in sec_analysis]):.3f}")
    print(f"    Mean run length: {np.mean([a['mean_run_length'] for a in sec_analysis]):.2f}")
    print(f"    Fibonacci run fraction: {np.mean([a['fib_run_fraction'] for a in sec_analysis]):.3f}")
    
    print("\n  FIBONACCI-BIASED:")
    fib_analysis = [analyze_sequence(s) for s in fib_proteins]
    print(f"    Hydrophobic fraction: {np.mean([a['hydrophobic_fraction'] for a in fib_analysis]):.3f}")
    print(f"    Mean run length: {np.mean([a['mean_run_length'] for a in fib_analysis]):.2f}")
    print(f"    Fibonacci run fraction: {np.mean([a['fib_run_fraction'] for a in fib_analysis]):.3f}")
    
    print("\n  RANDOM:")
    random_analysis = [analyze_sequence(s) for s in random_proteins]
    print(f"    Hydrophobic fraction: {np.mean([a['hydrophobic_fraction'] for a in random_analysis]):.3f}")
    print(f"    Mean run length: {np.mean([a['mean_run_length'] for a in random_analysis]):.2f}")
    print(f"    Fibonacci run fraction: {np.mean([a['fib_run_fraction'] for a in random_analysis]):.3f}")
    
    # Distance from real
    print("\n[6] Distance from real proteins...")
    
    def distance_from_real(analysis_list, real_analysis):
        """Compute how close generated is to real."""
        metrics = ['hydrophobic_fraction', 'mean_run_length', 'fib_run_fraction']
        distances = []
        for metric in metrics:
            gen_mean = np.mean([a[metric] for a in analysis_list])
            real_mean = np.mean([a[metric] for a in real_analysis])
            real_std = np.std([a[metric] for a in real_analysis])
            if real_std > 0:
                z = abs(gen_mean - real_mean) / real_std
            else:
                z = abs(gen_mean - real_mean)
            distances.append(z)
        return np.mean(distances)
    
    sec_dist = distance_from_real(sec_analysis, real_analysis)
    fib_dist = distance_from_real(fib_analysis, real_analysis)
    random_dist = distance_from_real(random_analysis, real_analysis)
    
    print(f"  SEC distance from real:        {sec_dist:.2f} σ")
    print(f"  Fibonacci distance from real:  {fib_dist:.2f} σ")
    print(f"  Random distance from real:     {random_dist:.2f} σ")
    
    results['distances'] = {
        'SEC': sec_dist,
        'Fibonacci': fib_dist,
        'Random': random_dist,
    }
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if sec_dist < random_dist:
        print(f"  SEC generates MORE protein-like sequences than random ✅")
        print(f"    ({sec_dist:.2f}σ vs {random_dist:.2f}σ from real)")
    else:
        print(f"  SEC does NOT outperform random ❌")
    
    if fib_dist < random_dist:
        print(f"  Fibonacci-biased generates MORE protein-like sequences ✅")
        print(f"    ({fib_dist:.2f}σ vs {random_dist:.2f}σ from real)")
    
    closest = min([(sec_dist, 'SEC'), (fib_dist, 'Fibonacci'), (random_dist, 'Random')])
    print(f"\n  Closest to real: {closest[1]} ({closest[0]:.2f}σ)")
    
    # Save example sequences
    results['examples'] = {
        'sec': sec_proteins[:3],
        'fibonacci': fib_proteins[:3],
        'random': random_proteins[:3],
    }
    
    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_09_sec_generator_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")


if __name__ == '__main__':
    run_experiment()

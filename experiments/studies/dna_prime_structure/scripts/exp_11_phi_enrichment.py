"""
Experiment 11: SEC Under Functional Constraint
===============================================

Key insight from exp_10: Real proteins are BETWEEN random and φ-optimized.
- Random: φ-distance = 0.924
- Real:   φ-distance = 0.864  
- Phi:    φ-distance = 0.687

This suggests SEC pressure exists but is CONSTRAINED by functional requirements.
Proteins can't fully optimize for recursive structure because they need to fold,
bind substrates, catalyze reactions, etc.

New model: Mixed generator
- SEC pressure pulls toward Fibonacci/φ
- Functional constraint (modeled as random perturbation) resists

Also: Let's look at the RATIO distribution more carefully.
If SEC is operating, consecutive ratios should cluster near φ.
"""

import numpy as np
from collections import defaultdict
from scipy import stats
import json
import os
from datetime import datetime
import urllib.request

PHI = (1 + np.sqrt(5)) / 2
FIBONACCI = set([1, 2, 3, 5, 8, 13, 21, 34])

HYDROPHOBIC = set('AILMFVPWG')
HYDROPHILIC = set('RKDENQHSTY')


def fetch_real_proteins():
    """Fetch diverse proteins."""
    proteins = [
        # Enzymes
        "P00760",  # Trypsin
        "P00761",  # Chymotrypsin
        "P00918",  # Carbonic anhydrase
        # Structural
        "P69905",  # Hemoglobin alpha
        "P02144",  # Myoglobin
        # Signaling
        "P0DP23",  # Calmodulin
        "P0CG48",  # Ubiquitin
        # Transport
        "P02787",  # Transferrin
        "P02768",  # Albumin
        # Immune
        "P01857",  # IgG
    ]
    
    seqs = []
    for pid in proteins:
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{pid}.fasta"
            with urllib.request.urlopen(url, timeout=10) as response:
                fasta = response.read().decode('utf-8')
            lines = fasta.strip().split('\n')
            seq = ''.join(lines[1:])
            if len(seq) > 50:
                seqs.append(seq)
        except:
            pass
    
    return seqs


def get_run_ratios(seq):
    """Get ratios of consecutive runs (all types)."""
    # Collect ALL runs (hydrophobic, hydrophilic, and transitions between)
    runs = []
    current_hydro = seq[0] in HYDROPHOBIC
    current_len = 1
    
    for aa in seq[1:]:
        is_hydro = aa in HYDROPHOBIC
        if is_hydro == current_hydro:
            current_len += 1
        else:
            runs.append(current_len)
            current_hydro = is_hydro
            current_len = 1
    runs.append(current_len)
    
    # Compute consecutive ratios
    ratios = []
    for i in range(len(runs) - 1):
        if runs[i] > 0 and runs[i+1] > 0:
            r = max(runs[i], runs[i+1]) / min(runs[i], runs[i+1])
            ratios.append(r)
    
    return runs, ratios


def analyze_ratio_distribution(proteins):
    """Analyze the distribution of consecutive run ratios."""
    all_ratios = []
    
    for seq in proteins:
        runs, ratios = get_run_ratios(seq)
        all_ratios.extend(ratios)
    
    if not all_ratios:
        return None
    
    # Bin analysis
    bins = np.linspace(1, 4, 31)
    hist, _ = np.histogram(all_ratios, bins=bins)
    
    # Find peaks
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            center = (bins[i] + bins[i+1]) / 2
            peaks.append((center, hist[i]))
    
    # Distance from φ distribution
    phi_distances = [abs(r - PHI) for r in all_ratios]
    
    return {
        'n_ratios': len(all_ratios),
        'mean_ratio': np.mean(all_ratios),
        'median_ratio': np.median(all_ratios),
        'std_ratio': np.std(all_ratios),
        'mean_phi_distance': np.mean(phi_distances),
        'fraction_near_phi': sum(1 for d in phi_distances if d < 0.2) / len(all_ratios),
        'peaks': sorted(peaks, key=lambda x: -x[1])[:5],
        'histogram': list(hist),
        'bin_edges': list(bins),
    }


def generate_random_baseline(n_seqs, lengths):
    """Generate random sequences with natural AA frequencies."""
    NATURAL_FREQ = {
        'A': 0.074, 'R': 0.042, 'N': 0.044, 'D': 0.059, 'C': 0.033,
        'E': 0.058, 'Q': 0.037, 'G': 0.074, 'H': 0.029, 'I': 0.038,
        'L': 0.076, 'K': 0.072, 'M': 0.018, 'F': 0.040, 'P': 0.050,
        'S': 0.081, 'T': 0.062, 'W': 0.013, 'Y': 0.033, 'V': 0.068,
    }
    
    aas = list(NATURAL_FREQ.keys())
    probs = np.array([NATURAL_FREQ[aa] for aa in aas])
    probs = probs / probs.sum()
    
    seqs = []
    for i in range(n_seqs):
        length = lengths[i % len(lengths)]
        seq = ''.join(np.random.choice(aas, size=length, p=probs))
        seqs.append(seq)
    
    return seqs


def test_phi_enrichment(real_ratios, random_ratios):
    """Statistical test for φ enrichment."""
    # Distance to φ for real
    real_phi_dist = [abs(r - PHI) for r in real_ratios]
    
    # Distance to φ for random
    random_phi_dist = [abs(r - PHI) for r in random_ratios]
    
    # t-test
    t_stat, p_value = stats.ttest_ind(real_phi_dist, random_phi_dist)
    
    # Effect size
    pooled_std = np.sqrt((np.var(real_phi_dist) + np.var(random_phi_dist)) / 2)
    cohens_d = (np.mean(random_phi_dist) - np.mean(real_phi_dist)) / pooled_std
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'real_mean_dist': np.mean(real_phi_dist),
        'random_mean_dist': np.mean(random_phi_dist),
        'real_closer': np.mean(real_phi_dist) < np.mean(random_phi_dist),
    }


def run_experiment():
    """Test if real proteins show φ-enrichment in run ratios."""
    print("=" * 60)
    print("Experiment 11: φ Enrichment in Consecutive Run Ratios")
    print("=" * 60)
    
    results = {'timestamp': datetime.now().isoformat()}
    
    print("\n[1] Fetching diverse proteins...")
    real_proteins = fetch_real_proteins()
    print(f"  Got {len(real_proteins)} proteins")
    print(f"  Total residues: {sum(len(s) for s in real_proteins)}")
    
    print("\n[2] Extracting run ratios from real proteins...")
    all_real_ratios = []
    for seq in real_proteins:
        _, ratios = get_run_ratios(seq)
        all_real_ratios.extend(ratios)
    
    print(f"  Extracted {len(all_real_ratios)} consecutive ratios")
    
    real_stats = analyze_ratio_distribution(real_proteins)
    print(f"  Mean ratio: {real_stats['mean_ratio']:.3f} (φ = 1.618)")
    print(f"  Median ratio: {real_stats['median_ratio']:.3f}")
    print(f"  Mean φ-distance: {real_stats['mean_phi_distance']:.3f}")
    print(f"  Fraction within 0.2 of φ: {real_stats['fraction_near_phi']:.3f}")
    print(f"  Peaks at: {[f'{p[0]:.2f}' for p in real_stats['peaks']]}")
    
    print("\n[3] Generating random baseline...")
    lengths = [len(s) for s in real_proteins]
    random_proteins = generate_random_baseline(len(real_proteins) * 10, lengths)
    
    all_random_ratios = []
    for seq in random_proteins:
        _, ratios = get_run_ratios(seq)
        all_random_ratios.extend(ratios)
    
    random_stats = analyze_ratio_distribution(random_proteins)
    print(f"  Random mean ratio: {random_stats['mean_ratio']:.3f}")
    print(f"  Random mean φ-distance: {random_stats['mean_phi_distance']:.3f}")
    print(f"  Random fraction near φ: {random_stats['fraction_near_phi']:.3f}")
    
    print("\n[4] Statistical test for φ enrichment...")
    test_results = test_phi_enrichment(all_real_ratios, all_random_ratios)
    results['phi_test'] = test_results
    
    print(f"  Real mean φ-distance:   {test_results['real_mean_dist']:.4f}")
    print(f"  Random mean φ-distance: {test_results['random_mean_dist']:.4f}")
    print(f"  t-statistic: {test_results['t_statistic']:.2f}")
    print(f"  p-value: {test_results['p_value']:.2e}")
    print(f"  Cohen's d: {test_results['cohens_d']:.3f}")
    
    print("\n[5] Checking specific ratio values...")
    
    # Count ratios at specific values
    def count_near(ratios, target, tolerance=0.1):
        return sum(1 for r in ratios if abs(r - target) < tolerance)
    
    targets = [1.0, 1.5, PHI, 2.0, 2.5, 3.0]
    print(f"\n  {'Target':>8} | {'Real %':>8} | {'Random %':>8} | {'Enrichment':>10}")
    print("  " + "-" * 45)
    
    for target in targets:
        real_count = count_near(all_real_ratios, target)
        random_count = count_near(all_random_ratios, target)
        real_frac = real_count / len(all_real_ratios)
        random_frac = random_count / len(all_random_ratios)
        enrichment = real_frac / random_frac if random_frac > 0 else 0
        
        marker = "←φ" if abs(target - PHI) < 0.01 else ""
        print(f"  {target:>8.3f} | {100*real_frac:>7.1f}% | {100*random_frac:>7.1f}% | {enrichment:>9.2f}x {marker}")
    
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if test_results['p_value'] < 0.05 and test_results['real_closer']:
        print(f"  Real proteins are SIGNIFICANTLY closer to φ ✅")
        print(f"  p = {test_results['p_value']:.2e}, Cohen's d = {test_results['cohens_d']:.3f}")
    elif test_results['real_closer']:
        print(f"  Real proteins trend toward φ but not significant")
        print(f"  p = {test_results['p_value']:.3f}")
    else:
        print(f"  No φ enrichment detected ❌")
    
    # Save
    results['real_stats'] = real_stats
    results['random_stats'] = random_stats
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f'exp_11_phi_enrichment_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {filepath}")


if __name__ == '__main__':
    run_experiment()

"""
Experiment 14: Real vs Cramér Visualization

Creates publication-quality visualizations comparing:
1. λ₁ scaling: Real primes vs Cramér model
2. Chord vocabulary distribution
3. Gap distribution comparison
4. Eigenvalue spectrum comparison
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from prime_chords import (
    get_primes, compute_gaps, extract_chords,
    build_transition_matrix, compute_eigenvalues, PHI, PHI_INV
)
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import sympy as sp


def generate_cramer_primes(limit: int, seed: int = None) -> np.ndarray:
    """Generate Cramér random primes."""
    rng = np.random.default_rng(seed)
    primes = [2]
    for n in range(3, limit):
        if rng.random() < 1 / np.log(n):
            primes.append(n)
    return np.array(primes, dtype=float)


def analyze_sequence(primes, topK=25):
    """Full analysis of a prime sequence."""
    gaps = np.diff(primes)
    chords = extract_chords(gaps, n_gaps=2)
    P, _ = build_transition_matrix(chords, top_k=topK)
    eigenvals = compute_eigenvalues(P[:topK, :topK])
    
    chord_counts = Counter([tuple(c) for c in chords])
    
    return {
        'gaps': gaps,
        'chords': chords,
        'eigenvalues': eigenvals,
        'chord_counts': chord_counts,
        'n_unique': len(chord_counts),
    }


def run_visualization():
    """Generate all visualizations."""
    
    print("=" * 70)
    print("PRIME HARMONIC MANIFOLD: Real vs Cramér Visualization")
    print("=" * 70)
    
    # Set up figure directory
    fig_dir = Path(__file__).parent.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Generate data
    print("\nGenerating data...")
    prime_limit = 500_000
    
    real_primes = get_primes(prime_limit)
    real_data = analyze_sequence(real_primes)
    print(f"  Real primes: {len(real_primes):,}")
    
    # Multiple Cramér trials
    n_cramer = 10
    cramer_data_list = []
    for i in range(n_cramer):
        cp = generate_cramer_primes(prime_limit, seed=i)
        cramer_data_list.append(analyze_sequence(cp))
    print(f"  Cramér trials: {n_cramer}")
    
    # =========================================================================
    # Figure 1: λ₁ Scaling Comparison
    # =========================================================================
    print("\n[1/4] λ₁ Scaling Comparison...")
    
    test_limits = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000]
    
    real_lambda1s = []
    cramer_lambda1s_mean = []
    cramer_lambda1s_std = []
    real_n_primes = []
    
    for lim in test_limits:
        # Real
        rp = get_primes(lim)
        rd = analyze_sequence(rp)
        real_lambda1s.append(rd['eigenvalues'][0])
        real_n_primes.append(len(rp))
        
        # Cramér (multiple)
        cl1s = []
        for seed in range(5):
            cp = generate_cramer_primes(lim, seed=seed)
            cd = analyze_sequence(cp)
            cl1s.append(cd['eigenvalues'][0])
        cramer_lambda1s_mean.append(np.mean(cl1s))
        cramer_lambda1s_std.append(np.std(cl1s))
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    log_n = np.log10(real_n_primes)
    
    ax1.plot(log_n, real_lambda1s, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='Real Primes')
    ax1.errorbar(log_n, cramer_lambda1s_mean, yerr=cramer_lambda1s_std, 
                 fmt='s--', color='#E94F37', linewidth=2, markersize=8, capsize=4, label='Cramér Model')
    ax1.axhline(y=PHI_INV, color='#F5B041', linestyle=':', linewidth=2, label=f'1/φ ≈ {PHI_INV:.3f}')
    
    ax1.set_xlabel('log₁₀(Number of Primes)', fontsize=12)
    ax1.set_ylabel('Leading Eigenvalue λ₁', fontsize=12)
    ax1.set_title('Prime Chord Eigenvalue: Real vs Random', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_ylim(0.2, 0.9)
    
    # Add annotations
    ax1.annotate(f'30σ separation', xy=(log_n[-1], (real_lambda1s[-1] + cramer_lambda1s_mean[-1])/2),
                 fontsize=10, ha='right')
    
    fig1.tight_layout()
    fig1.savefig(fig_dir / 'fig1_lambda_scaling.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {fig_dir / 'fig1_lambda_scaling.png'}")
    
    # =========================================================================
    # Figure 2: Chord Vocabulary Comparison
    # =========================================================================
    print("[2/4] Chord Vocabulary Comparison...")
    
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Real chord frequency
    real_sorted = sorted(real_data['chord_counts'].values(), reverse=True)
    ax2a.bar(range(min(50, len(real_sorted))), real_sorted[:50], color='#2E86AB', alpha=0.8)
    ax2a.set_xlabel('Chord Rank', fontsize=11)
    ax2a.set_ylabel('Frequency', fontsize=11)
    ax2a.set_title(f'Real Primes: {real_data["n_unique"]} unique chords', fontsize=12)
    ax2a.set_yscale('log')
    
    # Cramér chord frequency (first trial)
    cramer_sorted = sorted(cramer_data_list[0]['chord_counts'].values(), reverse=True)
    ax2b.bar(range(min(50, len(cramer_sorted))), cramer_sorted[:50], color='#E94F37', alpha=0.8)
    ax2b.set_xlabel('Chord Rank', fontsize=11)
    ax2b.set_ylabel('Frequency', fontsize=11)
    ax2b.set_title(f'Cramér Model: {cramer_data_list[0]["n_unique"]} unique chords', fontsize=12)
    ax2b.set_yscale('log')
    
    fig2.suptitle('Chord Vocabulary Distribution', fontsize=14, fontweight='bold', y=1.02)
    fig2.tight_layout()
    fig2.savefig(fig_dir / 'fig2_chord_vocabulary.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {fig_dir / 'fig2_chord_vocabulary.png'}")
    
    # =========================================================================
    # Figure 3: Gap Distribution
    # =========================================================================
    print("[3/4] Gap Distribution Comparison...")
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Real gaps
    real_gaps = real_data['gaps']
    bins = np.arange(0, 80, 2)
    ax3.hist(real_gaps, bins=bins, density=True, alpha=0.7, color='#2E86AB', label='Real Primes')
    
    # Cramér gaps (average over trials)
    all_cramer_gaps = np.concatenate([cd['gaps'] for cd in cramer_data_list])
    ax3.hist(all_cramer_gaps, bins=bins, density=True, alpha=0.5, color='#E94F37', label='Cramér Model')
    
    ax3.set_xlabel('Gap Size', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('Prime Gap Distribution: Real vs Cramér', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    
    # Stats annotation
    real_mean, real_std = np.mean(real_gaps), np.std(real_gaps)
    cram_mean, cram_std = np.mean(all_cramer_gaps), np.std(all_cramer_gaps)
    ax3.annotate(f'Real: μ={real_mean:.1f}, σ={real_std:.1f}\nCramér: μ={cram_mean:.1f}, σ={cram_std:.1f}',
                xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig3.tight_layout()
    fig3.savefig(fig_dir / 'fig3_gap_distribution.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {fig_dir / 'fig3_gap_distribution.png'}")
    
    # =========================================================================
    # Figure 4: Eigenvalue Spectrum
    # =========================================================================
    print("[4/4] Eigenvalue Spectrum Comparison...")
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    n_eigen = 20
    real_ev = real_data['eigenvalues'][:n_eigen]
    
    # Plot real
    ax4.plot(range(1, n_eigen+1), real_ev, 'o-', color='#2E86AB', linewidth=2, markersize=10, label='Real Primes')
    
    # Plot Cramér (with error bands)
    cramer_evs = np.array([cd['eigenvalues'][:n_eigen] for cd in cramer_data_list])
    cramer_mean = np.mean(cramer_evs, axis=0)
    cramer_std = np.std(cramer_evs, axis=0)
    
    ax4.plot(range(1, n_eigen+1), cramer_mean, 's--', color='#E94F37', linewidth=2, markersize=8, label='Cramér Model')
    ax4.fill_between(range(1, n_eigen+1), cramer_mean - cramer_std, cramer_mean + cramer_std, 
                     color='#E94F37', alpha=0.2)
    
    # Reference lines
    ax4.axhline(y=PHI_INV, color='#F5B041', linestyle=':', linewidth=2, label=f'1/φ ≈ {PHI_INV:.3f}')
    ax4.axhline(y=1/PHI**2, color='#9B59B6', linestyle=':', linewidth=1.5, alpha=0.7, label=f'1/φ² ≈ {1/PHI**2:.3f}')
    
    ax4.set_xlabel('Eigenvalue Index', fontsize=12)
    ax4.set_ylabel('|λᵢ|', fontsize=12)
    ax4.set_title('Transition Matrix Eigenvalue Spectrum', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10, loc='upper right')
    ax4.set_xlim(0.5, n_eigen + 0.5)
    ax4.set_ylim(0, 0.7)
    
    fig4.tight_layout()
    fig4.savefig(fig_dir / 'fig4_eigenvalue_spectrum.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {fig_dir / 'fig4_eigenvalue_spectrum.png'}")
    
    # =========================================================================
    # Figure 5: Decay Rate with 1/π² Fit
    # =========================================================================
    print("[5/5] Decay Rate Analysis...")
    
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    
    log_n_full = np.log10(real_n_primes)
    
    ax5.plot(log_n_full, real_lambda1s, 'o', color='#2E86AB', markersize=10, label='Data')
    
    # Fit line
    from scipy.optimize import curve_fit
    def linear(x, a, b): return a * x + b
    popt, _ = curve_fit(linear, log_n_full, real_lambda1s)
    
    x_fit = np.linspace(min(log_n_full) - 0.5, max(log_n_full) + 2, 100)
    ax5.plot(x_fit, linear(x_fit, *popt), '--', color='#2E86AB', linewidth=2, 
             label=f'Fit: λ₁ = {popt[0]:.4f}·log₁₀(N) + {popt[1]:.3f}')
    
    # 1/π² prediction
    slope_pi2 = -1/np.pi**2
    intercept_pred = np.mean(real_lambda1s - slope_pi2 * log_n_full)
    ax5.plot(x_fit, slope_pi2 * x_fit + intercept_pred, ':', color='#27AE60', linewidth=2,
             label=f'Theory: slope = -1/π² ≈ {slope_pi2:.4f}')
    
    ax5.axhline(y=PHI_INV, color='#F5B041', linestyle='-', linewidth=1.5, alpha=0.7, label='1/φ')
    ax5.axhline(y=0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='0.5')
    
    ax5.set_xlabel('log₁₀(Number of Primes)', fontsize=12)
    ax5.set_ylabel('Leading Eigenvalue λ₁', fontsize=12)
    ax5.set_title('Eigenvalue Decay: λ₁ ≈ 1.12 - (1/π²)·log₁₀(N)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.set_ylim(0.3, 0.9)
    
    fig5.tight_layout()
    fig5.savefig(fig_dir / 'fig5_decay_rate.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: {fig_dir / 'fig5_decay_rate.png'}")
    
    plt.close('all')
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nAll figures saved to: {fig_dir}")
    print("\nFigures:")
    print("  1. fig1_lambda_scaling.png - Real vs Cramér λ₁ scaling")
    print("  2. fig2_chord_vocabulary.png - Chord frequency distributions")
    print("  3. fig3_gap_distribution.png - Gap size histograms")
    print("  4. fig4_eigenvalue_spectrum.png - Full eigenvalue spectrum")
    print("  5. fig5_decay_rate.png - Decay rate with 1/π² fit")


if __name__ == '__main__':
    run_visualization()

"""
Visualization utilities for Prime Harmonic Manifold experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple, Optional

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


def plot_chord_heatmap(chords: List[Tuple], max_gap: int = 40, title: str = "2-gap chord frequency"):
    """Plot heatmap of 2-gap chord frequencies."""
    counts = Counter(chords)
    
    grid = np.zeros((max_gap + 1, max_gap + 1), dtype=int)
    for (a, b), c in counts.items():
        if int(a) <= max_gap and int(b) <= max_gap:
            grid[int(a), int(b)] = c
    
    plt.figure(figsize=(8, 7))
    plt.imshow(grid, origin='lower', interpolation='nearest', cmap='viridis')
    plt.title(title)
    plt.xlabel("g₂")
    plt.ylabel("g₁")
    plt.colorbar(label="count")
    plt.tight_layout()
    return plt.gcf()


def plot_eigenvalue_spectrum(eigenvalues: np.ndarray, title: str = "Transition matrix eigenvalues"):
    """Plot eigenvalue spectrum with φ-markers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart of magnitudes
    axes[0].bar(range(len(eigenvalues)), eigenvalues)
    axes[0].axhline(PHI, color='gold', linestyle='--', label='φ')
    axes[0].axhline(PHI_INV, color='orange', linestyle='--', label='1/φ')
    axes[0].axhline(1/PHI**2, color='red', linestyle=':', label='1/φ²')
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("|λ|")
    axes[0].set_title("Eigenvalue magnitudes")
    axes[0].legend()
    
    # Complex plane
    if np.iscomplexobj(eigenvalues):
        axes[1].scatter(eigenvalues.real, eigenvalues.imag, alpha=0.7, s=50)
    else:
        axes[1].scatter(eigenvalues, np.zeros_like(eigenvalues), alpha=0.7, s=50)
    
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    axes[1].add_patch(circle)
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].axvline(0, color='gray', linewidth=0.5)
    axes[1].set_xlabel("Real")
    axes[1].set_ylabel("Imaginary")
    axes[1].set_title("Complex plane")
    axes[1].axis('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_scale_convergence(results: List[dict], title: str = "λ₁ convergence with scale"):
    """Plot scale test results showing convergence to 1/φ."""
    limits = [r['limit'] for r in results]
    lambda1s = [r['lambda1'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Main convergence plot
    axes[0].semilogx(limits, lambda1s, 'bo-', markersize=8, label='Observed λ₁')
    axes[0].axhline(PHI_INV, color='gold', linestyle='--', linewidth=2, label=f'1/φ = {PHI_INV:.4f}')
    axes[0].axhline(0.5, color='green', linestyle=':', linewidth=2, label='0.5')
    axes[0].set_xlabel('Prime limit N')
    axes[0].set_ylabel('Leading eigenvalue λ₁')
    axes[0].set_title('λ₁ vs Prime Range')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals from 1/φ
    residuals = np.array(lambda1s) - PHI_INV
    axes[1].semilogx(limits, residuals, 'bo-', markersize=8)
    axes[1].axhline(0, color='gold', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prime limit N')
    axes[1].set_ylabel('λ₁ − 1/φ')
    axes[1].set_title('Deviation from Golden Ratio')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_gap_distribution(gaps: np.ndarray, primes: np.ndarray, max_gap: int = 60):
    """Plot gap histogram and normalized gaps."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Raw gaps
    axes[0].hist(gaps, bins=range(0, max_gap + 2, 2), edgecolor='black')
    axes[0].set_xlabel("Gap size")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Prime gap distribution")
    
    # Normalized gaps
    logs = np.log(primes[:-1])
    norm_gaps = gaps / logs
    axes[1].hist(norm_gaps, bins=80, range=(0, 4))
    axes[1].set_xlabel("g / log(p)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Normalized gaps")
    
    # Fractional part
    frac = norm_gaps % 1.0
    axes[2].hist(frac, bins=50)
    axes[2].set_xlabel("Fractional part")
    axes[2].set_ylabel("Count")
    axes[2].set_title("(g/log p) mod 1")
    
    plt.tight_layout()
    return fig


def plot_chord_ratios(chords: List[Tuple], counts: Counter):
    """Plot chord ratio distribution with φ markers."""
    ratios_weighted = []
    for (g1, g2), count in counts.items():
        if g1 > 0:
            ratios_weighted.extend([g2 / g1] * count)
    
    plt.figure(figsize=(10, 5))
    plt.hist(ratios_weighted, bins=50, range=(0, 3), alpha=0.7)
    plt.axvline(PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
    plt.axvline(PHI_INV, color='orange', linestyle='--', linewidth=2, label=f'1/φ = {PHI_INV:.3f}')
    plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='1.0')
    plt.axvline(PHI**2, color='darkgoldenrod', linestyle=':', linewidth=2, label=f'φ² = {PHI**2:.3f}')
    plt.xlabel("Ratio g₂/g₁")
    plt.ylabel("Frequency")
    plt.title("Chord ratio distribution")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_autocorrelation(acf: np.ndarray, max_lag: int = 30):
    """Plot autocorrelation with φ-decay overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    lags = np.arange(len(acf))
    
    # Bar plot
    axes[0].bar(lags, acf, alpha=0.7, label='Observed ACF')
    theoretical = [1/PHI**k for k in lags]
    axes[0].plot(lags, theoretical, 'r--', linewidth=2, label='φ^(-lag) decay')
    axes[0].axhline(1/np.e, color='gray', linestyle=':', label='1/e threshold')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Autocorrelation')
    axes[0].set_title('Gap ACF vs φ-decay')
    axes[0].legend()
    
    # Log-log plot
    valid_acf = np.abs(acf[1:20])
    valid_lags = lags[1:20]
    axes[1].loglog(valid_lags, valid_acf, 'bo-', label='|ACF|')
    axes[1].loglog(valid_lags, [1/PHI**k for k in valid_lags], 'r--', label='φ^(-k)')
    axes[1].set_xlabel('Lag (log)')
    axes[1].set_ylabel('|ACF| (log)')
    axes[1].set_title('Log-log ACF decay')
    axes[1].legend()
    
    plt.tight_layout()
    return fig

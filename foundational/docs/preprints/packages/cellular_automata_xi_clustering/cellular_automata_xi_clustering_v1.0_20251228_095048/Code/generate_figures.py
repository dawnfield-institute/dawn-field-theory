#!/usr/bin/env python3
"""
Generate publication figures for Cellular Automata Xi Clustering paper.

Generates:
- Figure 1: PAC phase space scatter plot (all 256 rules, colored by Wolfram class)
- Figure 2: Top 10 rules distance from Xi
- Figure 3: Class distribution by Xi-proximity (box plots)
- Figure 4: Statistical significance summary
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configuration
FIGURES_DIR = Path("../Figures")
RESULTS_DIR = Path("../Data/results")
XI_TARGET = 1.0571

# Wolfram class mapping (from paper)
WOLFRAM_CLASSES = {
    # Class IV (edge-of-chaos, Turing-complete)
    110: 'IV', 124: 'IV', 137: 'IV', 193: 'IV', 54: 'IV', 122: 'IV',

    # Class I (uniform, fixed point)
    0: 'I', 8: 'I', 32: 'I', 40: 'I', 128: 'I', 136: 'I', 160: 'I', 168: 'I',
    1: 'I', 2: 'I', 3: 'I', 4: 'I', 5: 'I', 6: 'I', 7: 'I', 9: 'I',

    # Class III (chaotic)
    18: 'III', 22: 'III', 30: 'III', 45: 'III', 60: 'III', 73: 'III', 90: 'III',
    105: 'III', 106: 'III', 129: 'III', 135: 'III', 146: 'III', 149: 'III', 150: 'III',
    182: 'III', 225: 'III',
}

def load_full_sweep():
    """Load the full 256-rule sweep data."""
    with open(RESULTS_DIR / "exp_02_full_sweep_20251220_090809.json") as f:
        data = json.load(f)
    return data

def load_definitive():
    """Load the definitive statistical analysis."""
    with open(RESULTS_DIR / "exp_07_definitive_20251220_094854.json") as f:
        data = json.load(f)
    return data

def get_wolfram_class(rule):
    """Get Wolfram class for a rule number."""
    return WOLFRAM_CLASSES.get(rule, 'II')  # Default to Class II if unknown

def plot_phase_space(data):
    """Figure 1: PAC phase space with all 256 rules."""
    fig, ax = plt.subplots(figsize=(12, 8))

    embeddings = data['results']['all_embeddings']

    # Organize by class
    class_data = {'I': [], 'II': [], 'III': [], 'IV': [], 'UNKNOWN': []}

    for rule_str, embedding in embeddings.items():
        rule = int(rule_str)
        wclass = get_wolfram_class(rule)
        pa_ratio = embedding['pa_ratio']
        xi = embedding['xi']

        # Clip extreme values for better visualization
        pa_ratio = min(pa_ratio, 50)

        class_data[wclass].append((pa_ratio, xi, rule))

    # Plot each class with different colors
    colors = {'I': '#1f77b4', 'II': '#7f7f7f', 'III': '#ff7f0e', 'IV': '#d62728'}
    markers = {'I': 'o', 'II': '.', 'III': 's', 'IV': '*'}
    sizes = {'I': 50, 'II': 20, 'III': 60, 'IV': 200}
    zorders = {'I': 3, 'II': 1, 'III': 2, 'IV': 4}

    for wclass in ['II', 'III', 'I', 'IV']:  # Plot IV last (on top)
        if not class_data[wclass]:
            continue
        points = np.array([(p, x) for p, x, r in class_data[wclass]])
        ax.scatter(points[:, 0], points[:, 1],
                  c=colors[wclass], marker=markers[wclass],
                  s=sizes[wclass], alpha=0.7,
                  label=f'Class {wclass} (n={len(points)})',
                  zorder=zorders[wclass])

    # Add Xi target line
    ax.axhline(y=0.05705, color='red', linestyle='--', linewidth=2,
               label=f'Ξ = {XI_TARGET} (PAC target)', zorder=5)

    # Highlight Rule 110 and 124
    for rule in [110, 124]:
        embedding = embeddings[str(rule)]
        pa = min(embedding['pa_ratio'], 50)
        ax.annotate(f'Rule {rule}\n(Turing-complete)',
                   xy=(pa, embedding['xi']),
                   xytext=(pa + 5, embedding['xi'] + 0.1),
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                   fontsize=10, fontweight='bold', color='darkred',
                   zorder=6)

    ax.set_xlabel('P/A Ratio (clipped at 50)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ξ (Xi Balance Operator)', fontsize=12, fontweight='bold')
    ax.set_title('Elementary CA Rules in PAC Phase Space\n' +
                'Class IV Rules Cluster Near Ξ = 1.0571',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_1_phase_space.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_1_phase_space.pdf', bbox_inches='tight')
    print("[OK] Generated Figure 1: Phase space scatter plot")
    plt.close()

def plot_top_10_distance(data):
    """Figure 2: Top 10 rules by distance from Xi."""
    fig, ax = plt.subplots(figsize=(10, 6))

    embeddings = data['results']['all_embeddings']

    # Calculate distances
    distances = []
    for rule_str, embedding in embeddings.items():
        rule = int(rule_str)
        pa_ratio = embedding['pa_ratio']
        distance = abs(pa_ratio - XI_TARGET)
        wclass = get_wolfram_class(rule)
        distances.append((distance, rule, pa_ratio, wclass))

    # Sort and take top 10
    distances.sort()
    top_10 = distances[:10]

    rules = [r for d, r, p, w in top_10]
    dists = [d for d, r, p, w in top_10]
    classes = [w for d, r, p, w in top_10]

    # Color by class
    colors_list = [{'I': 'blue', 'II': 'gray', 'III': 'orange', 'IV': 'red'}[c]
                   for c in classes]

    bars = ax.barh(range(len(rules)), dists, color=colors_list, alpha=0.7)

    # Add labels
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([f'Rule {r} (Class {c})' for r, c in zip(rules, classes)])
    ax.set_xlabel('Distance from Xi = 1.0571', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Rules Closest to Xi\n' +
                'All Top 4 are Class IV (p < 8.58x10^-8)',
                fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Add vertical line at 1% threshold
    ax.axvline(x=XI_TARGET * 0.01, color='green', linestyle='--', linewidth=2,
               label='1% threshold', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_2_top_10_distance.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_2_top_10_distance.pdf', bbox_inches='tight')
    print("[OK] Generated Figure 2: Top 10 rules by Xi-proximity")
    plt.close()

def plot_class_distributions(data):
    """Figure 3: Distance distributions by Wolfram class (box plots)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    embeddings = data['results']['all_embeddings']

    # Organize distances by class
    class_distances = {'I': [], 'II': [], 'III': [], 'IV': []}

    for rule_str, embedding in embeddings.items():
        rule = int(rule_str)
        pa_ratio = embedding['pa_ratio']
        distance = abs(pa_ratio - XI_TARGET)
        wclass = get_wolfram_class(rule)
        if wclass in class_distances:
            class_distances[wclass].append(distance)

    # Create box plots
    data_to_plot = [class_distances[c] for c in ['I', 'II', 'III', 'IV']]
    labels = [f'Class {c}\n(n={len(class_distances[c])})' for c in ['I', 'II', 'III', 'IV']]

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showfliers=True, notch=True)

    # Color boxes
    colors = ['#1f77b4', '#7f7f7f', '#ff7f0e', '#d62728']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Distance from Xi = 1.0571', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Xi-Distance by Wolfram Class\n' +
                'Class IV Shows Lowest Mean and Variance',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # Add mean markers
    means = [np.mean(d) for d in data_to_plot]
    ax.plot(range(1, 5), means, 'D', color='black', markersize=8,
            label='Mean', zorder=3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_3_class_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_3_class_distributions.pdf', bbox_inches='tight')
    print("[OK] Generated Figure 3: Class distribution box plots")
    plt.close()

def plot_statistical_summary(definitive_data):
    """Figure 4: Statistical significance summary."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Enrichment factors
    metrics = ['Top 4\nall Class IV', 'Binomial\n(top 10)', 'Mann-Whitney U',
               'Combined\n(Fisher)']
    p_values = [8.58e-8, 5.7e-5, 0.00916, 1.42e-10]

    # Convert to -log10(p) for better visualization
    neg_log_p = [-np.log10(p) for p in p_values]

    colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow'
              for p in p_values]

    bars = ax1.barh(metrics, neg_log_p, color=colors, alpha=0.7)
    ax1.set_xlabel('-log10(p-value)', fontsize=12, fontweight='bold')
    ax1.set_title('Statistical Significance Tests', fontsize=12, fontweight='bold')
    ax1.axvline(x=-np.log10(0.05), color='green', linestyle='--', linewidth=2,
                label='p = 0.05 threshold', alpha=0.7)
    ax1.axvline(x=-np.log10(0.001), color='red', linestyle='--', linewidth=2,
                label='p = 0.001 threshold', alpha=0.7)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')

    # Right panel: Enrichment factors
    enrichment_data = {
        'Rules within\n1% of Xi': {'baseline': 1.56, 'class_iv': 66.7, 'factor': 42.7},
        'Rules within\n0.5% of Xi': {'baseline': 0.0, 'class_iv': 66.7, 'factor': float('inf')}
    }

    x = np.arange(len(enrichment_data))
    width = 0.35

    baselines = [d['baseline'] for d in enrichment_data.values()]
    class_ivs = [d['class_iv'] for d in enrichment_data.values()]

    ax2.bar(x - width/2, baselines, width, label='Random Baseline',
            color='gray', alpha=0.7)
    ax2.bar(x + width/2, class_ivs, width, label='Class IV',
            color='red', alpha=0.7)

    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Class IV Enrichment at Xi', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(enrichment_data.keys())
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add enrichment factors as text
    for i, (metric, data) in enumerate(enrichment_data.items()):
        if data['factor'] == float('inf'):
            text = 'inf x'
        else:
            text = f"{data['factor']:.1f}x"
        ax2.text(i, max(baselines[i], class_ivs[i]) + 5, text,
                ha='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_4_statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_4_statistical_summary.pdf', bbox_inches='tight')
    print("[OK] Generated Figure 4: Statistical significance summary")
    plt.close()

def main():
    """Generate all figures."""
    print("Generating publication figures for CA Xi Clustering paper...")
    print()

    # Create figures directory
    FIGURES_DIR.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    full_sweep = load_full_sweep()
    definitive = load_definitive()
    print("[OK] Data loaded")
    print()

    # Generate figures
    print("Generating figures...")
    plot_phase_space(full_sweep)
    plot_top_10_distance(full_sweep)
    plot_class_distributions(full_sweep)
    plot_statistical_summary(definitive)

    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()

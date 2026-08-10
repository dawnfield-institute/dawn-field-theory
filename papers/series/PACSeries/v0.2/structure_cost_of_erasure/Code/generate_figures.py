"""
Generate publication-quality figures for PACSeries Paper 1:
"The Structural Cost of Erasure"

All figures load data from ../Data/results/*.json.
Run: python generate_figures.py
Output: ../Figures/*.png (300 DPI, publication-ready)
"""

import json
import numpy as np
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("ERROR: matplotlib required. Install with: pip install matplotlib")
    raise SystemExit(1)

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'Data', 'results')
FIGURES_DIR = os.path.join(BASE_DIR, '..', 'Figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

LN_PHI = np.log((1 + np.sqrt(5)) / 2)  # 0.48121182...
PHI = (1 + np.sqrt(5)) / 2              # 1.61803398...


def load_json(pattern):
    """Load the first JSON file matching a prefix pattern."""
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith(pattern) and f.endswith('.json'):
            path = os.path.join(RESULTS_DIR, f)
            with open(path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
    print(f"  WARNING: No JSON matching '{pattern}*' in {RESULTS_DIR}")
    return None


def fig1_coupling_topology():
    """§4.4 — Structure metrics across binding regimes (from exp_09)."""
    data = load_json('exp_09')
    if data is None:
        return
    mb = data['main_results']['metric_breakdown']

    metrics = list(mb.keys())
    rbf_vals = [mb[m]['RBF'] for m in metrics]
    lin_vals = [mb[m]['LIN'] for m in metrics]
    unb_vals = [mb[m]['UNB'] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, rbf_vals, width, label='RBF (nonlinear)', color='#2196F3', edgecolor='white')
    ax.bar(x, lin_vals, width, label='Linear', color='#FF9800', edgecolor='white')
    ax.bar(x + width, unb_vals, width, label='Unbound', color='#9E9E9E', edgecolor='white')

    ax.set_ylabel('Metric value')
    ax.set_title(r'Binding Regime Determines Structure Metrics (§4.4, exp\_09)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha='right')
    ax.legend()

    # Annotate balance operator
    xi_val = data['main_results']['balance_operator']
    ax.text(0.98, 0.95, f'Balance operator \u039e = {xi_val:.4f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD'))

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig1_coupling_topology.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig2_information_budget():
    """§4.5 — The PAC information budget (from exp_01)."""
    data = load_json('exp_01')
    if data is None:
        return
    pac = data['main_results']['pac_check']
    A = pac['actual']
    xi = pac['xi']
    theta = pac['residual']
    P = pac['potential']

    components = [r'Transfer $A$', r'Structure $\xi$', r'Thermal $\Theta$']
    values = [A, xi, theta]
    colors = ['#4CAF50', '#2196F3', '#9E9E9E']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios': [2, 1]})

    # Stacked bar
    bottom = 0
    for comp, val, col in zip(components, values, colors):
        ax1.bar('1-bit erasure', val, bottom=bottom, color=col, label=comp,
                edgecolor='white', width=0.5)
        ax1.text(0, bottom + val/2, f'{val:.3f}', ha='center', va='center',
                fontweight='bold', fontsize=11, color='white')
        bottom += val

    ax1.set_ylabel('Information (bits)')
    ax1.set_title(r'PAC Budget: $P = A + \xi + \Theta$ (§4.5)')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=P, color='red', linestyle='--', alpha=0.5,
                label=f'P = {P:.3f} bit')
    ax1.legend(loc='upper right')
    ax1.set_xticks([])

    # Pie chart
    ax2.pie(values, labels=components, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 9})
    ax2.set_title('Proportion')

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_information_budget.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig3_decay_ratio_sweep():
    r"""§6 — A/(A+ξ) converges to ln(φ) at optimal decay ratio (from exp_05)."""
    data = load_json('exp_05')
    if data is None:
        return

    sweep = data['best_by_decay_ratio']
    ratios = [entry['decay_ratio'] for entry in sweep]
    results = [entry['result_ratio'] for entry in sweep]
    diffs = [entry['diff_pct'] for entry in sweep]

    # Sort by decay_ratio for clean line
    order = np.argsort(ratios)
    ratios = [ratios[i] for i in order]
    results = [results[i] for i in order]
    diffs = [diffs[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), height_ratios=[2, 1])

    # Top: A/(A+ξ) vs decay ratio
    ax1.plot(ratios, results, 'o-', color='#2196F3', markersize=8,
             linewidth=2, zorder=5)
    ax1.axhline(y=LN_PHI, color='red', linestyle='--', alpha=0.7,
                label=f'ln(\u03c6) = {LN_PHI:.4f}')

    # Highlight the closest-to-phi point
    best_idx = np.argmin(diffs)
    ax1.plot(ratios[best_idx], results[best_idx], 'D', color='#E91E63',
             markersize=12, zorder=6,
             label=f'Best: ratio={ratios[best_idx]:.2f}, dev={diffs[best_idx]:.2f}%')

    ax1.set_ylabel(r'$A/(A+\xi)$')
    ax1.set_title(r'Decay Ratio Sweep: Convergence to ln($\varphi$) (§6)')
    ax1.legend()

    # Bottom: deviation %
    colors = ['#E91E63' if i == best_idx else '#FF9800' for i in range(len(diffs))]
    ax2.bar([f'{r:.2f}' for r in ratios], diffs, color=colors, edgecolor='white')
    ax2.set_xlabel('Decay ratio (flip / correlation)')
    ax2.set_ylabel('% deviation from ln(\u03c6)')
    ax2.set_title('Deviation Minimizes Near \u03c6')

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_decay_ratio_sweep.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig4_cascade_amplification():
    """§10.3 — Cascade amplification of ξ (from exp_10)."""
    data = load_json('exp_10')
    if data is None:
        return

    gens = data['main_results']['cascade_table']['generations']
    summary = data['main_results']['summary']

    gen_nums = [g['gen'] for g in gens]
    cum_xi = [g['cumulative_xi'] for g in gens]
    per_gen_xi = [g['xi'] for g in gens]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: cumulative ξ across generations
    ax1.fill_between(gen_nums, cum_xi, alpha=0.3, color='#2196F3')
    ax1.plot(gen_nums, cum_xi, 'o-', color='#2196F3', markersize=6, linewidth=2)
    ax1.axhline(y=summary['single_event_xi'], color='#9E9E9E', linestyle='--',
                alpha=0.7, label=f'Single event \u03be = {summary["single_event_xi"]:.4f}')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel(r'Cumulative $\xi$ (bits)')
    ax1.set_title(r'Cascade Amplification of $\xi$ (§10.3)')
    ax1.legend()

    # Annotate final amplification
    amp = summary['amplification_ratio']
    p_val = summary['p_value']
    ax1.text(0.55, 0.25, f'{amp:.0f}\u00d7 amplification\np = {p_val:.2e}',
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             color='#E91E63', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FCE4EC'))

    # Right: per-generation ξ
    ax2.bar(gen_nums, per_gen_xi, color='#FF9800', edgecolor='white')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel(r'$\xi$ per generation (bits)')
    ax2.set_title('Structure Produced per Generation')

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_cascade_amplification.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig5_dense_sparse_regimes():
    """§11.2 — Dense vs sparse ξ-production regimes (from exp_11)."""
    data = load_json('exp_11')
    if data is None:
        return

    dense = data['main_results']['dense_regime']['ticks']
    sparse = data['main_results']['sparse_regime']['ticks']

    dense_ticks = [t['tick'] for t in dense]
    dense_xi = [t['xi'] for t in dense]
    dense_tc = [t['TC'] for t in dense]

    sparse_ticks = [t['tick'] for t in sparse]
    sparse_xi = [t['xi'] for t in sparse]
    sparse_tc = [t['TC'] for t in sparse]

    comp = data['main_results']['comparison']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: ξ per tick
    ax1.plot(dense_ticks, dense_xi, 'o-', color='#FF5722', linewidth=2,
             label='Dense (strong coupling)')
    ax1.plot(sparse_ticks, sparse_xi, 's-', color='#03A9F4', linewidth=2,
             label='Sparse (weak coupling)')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel(r'$\xi$ per tick (bits)')
    ax1.set_title(r'$\xi$ Production Rate by Regime (§11.2)')
    ax1.legend()

    # Annotate ratio
    ratio = comp['ratio']
    p_val = comp['p_value']
    ax1.text(0.55, 0.85, f'{ratio:.1f}\u00d7 ratio\np = {p_val:.2e}',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             color='#E91E63',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FCE4EC'))

    # Right: cumulative TC
    ax2.plot(dense_ticks, dense_tc, 'o-', color='#FF5722', linewidth=2,
             label='Dense')
    ax2.plot(sparse_ticks, sparse_tc, 's-', color='#03A9F4', linewidth=2,
             label='Sparse')
    ax2.set_xlabel('Tick')
    ax2.set_ylabel('Cumulative TC')
    ax2.set_title('Total Computation Diverges')
    ax2.legend()

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_dense_sparse_regimes.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig6_pac_ratio_stability():
    """§9.2 — PAC ratio stability: lag comparison + shuffled control (from exp_12, exp_13)."""
    d12 = load_json('exp_12')
    d13 = load_json('exp_13')
    if d12 is None or d13 is None:
        return

    # Panel A: multi-seed lag deviations from exp_12
    lag_stats = d12['main_results']['multi_seed_validation']['lag_stats']
    lag_labels = [f'Lag {s["lag"]}' for s in lag_stats]
    lag_means = [s['mean_dev_pct'] for s in lag_stats]
    lag_stds = [s['std_pct'] for s in lag_stats]

    # Panel B: ratio values — single-run lags + shuffled control from exp_13
    lags = d12['main_results']['single_run_lag_comparison']['lags']
    shuf = d13['main_results']['shuffled_control']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: deviation from ln(φ) by lag
    colors_a = ['#F44336', '#4CAF50', '#FF9800']
    bars = ax1.bar(lag_labels, lag_means, yerr=lag_stds, color=colors_a,
                   edgecolor='white', width=0.5, capsize=8, error_kw={'linewidth': 2})
    ax1.set_ylabel('Mean % deviation from ln(\u03c6)')
    ax1.set_title('(a) Lag=1 Minimizes Deviation (20 seeds)')

    # Annotate t-test
    ttest = d12['main_results']['multi_seed_validation']['statistical_tests']
    t01 = ttest['lag0_vs_lag1']
    ax1.text(0.5, 0.92, f'Lag0 vs Lag1: t={t01["t"]:.2f}, p={t01["p"]:.4f}',
             transform=ax1.transAxes, ha='center', fontsize=9, color='#666')

    # Panel B: A/(A+ξ) for each condition
    conditions = [f'Lag {l["lag"]}' for l in lags] + ['Shuffled']
    ratios = [l['A_over_A_plus_xi'] for l in lags] + [shuf['shuffled_ratio']]
    bar_colors = ['#4CAF50' if abs(r - LN_PHI) / LN_PHI < 0.10 else '#FF9800'
                  for r in ratios]
    bar_colors[-1] = '#F44336'  # Shuffled always red

    ax2.bar(conditions, ratios, color=bar_colors, edgecolor='white', width=0.5)
    ax2.axhline(y=LN_PHI, color='red', linestyle='--', alpha=0.7,
                label=f'ln(\u03c6) = {LN_PHI:.4f}')
    ax2.set_ylabel(r'$A/(A+\xi)$')
    ax2.set_title(r'(b) Ratio $A/(A+\xi)$: Shuffling Breaks Signal')
    ax2.set_ylim(0.35, 0.65)
    ax2.legend()

    # Annotate shuffled deviation
    ax2.annotate(f'{shuf["shuffled_deviation_pct"]:.1f}% dev\n(causal link broken)',
                xy=(4, shuf['shuffled_ratio']), xytext=(2.5, 0.62),
                arrowprops=dict(arrowstyle='->', color='#F44336'),
                fontsize=9, color='#F44336')

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig6_pac_ratio_stability.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    print("Generating figures for PACSeries Paper 1...")
    print(f"  Data: {os.path.abspath(RESULTS_DIR)}")
    print("=" * 50)

    fig1_coupling_topology()
    fig2_information_budget()
    fig3_decay_ratio_sweep()
    fig4_cascade_amplification()
    fig5_dense_sparse_regimes()
    fig6_pac_ratio_stability()

    print("=" * 50)
    print(f"All 6 figures saved to {os.path.abspath(FIGURES_DIR)}")

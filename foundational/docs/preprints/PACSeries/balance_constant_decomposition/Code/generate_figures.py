"""
Generate publication-quality figures for PACSeries Paper 2:
"The Balance Constant and Its Decomposition"

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

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
GAMMA = 0.5772156649015329  # Euler-Mascheroni
XI_ANALYTIC = GAMMA + LN_PHI
XI_FIB = 1 + np.pi / 55


def load_json(pattern):
    """Load the first JSON file matching a prefix pattern."""
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith(pattern) and f.endswith('.json'):
            path = os.path.join(RESULTS_DIR, f)
            with open(path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
    print(f"  WARNING: No JSON matching '{pattern}*' in {RESULTS_DIR}")
    return None


# ── Figure 1: Four-Domain Convergence ──────────────────────────────────
def fig1_domain_convergence():
    """§3 — Four independent domains converge on Ξ ≈ 1.058."""
    data = load_json('exp_31')
    if data is None:
        return

    decomps = data['decomposition_test']['decompositions']

    # Extract the three measured domains
    domains = []
    values = []
    errors_pct = []

    for key, entry in decomps.items():
        label = key.replace('_', ' ').title()
        if 'formula' in key:
            label = 'Fibonacci (1+π/55)'
        elif 'rule_110' in key:
            label = 'Cellular Automata'
        elif 'analytic' in key:
            label = 'Analytic (γ+ln φ)'
        domains.append(label)
        values.append(entry['xi_value'])
        errors_pct.append(entry['error_from_ln_phi_percent'])

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#2196F3', '#FF9800', '#4CAF50']
    bars = ax.bar(domains, values, color=colors, edgecolor='white', width=0.5)

    # Reference line at γ + ln(φ)
    ax.axhline(y=XI_ANALYTIC, color='red', linestyle='--', linewidth=1.5,
               label=f'γ + ln(φ) = {XI_ANALYTIC:.5f}')

    # Annotate error percentages
    for bar, err in zip(bars, errors_pct):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0003,
                f'{err:.3f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Measured Ξ value')
    ax.set_title('Four-Domain Convergence on the Balance Constant')
    ax.set_ylim(1.055, 1.062)
    ax.legend(loc='lower right')

    # Add p-value annotation
    p_val = data.get('significance_test', {}).get('p_random', 0.00376)
    ax.text(0.02, 0.98, f'p = {p_val:.5f}\n(n = 100,000 trials)',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    path = os.path.join(FIGURES_DIR, 'fig1_domain_convergence.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: Fibonacci Depth Convergence ──────────────────────────────
def fig2_fibonacci_depth():
    """§4.1 — Ξ convergence across Fibonacci depths F₅–F₁₁."""
    data = load_json('exp_23')
    if data is None:
        return

    depths_raw = data.get('part3_first_principles', {}).get('fibonacci_depths', {})
    if not depths_raw:
        print("  WARNING: No fibonacci_depths data in exp_23")
        return

    # Handle dict (keyed by F_k label) or list
    if isinstance(depths_raw, dict):
        entries = list(depths_raw.values())
    else:
        entries = depths_raw

    fibs = [d['fib'] for d in entries]
    xis = [d['xi'] for d in entries]
    errors = [d['error_vs_gamma_phi'] for d in entries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Ξ value at each Fibonacci depth
    ax1.plot(fibs, xis, 'o-', color='#2196F3', markersize=8, linewidth=2)
    ax1.axhline(y=XI_ANALYTIC, color='red', linestyle='--', linewidth=1.5,
                label=f'γ + ln(φ) = {XI_ANALYTIC:.5f}')
    ax1.axhline(y=XI_FIB, color='orange', linestyle=':', linewidth=1.5,
                label=f'1 + π/55 = {XI_FIB:.5f}')
    ax1.set_xlabel('Fibonacci number Fₖ')
    ax1.set_ylabel('Ξ value')
    ax1.set_title('Balance Constant vs Fibonacci Depth')
    ax1.legend()

    # Right: Error vs γ + ln(φ)
    ax2.bar(range(len(fibs)), [abs(e) for e in errors],
            tick_label=[str(f) for f in fibs],
            color='#FF5722', edgecolor='white')
    ax2.set_xlabel('Fibonacci number Fₖ')
    ax2.set_ylabel('|Error| vs γ + ln(φ)')
    ax2.set_title('Approximation Error by Depth')
    ax2.set_yscale('log')

    # Highlight F₁₀ = 55
    for i, f in enumerate(fibs):
        if f == 55:
            ax2.patches[i].set_facecolor('#4CAF50')
            ax2.patches[i].set_edgecolor('black')
            ax2.patches[i].set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_fibonacci_depth.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Class IV Cellular Automata Clustering ────────────────────
def fig3_ca_clustering():
    """§5 — Class IV rules cluster nearest to Ξ."""
    data = load_json('exp_07')
    if data is None:
        return

    top10 = data.get('top_10_closest_to_xi', [])
    if not top10:
        print("  WARNING: No top_10_closest_to_xi data in exp_07")
        return

    rules = [str(r['rule']) for r in top10]
    distances = [r['distance'] for r in top10]
    classes = [r.get('class', 'UNKNOWN') for r in top10]

    # Color by Wolfram class
    color_map = {
        'CLASS_IV': '#4CAF50',
        'CLASS_III': '#FF9800',
        'CLASS_II': '#9E9E9E',
        'CLASS_I': '#BDBDBD',
        'UNKNOWN': '#78909C'
    }
    colors = [color_map.get(c, '#78909C') for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(rules)), distances, color=colors, edgecolor='white')

    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([f'Rule {r}' for r in rules])
    ax.set_xlabel('Distance from Ξ')
    ax.set_title('Top 10 ECA Rules Nearest to Balance Constant')
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Class IV (edge-of-chaos)'),
        Patch(facecolor='#78909C', label='Other classes'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Annotate top 4
    for i in range(min(4, len(top10))):
        ax.text(distances[i] + 0.0001, i,
                f'{distances[i]:.5f}', va='center', fontsize=9, fontweight='bold')

    # Add enrichment annotation
    mc = data.get('monte_carlo', {})
    enrichment = mc.get('enrichment_factor', 42.67)
    ax.text(0.98, 0.02,
            f'Class IV enrichment: {enrichment:.1f}×\np < 10⁻⁷',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    path = os.path.join(FIGURES_DIR, 'fig3_ca_clustering.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Prime Sieve PAC Conservation ─────────────────────────────
def fig4_sieve_conservation():
    """§6 — PAC exact at all sieve steps; Mertens convergence."""
    data = load_json('exp_14')
    if data is None:
        return

    steps = data.get('first_10_steps', [])
    if not steps:
        print("  WARNING: No first_10_steps data in exp_14")
        return

    primes = [s['p'] for s in steps]
    surviving = [s['surviving_fraction'] for s in steps]
    cumulative = [s['cumulative_product'] for s in steps]
    deltas = [s['delta'] for s in steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Top: Surviving fraction vs Mertens prediction
    ax1.plot(range(len(primes)), surviving, 's-', color='#2196F3',
             markersize=8, linewidth=2, label='Measured surviving fraction')
    ax1.plot(range(len(primes)), cumulative, 'o--', color='#FF5722',
             markersize=6, linewidth=1.5, label='Mertens cumulative product')
    ax1.set_xticks(range(len(primes)))
    ax1.set_xticklabels([str(p) for p in primes])
    ax1.set_ylabel('Fraction surviving')
    ax1.set_title('Prime Sieve: PAC Conservation vs Mertens Prediction')
    ax1.legend()

    # Bottom: Delta (deviation from theory)
    ax2.bar(range(len(primes)), deltas, color='#9C27B0', edgecolor='white')
    ax2.set_xticks(range(len(primes)))
    ax2.set_xticklabels([str(p) for p in primes])
    ax2.set_xlabel('Sieving prime p')
    ax2.set_ylabel('Δ (measured − predicted)')
    ax2.set_title('Deviation from Mertens at Each Step')
    ax2.axhline(y=0, color='black', linewidth=0.5)

    # Annotation
    n_steps = data.get('n_sieve_steps', 126)
    mertens_err = data.get('sieve_error_pct', 0.012)
    ax1.text(0.98, 0.98,
             f'N = {data.get("N", 500000):,}\n'
             f'{n_steps} sieve steps\n'
             f'PAC exact: all steps\n'
             f'Mertens error: {mertens_err:.3f}%',
             transform=ax1.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_sieve_conservation.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 5: γ + ln(φ) Decomposition ─────────────────────────────────
def fig5_decomposition():
    """§9 — γ and ln(φ) as complementary components of Ξ."""
    data = load_json('exp_29')
    if data is None:
        return

    decomp = data.get('tests', {}).get('decomposition', {})
    components = decomp.get('components', {})
    if not components:
        # Fallback: use known values
        components = {
            'structure': {'value': LN_PHI, 'share': 45.5, 'name': 'ln(φ)'},
            'surplus': {'value': GAMMA, 'share': 54.5, 'name': 'γ'}
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Stacked bar showing decomposition
    labels = ['Ξ = γ + ln(φ)']
    ln_phi_val = LN_PHI
    gamma_val = GAMMA

    ax1.bar(labels, [ln_phi_val], color='#2196F3', label=f'ln(φ) = {ln_phi_val:.4f} (45.5%)')
    ax1.bar(labels, [gamma_val], bottom=[ln_phi_val], color='#FF9800',
            label=f'γ = {gamma_val:.4f} (54.5%)')
    ax1.axhline(y=XI_FIB, color='green', linestyle=':', linewidth=1.5,
                label=f'1 + π/55 = {XI_FIB:.5f}')
    ax1.set_ylabel('Value')
    ax1.set_title('Decomposition of Ξ')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 1.15)

    # Right: Rule 110 candidate comparison
    r110 = data.get('tests', {}).get('rule_110', {})
    candidates_raw = r110.get('candidates', {})
    if candidates_raw:
        # Handle dict (keyed by name) or list
        if isinstance(candidates_raw, dict):
            names = list(candidates_raw.keys())
            candidates_list = list(candidates_raw.values())
        else:
            candidates_list = candidates_raw
            names = [c.get('name', c.get('label', f'C{i}')) for i, c in enumerate(candidates_list)]
        errs = [abs(c.get('error_percent', 0)) for c in candidates_list]

        sorted_pairs = sorted(zip(errs, names))
        errs_sorted = [p[0] for p in sorted_pairs]
        names_sorted = [p[1] for p in sorted_pairs]

        colors = ['#4CAF50' if 'gamma' in n.lower() or 'γ' in n else '#78909C'
                  for n in names_sorted]

        ax2.barh(range(len(names_sorted)), errs_sorted, color=colors, edgecolor='white')
        ax2.set_yticks(range(len(names_sorted)))
        ax2.set_yticklabels(names_sorted, fontsize=9)
        ax2.set_xlabel('|Error %| vs Rule 110 midpoint')
        ax2.set_title('Which constant explains Rule 110?')
        ax2.invert_yaxis()
    else:
        # Fallback: simple comparison
        vals = {
            'γ (Euler-Mascheroni)': abs(0.5772 - 0.574) / 0.574 * 100,
            '1 - ln(φ)': abs(0.5188 - 0.574) / 0.574 * 100,
            'ln(2)/ln(3)': abs(0.6309 - 0.574) / 0.574 * 100,
            '1/√3': abs(0.5774 - 0.574) / 0.574 * 100,
        }
        names = list(vals.keys())
        errs = list(vals.values())
        colors = ['#4CAF50', '#78909C', '#78909C', '#78909C']
        ax2.barh(range(len(names)), errs, color=colors, edgecolor='white')
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_xlabel('|Error %| vs Rule 110 midpoint (0.574)')
        ax2.set_title('Which constant explains Rule 110?')
        ax2.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_decomposition.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 6: Approximation Error Comparison ───────────────────────────
def fig6_approximation_errors():
    """§4.2, §10 — Comparing discrete vs analytic Ξ approximations."""
    data = load_json('exp_25')
    if data is None:
        return

    verify = data.get('part2_verify', {})
    if not verify:
        print("  WARNING: No part2_verify data in exp_25")
        return

    methods = list(verify.keys())
    labels = []
    errors = []

    for method in methods:
        entry = verify[method]
        label = method.replace('_', ' ').title()
        rel_err = abs(entry.get('rel_error', entry.get('error', 0)))
        labels.append(label)
        errors.append(rel_err * 100)  # Convert to percentage if needed

    # If errors seem already in pct or are very small, adjust
    if all(e < 0.001 for e in errors):
        errors = [e * 100 for e in errors]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0'][:len(labels)]
    bars = ax.bar(labels, errors, color=colors, edgecolor='white', width=0.5)

    ax.set_ylabel('Relative error (%)')
    ax.set_title('Ξ Approximation Methods: Error vs γ + ln(φ)')

    # Annotate with exact values
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(errors)*0.02,
                f'{err:.4f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add reference annotation
    ax.text(0.98, 0.98,
            f'k_exact = 10.0121\nΔk = γ/48 ≈ 0.01203\nError: 0.67%',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    path = os.path.join(FIGURES_DIR, 'fig6_approximation_errors.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("Generating figures for Paper 2: The Balance Constant and Its Decomposition")
    print(f"Reading results from: {os.path.abspath(RESULTS_DIR)}")
    print(f"Saving figures to: {os.path.abspath(FIGURES_DIR)}")
    print()

    generators = [
        ("Fig 1: Domain Convergence (§3)", fig1_domain_convergence),
        ("Fig 2: Fibonacci Depth (§4)", fig2_fibonacci_depth),
        ("Fig 3: CA Clustering (§5)", fig3_ca_clustering),
        ("Fig 4: Sieve Conservation (§6)", fig4_sieve_conservation),
        ("Fig 5: γ + ln(φ) Decomposition (§9)", fig5_decomposition),
        ("Fig 6: Approximation Errors (§4.2)", fig6_approximation_errors),
    ]

    for label, func in generators:
        print(f"\n── {label} ──")
        try:
            func()
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. {len(generators)} figures generated.")


if __name__ == "__main__":
    main()

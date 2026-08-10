"""
exp_17 -- Random Graph Paradox: Why ADE Is Less Constrained

Milestone 13.5, Investigation Experiment

Central question: exp_15 T3 found that random graphs have LOWER rate variance
(CV=0.12) than ADE graphs (CV=0.19) at large rank. This is counterintuitive:
ADE graphs are highly structured, so their rates should be MORE constrained,
not less. What explains the paradox?

Hypothesis: ADE graphs span multiple FAMILIES (A, D, E) with different topologies
(chain, branching, exceptional), while random graphs at high density are all
topologically similar (high connectivity washes out structural differences).
The ADE variance is BETWEEN-family, not within-family.

Tests:
  T1: Decompose ADE variance into within-family and between-family components
  T2: Random graph rate CV as a function of density (does high density -> low CV?)
  T3: Random graphs vs ADE at MATCHED sizes (control for size distribution)
  T4: Spectral radius distribution: ADE < 2 constraint vs random unconstrained
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from identity_complement import (
    PHI, LN_PHI,
    DynkinDiagram,
    max_deformation_rate,
    generate_random_connected_graph,
    save_m13_results, _convert_numpy,
)


def test_T1_variance_decomposition():
    """T1: Decompose ADE rate variance into within-family and between-family."""

    # Collect rates by family for rank 10-20
    family_rates = {'A': [], 'D': []}

    for n in range(10, 21):
        for family in ['A', 'D']:
            d = DynkinDiagram(family, n)
            r = max_deformation_rate(d.adjacency)
            family_rates[family].append({'rank': n, 'rate': float(r)})

    # Within-family variance
    a_values = [r['rate'] for r in family_rates['A']]
    d_values = [r['rate'] for r in family_rates['D']]

    a_cv = float(np.std(a_values) / np.mean(a_values))
    d_cv = float(np.std(d_values) / np.mean(d_values))

    print(f"  A-family (rank 10-20): mean={np.mean(a_values):.4f}, CV={a_cv:.4f}")
    print(f"  D-family (rank 10-20): mean={np.mean(d_values):.4f}, CV={d_cv:.4f}")

    # Between-family: variance of family means
    family_means = [np.mean(a_values), np.mean(d_values)]
    between_cv = float(np.std(family_means) / np.mean(family_means))

    print(f"  Between-family CV: {between_cv:.4f}")

    # Combined (mixing A and D as exp_15 T3 did)
    all_ade = a_values + d_values
    combined_cv = float(np.std(all_ade) / np.mean(all_ade))
    print(f"  Combined ADE CV: {combined_cv:.4f}")

    # The paradox is explained if between-family variance dominates
    within_avg_cv = (a_cv + d_cv) / 2
    between_dominates = between_cv > within_avg_cv

    print(f"  Within-family avg CV: {within_avg_cv:.4f}")
    print(f"  Between > Within: {between_dominates}")

    result = {
        'test': 'T1_variance_decomposition',
        'a_family': family_rates['A'],
        'd_family': family_rates['D'],
        'a_cv': a_cv,
        'd_cv': d_cv,
        'between_family_cv': between_cv,
        'combined_cv': combined_cv,
        'within_avg_cv': within_avg_cv,
        'between_dominates': between_dominates,
        'note': 'If between-family variance dominates, the ADE "high variance" is a '
                'mixing effect: A and D families have different rates, not that individual '
                'families are noisy.',
        'PASS': between_dominates,
    }
    return result


def test_T2_random_cv_vs_density():
    """T2: Random graph rate CV as a function of density."""

    densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    n_graphs = 15
    n_vertices = 15

    density_results = []

    for dens in densities:
        rates = []
        for seed in range(n_graphs):
            try:
                G = generate_random_connected_graph(n_vertices, density=dens, seed=seed + 1000)
                r = max_deformation_rate(G)
                rates.append(float(r))
            except RuntimeError:
                pass

        if len(rates) >= 5:
            cv = float(np.std(rates) / np.mean(rates))
            mean_rate = float(np.mean(rates))
        else:
            cv = float('nan')
            mean_rate = float('nan')

        density_results.append({
            'density': dens,
            'n_successful': len(rates),
            'mean_rate': mean_rate,
            'cv': cv,
        })
        print(f"    density={dens:.1f}: n={len(rates)}, mean={mean_rate:.4f}, CV={cv:.4f}")

    # Check if CV decreases with density (higher density -> more similar -> lower CV)
    valid = [(r['density'], r['cv']) for r in density_results if not np.isnan(r['cv'])]
    if len(valid) >= 4:
        densities_arr = np.array([v[0] for v in valid])
        cvs_arr = np.array([v[1] for v in valid])
        # Correlation between density and CV
        corr = float(np.corrcoef(densities_arr, cvs_arr)[0, 1])
    else:
        corr = 0.0

    cv_decreases_with_density = corr < -0.3
    print(f"\n  Density-CV correlation: r={corr:.4f}")
    print(f"  CV decreases with density: {cv_decreases_with_density}")

    result = {
        'test': 'T2_random_cv_vs_density',
        'n_vertices': n_vertices,
        'density_results': density_results,
        'density_cv_correlation': corr,
        'cv_decreases_with_density': cv_decreases_with_density,
        'note': 'If CV decreases with density, the paradox is explained: high-density '
                'random graphs are topologically similar (many edges -> similar spectra), '
                'so their rates converge. ADE at large rank are sparse (2 edges/vertex), '
                'preserving structural diversity.',
        'PASS': cv_decreases_with_density,
    }
    return result


def test_T3_matched_size_comparison():
    """T3: Random vs ADE at matched sizes (controlling for size distribution)."""

    # The exp_15 comparison used mixed sizes (randint 10-20 for random vs
    # exactly 10-20 for ADE). Let's match exactly.

    sizes = [10, 12, 14, 16, 18, 20]
    n_random_per_size = 5

    ade_rates_by_size = {}
    random_rates_by_size = {}

    for n in sizes:
        # ADE rates at this size
        ade_rates = []
        for family in ['A', 'D']:
            d = DynkinDiagram(family, n)
            r = max_deformation_rate(d.adjacency)
            ade_rates.append(float(r))
        ade_rates_by_size[n] = ade_rates

        # Random rates at this size, multiple densities
        rand_rates = []
        for seed in range(n_random_per_size):
            for dens in [0.2, 0.3, 0.4]:
                try:
                    G = generate_random_connected_graph(n, density=dens, seed=seed + n * 100)
                    r = max_deformation_rate(G)
                    rand_rates.append(float(r))
                except RuntimeError:
                    pass
        random_rates_by_size[n] = rand_rates

        print(f"    n={n}: ADE rates={[f'{r:.3f}' for r in ade_rates]}, "
              f"random mean={np.mean(rand_rates):.3f} (n={len(rand_rates)})")

    # Compute CVs with matched sizes
    all_ade = []
    all_random = []
    for n in sizes:
        all_ade.extend(ade_rates_by_size[n])
        all_random.extend(random_rates_by_size[n])

    ade_cv = float(np.std(all_ade) / np.mean(all_ade))
    random_cv = float(np.std(all_random) / np.mean(all_random))

    print(f"\n  Matched-size ADE CV: {ade_cv:.4f} ({len(all_ade)} values)")
    print(f"  Matched-size random CV: {random_cv:.4f} ({len(all_random)} values)")

    # With matched sizes and mixed densities, does the paradox persist?
    paradox_persists = random_cv < ade_cv

    result = {
        'test': 'T3_matched_size_comparison',
        'sizes': sizes,
        'ade_rates_by_size': {str(k): v for k, v in ade_rates_by_size.items()},
        'random_rates_by_size': {str(k): v for k, v in random_rates_by_size.items()},
        'ade_cv': ade_cv,
        'random_cv': random_cv,
        'paradox_persists': paradox_persists,
        'note': 'Size-matched comparison: if random still has lower CV than ADE, '
                'the paradox is robust (not a size-distribution artifact). If ADE '
                'now has lower CV, the original result was confounded by size mixing.',
        'PASS': not paradox_persists,  # PASS if paradox is resolved by controlling size
    }
    return result


def test_T4_spectral_radius_constraint():
    """T4: ADE spectral radius < 2 constraint vs random unconstrained."""

    # ADE Dynkin diagrams have spectral radius < 2 (by definition).
    # Random graphs have no such constraint. Does this explain rate differences?

    sizes = list(range(10, 21))

    ade_data = []
    for n in sizes:
        for family in ['A', 'D']:
            d = DynkinDiagram(family, n)
            adj = d.adjacency
            spec_rad = float(np.max(np.abs(np.linalg.eigvalsh(adj))))
            rate = max_deformation_rate(adj)
            ade_data.append({
                'name': f'{family}_{n}',
                'n': n,
                'spectral_radius': spec_rad,
                'rate': float(rate),
            })

    random_data = []
    for seed in range(30):
        n = 10 + (seed % 11)
        try:
            G = generate_random_connected_graph(n, density=0.3, seed=seed + 500)
            spec_rad = float(np.max(np.abs(np.linalg.eigvalsh(G))))
            rate = max_deformation_rate(G)
            random_data.append({
                'seed': seed,
                'n': n,
                'spectral_radius': spec_rad,
                'rate': float(rate),
            })
        except RuntimeError:
            pass

    # Statistics
    ade_spec_rads = [d['spectral_radius'] for d in ade_data]
    random_spec_rads = [d['spectral_radius'] for d in random_data]
    ade_rates = [d['rate'] for d in ade_data]
    random_rates = [d['rate'] for d in random_data]

    ade_spec_mean = float(np.mean(ade_spec_rads))
    random_spec_mean = float(np.mean(random_spec_rads))

    print(f"  ADE spectral radius: mean={ade_spec_mean:.3f}, "
          f"max={max(ade_spec_rads):.3f}, all < 2: {all(s < 2 for s in ade_spec_rads)}")
    print(f"  Random spectral radius: mean={random_spec_mean:.3f}, "
          f"max={max(random_spec_rads):.3f}")

    # Correlation between spectral radius and rate
    all_spec = ade_spec_rads + random_spec_rads
    all_rates_combined = ade_rates + random_rates
    spec_rate_corr = float(np.corrcoef(all_spec, all_rates_combined)[0, 1])

    print(f"  Spectral radius - rate correlation: r={spec_rate_corr:.4f}")

    # The ADE spectral radius constraint (<2) restricts rates to a narrow band.
    # But WITHIN that band, the A/D family difference creates variance.
    # Random graphs at density 0.3 have higher spectral radii but more uniform
    # topology -> lower rate variance.
    ade_all_below_2 = all(s < 2.0 for s in ade_spec_rads)
    random_some_above_2 = any(s > 2.0 for s in random_spec_rads)
    strong_correlation = abs(spec_rate_corr) > 0.5

    print(f"  ADE all below 2: {ade_all_below_2}")
    print(f"  Random some above 2: {random_some_above_2}")
    print(f"  Strong rate-spectral correlation: {strong_correlation}")

    result = {
        'test': 'T4_spectral_radius_constraint',
        'ade_data': ade_data,
        'random_data': random_data,
        'ade_spec_mean': ade_spec_mean,
        'random_spec_mean': random_spec_mean,
        'spec_rate_correlation': spec_rate_corr,
        'ade_all_below_2': ade_all_below_2,
        'random_some_above_2': random_some_above_2,
        'strong_correlation': strong_correlation,
        'note': 'ADE spectral radius < 2 (defining property). Random graphs are '
                'unconstrained. If rate correlates with spectral radius AND random '
                'graphs cluster at a single spectral radius, the paradox is explained.',
        'PASS': strong_correlation and random_some_above_2,
    }
    return result


def main():
    print("=" * 70)
    print("EXP 17 -- Random Graph Paradox")
    print("Milestone 13.5, Investigation Experiment")
    print("=" * 70)

    results = {}
    score = 0
    total = 4

    for name, test_fn in [
        ('T1', test_T1_variance_decomposition),
        ('T2', test_T2_random_cv_vs_density),
        ('T3', test_T3_matched_size_comparison),
        ('T4', test_T4_spectral_radius_constraint),
    ]:
        print(f"\n--- {name}: {test_fn.__doc__.strip().split(chr(10))[0]} ---")
        r = test_fn()
        results[name] = r
        if r['PASS']:
            score += 1
            print(f"  PASS")
        else:
            print(f"  FAIL")

    final = {
        'experiment': 'exp_17_random_graph_paradox',
        'milestone': 'milestone13.5',
        'block': 'investigation',
        'version': 'v0.1',
        'score': score,
        'total': total,
        'tests': results,
    }

    filename = save_m13_results('exp_17_random_graph_paradox', _convert_numpy(final))
    print(f"\nScore: {score}/{total}")
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()

"""
exp_13 -- Perspectival Barrier: Identity Reconciliation

Milestone R, Block C (Novel Physics)

Thesis: The decay barrier isn't just topological (all d edges decouple
simultaneously). It's PERSPECTIVAL -- the information-theoretic gap between
how a vertex sees itself (local identity / SEC) and how the global system
accounts for it (PAC conservation). The "tunneling" is identity reconciliation.

Jensen-Shannon divergence between a vertex's local random-walk distribution
and global equilibrium measures this perspective gap. JSD captures more than
raw degree: it encodes the vertex's POSITION in the graph, not just its
connectivity count. Symmetric positions (hubs in D_n) have lower JSD because
their local view is closer to global; asymmetric positions (endpoints) have
higher JSD because their local view is concentrated.

Connected to M13 definitional parallax: different observers compute different
complements of the same target. The barrier is the parallax between local
identity and global accounting.

Tests:
  T1: JSD predicts barrier FPT better than degree
  T2: GN scaling from perspectival barrier (hub vertex, positive slope)
  T3: Symmetric positions have lower perspective divergence
  T4: Perspectival barrier walk outperforms topological barrier walk
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from radiation_physics import (
    PHI, INV_PHI, XI_BALANCE, LN_PHI, LN2, PI,
    pac_deficit,
    stochastic_barrier_walk,
    perspective_divergence,
    perspectival_barrier_walk,
    ade_graphs, redistribute_on_graph,
    save_mr_results,
)


def test_T1_jsd_predicts_fpt_better_than_degree():
    """T1: JSD correlates with barrier FPT better than degree."""
    print("\n  T1: Perspective divergence predicts FPT better than degree")
    results = {'description': 'R^2(JSD vs log(FPT)) > R^2(degree vs log(FPT)) for majority of graphs'}

    n_trials = 50
    per_graph = []

    for name, adj in ade_graphs(max_rank=8):
        n = adj.shape[0]
        if n < 4:
            continue

        graph_jsds = []
        graph_degrees = []
        graph_fpts = []

        for v in range(n):
            degree = int(np.sum(adj[v] > 0))
            jsd = perspective_divergence(adj, v, horizon=2)

            # Run barrier walks for FPT
            fpts = []
            for trial in range(n_trials):
                initial = np.ones(n) / n
                initial[v] += 1.0 / n
                initial = initial / np.sum(initial)

                result = stochastic_barrier_walk(
                    adj, v, initial,
                    noise_amplitude=0.01,
                    max_steps=5000,
                    seed=trial + v * 10000 + abs(hash(name)) % 100000,
                )
                if result['converged']:
                    fpts.append(result['first_passage_time'])

            if len(fpts) >= n_trials // 4:
                med_fpt = np.median(fpts)
                graph_jsds.append(jsd)
                graph_degrees.append(degree)
                graph_fpts.append(med_fpt)

        # Per-graph R^2 comparison
        if len(graph_fpts) >= 3 and len(set(graph_jsds)) > 1 and len(set(graph_degrees)) > 1:
            log_fpts = np.log10(np.array(graph_fpts) + 1)

            _, _, r_jsd, _, _ = stats.linregress(graph_jsds, log_fpts)
            r2_jsd = r_jsd ** 2

            _, _, r_deg, _, _ = stats.linregress(graph_degrees, log_fpts)
            r2_deg = r_deg ** 2

            jsd_wins = r2_jsd > r2_deg
            per_graph.append({
                'graph': name,
                'r2_jsd': round(float(r2_jsd), 4),
                'r2_degree': round(float(r2_deg), 4),
                'jsd_wins': jsd_wins,
                'n_vertices': len(graph_fpts),
            })
            print(f"    {name}: R^2(JSD)={r2_jsd:.4f} vs R^2(deg)={r2_deg:.4f} "
                  f"-> {'JSD' if jsd_wins else 'degree'}")
        elif len(graph_fpts) >= 3 and len(set(graph_degrees)) <= 1:
            # All same degree (e.g. chain interior) -- JSD wins by default
            # since degree has zero variance
            log_fpts = np.log10(np.array(graph_fpts) + 1)
            if len(set(graph_jsds)) > 1:
                _, _, r_jsd, _, _ = stats.linregress(graph_jsds, log_fpts)
                r2_jsd = r_jsd ** 2
            else:
                r2_jsd = 0.0
            per_graph.append({
                'graph': name,
                'r2_jsd': round(float(r2_jsd), 4),
                'r2_degree': 0.0,
                'jsd_wins': r2_jsd > 0,
                'n_vertices': len(graph_fpts),
                'note': 'degree has zero variance',
            })
            print(f"    {name}: R^2(JSD)={r2_jsd:.4f} vs R^2(deg)=N/A (constant degree) "
                  f"-> {'JSD' if r2_jsd > 0 else 'tie'}")

    jsd_win_count = sum(1 for pg in per_graph if pg['jsd_wins'])
    total = len(per_graph)
    passed = total > 0 and jsd_win_count > total / 2

    results['per_graph'] = per_graph
    results['jsd_wins'] = jsd_win_count
    results['total_graphs'] = total
    results['PASS'] = passed
    print(f"    JSD wins {jsd_win_count}/{total} graphs")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T2_perspectival_gn_scaling():
    """T2: Perspectival barrier walk produces GN-like scaling on hub vertex."""
    print("\n  T2: GN scaling from perspectival barrier (D_8 hub)")
    results = {'description': 'log(FPT) vs 1/sqrt(deficit): positive slope AND R^2 > 0.7'}

    # Use D_8 graph, target the HUB vertex (degree 3)
    test_graph = None
    for name, adj in ade_graphs(max_rank=8):
        if name == 'D_8':
            test_graph = adj
            break

    if test_graph is None:
        results['PASS'] = False
        results['error'] = 'D_8 not found'
        print("    ERROR: D_8 graph not found")
        return results

    n = test_graph.shape[0]
    # Hub in D_n is at index n-3
    target = n - 3
    target_degree = int(np.sum(test_graph[target] > 0))
    target_jsd = perspective_divergence(test_graph, target, horizon=2)
    print(f"    Graph: D_8, target vertex {target} (degree {target_degree}, JSD={target_jsd:.6f})")

    deficit_levels = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    n_trials = 100
    median_fpts = []
    actual_deficits = []

    for pert in deficit_levels:
        fpts = []
        for trial in range(n_trials):
            initial = np.ones(n) / n
            initial[target] += pert / n
            initial = initial / np.sum(initial)

            result = perspectival_barrier_walk(
                test_graph, target, initial,
                horizon=2,
                noise_amplitude=0.01,
                jsd_threshold=0.005,
                max_steps=8000,
                seed=trial * 1000 + int(pert * 100),
            )
            if result['converged']:
                fpts.append(result['first_passage_time'])

        if len(fpts) >= n_trials // 4:
            med = np.median(fpts)
            median_fpts.append(med)
            actual_deficits.append(pert)
            print(f"      deficit={pert:.1f}: median_fpt={med:.0f} "
                  f"({len(fpts)}/{n_trials} converged)")
        else:
            print(f"      deficit={pert:.1f}: too few converged ({len(fpts)}/{n_trials})")

    # Regression: log(FPT) vs 1/sqrt(deficit)
    slope = float('nan')
    r2 = 0.0
    passed = False
    if len(median_fpts) >= 4:
        x = 1.0 / np.sqrt(np.array(actual_deficits))
        y = np.log10(np.array(median_fpts))
        slope, intercept, r, p, se = stats.linregress(x, y)
        r2 = r ** 2
        passed = slope > 0 and r2 > 0.7

    results['target_vertex'] = target
    results['target_degree'] = target_degree
    results['target_jsd'] = round(float(target_jsd), 6)
    results['slope'] = round(float(slope), 4) if not np.isnan(slope) else None
    results['r2'] = round(float(r2), 4)
    results['deficit_levels'] = [float(d) for d in actual_deficits]
    results['median_fpts'] = [float(f) for f in median_fpts]
    results['n_trials'] = n_trials
    results['PASS'] = passed
    print(f"    log(FPT) vs 1/sqrt(deficit): slope={slope:.4f}, R^2={r2:.4f}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


def test_T3_symmetry_reduces_divergence():
    """T3: Hub vertices (high symmetry) have lower JSD than endpoints."""
    print("\n  T3: Symmetry reduces perspectival divergence")
    results = {'description': 'Hub JSD < mean endpoint JSD for D_4, D_5, D_6'}

    target_graphs = ['D_4', 'D_5', 'D_6']
    details = []
    all_pass = True

    for graph_name in target_graphs:
        adj = None
        for name, a in ade_graphs(max_rank=8):
            if name == graph_name:
                adj = a
                break
        if adj is None:
            continue

        n = adj.shape[0]
        hub_idx = n - 3  # D_n hub is at index n-3

        vertex_jsds = []
        for v in range(n):
            jsd = perspective_divergence(adj, v, horizon=2)
            degree = int(np.sum(adj[v] > 0))
            vertex_jsds.append({
                'vertex': v,
                'degree': degree,
                'jsd': round(jsd, 6),
                'is_hub': v == hub_idx,
                'is_endpoint': degree == 1,
            })

        hub_jsd = vertex_jsds[hub_idx]['jsd']
        endpoint_jsds = [vj['jsd'] for vj in vertex_jsds if vj['is_endpoint']]
        mean_endpoint_jsd = float(np.mean(endpoint_jsds)) if endpoint_jsds else 0.0

        ok = hub_jsd < mean_endpoint_jsd
        if not ok:
            all_pass = False

        details.append({
            'graph': graph_name,
            'hub_vertex': hub_idx,
            'hub_degree': int(np.sum(adj[hub_idx] > 0)),
            'hub_jsd': round(float(hub_jsd), 6),
            'mean_endpoint_jsd': round(float(mean_endpoint_jsd), 6),
            'all_vertex_jsds': vertex_jsds,
            'hub_lower': ok,
        })
        print(f"    {graph_name}: hub JSD={hub_jsd:.6f}, "
              f"mean endpoint JSD={mean_endpoint_jsd:.6f} "
              f"-> {'hub < endpoint' if ok else 'FAIL: hub >= endpoint'}")

    results['details'] = details
    results['PASS'] = all_pass and len(details) == len(target_graphs)
    print(f"    -> {'PASS' if results['PASS'] else 'FAIL'}")
    return results


def test_T4_perspectival_vs_topological():
    """T4: Perspectival FPT correlates with deficit better than topological FPT."""
    print("\n  T4: Perspectival barrier vs topological barrier (head-to-head)")
    results = {'description': 'Perspectival R^2 > topological R^2 for majority of graphs'}

    deficit_levels = [0.5, 1.0, 2.0, 3.0, 5.0]
    n_trials = 100
    per_graph = []

    test_graphs = ['D_5', 'D_6', 'D_7', 'E_6', 'E_7']

    for graph_name in test_graphs:
        adj = None
        for name, a in ade_graphs(max_rank=8):
            if name == graph_name:
                adj = a
                break
        if adj is None:
            continue

        n = adj.shape[0]
        # Target the highest-degree vertex (hub)
        degrees_arr = np.sum(adj > 0, axis=1).astype(int)
        target = int(np.argmax(degrees_arr))
        target_degree = int(degrees_arr[target])

        print(f"    {graph_name}: target v={target} (degree {target_degree})")

        perspectival_fpts = []
        topological_fpts = []
        persp_deficits = []
        topo_deficits = []

        for pert in deficit_levels:
            p_fpts = []
            t_fpts = []

            for trial in range(n_trials):
                initial = np.ones(n) / n
                initial[target] += pert / n
                initial = initial / np.sum(initial)

                seed = trial * 1000 + int(pert * 100)

                # Perspectival
                r_persp = perspectival_barrier_walk(
                    adj, target, initial,
                    horizon=2, noise_amplitude=0.01,
                    jsd_threshold=0.005,
                    max_steps=5000,
                    seed=seed,
                )
                if r_persp['converged']:
                    p_fpts.append(r_persp['first_passage_time'])

                # Topological
                r_topo = stochastic_barrier_walk(
                    adj, target, initial,
                    noise_amplitude=0.01,
                    max_steps=5000,
                    seed=seed,
                )
                if r_topo['converged']:
                    t_fpts.append(r_topo['first_passage_time'])

            if len(p_fpts) >= n_trials // 4:
                perspectival_fpts.append(float(np.median(p_fpts)))
                persp_deficits.append(pert)
            if len(t_fpts) >= n_trials // 4:
                topological_fpts.append(float(np.median(t_fpts)))
                topo_deficits.append(pert)

        # Compare R^2 for GN-like scaling
        r2_persp = 0.0
        r2_topo = 0.0

        if len(perspectival_fpts) >= 3:
            x = 1.0 / np.sqrt(np.array(persp_deficits))
            y = np.log10(np.array(perspectival_fpts))
            _, _, r, _, _ = stats.linregress(x, y)
            r2_persp = r ** 2

        if len(topological_fpts) >= 3:
            x = 1.0 / np.sqrt(np.array(topo_deficits))
            y = np.log10(np.array(topological_fpts))
            _, _, r, _, _ = stats.linregress(x, y)
            r2_topo = r ** 2

        persp_wins = r2_persp > r2_topo
        per_graph.append({
            'graph': graph_name,
            'r2_perspectival': round(float(r2_persp), 4),
            'r2_topological': round(float(r2_topo), 4),
            'perspectival_wins': persp_wins,
            'persp_convergence': f"{len(perspectival_fpts)}/{len(deficit_levels)}",
            'topo_convergence': f"{len(topological_fpts)}/{len(deficit_levels)}",
        })
        print(f"      R^2 perspectival={r2_persp:.4f}, topological={r2_topo:.4f} "
              f"-> {'perspectival' if persp_wins else 'topological'}")

    persp_win_count = sum(1 for pg in per_graph if pg['perspectival_wins'])
    total = len(per_graph)
    passed = total > 0 and persp_win_count > total / 2

    results['per_graph'] = per_graph
    results['perspectival_wins'] = persp_win_count
    results['total_graphs'] = total
    results['PASS'] = passed
    print(f"    Perspectival wins {persp_win_count}/{total}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("exp_13: Perspectival Barrier -- Identity Reconciliation")
    print("=" * 60)

    t1 = test_T1_jsd_predicts_fpt_better_than_degree()
    t2 = test_T2_perspectival_gn_scaling()
    t3 = test_T3_symmetry_reduces_divergence()
    t4 = test_T4_perspectival_vs_topological()

    score = sum(1 for t in [t1, t2, t3, t4] if t['PASS'])
    print(f"\n{'=' * 60}")
    print(f"  Overall: {score}/4")
    print(f"{'=' * 60}")

    data = {
        'experiment': 'exp_13_perspectival_barrier',
        'timestamp': datetime.now().isoformat(),
        'block': 'C',
        'thesis': 'The decay barrier is PERSPECTIVAL: the information-theoretic gap '
                  'between local identity (SEC) and global accounting (PAC). '
                  'JSD between local random-walk distribution and global equilibrium '
                  'predicts FPT better than degree, and gives GN-like scaling when '
                  'targeting high-degree hub vertices. Connected to M13 definitional '
                  'parallax: the barrier is identity reconciliation.',
        'test_results': {'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4},
        'overall_score': f"{score}/4",
    }
    save_mr_results(data, 'exp_13_perspectival_barrier')
